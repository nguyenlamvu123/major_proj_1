"""Nhánh 07 — công cụ kiểm chứng cấu hình & một batch forward trước khi train (parseq).

Vai trò: **không** huấn luyện lại logic của parseq (dùng `third_party/parseq/train.py`).
File này chỉ:

1. **Compose** config hydra từ `configs/hydra/` (tự chứa: `tm_crnn`, `tm_trba`,
   `tm_parseq`, `tm_svtrctc`, ...).
2. **Dry-run**: instantiate model + datamodule, lấy **1 batch** train, tính loss
   (CTC/CE tuỳ hệ), in model summary (số tham số) và decode thử → bảo đảm wiring
   đúng trước khi mất 30 phút train trên Colab.
3. In ra **lệnh train thật** tương ứng.

Train thật (lặp lại logic của parseq, không sửa code bên thứ ba)::

    python3 third_party/parseq/train.py --config-dir configs/hydra --config-name tm_crnn

CLI
---
    python3 -m trademark_ocr.train_utils --config tm_crnn --dry-run
    python3 -m trademark_ocr.train_utils --config tm_crnn --dry-run -o trainer.accelerator=cpu
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence

# Shim NumPy-2 cho imgaug (RandAugment) — cần TRƯỚC khi import strhub.data.augment.
from trademark_ocr._numpy2_compat import apply_numpy2_compat  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
PARSEQ_ROOT = REPO_ROOT / "third_party" / "parseq"
CFG_DIR = REPO_ROOT / "configs" / "hydra"

if str(PARSEQ_ROOT) not in sys.path:
    sys.path.insert(0, str(PARSEQ_ROOT))

DEFAULT_DATA_ROOT: str = os.environ.get(
    "DATA_ROOT", "/media/zaibachkhoa/AC84-F995/trademark_data"
)


def _require_strhub() -> None:
    if not (PARSEQ_ROOT / "strhub" / "__init__.py").exists():
        raise SystemExit(
            "Không thấy third_party/parseq/strhub — hãy:\n"
            f"  git clone https://github.com/baudm/parseq {PARSEQ_ROOT}"
        )


def available_configs() -> list[str]:
    return sorted(p.stem for p in CFG_DIR.glob("tm_*.yaml"))


def compose_config(name: str, overrides: Sequence[str] = ()):
    """Compose config `name` từ `configs/hydra` (hydra 1.2)."""
    import hydra
    from hydra import compose, initialize_config_dir

    if not (CFG_DIR / f"{name}.yaml").exists():
        raise SystemExit(
            f"Không tìm thấy config {name!r}. Có: {', '.join(available_configs())}")
    # hydra là singleton: phải đóng initialize trước lần compose kế.
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base="1.2"):
        cfg = compose(config_name=name, overrides=list(overrides))
    # root_dir đã tuyệt đối nhờ ${oc.env:DATA_ROOT,...}; nhưng vẫn normalize.
    if cfg.data.get("root_dir"):
        cfg.data.root_dir = str(Path(str(cfg.data.root_dir)).expanduser().resolve())
    return cfg


def print_summary(model: Any) -> int:
    from pytorch_lightning.utilities.model_summary import summarize

    print(summarize(model, max_depth=2))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[dry-run] total params = {n_params:,}")
    return n_params


def dry_run(cfg: Any, *, num_batches: int = 1, device: str = "cpu",
            num_images: int = 16) -> dict[str, Any]:
    import hydra
    import torch

    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)
    n_params = print_summary(model)

    datamodule = hydra.utils.instantiate(cfg.data)
    loader = datamodule.train_dataloader()
    print(f"[dry-run] root_dir={cfg.data.root_dir} train_dir={cfg.data.train_dir}")
    print(f"[dry-run] charset_train={len(cfg.model.charset_train)} ký tự, "
          f"normalize_unicode={cfg.data.normalize_unicode}, "
          f"remove_whitespace={cfg.data.remove_whitespace}")

    # ĐÁNH GIÁ Ở EVAL MODE.
    # Lý do: `CrossEntropySystem.forward_logits_loss` -> `model(images)` và với TRBA,
    # nhánh `training` của Attention (`trba/prediction.py`) đòi `text` 2D trong khi
    # `TRBA.forward` chỉ tạo `text` 1D -> chỉ hợp lệ khi `model.eval()` (chính là
    # đường parseq dùng khi validate).
    model.eval()
    results: dict[str, Any] = {"n_params": n_params, "losses": []}
    batch = None
    for step, batch in enumerate(loader):
        if step >= num_batches:
            break
        images, labels = batch
        if num_images:
            images, labels = images[:num_images], labels[:num_images]
        images = images.to(device)
        _, loss, numel = model.forward_logits_loss(images, labels)
        loss.backward()  # kiểm chứng backward chạy được (gradients tồn tại)
        has_grad = any(p.grad is not None for p in model.parameters())
        n_grad = sum(p.grad.abs().sum().item() for p in model.parameters()
                     if p.grad is not None)
        print(f"[dry-run] batch {step}: images={tuple(images.shape)} "
              f"labels={len(labels)} loss={loss.item():.4f} numel={numel} "
              f"grad_ok={has_grad} grad_sum={n_grad:.3e}")
        assert torch.isfinite(loss) and loss.item() >= 0, "loss không hợp lệ"
        results["losses"].append(float(loss.item()))
        model.zero_grad(set_to_none=True)

    # Decode thử để chắc tokenizer/charset khớp.
    with torch.no_grad():
        images, labels = batch
        if num_images:
            images, labels = images[:num_images], labels[:num_images]
        logits = model(images.to(device))
        probs = logits.softmax(-1)
        preds, _ = model.tokenizer.decode(probs)
        for i, (pred, gt) in enumerate(zip(preds[:4], labels[:4])):
            print(f"[dry-run] pred[{i}]={pred!r:<32} gt={gt!r}")
    results["preds"] = preds[:4]
    return results


def train_command(config: str, extra: Sequence[str] = ()) -> str:
    parts = [
        "python3 third_party/parseq/train.py",
        f"--config-dir configs/hydra --config-name {config}",
        *extra,
    ]
    return " \\\n  ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Compose config + dry-run 1 batch trước khi train (nhánh 07).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", default="tm_crnn", choices=available_configs() or ["tm_crnn"])
    ap.add_argument("--dry-run", action="store_true", help="1 batch forward + loss + decode")
    ap.add_argument("--num-batches", type=int, default=1)
    ap.add_argument("--num-images", type=int, default=16,
                    help="giới hạn ảnh/batch cho dry-run (0 = toàn batch)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("-o", "--override", action="append", default=[],
                    metavar="KEY=VALUE", help="hydra override, ví dụ trainer.accelerator=cpu")
    ap.add_argument("--list-configs", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.list_configs:
        print("\n".join(available_configs()))
        return 0
    _require_strhub()
    import torch

    cfg = compose_config(a.config, a.override)
    print(f"[train_utils] config={a.config} torch={torch.__version__} "
          f"cuda={torch.cuda.is_available()} DATA_ROOT={DEFAULT_DATA_ROOT}")
    if a.dry_run:
        dry_run(cfg, num_batches=a.num_batches, device=a.device,
                num_images=a.num_images)
        print("[dry-run] OK")
    print("[train] lệnh train thật:")
    print("  " + train_command(a.config, a.override))
    return 0


if __name__ == "__main__":
    sys.exit(main())
