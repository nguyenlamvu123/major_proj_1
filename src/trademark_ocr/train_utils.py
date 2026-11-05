"""Nhánh 07 — công cụ kiểm chứng cấu hình & một batch forward trước khi train (parseq).

Vai trò: **không** huấn luyện lại logic của parseq (dùng `third_party/parseq/train.py`).
File này chỉ:

1. **Compose** config hydra từ `configs/hydra/` (tự chứa: `tm_crnn`, `tm_trba`,
   `tm_parseq`, `tm_svtrctc`, ...).
2. **Dry-run**: instantiate model + datamodule, lấy **1 batch** train, tính loss
   (CTC/CE tuỳ hệ), in model summary (số tham số) và decode thử → bảo đảm wiring
   đúng trước khi mất 30 phút train trên Colab.
3. In ra **lệnh train thật** tương ứng.
4. (nhánh 16.8) Phục vụ ``scripts/09_full_pipeline_cpu.sh`` khi train theo **ngân sách giờ**:
   :func:`benchmark` đo giây/batch thật trên máy, :func:`epochs_for_budget` đổi ngân sách ra
   số epoch (để lịch LR OneCycle chạy trọn thay vì bị cắt ngang), :func:`ckpt_progress` đọc
   epoch + thời gian đã train từ ``last.ckpt`` (biết run đã xong hay phải train tiếp).
5. (nhánh 16.9) :func:`train_verdict` xem run đã train xong có thực sự học được gì không
   (``val_accuracy`` tốt nhất trong tên checkpoint) — pipeline ghi ``UNDERTRAINED`` thay vì
   ``DONE`` nếu không, để báo cáo không coi model 0% là kết quả.

Train thật (lặp lại logic của parseq, không sửa code bên thứ ba)::

    python3 third_party/parseq/train.py --config-dir configs/hydra --config-name tm_crnn

CLI
---
    python3 -m trademark_ocr.train_utils --config tm_crnn --dry-run
    python3 -m trademark_ocr.train_utils --config tm_crnn --dry-run -o trainer.accelerator=cpu
    python3 -m trademark_ocr.train_utils --config tm_parseq_ft --benchmark --budget-hours 3 \
        --max-epochs 10 --json-out plan.json -o model.batch_size=64 -o trainer.accelerator=cpu
    python3 -m trademark_ocr.train_utils --ckpt-info runs/crnn_ft/<ts>/checkpoints/last.ckpt
    python3 -m trademark_ocr.train_utils --train-verdict runs/parseq_ft/<ts>
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from trademark_ocr import DEFAULT_DATA_ROOT
# Shim NumPy-2 cho imgaug (RandAugment) — cần TRƯỚC khi import strhub.data.augment.
from trademark_ocr._numpy2_compat import apply_numpy2_compat  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
PARSEQ_ROOT = REPO_ROOT / "third_party" / "parseq"
CFG_DIR = REPO_ROOT / "configs" / "hydra"

if str(PARSEQ_ROOT) not in sys.path:
    sys.path.insert(0, str(PARSEQ_ROOT))

#: Train xong mà ``val_accuracy`` (%) tốt nhất vẫn dưới mức này = model chưa học được gì.
#: Lần chạy 04/11: parseq_ft 2 epoch đứng ở 0,0000 cả hai epoch; model học được thì vượt xa
#: (crnn_ft 60,9% ngay epoch thứ 2), nên 1% tách hai trường hợp mà không cần tinh chỉnh.
MIN_VAL_ACCURACY = 1.0

#: Ngân sách cho ít hơn số epoch này thì cảnh báo trước khi train. Fine-tune khởi tạo lại
#: head theo charset 217 ký tự: crnn_ft 0% sau epoch 1, 60,9% sau epoch 2 (lần chạy 04/11).
MIN_USEFUL_EPOCHS = 3

# parseq/train.py đặt tên checkpoint '{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}', Lightning
# chèn "tên=" trước mỗi giá trị -> "epoch=1-step=1602-val_accuracy=0.0000-val_NED=7.0307.ckpt"
_VAL_ACC_IN_NAME = re.compile(r"val_accuracy=([0-9]+(?:\.[0-9]+)?)")



def _require_strhub() -> None:
    if not (PARSEQ_ROOT / "strhub" / "__init__.py").exists():
        raise SystemExit(
            "Không thấy third_party/parseq/strhub — hãy chạy: bash scripts/00_get_parseq.sh"
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


def benchmark(cfg: Any, *, warmup: int = 2, num_batches: int = 8,
              val_batches: int = 3) -> dict[str, Any]:
    """Đo tốc độ train THẬT trên máy này với đúng config (ảnh, augment, dataloader, batch).

    Đo cả thời gian chờ dataloader (RandAugment + giải mã ảnh chạy trên CPU cũng là chi phí
    thật của một epoch), bỏ ``warmup`` batch đầu (khởi động worker, cấp phát bộ nhớ).
    Optimizer là AdamW thường: ``configure_optimizers`` của parseq cần ``Trainer`` gắn sẵn,
    còn chi phí một bước optimizer nhỏ so với forward/backward.
    """
    import hydra
    import torch
    from omegaconf import open_dict

    # Trọng số khởi tạo không ảnh hưởng tốc độ: tắt nạp pretrained để không tải mạng khi đo.
    if "init_from" in cfg.model:
        with open_dict(cfg):
            cfg.model.init_from = None
    gpu = cfg.trainer.get("accelerator") == "gpu"
    device = "cuda" if gpu else "cpu"
    model = hydra.utils.instantiate(cfg.model).to(device)
    datamodule = hydra.utils.instantiate(cfg.data)

    def batches(loader):
        while True:                      # tập nhỏ hơn warmup + num_batches thì quay vòng
            yield from loader

    def sync():
        if gpu:
            torch.cuda.synchronize()

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    it = batches(datamodule.train_dataloader())
    for i in range(warmup + num_batches):
        if i == warmup:
            sync()
            t0 = time.perf_counter()
        images, labels = next(it)
        loss = model.training_step((images.to(device), labels), i)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    sync()
    train_spb = (time.perf_counter() - t0) / num_batches

    model.eval()
    it = batches(datamodule.val_dataloader())
    with torch.no_grad():
        next(it)                         # khởi động worker của loader val
        sync()
        t0 = time.perf_counter()
        for _ in range(val_batches):
            images, _labels = next(it)
            model(images.to(device))
        sync()
    val_spb = (time.perf_counter() - t0) / val_batches
    return {
        "train_samples": len(datamodule.train_dataset),
        "val_samples": len(datamodule.val_dataset),
        "batch_size": int(cfg.model.batch_size),
        "sec_per_train_batch": round(train_spb, 4),
        "sec_per_val_batch": round(val_spb, 4),
    }


def epochs_for_budget(*, budget_hours: float, max_epochs: int, train_samples: int,
                      val_samples: int, batch_size: int, sec_per_train_batch: float,
                      sec_per_val_batch: float, safety: float = 0.85) -> dict[str, Any]:
    """Số epoch lớn nhất (≤ ``max_epochs``, ≥ 1) vừa ngân sách ``budget_hours``.

    Một epoch = mọi batch train + một lượt validate (``val_check_interval: 1.0``). ``safety``
    chừa phần cho khởi động, lưu checkpoint và dao động tốc độ. ``over_budget`` = kể cả 1 epoch
    cũng vượt ngân sách (trần ``max_time`` của Lightning sẽ cắt giữa chừng).
    """
    train_batches = math.ceil(train_samples / batch_size)
    val_batches = math.ceil(val_samples / batch_size)
    # NOTICE: số giây một epoch = batch train × giây/batch + batch val × giây/batch val;
    # chưa gồm thời gian dừng lưu checkpoint (vài giây) — phần đó nằm trong `safety`.
    epoch_seconds = train_batches * sec_per_train_batch + val_batches * sec_per_val_batch
    fit = math.floor(budget_hours * 3600 * safety / epoch_seconds) if epoch_seconds > 0 else max_epochs
    epochs = max(1, min(max_epochs, fit))
    return {
        "epochs": epochs,
        "train_batches_per_epoch": train_batches,
        "epoch_minutes": round(epoch_seconds / 60, 1),
        "estimated_hours": round(epochs * epoch_seconds / 3600, 2),
        "fits_max_epochs": fit >= max_epochs,
        "over_budget": fit < 1,
        # Chỉ khi NGÂN SÁCH là thứ giới hạn: người dùng tự đặt --max-epochs 2 (smoke test) thì
        # không cảnh báo.
        "too_few_epochs": epochs < min(MIN_USEFUL_EPOCHS, max_epochs),
    }


def ckpt_progress(path: Path | str) -> dict[str, Any]:
    """Tiến độ đã lưu trong checkpoint Lightning: ``epoch`` (0-based, epoch vừa validate),
    ``global_step`` và giây đã train (từ trạng thái callback ``Timer``; 0 nếu run không đặt
    ``max_time``)."""
    import torch

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    elapsed = 0.0
    for key, state in (ckpt.get("callbacks") or {}).items():
        # Lightning lưu trạng thái callback theo `state_key` (= tên lớp với Timer)
        if key.startswith("Timer") and isinstance(state, dict):
            elapsed = float(state.get("time_elapsed", {}).get("train", 0.0))
    return {"epoch": int(ckpt.get("epoch", -1)), "global_step": int(ckpt.get("global_step", 0)),
            "train_seconds": round(elapsed, 1)}


def train_verdict(run_dir: Path | str, *, min_val_accuracy: float = MIN_VAL_ACCURACY) -> dict[str, Any]:
    """Run đã train xong có học được gì không, đọc từ TÊN các checkpoint top-k trong
    ``<run>/checkpoints/`` (không nạp checkpoint nên nhanh, không cần torch).

    ``verdict``: ``ok`` | ``undertrained`` (``val_accuracy`` tốt nhất < ``min_val_accuracy``, đơn
    vị %) | ``unknown`` (không checkpoint nào mang ``val_accuracy`` trong tên, vd chỉ có
    ``last.ckpt`` — không đủ căn cứ nên pipeline coi như xong, giữ hành vi cũ).
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    scores = sorted(float(m.group(1)) for p in ckpt_dir.glob("*.ckpt")
                    if (m := _VAL_ACC_IN_NAME.search(p.name)))
    if not scores:
        return {"verdict": "unknown", "best_val_accuracy": None, "n_scored_ckpt": 0,
                "min_val_accuracy": min_val_accuracy}
    best = scores[-1]   # save_top_k giữ k checkpoint tốt nhất -> max của chúng = tốt nhất cả run
    return {"verdict": "ok" if best >= min_val_accuracy else "undertrained",
            "best_val_accuracy": best, "n_scored_ckpt": len(scores),
            "min_val_accuracy": min_val_accuracy}


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
    ap.add_argument("--benchmark", action="store_true",
                    help="đo giây/batch train + val thật trên máy (cần LMDB)")
    ap.add_argument("--budget-hours", type=float, default=None,
                    help="cùng --benchmark: chọn số epoch vừa ngân sách giờ")
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--json-out", default=None, help="ghi kết quả benchmark / ckpt-info ra JSON")
    ap.add_argument("--ckpt-info", default=None, metavar="CKPT",
                    help="in epoch, global_step, giây đã train của một checkpoint")
    ap.add_argument("--train-verdict", default=None, metavar="RUN_DIR",
                    help="run đã train xong có học được gì không (val_accuracy trong tên checkpoint)")
    return ap


def _emit(result: dict[str, Any], json_out: str | None) -> None:
    text = json.dumps(result, ensure_ascii=False)
    print(text)
    if json_out:
        Path(json_out).write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.list_configs:
        print("\n".join(available_configs()))
        return 0
    if a.ckpt_info:
        _emit(ckpt_progress(a.ckpt_info), a.json_out)
        return 0
    if a.train_verdict:
        _emit(train_verdict(a.train_verdict), a.json_out)
        return 0
    _require_strhub()
    import torch

    cfg = compose_config(a.config, a.override)
    print(f"[train_utils] config={a.config} torch={torch.__version__} "
          f"cuda={torch.cuda.is_available()} DATA_ROOT={DEFAULT_DATA_ROOT}")
    if a.benchmark:
        result = benchmark(cfg)
        if a.budget_hours is not None:
            result.update(epochs_for_budget(budget_hours=a.budget_hours, max_epochs=a.max_epochs,
                                            **result))
        _emit(result, a.json_out)
        return 0
    if a.dry_run:
        dry_run(cfg, num_batches=a.num_batches, device=a.device,
                num_images=a.num_images)
        print("[dry-run] OK")
    print("[train] lệnh train thật:")
    print("  " + train_command(a.config, a.override))
    return 0


if __name__ == "__main__":
    sys.exit(main())
