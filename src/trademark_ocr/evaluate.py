"""Nhánh 10 (phần 2, viết lại ở nhánh 16) — chấm điểm model trên LMDB của đồ án.

Nạp checkpoint (model của parseq hoặc ``trademark_ocr.svtr_ctc.SVTRCTC``), chạy trên
``lmdb/test/<subset>`` (hoặc ``lmdb/val``), ghi CSV từng mẫu (``REPORT_COLUMNS``) +
``<out>.summary.json`` để nhánh 12/13/14 dùng lại.

Khác ``test.py`` của parseq ở ba điểm, có chủ ý (REVIEW.md §4.5):

1. Chấm **mọi** mẫu trong LMDB với **nhãn gốc** (NFC). ``LmdbDataset`` của parseq bỏ nhãn
   dài hơn ``max_label_length`` và xoá ký tự ngoài charset khỏi chính nhãn đúng — làm tập
   VN co từ 50 còn 36 mẫu mà không báo. Ở đây mẫu ngoài phạm vi vẫn được chấm (model không
   thể đúng), được đánh dấu ``in_scope=0``, và summary ghi ``coverage``.
2. ``idx`` là **chỉ số trong LMDB** (từ 1); ``serial_number``/``drawing_kind`` đọc từ khoá
   ``meta-*`` (``build_lmdb.read_meta``) → truy ngược được về ảnh gốc khi phân tích lỗi.
3. ``img_size``, ``max_label_length``, charset lấy từ **hparams của checkpoint**, không viết
   cứng ``(32, 128)`` / ``25``.

Call stack::

    run
      -> load_model -> resolve_model_class
      -> build_loader -> LmdbEvalDataset
      -> iter_predictions
      -> metrics.per_sample / metrics.evaluate_metrics

CLI
---
    python3 -m trademark_ocr.evaluate --ckpt runs/crnn/.../last.ckpt --subset test_us --out reports/crnn_us.csv
    python3 -m trademark_ocr.evaluate --ckpt copied.ckpt --model-class trademark_ocr.svtr_ctc.SVTRCTC ...
"""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterator

from trademark_ocr import DEFAULT_DATA_ROOT
from trademark_ocr.build_lmdb import read_meta
from trademark_ocr.charset import CHARSET_VI
from trademark_ocr.metrics import REPORT_COLUMNS, evaluate_metrics, format_summary, per_sample

PARSEQ_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "parseq"
if str(PARSEQ_ROOT) not in sys.path:
    sys.path.insert(0, str(PARSEQ_ROOT))


#: subset -> thư mục LMDB (khớp build_lmdb.lmdb_path)
def lmdb_dir(data_root: Path | str, subset: str) -> Path:
    root = Path(data_root) / "lmdb"
    if subset == "train":
        return root / "train" / "trademark"
    if subset == "val":
        return root / "val"
    return root / "test" / subset


def _import_class(dotted: str) -> type:
    module, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(module), name)


def resolve_model_class(ckpt: Path | str, model_class: str | None = None) -> type:
    """Lớp ``LightningModule`` để nạp ``ckpt``.

    Thứ tự ưu tiên:

    1. ``model_class`` (dotted path) truyền tay — cho ckpt đã chép ra khỏi thư mục run.
    2. ``<run>/config/config.yaml`` do hydra lưu cạnh ``<run>/checkpoints/*.ckpt``
       (``model._target_``) — nguồn chính xác cho mọi lần train qua ``scripts/05_train.sh``.
    3. Quy ước tên của parseq (``'crnn' in path`` …) — chỉ còn để nạp ckpt chính thức của
       parseq; đoán theo đường dẫn nên xếp cuối.

    Không xác định được thì dừng với hướng dẫn, thay vì đoán (bản cũ rơi sang ``SVTRCTC``
    với mọi lỗi và che mất lỗi thật — REVIEW.md §4.8).
    """
    if model_class:
        return _import_class(model_class)
    cfg_path = Path(ckpt).resolve().parent.parent / "config" / "config.yaml"
    if cfg_path.exists():
        import yaml
        from trademark_ocr.pretrained import FINETUNE_TARGET
        model_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["model"]
        target = model_cfg["_target_"]
        # Run fine-tune: _target_ là factory, lớp thật nằm ở `target` (DESIGN.md 16.6).
        if target == FINETUNE_TARGET:
            target = model_cfg["target"]
        return _import_class(target)
    from strhub.models.utils import InvalidModelError, _get_model_class
    try:
        return _get_model_class(str(ckpt))
    except InvalidModelError:
        raise SystemExit(
            f"Không xác định được lớp model cho {ckpt}: không có {cfg_path} và tên file không "
            "theo quy ước parseq. Truyền --model-class, vd trademark_ocr.svtr_ctc.SVTRCTC"
        ) from None


def load_model(ckpt: Path | str, device: str = "cpu", model_class: str | None = None) -> Any:
    """Nạp model từ ``ckpt`` (xem :func:`resolve_model_class`), ở chế độ eval."""
    cls = resolve_model_class(ckpt, model_class)
    # charset_test ghi đè = CHARSET_VI: ckpt train bằng main.yaml mặc định của parseq có
    # charset_test 36 ký tự thường → CharsetAdapter sẽ hạ chữ thường + xoá dấu trên dự đoán.
    model = cls.load_from_checkpoint(str(ckpt), charset_test=CHARSET_VI)
    return model.eval().to(device)


class LmdbEvalDataset:
    """Mọi mẫu của một LMDB, **không lọc**; ``__getitem__`` trả ``(ảnh tensor, vị trí)``.

    ``samples[i]`` = ``{idx, gt, serial_number, drawing_kind}`` đọc sẵn lúc khởi tạo; ảnh
    đọc lười theo vị trí để DataLoader nhiều worker vẫn dùng được.
    """

    def __init__(self, root: Path | str, transform: Callable[[Any], Any], limit: int = 0):
        import lmdb

        self.root = str(root)
        self.transform = transform
        self.samples: list[dict[str, Any]] = []
        env = lmdb.open(self.root, readonly=True, lock=False, readahead=False, meminit=False)
        try:
            with env.begin() as txn:
                raw = txn.get(b"num-samples")
                if raw is None:
                    raise SystemExit(f"{self.root}: LMDB thiếu khoá num-samples (build dở?)")
                n = int(raw)
                if limit:
                    n = min(n, limit)
                for idx in range(1, n + 1):
                    gt = txn.get(f"label-{idx:09d}".encode()).decode("utf-8")
                    meta = read_meta(txn, idx)
                    self.samples.append({
                        "idx": idx,
                        "gt": unicodedata.normalize("NFC", gt),
                        "serial_number": meta.get("serial_number", ""),
                        "drawing_kind": meta.get("drawing_kind", ""),
                    })
        finally:
            env.close()
        # Env mở lười trong từng tiến trình: handle LMDB không được dùng chung qua fork.
        self._env = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, pos: int) -> tuple[Any, int]:
        import lmdb
        from PIL import Image

        if self._env is None:
            self._env = lmdb.open(self.root, readonly=True, lock=False, readahead=False,
                                  meminit=False)
        with self._env.begin() as txn:
            buf = txn.get(f"image-{self.samples[pos]['idx']:09d}".encode())
        return self.transform(Image.open(io.BytesIO(buf)).convert("RGB")), pos


def build_loader(subset: str, *, data_root: Path | str, img_size: tuple[int, int],
                 batch_size: int = 128, num_workers: int = 0, limit: int = 0,
                 device: str = "cpu") -> Any:
    from torch.utils.data import DataLoader
    from strhub.data.module import SceneTextDataModule

    path = lmdb_dir(data_root, subset)
    if not (path / "data.mdb").exists():
        raise SystemExit(f"Không thấy LMDB: {path} (chạy build_lmdb / nhánh 05 trước)")
    transform = SceneTextDataModule.get_transform(tuple(img_size), augment=False)
    ds = LmdbEvalDataset(path, transform, limit=limit)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      shuffle=False, pin_memory=(device != "cpu"))


def iter_predictions(model: Any, loader: Any, device: str = "cpu") -> Iterator[tuple[int, str, float]]:
    """Sinh ``(vị trí mẫu, pred, confidence)``; ``pred`` qua ``charset_adapter`` như parseq."""
    import torch

    with torch.inference_mode():
        for images, positions in loader:
            probs = model(images.to(device)).softmax(-1)
            preds, conf = model.tokenizer.decode(probs)
            for pos, pred, c in zip(positions.tolist(), preds, conf):
                yield pos, model.charset_adapter(pred), float(c.prod())


def run(*, ckpt: Path | str, subset: str, out: Path | str,
        data_root: Path | str = DEFAULT_DATA_ROOT, batch_size: int = 128,
        num_workers: int = 0, limit: int = 0, device: str = "cpu",
        model_class: str | None = None,
        log: Callable[[str], None] = print) -> dict[str, Any]:
    model = load_model(ckpt, device=device, model_class=model_class)
    hp = model.hparams
    img_size = [int(v) for v in hp.img_size]
    max_len = int(hp.max_label_length)
    charset = set(hp.charset_train)
    loader = build_loader(subset, data_root=data_root, img_size=tuple(img_size),
                          batch_size=batch_size, num_workers=num_workers, limit=limit,
                          device=device)
    samples = loader.dataset.samples

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(REPORT_COLUMNS))
        w.writeheader()
        for pos, pred, conf in iter_predictions(model, loader, device):
            s = samples[pos]
            gt = s["gt"]
            # Ngoài phạm vi = model KHÔNG THỂ đúng: dài hơn số ký tự model phát được, hoặc
            # chứa ký tự không có trong charset huấn luyện. Vẫn chấm (tính là sai).
            row = {
                "idx": s["idx"], "serial_number": s["serial_number"],
                "drawing_kind": s["drawing_kind"], **per_sample(pred, gt),
                "in_scope": int(len(gt) <= max_len and set(gt) <= charset),
                "confidence": round(conf, 6),
            }
            w.writerow(row)
            rows.append(row)

    in_scope = [r for r in rows if r["in_scope"]]
    metrics: dict[str, Any] = evaluate_metrics([r["pred"] for r in rows], [r["gt"] for r in rows])
    metrics.update({
        "coverage": len(in_scope) / len(rows) if rows else 0.0,
        "n_too_long": sum(len(r["gt"]) > max_len for r in rows),
        "n_unsupported_char": sum(not set(r["gt"]) <= charset for r in rows),
        "in_scope": evaluate_metrics([r["pred"] for r in in_scope], [r["gt"] for r in in_scope]),
        "ckpt": str(ckpt), "subset": subset, "report": str(out),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "img_size": img_size, "max_label_length": max_len,
    })
    summary_path = out.with_suffix(".summary.json") if out.suffix else \
        out.with_name(out.name + ".summary.json")
    summary_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    log(f"[evaluate] {type(model).__name__} / {subset}: {format_summary(metrics)} "
        f"coverage={metrics['coverage']:.3f} "
        f"(quá dài {metrics['n_too_long']}, ký tự lạ {metrics['n_unsupported_char']})")
    log(f"[evaluate] CSV -> {out}  |  summary -> {summary_path}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Chấm điểm model trên LMDB của đồ án (nhánh 10, viết lại ở 16).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subset", default="test_us",
                    choices=["test_us", "test_vn", "val"])
    ap.add_argument("--out", default="reports/eval.csv")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-class", default=None,
                    help="dotted path lớp model khi ckpt không nằm trong thư mục run hydra")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    run(ckpt=a.ckpt, subset=a.subset, out=a.out, data_root=a.data_root,
        batch_size=a.batch_size, num_workers=a.num_workers, limit=a.limit,
        device=a.device, model_class=a.model_class)
    return 0


if __name__ == "__main__":
    sys.exit(main())
