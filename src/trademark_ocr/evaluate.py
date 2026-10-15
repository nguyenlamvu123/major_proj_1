"""Nhánh 10 (phần 2) — chấm điểm model trên tập LMDB của đồ án.

Nạp checkpoint parseq (`strhub.models.utils.load_from_checkpoint`) hoặc model tự viết
(`trademark_ocr.svtr_ctc.SVTRCTC`), chạy trên `lmdb/test/<subset>` (hoặc `lmdb/val`),
ghi CSV từng mẫu (`REPORT_COLUMNS`) để nhánh 12/14 dùng lại.

CLI
---
    python3 -m trademark_ocr.evaluate --ckpt runs/crnn/.../last.ckpt --subset test_us --out reports/crnn_us.csv
    python3 -m trademark_ocr.evaluate --ckpt ... --subset test_us --out reports/crnn_us.csv --limit 200
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from trademark_ocr.charset import CHARSET_VI
from trademark_ocr.metrics import (
    REPORT_COLUMNS,
    edit_distance,
    evaluate_metrics,
    format_summary,
    normalized_edit_distance,
)

DEFAULT_DATA_ROOT: str = os.environ.get(
    "DATA_ROOT", "/media/zaibachkhoa/AC84-F995/trademark_data"
)
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


def load_model(ckpt: Path | str, device: str = "cpu") -> Any:
    """Nạp model từ ckpt: parseq (crnn/trba/parseq/...) hoặc SVTRCTC tự viết."""
    ckpt = str(ckpt)
    kw = {"charset_test": CHARSET_VI}
    try:
        from strhub.models.utils import _get_model_class, load_from_checkpoint
        ModelClass = _get_model_class(ckpt)
        model = ModelClass.load_from_checkpoint(ckpt, **kw)
    except Exception:
        from trademark_ocr.svtr_ctc import SVTRCTC
        model = SVTRCTC.load_from_checkpoint(ckpt, charset_test=CHARSET_VI)
    return model.eval().to(device)


def iter_predictions(
    model: Any, loader: Any, device: str = "cpu",
) -> Iterator[tuple[str, str, float]]:
    """Sinh `(pred, gt, confidence)`; `pred` đã qua `charset_adapter` như parseq eval."""
    import torch

    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = logits.softmax(-1)
            preds, conf = model.tokenizer.decode(probs)
            for i, (pred, gt) in enumerate(zip(preds, labels)):
                pred = model.charset_adapter(pred)
                p = float(conf[i].prod()) if len(conf) else float(probs[i].max())
                yield pred, gt, p


def build_loader(subset: str, *, data_root: Path | str, batch_size: int = 128,
                 num_workers: int = 0, limit: int = 0, device: str = "cpu") -> Any:
    import torch
    from torch.utils.data import DataLoader
    from strhub.data.dataset import LmdbDataset
    from strhub.data.module import SceneTextDataModule

    path = lmdb_dir(data_root, subset)
    if not path.exists():
        raise SystemExit(f"Không thấy LMDB: {path} (chạy nhánh 05 trước)")
    transform = SceneTextDataModule.get_transform((32, 128), augment=False)
    ds = LmdbDataset(str(path), CHARSET_VI, max_label_len=25,
                     remove_whitespace=False, normalize_unicode=False,
                     transform=transform)
    if limit:
        ds = torch.utils.data.Subset(ds, range(min(limit, len(ds))))
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      shuffle=False, pin_memory=(device != "cpu"))


def run(*, ckpt: Path | str, subset: str, out: Path | str,
        data_root: Path | str = DEFAULT_DATA_ROOT, batch_size: int = 128,
        num_workers: int = 0, limit: int = 0, device: str = "cpu",
        log: Callable[[str], None] = print) -> dict[str, Any]:
    model = load_model(ckpt, device=device)
    loader = build_loader(subset, data_root=data_root, batch_size=batch_size,
                          num_workers=num_workers, limit=limit, device=device)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    preds, gts = [], []
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(REPORT_COLUMNS))
        w.writeheader()
        for i, (pred, gt, conf) in enumerate(iter_predictions(model, loader, device)):
            preds.append(pred)
            gts.append(gt)
            w.writerow({
                "idx": i, "pred": pred, "gt": gt,
                "correct": int(pred == gt),
                "ned": normalized_edit_distance(pred, gt),
                "char_err": (edit_distance(pred, gt) / len(gt)) if gt else 0.0,
                "pred_len": len(pred), "gt_len": len(gt),
            })

    metrics = evaluate_metrics(preds, gts)
    metrics.update({"ckpt": str(ckpt), "subset": subset, "report": str(out)})
    summary_path = out.with_suffix(".summary.json") if out.suffix else \
        out.with_name(out.name + ".summary.json")
    summary_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    log(f"[evaluate] {Path(ckpt).parent.name or ckpt} / {subset}: {format_summary(metrics)}")
    log(f"[evaluate] CSV -> {out}  |  summary -> {summary_path}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Chấm điểm model trên LMDB của đồ án (nhánh 10).",
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
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    run(ckpt=a.ckpt, subset=a.subset, out=a.out, data_root=a.data_root,
        batch_size=a.batch_size, num_workers=a.num_workers, limit=a.limit,
        device=a.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
