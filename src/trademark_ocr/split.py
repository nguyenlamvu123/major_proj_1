"""Nhánh 05 (phần 1) — chia tập train/val/test không rò rỉ nhãn (label leakage).

Vì sao chia theo **nhóm nhãn**: nhiều đơn nhãn hiệu có cùng chữ (ví dụ "APPLE").
Nếu cùng một nhãn xuất hiện ở cả train lẫn val/test thì điểm val/test bị "ăn may"
(mô hình đã thấy nhãn đó). Do đó:

1. Gom mọi dòng theo ``word_mark`` (nhóm nhãn).
2. Trộn **các nhóm nhãn** (không trộn từng dòng) bằng seed cố định.
3. Cắt 90% nhóm → train, 5% → val, 5% → test.

⇒ Không nhãn nào xuất hiện ở ≥2 tập (report in overlap = 0).

Đầu vào: ``$DATA_ROOT/processed/clean.csv`` (nhánh 04).
Đầu ra: ``processed/split_{train,val,test_us}.csv`` + ``processed/split_report.json``.

CLI
---
    python3 -m trademark_ocr.split --seed 1111
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from trademark_ocr import DEFAULT_DATA_ROOT
from trademark_ocr.clean import CLEAN_COLUMNS, write_clean_csv


SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test_us")


def group_by_label(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["word_mark"]].append(r)
    return dict(groups)


def split_rows(
    rows: Sequence[dict[str, Any]],
    *,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    seed: int = 1111,
) -> dict[str, list[dict[str, Any]]]:
    """Chia theo nhóm nhãn; trả ``{split_name: rows}`` (deterministic theo seed)."""
    groups = group_by_label(rows)
    labels = sorted(groups)  # sắp trước khi trộn -> ổn định
    random.Random(seed).shuffle(labels)

    n = len(labels)
    n_train = max(0, min(n, int(n * train_ratio)))
    n_val = max(0, min(n - n_train, int(n * val_ratio)))

    buckets = {
        "train": labels[:n_train],
        "val": labels[n_train:n_train + n_val],
        "test_us": labels[n_train + n_val:],
    }
    out: dict[str, list[dict[str, Any]]] = {}
    for name, ls in buckets.items():
        out[name] = [r for lab in ls for r in groups[lab]]
    return out


def _overlap(splits: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    label_sets = {k: {r["word_mark"] for r in v} for k, v in splits.items()}
    serial_sets = {k: {r["serial_number"] for r in v} for k, v in splits.items()}
    return {
        "train_val_labels": len(label_sets["train"] & label_sets["val"]),
        "train_test_labels": len(label_sets["train"] & label_sets["test_us"]),
        "val_test_labels": len(label_sets["val"] & label_sets["test_us"]),
        "any_serial": (
            len(serial_sets["train"] & serial_sets["val"])
            + len(serial_sets["train"] & serial_sets["test_us"])
            + len(serial_sets["val"] & serial_sets["test_us"])
        ),
    }


def run(
    *,
    clean_path: Path | str | None = None,
    out_dir: Path | str | None = None,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    seed: int = 1111,
    limit: int = 0,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    root = Path(DEFAULT_DATA_ROOT)
    clean_path = Path(clean_path) if clean_path else root / "processed" / "clean.csv"
    out_dir = Path(out_dir) if out_dir else root / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    with clean_path.open(encoding="utf-8", newline="") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    if limit:
        rows = rows[:limit]

    splits = split_rows(rows, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    for name in SPLIT_NAMES:
        write_clean_csv(splits[name], out_dir / f"split_{name}.csv", CLEAN_COLUMNS)

    overlap = _overlap(splits)
    report: dict[str, Any] = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "counts": {k: len(v) for k, v in splits.items()},
        "n_labels": {k: len({r["word_mark"] for r in v}) for k, v in splits.items()},
        "overlap": overlap,
        "total": len(rows),
    }
    (out_dir / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"[split] counts={report['counts']} overlap={overlap}")
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Chia train/val/test không rò rỉ nhãn (nhánh 05).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--clean", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1111)
    ap.add_argument("--limit", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    run(clean_path=a.clean, out_dir=a.out_dir, train_ratio=a.train_ratio,
        val_ratio=a.val_ratio, seed=a.seed, limit=a.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
