"""Nhánh 05 (phần 2) — đóng gói split thành LMDB theo đúng định dạng `baudm/parseq`.

Định dạng LMDB (khớp ``strhub/data/dataset.py::LmdbDataset``)
-----------------------------------------------------------
- khoá ``num-samples``         : chuỗi số nguyên (số mẫu).
- khoá ``image-{idx:09d}``     : bytes ảnh (PNG) — parseq mở lại bằng ``PIL``.
- khoá ``label-{idx:09d}``     : chuỗi nhãn **UTF-8** (ta lưu NFC; config sẽ đặt
  ``data.normalize_unicode: false`` để không mất dấu).
- khoá ``meta-{idx:09d}``      : JSON ``{"serial_number", "drawing_kind"}`` (chỉ trường có
  giá trị) — **của đồ án**, parseq không đọc; ``evaluate`` dùng để truy ngược mẫu về
  serial/ảnh gốc và tách kết quả theo ``drawing_kind``. Đọc bằng :func:`read_meta`.
- ``idx`` bắt đầu từ **1**.

Vị trí (khớp ``strhub/data/module.py::SceneTextDataModule``)
-----------------------------------------------------------
``<DATA_ROOT>/lmdb/train/<name>/data.mdb``  (train_dir = ``<name>``)
``<DATA_ROOT>/lmdb/val/data.mdb``
``<DATA_ROOT>/lmdb/test/<subset>/data.mdb``

CLI
---
    python3 -m trademark_ocr.build_lmdb --subset train --limit 500
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

import lmdb

from trademark_ocr import DEFAULT_DATA_ROOT


#: subset -> hàm tạo đường dẫn (root, name)
def lmdb_path(data_root: Path | str, subset: str, name: str = "trademark") -> Path:
    root = Path(data_root) / "lmdb"
    if subset == "train":
        return root / "train" / name
    if subset == "val":
        return root / "val"
    if subset.startswith("test"):
        return root / "test" / subset
    raise ValueError(f"subset không hợp lệ: {subset!r}")


#: cột của split CSV được chép vào khoá ``meta-*`` (xem docstring module)
META_FIELDS: tuple[str, ...] = ("serial_number", "drawing_kind")


def read_meta(txn: Any, idx: int) -> dict[str, str]:
    """Đọc khoá ``meta-{idx}``; ``{}`` khi LMDB build trước nhánh 16 (chưa có khoá này)."""
    raw = txn.get(f"meta-{idx:09d}".encode())
    return json.loads(raw.decode("utf-8")) if raw else {}


def split_csv_path(data_root: Path | str, subset: str) -> Path:
    return Path(data_root) / "processed" / f"split_{subset}.csv"


def _resolve_image(path_str: str, data_root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (data_root / p)


def create_lmdb(
    rows: list[dict[str, str]],
    out_dir: Path | str,
    *,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    map_size: int | None = None,
    commit_every: int = 1000,
) -> dict[str, Any]:
    """Ghi LMDB. Trả ``{out, num_samples, skipped, map_size}``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in rows if r.get("image_path")]
    if map_size is None:
        total = sum(
            _resolve_image(r["image_path"], Path(data_root)).stat().st_size
            for r in rows
            if _resolve_image(r["image_path"], Path(data_root)).exists()
        )
        map_size = max(1 << 30, int(total * 1.3) + (1 << 20))
    env = lmdb.open(str(out_dir), map_size=map_size, subdir=True, readonly=False)
    written = 0
    # NOTICE:
    # Transaction được quản lý tay, không dùng `with env.begin(write=True) as txn`.
    # Commit theo lô phải mở transaction mới, nhưng `with` chỉ gọi __exit__ của transaction
    # ban đầu → lô cuối + `num-samples` không được commit, LMDB ≥ commit_every mẫu đọc ra 0
    # (REVIEW.md §4.1, tests/test_lmdb.py::test_create_lmdb_commits_tail_after_intermediate_commit).
    #
    # `num-samples` được ghi trong lô CUỐI: nếu dừng giữa chừng, các lô trước đã commit nhưng
    # không có `num-samples` → parseq báo lỗi ngay thay vì đọc một tập thiếu mẫu.
    txn = env.begin(write=True)
    try:
        for row in rows:
            img = _resolve_image(row["image_path"], Path(data_root))
            if not img.exists():
                continue
            idx = written + 1
            txn.put(f"image-{idx:09d}".encode(), img.read_bytes())
            txn.put(f"label-{idx:09d}".encode(), row["word_mark"].encode("utf-8"))
            meta = {k: row[k] for k in META_FIELDS if row.get(k)}
            if meta:
                txn.put(f"meta-{idx:09d}".encode(),
                        json.dumps(meta, ensure_ascii=False).encode("utf-8"))
            written += 1
            if commit_every and written % commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.put("num-samples".encode(), str(written).encode())
        txn.commit()
    except BaseException:
        txn.abort()
        raise
    finally:
        env.sync()
        env.close()
    return {"out": str(out_dir), "num_samples": written,
            "skipped": len(rows) - written, "map_size": map_size}


def iter_lmdb(out_dir: Path | str) -> Iterator[tuple[int, str, bytes]]:
    """Đọc lại LMDB: sinh ``(idx, label, image_bytes)``."""
    env = lmdb.open(str(out_dir), readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as txn:
            n = int(txn.get("num-samples".encode()) or 0)
            for i in range(1, n + 1):
                label = txn.get(f"label-{i:09d}".encode()).decode("utf-8")
                img = txn.get(f"image-{i:09d}".encode())
                yield i, label, img
    finally:
        env.close()


def run(
    subset: str,
    *,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    split_csv: Path | str | None = None,
    out_dir: Path | str | None = None,
    name: str = "trademark",
    limit: int = 0,
    map_size: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    data_root = Path(data_root)
    split_csv = Path(split_csv) if split_csv else split_csv_path(data_root, subset)
    out_dir = Path(out_dir) if out_dir else lmdb_path(data_root, subset, name)

    with split_csv.open(encoding="utf-8", newline="") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    if limit:
        rows = rows[:limit]

    res = create_lmdb(rows, out_dir, data_root=data_root, map_size=map_size)
    res.update({"subset": subset, "input": str(split_csv), "requested": len(rows)})
    log(f"[lmdb] {subset}: num_samples={res['num_samples']} "
        f"(yêu cầu {len(rows)}) -> {res['out']}")
    return res


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Đóng gói split -> LMDB đúng định dạng parseq (nhánh 05).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--subset", required=True,
                    choices=["train", "val", "test_us", "test_vn"])
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--name", default="trademark", help="tên train_dir cho subset=train")
    ap.add_argument("--split-csv", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--map-size", type=int, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    run(a.subset, data_root=a.data_root, split_csv=a.split_csv, out_dir=a.out,
        name=a.name, limit=a.limit, map_size=a.map_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
