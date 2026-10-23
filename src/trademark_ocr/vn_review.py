"""Nhánh 16 — kiểm tra tay nhãn tập test VN (REVIEW.md §4.7).

Trường ``rs-MK`` của WIPO Publish là phần chữ **được đăng ký**, không phải bản chép lại chữ
nhìn thấy trong ảnh (có thể thiếu/thừa chữ, khác hoa/thường, ảnh nhiều dòng...). Tập test
"mới" phải có nhãn đúng với ảnh, nên cần một vòng duyệt tay và báo cáo tỉ lệ nhãn sai.

Quy trình hai bước, trạng thái nằm trong file CSV (người duyệt sửa bằng bảng tính bất kỳ):

1. ``make``  — lấy mẫu từ ``processed/split_test_vn.csv`` → ``processed/review_vn.csv``
   (cột ``visible_text`` điền sẵn = ``word_mark``) + ``processed/review_vn.html`` (ảnh cạnh
   nhãn để xem nhanh). **Không ghi đè** ``review_vn.csv`` đã có (tránh mất công duyệt).
2. Người duyệt sửa ``visible_text`` thành đúng chữ trong ảnh (giữ nguyên nếu đúng), để
   **trống** nếu ảnh không dùng được (không đọc được, không có chữ...).
3. ``apply`` — đọc ``review_vn.csv`` → ``processed/split_test_vn_verified.csv`` (nhãn đã
   sửa, bỏ dòng trống) + ``processed/review_vn_report.json`` (tỉ lệ nhãn sai).

Sau đó build LMDB từ bản đã xác minh::

    python3 -m trademark_ocr.build_lmdb --subset test_vn \
        --split-csv $DATA_ROOT/processed/split_test_vn_verified.csv

CLI
---
    python3 -m trademark_ocr.vn_review make --n 300
    python3 -m trademark_ocr.vn_review apply
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable

from trademark_ocr.charset import collapse_ws

DEFAULT_DATA_ROOT: str = os.environ.get(
    "DATA_ROOT", "/media/zaibachkhoa/AC84-F995/trademark_data"
)

REVIEW_COLUMNS: tuple[str, ...] = ("serial_number", "image_path", "word_mark", "visible_text")
#: cột split đã xác minh — khớp ``fetch_vn.SPLIT_COLUMNS`` để build_lmdb đọc như cũ
VERIFIED_COLUMNS: tuple[str, ...] = ("serial_number", "image_path", "word_mark", "n_chars")


def _paths(data_root: Path | str) -> dict[str, Path]:
    processed = Path(data_root) / "processed"
    return {
        "split": processed / "split_test_vn.csv",
        "review": processed / "review_vn.csv",
        "html": processed / "review_vn.html",
        "verified": processed / "split_test_vn_verified.csv",
        "report": processed / "review_vn_report.json",
    }


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _write(rows: list[dict[str, Any]], path: Path, columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def make(*, data_root: Path | str = DEFAULT_DATA_ROOT, n: int = 300, seed: int = 1111,
         overwrite: bool = False, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Sinh ``review_vn.csv`` + ``review_vn.html`` cho ``n`` mẫu ngẫu nhiên (0 = tất cả)."""
    p = _paths(data_root)
    if p["review"].exists() and not overwrite:
        raise SystemExit(f"{p['review']} đã có (có thể đang duyệt dở) — dùng --overwrite "
                         "nếu thật sự muốn lấy mẫu lại")
    rows = _read(p["split"])
    if n and n < len(rows):
        # sắp theo serial trước khi trộn → cùng seed cho cùng mẫu dù thứ tự crawl khác
        rows = random.Random(seed).sample(sorted(rows, key=lambda r: r["serial_number"]), n)
    review = [{**r, "visible_text": r["word_mark"]} for r in rows]
    _write(review, p["review"], REVIEW_COLUMNS)

    # Ảnh nằm dưới DATA_ROOT (raw/images_vn/...), HTML nằm ở processed/ → đường dẫn "../".
    cells = "\n".join(
        f"<tr><td>{i}</td><td><img src=\"../{html.escape(r['image_path'])}\"></td>"
        f"<td>{html.escape(r['word_mark'])}</td><td>{html.escape(r['serial_number'])}</td></tr>"
        for i, r in enumerate(review, start=1)
    )
    p["html"].write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>review_vn</title>"
        "<style>td{border-bottom:1px solid #ccc;padding:4px}img{max-height:80px}</style>"
        "<table><tr><th>#</th><th>ảnh</th><th>word_mark (rs-MK)</th><th>serial</th></tr>"
        f"{cells}</table>\n", encoding="utf-8")
    log(f"[vn_review] {len(review)} mẫu -> {p['review']} | xem ảnh: {p['html']}")
    return {"n": len(review), "review": str(p["review"]), "html": str(p["html"])}


def apply(*, data_root: Path | str = DEFAULT_DATA_ROOT,
          log: Callable[[str], None] = print) -> dict[str, Any]:
    """Đọc ``review_vn.csv`` đã duyệt → split đã xác minh + báo cáo tỉ lệ nhãn sai."""
    p = _paths(data_root)
    reviewed = _read(p["review"])
    verified: list[dict[str, Any]] = []
    n_same = n_fixed = n_dropped = 0
    for r in reviewed:
        visible = collapse_ws(unicodedata.normalize("NFC", r.get("visible_text", "")))
        if not visible:
            n_dropped += 1
            continue
        if visible == r["word_mark"]:
            n_same += 1
        else:
            n_fixed += 1
        verified.append({**r, "word_mark": visible, "n_chars": len(visible)})
    _write(verified, p["verified"], VERIFIED_COLUMNS)

    n = len(reviewed)
    report = {
        "n_reviewed": n, "n_same": n_same, "n_fixed": n_fixed, "n_dropped": n_dropped,
        # tỉ lệ nhãn rs-MK KHÔNG khớp chữ trong ảnh (sửa hoặc bỏ) — số cần đưa vào báo cáo
        "label_noise_rate": (n_fixed + n_dropped) / n if n else 0.0,
        "verified_csv": str(p["verified"]),
    }
    p["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    log(f"[vn_review] {n} mẫu: giữ {n_same}, sửa {n_fixed}, bỏ {n_dropped} "
        f"(nhãn sai {report['label_noise_rate']:.1%}) -> {p['verified']}")
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Kiểm tra tay nhãn tập test VN (nhánh 16).")
    ap.add_argument("action", choices=["make", "apply"])
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--n", type=int, default=300, help="số mẫu cần duyệt (0 = tất cả)")
    ap.add_argument("--seed", type=int, default=1111)
    ap.add_argument("--overwrite", action="store_true", help="cho phép ghi đè review_vn.csv")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.action == "make":
        make(data_root=a.data_root, n=a.n, seed=a.seed, overwrite=a.overwrite)
    else:
        apply(data_root=a.data_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
