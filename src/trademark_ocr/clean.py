"""Nhánh 04 — lọc & chuẩn hoá nhãn, ghép metadata với ảnh đã tải.

Đầu vào
-------
- ``$DATA_ROOT/raw/meta/us_marks.csv``      (nhánh 02)
- ``$DATA_ROOT/raw/images_us/index.csv``    + ảnh ``{serial}.png`` (nhánh 03)

Công việc
---------
1. Chuẩn hoá nhãn: NFC + gộp khoảng trắng + strip (nhất quán với charset).
2. Lọc bỏ mẫu không dùng được, **ghi rõ lý do** rớt:
   ``no_text``, ``too_short``, ``too_long``, ``unsupported_char``,
   ``missing_image``, ``bad_image``, ``duplicate_label``.
3. Ghép với ảnh (đường dẫn **tương đối so với DATA_ROOT** để dùng được cả local & Colab).
4. Ghi ``processed/clean.csv`` + ``processed/clean_report.json`` (có ``keep_ratio``).

CLI
---
    python3 -m trademark_ocr.clean --limit 5000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from trademark_ocr import DEFAULT_DATA_ROOT
from trademark_ocr.charset import CHARSET_VI, collapse_ws, contains_only_supported, to_nfc
from trademark_ocr.fetch_images import probe_image, read_index


CLEAN_COLUMNS: list[str] = [
    "serial_number",
    "image_path",
    "word_mark",
    "n_chars",
    "drawing_kind",
    "mark_drawing_cd",
    "publication_date",
]

#: Thứ tự cố định cho report (dễ so sánh giữa các lần chạy).
DROP_REASONS: tuple[str, ...] = (
    "no_text",
    "too_short",
    "too_long",
    "unsupported_char",
    "missing_image",
    "bad_image",
    "duplicate_label",
)


# --------------------------------------------------------------------------
# 1. Chuẩn hoá nhãn
# --------------------------------------------------------------------------

def normalize_label(text: str, charset: str = CHARSET_VI) -> tuple[str, str]:
    """NFC + gộp space. Trả ``(label, reason)`` — ``reason`` "" nếu ổn về text."""
    label = collapse_ws(to_nfc(text or ""))
    if not label:
        return label, "no_text"
    return label, ""


# --------------------------------------------------------------------------
# 2. Lọc 1 dòng
# --------------------------------------------------------------------------

def _rel(path: Path, data_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(data_root).resolve()))
    except ValueError:
        return str(path)


def clean_row(
    row: dict[str, str],
    index: dict[str, dict[str, str]],
    data_root: Path | str,
    *,
    charset: str = CHARSET_VI,
    min_len: int = 2,
    max_len: int = 25,
    require_image: bool = True,
    check_image: bool = True,
) -> tuple[dict[str, Any] | None, str]:
    """Trả ``(clean_row, "")`` nếu giữ, hoặc ``(None, reason)`` nếu rớt."""
    label, reason = normalize_label(row.get("word_mark", ""), charset)
    if reason:
        return None, reason
    if len(label) < min_len:
        return None, "too_short"
    if len(label) > max_len:
        return None, "too_long"
    if not contains_only_supported(label, charset):
        return None, "unsupported_char"

    serial = row.get("serial_number", "")
    entry = index.get(serial)
    file_name = (entry or {}).get("file_name") or ""
    path = Path(data_root) / "raw" / "images_us" / file_name if file_name else None
    available = bool(
        entry and entry.get("status") == "ok" and file_name and path and path.exists()
    )
    img_path_rel = ""
    if available:
        if check_image:
            ok, _w, _h, _r = probe_image(path)
            if not ok:
                return None, "bad_image"
        img_path_rel = _rel(path, data_root)
    elif require_image:
        return None, "missing_image"

    return {
        "serial_number": serial,
        "image_path": img_path_rel,
        "word_mark": label,
        "n_chars": len(label),
        "drawing_kind": row.get("drawing_kind", ""),
        "mark_drawing_cd": row.get("mark_drawing_cd", ""),
        "publication_date": row.get("publication_date", ""),
    }, ""


# --------------------------------------------------------------------------
# 3. Vòng chạy chính
# --------------------------------------------------------------------------

def clean(
    meta_rows: Iterable[dict[str, str]],
    index: dict[str, dict[str, str]],
    data_root: Path | str,
    *,
    charset: str = CHARSET_VI,
    min_len: int = 2,
    max_len: int = 25,
    limit: int = 0,
    require_image: bool = True,
    check_image: bool = True,
    dedupe: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Lọc toàn bộ; trả ``(kept_rows, report)``."""
    rows = list(meta_rows)
    if limit:
        rows = rows[:limit]

    reasons: dict[str, int] = {r: 0 for r in DROP_REASONS}
    kept: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for row in rows:
        out, reason = clean_row(
            row, index, data_root, charset=charset, min_len=min_len,
            max_len=max_len, require_image=require_image, check_image=check_image,
        )
        if out is None:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        if dedupe and out["word_mark"] in seen_labels:
            reasons["duplicate_label"] += 1
            continue
        seen_labels.add(out["word_mark"])
        kept.append(out)

    total = len(rows)
    by_kind: dict[str, int] = {}
    for r in kept:
        by_kind[r["drawing_kind"]] = by_kind.get(r["drawing_kind"], 0) + 1
    lens = [r["n_chars"] for r in kept]
    report: dict[str, Any] = {
        "total": total,
        "kept": len(kept),
        "keep_ratio": round(len(kept) / total, 6) if total else 0.0,
        "drop_reasons": {k: reasons[k] for k in DROP_REASONS},
        "by_drawing_kind": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])),
        "label_len": {
            "min": min(lens) if lens else 0,
            "max": max(lens) if lens else 0,
            "mean": round(sum(lens) / len(lens), 3) if lens else 0.0,
        },
        "params": {
            "min_len": min_len, "max_len": max_len, "limit": limit,
            "require_image": require_image, "check_image": check_image,
            "dedupe": dedupe, "n_charset": len(charset),
        },
    }
    return kept, report


def write_clean_csv(rows: Sequence[dict[str, Any]], path: Path | str,
                    columns: Sequence[str] = CLEAN_COLUMNS) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    tmp.replace(path)
    return path


def run(
    *,
    meta_path: Path | str | None = None,
    index_path: Path | str | None = None,
    out_path: Path | str | None = None,
    report_path: Path | str | None = None,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    limit: int = 0,
    min_len: int = 2,
    max_len: int = 25,
    require_image: bool = True,
    check_image: bool = True,
    dedupe: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    data_root = Path(data_root)
    meta_path = Path(meta_path) if meta_path else data_root / "raw" / "meta" / "us_marks.csv"
    index_path = Path(index_path) if index_path else data_root / "raw" / "images_us" / "index.csv"
    out_path = Path(out_path) if out_path else data_root / "processed" / "clean.csv"
    report_path = Path(report_path) if report_path else data_root / "processed" / "clean_report.json"

    with meta_path.open(encoding="utf-8", newline="") as fh:
        meta_rows = [dict(r) for r in csv.DictReader(fh)]
    index = read_index(index_path)

    kept, report = clean(
        meta_rows, index, data_root, min_len=min_len, max_len=max_len, limit=limit,
        require_image=require_image, check_image=check_image, dedupe=dedupe,
    )
    write_clean_csv(kept, out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    log(f"[clean] kept {report['kept']}/{report['total']} "
        f"(keep_ratio={report['keep_ratio']}) -> {out_path}")
    log(f"[clean] drop: {report['drop_reasons']}")
    return report


# --------------------------------------------------------------------------
# 4. CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    root = Path(DEFAULT_DATA_ROOT)
    ap = argparse.ArgumentParser(
        description="Lọc & chuẩn hoá nhãn, ghép metadata với ảnh (nhánh 04).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-root", default=str(root))
    ap.add_argument("--meta", default=None)
    ap.add_argument("--index", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=25)
    ap.add_argument("--no-require-image", dest="require_image", action="store_false")
    ap.add_argument("--no-check-image", dest="check_image", action="store_false")
    ap.add_argument("--no-dedupe", dest="dedupe", action="store_false")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(
        meta_path=args.meta, index_path=args.index, out_path=args.out,
        report_path=args.report, data_root=args.data_root, limit=args.limit,
        min_len=args.min_len, max_len=args.max_len,
        require_image=args.require_image, check_image=args.check_image,
        dedupe=args.dedupe,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
