"""Test offline cho nhánh 04 — `trademark_ocr.clean`."""

from __future__ import annotations

import csv
import io
import json

from PIL import Image

from trademark_ocr.clean import (
    CLEAN_COLUMNS,
    clean,
    clean_row,
    normalize_label,
    run,
    write_clean_csv,
)


def _png(path, size=(64, 64)):
    import os as _os
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    Image.frombytes("RGB", (w, h), _os.urandom(w * h * 3)).save(path, "PNG")


def _row(serial="10000000", word="MARK", kind="stylized_words", cd="5"):
    return {
        "serial_number": serial,
        "word_mark": word,
        "image_status": "ok",
        "mark_drawing_cd": cd,
        "drawing_kind": kind,
        "publication_date": "2026-10-13",
        "image_url": f"https://cdn.uspto.gov/img/{serial}/2026-10-13/original",
    }


def _index(serials, file_name_fmt="{}.png", status="ok"):
    return {
        s: {"serial_number": s, "file_name": file_name_fmt.format(s),
            "status": status, "word_mark": "X"}
        for s in serials
    }


# --------------------------------------------------------------------------
# 1. normalize_label
# --------------------------------------------------------------------------

def test_normalize_label_nfc_and_collapse():
    assert normalize_label("  CÀ  PHÊ ") == ("CÀ PHÊ", "")
    assert normalize_label("a\u0301") == ("á", "")  # NFD -> NFC
    assert normalize_label("   ") == ("", "no_text")
    assert normalize_label(None) == ("", "no_text")


# --------------------------------------------------------------------------
# 2. clean_row
# --------------------------------------------------------------------------

def test_clean_row_keeps_and_sets_relative_path(tmp_path):
    _png(tmp_path / "raw" / "images_us" / "10000000.png")
    out, reason = clean_row(_row(), _index(["10000000"]), tmp_path)
    assert reason == "" and out["word_mark"] == "MARK"
    assert out["image_path"] == "raw/images_us/10000000.png"
    assert out["n_chars"] == 4


def test_clean_row_missing_image(tmp_path):
    out, reason = clean_row(_row(), {}, tmp_path)
    assert out is None and reason == "missing_image"


def test_clean_row_bad_image(tmp_path):
    p = tmp_path / "raw" / "images_us" / "10000000.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"khong-phai-anh")
    out, reason = clean_row(_row(), _index(["10000000"]), tmp_path)
    assert out is None and reason == "bad_image"


def test_clean_row_without_image_when_not_required(tmp_path):
    out, reason = clean_row(_row(), {}, tmp_path, require_image=False)
    assert reason == "" and out["image_path"] == ""


def test_clean_row_text_reasons(tmp_path):
    assert clean_row(_row(word=""), {}, tmp_path, require_image=False)[1] == "no_text"
    assert clean_row(_row(word="A"), {}, tmp_path, require_image=False)[1] == "too_short"
    assert clean_row(_row(word="A" * 30), {}, tmp_path, require_image=False)[1] == "too_long"
    r = clean_row(_row(word="ABC™"), {}, tmp_path, require_image=False)[1]
    assert r == "unsupported_char"


# --------------------------------------------------------------------------
# 3. clean()
# --------------------------------------------------------------------------

def test_clean_report_counts_all_reasons(tmp_path):
    imgs = tmp_path / "raw" / "images_us"
    for s in ("10000001", "10000002"):
        _png(imgs / f"{s}.png")
    rows = [
        _row("10000001", "GOOD ONE"),          # keep
        _row("10000002", "GOOD TWO"),          # keep
        _row("10000003", ""),                  # no_text
        _row("10000004", "A"),                 # too_short
        _row("10000005", "B" * 40),            # too_long
        _row("10000006", "BAD™"),              # unsupported
        _row("10000007", "NO IMAGE"),          # missing_image
    ]
    idx = _index(["10000001", "10000002", "10000003", "10000004",
                  "10000005", "10000006"])
    kept, report = clean(rows, idx, tmp_path)
    assert report["kept"] == 2
    assert report["total"] == 7
    assert report["keep_ratio"] == round(2 / 7, 6)
    dr = report["drop_reasons"]
    assert dr["no_text"] == 1 and dr["too_short"] == 1 and dr["too_long"] == 1
    assert dr["unsupported_char"] == 1 and dr["missing_image"] == 1
    assert report["label_len"]["min"] == 8  # "GOOD ONE"/"GOOD TWO"


def test_clean_dedupes_identical_labels(tmp_path):
    _png(tmp_path / "raw" / "images_us" / "10000001.png")
    _png(tmp_path / "raw" / "images_us" / "10000002.png")
    rows = [_row("10000001", "SAME"), _row("10000002", "SAME")]
    idx = _index(["10000001", "10000002"])
    kept, report = clean(rows, idx, tmp_path)
    assert len(kept) == 1 and report["drop_reasons"]["duplicate_label"] == 1
    kept2, report2 = clean(rows, idx, tmp_path, dedupe=False)
    assert len(kept2) == 2 and report2["drop_reasons"]["duplicate_label"] == 0


def test_clean_limit(tmp_path):
    rows = [_row(f"1000000{i}", f"W{i}") for i in range(10)]
    kept, report = clean(rows, {}, tmp_path, limit=4)
    assert report["total"] == 4


# --------------------------------------------------------------------------
# 4. run() + I/O
# --------------------------------------------------------------------------

def _write_meta(path, rows):
    cols = ["serial_number", "word_mark", "image_status", "mark_drawing_cd",
            "drawing_kind", "publication_date", "image_url"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _write_index(path, serials):
    cols = ["serial_number", "file_name", "word_mark", "mark_drawing_cd",
            "drawing_kind", "publication_date", "status", "http_status",
            "width", "height", "bytes", "sha1", "reason", "url"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for s in serials:
            w.writerow({"serial_number": s, "file_name": f"{s}.png", "status": "ok",
                        "word_mark": "X"})


def test_run_writes_clean_and_report(tmp_path):
    _write_meta(tmp_path / "raw" / "meta" / "us_marks.csv",
                [_row("10000001", "KEEP ME"), _row("10000002", "DROP ME")])
    _write_index(tmp_path / "raw" / "images_us" / "index.csv", ["10000001"])
    _png(tmp_path / "raw" / "images_us" / "10000001.png")

    report = run(data_root=tmp_path, log=lambda *_: None)
    assert report["kept"] == 1

    clean_rows = list(csv.DictReader(
        open(tmp_path / "processed" / "clean.csv", encoding="utf-8")))
    assert len(clean_rows) == 1
    assert list(clean_rows[0].keys()) == CLEAN_COLUMNS
    assert clean_rows[0]["serial_number"] == "10000001"
    assert clean_rows[0]["image_path"] == "raw/images_us/10000001.png"

    saved = json.loads(
        (tmp_path / "processed" / "clean_report.json").read_text(encoding="utf-8"))
    assert saved["keep_ratio"] == 0.5
    assert saved["drop_reasons"]["missing_image"] == 1


def test_write_clean_csv_roundtrip(tmp_path):
    p = tmp_path / "clean.csv"
    write_clean_csv([{"serial_number": "1", "word_mark": "A", "n_chars": 1}], p)
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert rows[0]["serial_number"] == "1" and "image_path" in rows[0]
