"""Test offline cho nhánh 05 — `trademark_ocr.split` + `trademark_ocr.build_lmdb`.

Bao gồm kiểm tra tích hợp với `strhub.data.dataset.LmdbDataset` (parseq).
"""

from __future__ import annotations

import csv
import io
import os

from PIL import Image

from trademark_ocr.charset import CHARSET_VI
from trademark_ocr.build_lmdb import create_lmdb, iter_lmdb, lmdb_path, run as lmdb_run
from trademark_ocr.clean import CLEAN_COLUMNS, write_clean_csv
from trademark_ocr.split import split_rows, run as split_run


def _png(path, size=(64, 32), seed=0):
    import random
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    rnd = random.Random(seed)
    data = bytes(rnd.randrange(256) for _ in range(w * h * 3))
    Image.frombytes("RGB", (w, h), data).save(path, "PNG")


def _rows(n=20, dup_every=4):
    rows = []
    for i in range(n):
        label = f"MARK{i // dup_every:03d}"  # nhiều serial cùng nhãn
        rows.append({
            "serial_number": f"{10000000 + i}",
            "image_path": f"raw/images_us/{10000000 + i}.png",
            "word_mark": label,
            "n_chars": len(label),
            "drawing_kind": "stylized_words",
            "mark_drawing_cd": "5",
            "publication_date": "2026-10-13",
        })
    return rows


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------

def test_split_no_label_or_serial_overlap():
    splits = split_rows(_rows(24), seed=1111)
    labels = {k: {r["word_mark"] for r in v} for k, v in splits.items()}
    serials = {k: {r["serial_number"] for r in v} for k, v in splits.items()}
    assert not (labels["train"] & labels["val"])
    assert not (labels["train"] & labels["test_us"])
    assert not (labels["val"] & labels["test_us"])
    assert not (serials["train"] & serials["test_us"])
    assert sum(len(v) for v in splits.values()) == 24


def test_split_is_deterministic():
    a = split_rows(_rows(30), seed=1111)
    b = split_rows(_rows(30), seed=1111)
    c = split_rows(_rows(30), seed=2222)
    assert a == b
    assert a != c or all(len(a[k]) == len(c[k]) for k in a)


def test_split_run_writes_files_and_report(tmp_path):
    clean = tmp_path / "processed" / "clean.csv"
    write_clean_csv(_rows(20), clean, CLEAN_COLUMNS)
    report = split_run(clean_path=clean, out_dir=tmp_path / "processed", seed=7,
                       log=lambda *_: None)
    for name in ("train", "val", "test_us"):
        assert (tmp_path / "processed" / f"split_{name}.csv").exists()
    assert report["overlap"]["train_val_labels"] == 0
    assert report["overlap"]["any_serial"] == 0


# --------------------------------------------------------------------------
# build_lmdb
# --------------------------------------------------------------------------

def test_create_lmdb_roundtrip_and_nfc(tmp_path):
    rows = _rows(5)
    for r in rows:
        _png(tmp_path / r["image_path"])
    rows[0]["word_mark"] = "ĐÔNG Á"
    res = create_lmdb(rows, tmp_path / "lmdb" / "train" / "trademark",
                      data_root=tmp_path)
    assert res["num_samples"] == 5
    got = list(iter_lmdb(tmp_path / "lmdb" / "train" / "trademark"))
    assert len(got) == 5
    assert got[0][1] == "ĐÔNG Á"                       # giữ dấu (NFC)
    img = Image.open(io.BytesIO(got[0][2]))
    assert img.size == (64, 32)                        # ảnh decode được


def test_lmdb_paths_match_parseq_layout(tmp_path):
    assert lmdb_path(tmp_path, "train", "trademark") == tmp_path / "lmdb" / "train" / "trademark"
    assert lmdb_path(tmp_path, "val") == tmp_path / "lmdb" / "val"
    assert lmdb_path(tmp_path, "test_us") == tmp_path / "lmdb" / "test" / "test_us"


def test_lmdb_run_limit(tmp_path):
    clean_rows = _rows(10)
    for r in clean_rows:
        _png(tmp_path / r["image_path"])
    write_clean_csv(clean_rows, tmp_path / "processed" / "split_train.csv", CLEAN_COLUMNS)
    res = lmdb_run("train", data_root=tmp_path, limit=4, log=lambda *_: None)
    assert res["num_samples"] == 4


def test_lmdb_skips_missing_images(tmp_path):
    rows = _rows(3)
    _png(tmp_path / rows[0]["image_path"])  # chỉ 1 ảnh tồn tại
    res = create_lmdb(rows, tmp_path / "out", data_root=tmp_path)
    assert res["num_samples"] == 1 and res["skipped"] == 2


# --------------------------------------------------------------------------
# Tích hợp với parseq LmdbDataset
# --------------------------------------------------------------------------

def test_parseq_lmdbdataset_reads_our_lmdb(tmp_path):
    from strhub.data.dataset import LmdbDataset

    rows = _rows(6)
    for r in rows:
        _png(tmp_path / r["image_path"])
    rows[1]["word_mark"] = "ĐÔNG Á"
    out = tmp_path / "lmdb" / "val"
    create_lmdb(rows, out, data_root=tmp_path)

    ds = LmdbDataset(str(out), CHARSET_VI, max_label_len=25,
                     remove_whitespace=False, normalize_unicode=False)
    assert len(ds) == 6
    img, label = ds[1]
    assert label == "ĐÔNG Á"          # parseq giữ đúng dấu khi tắt normalize
    assert isinstance(img, Image.Image) and img.size == (64, 32)
