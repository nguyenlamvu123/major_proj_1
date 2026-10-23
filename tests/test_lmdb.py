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


# --------------------------------------------------------------------------
# Hồi quy: LMDB ≥ commit_every mẫu (REVIEW.md §4.1)
# --------------------------------------------------------------------------

def _tiny_rows(tmp_path, n):
    img = tmp_path / "one.png"
    _png(img, size=(8, 4))
    return [{"image_path": str(img), "word_mark": f"MARK{i:05d}",
             "serial_number": f"{30000000 + i}", "drawing_kind": "stylized_words"}
            for i in range(n)]


def test_create_lmdb_commits_tail_after_intermediate_commit(tmp_path):
    # ROOT CAUSE:
    #
    # Với >= commit_every mẫu, LMDB đọc lại ra 0 mẫu (parseq: int(None) -> TypeError).
    # create_lmdb gán lại `txn` BÊN TRONG `with env.begin(write=True) as txn:`; `with`
    # giữ __exit__ của transaction CŨ (đã commit) nên transaction mới — chứa phần đuôi và
    # khoá `num-samples` — không bao giờ được commit và bị huỷ khi env.close().
    #
    #   with env.begin(write=True) as txn:
    #       ... txn.commit(); txn = env.begin(write=True)
    #       txn.put(b"num-samples", ...)          # mất
    #
    # Đã sửa bằng cách tự quản transaction: commit theo lô, commit lô cuối + num-samples
    # tường minh, abort nếu có ngoại lệ.
    out = tmp_path / "lmdb"
    res = create_lmdb(_tiny_rows(tmp_path, 7), out, data_root=tmp_path, commit_every=3)
    got = list(iter_lmdb(out))
    assert res["num_samples"] == 7
    assert [lab for _i, lab, _img in got] == [f"MARK{i:05d}" for i in range(7)]


def test_create_lmdb_default_batch_over_1000_readable_by_parseq(tmp_path):
    from strhub.data.dataset import LmdbDataset

    out = tmp_path / "lmdb"
    res = create_lmdb(_tiny_rows(tmp_path, 1500), out, data_root=tmp_path)
    assert res["num_samples"] == 1500
    ds = LmdbDataset(str(out), CHARSET_VI, max_label_len=25,
                     remove_whitespace=False, normalize_unicode=False)
    assert len(ds) == 1500
    assert ds.labels[-1] == "MARK01499"


def test_create_lmdb_stores_meta_for_traceability(tmp_path):
    import lmdb
    from trademark_ocr.build_lmdb import read_meta

    rows = _rows(3)
    for r in rows:
        _png(tmp_path / r["image_path"])
    rows[2]["drawing_kind"] = ""                       # trường rỗng không được ghi
    out = tmp_path / "lmdb"
    create_lmdb(rows, out, data_root=tmp_path)
    env = lmdb.open(str(out), readonly=True, lock=False)
    with env.begin() as txn:
        assert read_meta(txn, 1) == {"serial_number": "10000000", "drawing_kind": "stylized_words"}
        assert read_meta(txn, 3) == {"serial_number": "10000002"}
        assert read_meta(txn, 4) == {}                 # ngoài phạm vi / LMDB cũ
    env.close()
