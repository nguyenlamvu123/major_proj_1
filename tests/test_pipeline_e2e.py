"""Test end-to-end offline cho chuỗi dữ liệu đã chạy đúng (nhánh 04 → 05 → parseq).

meta CSV + index ảnh (nhánh 02/03) → ``clean.run`` → ``split.run`` → ``build_lmdb.run``
→ ``SceneTextDataModule`` của parseq (đúng config ``tm_crnn``) → 1 batch tensor.

Các test module riêng đã kiểm từng bước; file này khoá **hợp đồng giữa các bước**: tên
cột, đường dẫn tương đối theo ``DATA_ROOT``, layout LMDB mà parseq đọc, và nhãn tiếng Việt
có dấu + khoảng trắng đi qua cả chuỗi mà không bị đổi.

Số mẫu giữ dưới 1.000 vì ``create_lmdb`` chỉ đúng ở dải này (xem REVIEW.md §4.1).
"""

from __future__ import annotations

import csv
import shutil

import hydra
import pytest
import torch
from PIL import Image

from trademark_ocr import build_lmdb, clean, split
from trademark_ocr.fetch_images import write_index
from trademark_ocr.train_utils import compose_config

META_COLUMNS = ["serial_number", "word_mark", "image_status", "mark_drawing_cd",
                "drawing_kind", "publication_date"]

#: nhãn tốt (giữ lại) — có dấu tiếng Việt, khoảng trắng, dấu câu trong charset
GOOD_LABELS = ["ĐÔNG Á", "NÔNG TRẠI CẨM THẠCH", "LJ LEJET CAFÉ BISTRO", "1P'S BAKERY & CO"] + [
    f"MARK {i:02d}" for i in range(36)
]


def _png(path, size=(120, 48), shade=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (shade % 256, 128, 255 - shade % 256)).save(path, "PNG")


def _write_raw(data_root):
    """Dựng ``raw/meta/us_marks.csv`` + ``raw/images_us/{index.csv,*.png}`` giả.

    Trả số dòng bị rớt mong đợi theo từng lý do (khớp ``clean.DROP_REASONS``).
    """
    meta, index = [], []

    def add(serial, label, with_image=True):
        meta.append({"serial_number": serial, "word_mark": label, "image_status": "ok",
                     "mark_drawing_cd": "5", "drawing_kind": "stylized_words",
                     "publication_date": "2026-10-13"})
        if with_image:
            _png(data_root / "raw" / "images_us" / f"{serial}.png", shade=int(serial) % 97)
            index.append({"serial_number": serial, "file_name": f"{serial}.png",
                          "word_mark": label, "status": "ok"})

    serial = 20000000
    for label in GOOD_LABELS:
        add(str(serial), label)
        serial += 1
    # mỗi lý do rớt một mẫu
    add(str(serial), "   "); serial += 1                          # no_text
    add(str(serial), "A"); serial += 1                            # too_short
    add(str(serial), "X" * 26); serial += 1                       # too_long
    add(str(serial), "BRAND©"); serial += 1                       # unsupported_char
    add(str(serial), "NO IMAGE", with_image=False); serial += 1   # missing_image
    add(str(serial), "ĐÔNG Á"); serial += 1                       # duplicate_label

    meta_path = data_root / "raw" / "meta" / "us_marks.csv"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=META_COLUMNS)
        w.writeheader()
        w.writerows(meta)
    write_index(index, data_root / "raw" / "images_us" / "index.csv")
    return {"no_text": 1, "too_short": 1, "too_long": 1, "unsupported_char": 1,
            "missing_image": 1, "bad_image": 0, "duplicate_label": 1}


def _build(data_root):
    """Chạy clean → split → build_lmdb như ACCEPTANCE nhánh 04/05."""
    noop = lambda *_: None  # noqa: E731
    clean_report = clean.run(data_root=data_root, log=noop)
    split_report = split.run(clean_path=data_root / "processed" / "clean.csv",
                             out_dir=data_root / "processed", seed=1111, log=noop)
    lmdb = {sub: build_lmdb.run(sub, data_root=data_root, log=noop)
            for sub in ("train", "val", "test_us")}
    return clean_report, split_report, lmdb


@pytest.fixture()
def built(tmp_path):
    expected_drops = _write_raw(tmp_path)
    clean_report, split_report, lmdb = _build(tmp_path)
    return tmp_path, expected_drops, clean_report, split_report, lmdb


def _labels(dataset) -> list[str]:
    # build_tree_dataset của parseq trả ConcatDataset gồm các LmdbDataset
    return [lab for d in dataset.datasets for lab in d.labels]


def test_clean_report_counts_each_drop_reason(built):
    _root, expected, clean_report, _split, _lmdb = built
    assert clean_report["drop_reasons"] == expected
    assert clean_report["kept"] == len(GOOD_LABELS)


def test_split_and_lmdb_counts_agree(built):
    _root, _exp, clean_report, split_report, lmdb = built
    counts = split_report["counts"]
    assert sum(counts.values()) == clean_report["kept"]
    assert all(v == 0 for v in split_report["overlap"].values())
    assert lmdb["train"]["num_samples"] == counts["train"]
    assert lmdb["val"]["num_samples"] == counts["val"]
    assert lmdb["test_us"]["num_samples"] == counts["test_us"]


def test_parseq_datamodule_reads_pipeline_output(built):
    root, *_ = built
    cfg = compose_config("tm_crnn", [
        f"data.root_dir={root / 'lmdb'}",
        "data.augment=false",
        "data.num_workers=0",
        "model.batch_size=4",
    ])
    dm = hydra.utils.instantiate(cfg.data)

    # Mọi nhãn tốt đi qua chuỗi + bộ lọc của parseq mà không bị đổi (dấu, space, dấu câu).
    seen = set(_labels(dm.train_dataset)) | set(_labels(dm.val_dataset))
    test_us = set(lab for _i, lab, _img in build_lmdb.iter_lmdb(root / "lmdb" / "test" / "test_us"))
    assert seen | test_us == set(GOOD_LABELS)

    images, labels = next(iter(dm.train_dataloader()))
    assert images.shape == (4, 3, 32, 128)
    assert images.dtype == torch.float32
    assert -1.0 <= images.min() and images.max() <= 1.0     # Normalize(0.5, 0.5)
    assert all(lab in GOOD_LABELS for lab in labels)


def test_relative_image_paths_survive_moving_data_root(built, tmp_path_factory):
    """``clean.csv`` lưu đường dẫn tương đối → đổi DATA_ROOT (local ↔ Colab) vẫn build được."""
    root, _exp, _clean, split_report, _lmdb = built
    moved = tmp_path_factory.mktemp("moved") / "trademark_data"
    shutil.copytree(root / "raw", moved / "raw")
    shutil.copytree(root / "processed", moved / "processed")

    with (moved / "processed" / "clean.csv").open(encoding="utf-8") as fh:
        paths = [r["image_path"] for r in csv.DictReader(fh)]
    assert paths and all(p.startswith("raw/images_us/") for p in paths)

    res = build_lmdb.run("train", data_root=moved, log=lambda *_: None)
    assert res["num_samples"] == split_report["counts"]["train"]
    assert res["skipped"] == 0
