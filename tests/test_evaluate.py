"""Test offline cho ``trademark_ocr.evaluate`` (viết lại ở nhánh 16, REVIEW.md §4.5 + §4.8).

Dựng checkpoint THẬT (SVTR-CTC cỡ nhỏ, chưa train) trong một thư mục run giống hydra của
parseq, LMDB test có cả nhãn ngoài phạm vi, rồi chạy ``evaluate.run`` end-to-end.
"""

from __future__ import annotations

import csv
import json

import pytest
import yaml
from PIL import Image

from trademark_ocr import evaluate as ev
from trademark_ocr.build_lmdb import create_lmdb
from trademark_ocr.charset import CHARSET_VI

#: (nhãn, serial, in_scope) — in_scope: len ≤ 25 và mọi ký tự thuộc charset
VN_ROWS = [
    ("ĐÔNG Á", "VN1", 1),
    ("Hoff", "VN2", 1),
    ("KP-Food TINH HOA TỪ BIỂN - TRỌN VỊ QUÊ HƯƠNG", "VN3", 0),   # 45 ký tự > 25
    ("BRAND™", "VN4", 0),                                          # '™' ngoài charset
]
#: khác 32×128 mặc định: chỉ chạy được nếu evaluate đọc img_size từ hparams
IMG_SIZE = [32, 160]


def _save_ckpt(model, path):
    import pytorch_lightning as pl

    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.strategy.connect(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(path))


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """``runs/svtrctc/<ts>/{checkpoints/last.ckpt, config/config.yaml}`` như parseq train.py."""
    from trademark_ocr.svtr_ctc import SVTRCTC

    root = tmp_path_factory.mktemp("runs") / "svtrctc" / "2026-10-23_00-00-00"
    model = SVTRCTC(charset_train=CHARSET_VI, charset_test=CHARSET_VI, max_label_length=25,
                    batch_size=2, lr=1e-3, warmup_pct=0.1, weight_decay=0.0,
                    img_size=IMG_SIZE, embed_dim=32, num_heads=2)
    _save_ckpt(model, root / "checkpoints" / "last.ckpt")
    (root / "config").mkdir()
    (root / "config" / "config.yaml").write_text(
        yaml.safe_dump({"model": {"_target_": "trademark_ocr.svtr_ctc.SVTRCTC",
                                  "name": "svtrctc"}}), encoding="utf-8")
    return root


@pytest.fixture()
def vn_lmdb(tmp_path):
    rows = []
    for label, serial, _ in VN_ROWS:
        img = tmp_path / "raw" / "images_vn" / f"{serial}.jpg"
        img.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (200, 47), "white").save(img)
        rows.append({"image_path": str(img), "word_mark": label, "serial_number": serial,
                     "drawing_kind": "vn_combined"})
    create_lmdb(rows, tmp_path / "lmdb" / "test" / "test_vn", data_root=tmp_path)
    return tmp_path


# ---------------------------- resolve_model_class ----------------------------

def test_resolve_prefers_hydra_config_next_to_ckpt(run_dir):
    from trademark_ocr.svtr_ctc import SVTRCTC

    assert ev.resolve_model_class(run_dir / "checkpoints" / "last.ckpt") is SVTRCTC


def test_resolve_explicit_class_wins(run_dir):
    from strhub.models.crnn.system import CRNN

    got = ev.resolve_model_class(run_dir / "checkpoints" / "last.ckpt",
                                 model_class="strhub.models.crnn.system.CRNN")
    assert got is CRNN


def test_resolve_falls_back_to_parseq_naming_without_config(tmp_path):
    from strhub.models.crnn.system import CRNN

    assert ev.resolve_model_class(tmp_path / "crnn-pretrained.ckpt") is CRNN


def test_resolve_unknown_ckpt_fails_loudly(tmp_path):
    # ROOT CAUSE:
    #
    # load_model bọc mọi lỗi bằng `except Exception` rồi thử SVTRCTC, nên ckpt không rõ lớp
    # (hay ckpt hỏng) báo lỗi của SVTRCTC — che mất lỗi thật.
    #
    # Đã sửa: không còn except rộng; không xác định được lớp thì dừng với hướng dẫn rõ.
    with pytest.raises(SystemExit, match="--model-class"):
        ev.resolve_model_class(tmp_path / "copied" / "last.ckpt")


# --------------------------------- run ---------------------------------

def test_run_scores_every_sample_and_keeps_ground_truth(run_dir, vn_lmdb):
    # ROOT CAUSE:
    #
    # evaluate dùng LmdbDataset của parseq: nhãn > max_label_length bị BỎ (VN: 50 → 36 mẫu)
    # và ký tự ngoài charset bị XOÁ khỏi nhãn đúng ("BRAND™" -> "BRAND"); cột idx là số thứ
    # tự sau lọc nên không truy ngược được về ảnh. Ảnh luôn resize về (32, 128) viết cứng.
    #
    # Đã sửa: dataset riêng chấm MỌI mẫu với nhãn gốc, idx = chỉ số LMDB + serial/drawing_kind
    # từ khoá meta-*, đánh dấu in_scope, ghi coverage; img_size/max_len lấy từ hparams.
    out = vn_lmdb / "reports" / "svtrctc_vn.csv"
    summary = ev.run(ckpt=run_dir / "checkpoints" / "last.ckpt", subset="test_vn", out=out,
                     data_root=vn_lmdb, batch_size=3, log=lambda *_: None)

    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == list(ev.REPORT_COLUMNS)
    assert [r["gt"] for r in rows] == [lab for lab, _, _ in VN_ROWS]       # không lọc/không sửa
    assert [r["idx"] for r in rows] == ["1", "2", "3", "4"]                # chỉ số LMDB
    assert [r["serial_number"] for r in rows] == [s for _, s, _ in VN_ROWS]
    assert {r["drawing_kind"] for r in rows} == {"vn_combined"}
    assert [int(r["in_scope"]) for r in rows] == [sc for _, _, sc in VN_ROWS]
    assert [int(r["gt_has_diacritics"]) for r in rows] == [1, 0, 1, 0]

    assert summary["n"] == 4
    assert summary["coverage"] == pytest.approx(0.5)
    assert summary["n_too_long"] == 1 and summary["n_unsupported_char"] == 1
    assert summary["in_scope"]["n"] == 2
    assert summary["img_size"] == IMG_SIZE and summary["max_label_length"] == 25
    assert summary["model_class"] == "trademark_ocr.svtr_ctc.SVTRCTC"
    saved = json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert saved["coverage"] == pytest.approx(0.5)


def test_run_limit_takes_first_samples(run_dir, vn_lmdb):
    out = vn_lmdb / "reports" / "lim_vn.csv"
    summary = ev.run(ckpt=run_dir / "checkpoints" / "last.ckpt", subset="test_vn", out=out,
                     data_root=vn_lmdb, limit=2, log=lambda *_: None)
    assert summary["n"] == 2 and summary["coverage"] == 1.0


def test_missing_lmdb_is_reported(run_dir, tmp_path):
    with pytest.raises(SystemExit, match="LMDB"):
        ev.run(ckpt=run_dir / "checkpoints" / "last.ckpt", subset="test_us",
               out=tmp_path / "x.csv", data_root=tmp_path, log=lambda *_: None)
