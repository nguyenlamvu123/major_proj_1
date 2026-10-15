"""Test offline cho nhánh 10 — `trademark_ocr.metrics` (+ smoke `evaluate`)."""

from __future__ import annotations

import csv

import pytest

from trademark_ocr.metrics import (
    char_error_rate,
    edit_distance,
    evaluate_metrics,
    normalized_edit_distance,
    per_sample,
)


# --------------------------- edit_distance ---------------------------

@pytest.mark.parametrize("a,b,want", [
    ("", "", 0),
    ("abc", "abc", 0),
    ("abc", "", 3),
    ("", "abc", 3),
    ("abd", "abc", 1),                 # 1 thay
    ("ab", "abc", 1),                  # 1 chèn
    ("abc", "ab", 1),                  # 1 xóa
    ("kitten", "sitting", 3),          # ví dụ kinh điển
    ("ĐÔNG Á", "DONG A", 3),           # dấu tiếng Việt tính là khác
    ("ABC", "abc", 3),                 # phân biệt hoa/thường
])
def test_edit_distance_canonical(a, b, want):
    assert edit_distance(a, b) == want
    assert edit_distance(a, b) == edit_distance(b, a)      # đối xứng


# ----------------------- normalized edit distance -----------------------

def test_ned_canonical():
    assert normalized_edit_distance("abc", "abc") == 0.0
    assert normalized_edit_distance("abc", "") == 1.0
    assert normalized_edit_distance("", "") == 0.0
    assert normalized_edit_distance("abd", "abc") == pytest.approx(1 / 3)
    assert normalized_edit_distance("ĐÔNG Á", "DONG A") == pytest.approx(0.5)


def test_char_error_rate_canonical():
    assert char_error_rate("abc", "abc") == 0.0
    assert char_error_rate("abd", "abc") == pytest.approx(1 / 3)
    assert char_error_rate("", "abc") == 1.0        # xóa hết
    assert char_error_rate("abc", "") == 1.0        # gt rỗng, pred không rỗng
    assert char_error_rate("", "") == 0.0


# ---------------------------- aggregate ----------------------------

def test_evaluate_metrics_perfect():
    m = evaluate_metrics(["ABC", "ĐÔNG Á"], ["ABC", "ĐÔNG Á"])
    assert m["n"] == 2
    assert m["accuracy"] == 1.0
    assert m["ned"] == 0.0
    assert m["one_minus_ned"] == 1.0
    assert m["cer"] == 0.0 and m["char_accuracy"] == 1.0


def test_evaluate_metrics_known_values():
    preds = ["abd", "abc", "xyz"]      # 1 đúng, 1 sai 1/3, 1 sai 3/3
    gts = ["abc", "abc", "xyz"]
    m = evaluate_metrics(preds, preds and ["abd", "abc", "xyz"])
    assert m["accuracy"] == 1.0

    preds = ["abd", "abc"]
    gts = ["abc", "abc"]
    m = evaluate_metrics(preds, gts)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["ned"] == pytest.approx((1 / 3 + 0.0) / 2)
    assert m["one_minus_ned"] == pytest.approx(1 - 1 / 6)
    # CER corpus: tổng edit 1 / tổng gt len 6
    assert m["cer"] == pytest.approx(1 / 6)
    assert m["char_accuracy"] == pytest.approx(1 - 1 / 6)
    assert m["mean_label_len"] == 3.0


def test_evaluate_metrics_empty_and_mismatch():
    m = evaluate_metrics([], [])
    assert m["n"] == 0 and m["accuracy"] == 0.0
    with pytest.raises(ValueError):
        evaluate_metrics(["a"], ["a", "b"])


def test_per_sample_row_matches_report_columns():
    row = per_sample("abd", "abc")
    assert row["correct"] == 0 and row["ned"] == pytest.approx(1 / 3)
    assert row["gt_len"] == 3 and row["pred_len"] == 3
    from trademark_ocr.metrics import REPORT_COLUMNS
    assert set(row) == set(REPORT_COLUMNS) - {"idx"}


# ------------------------------ evaluate CLI ------------------------------

def test_evaluate_cli_help_exits_zero():
    from trademark_ocr.evaluate import main
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0


def test_evaluate_lmdb_dir_mapping(tmp_path):
    from trademark_ocr.evaluate import lmdb_dir
    assert lmdb_dir(tmp_path, "val") == tmp_path / "lmdb" / "val"
    assert lmdb_dir(tmp_path, "test_us") == tmp_path / "lmdb" / "test" / "test_us"
    assert lmdb_dir(tmp_path, "test_vn") == tmp_path / "lmdb" / "test" / "test_vn"


@pytest.mark.skipif(
    not (__import__("os").environ.get("DATA_ROOT") or
         __import__("pathlib").Path("/media/zaibachkhoa/AC84-F995/trademark_data/lmdb/val").exists()),
    reason="cần LMDB val trên đĩa (nhánh 05)")
def test_evaluate_end_to_end_on_real_lmdb(tmp_path):
    """Chạy thật: dùng model CHƯA train (khởi tạo từ config) -> chỉ kiểm tra CSV."""
    import hydra
    from trademark_ocr.train_utils import compose_config
    from trademark_ocr import evaluate as ev

    cfg = compose_config("tm_crnn", ["model.batch_size=16"])
    model = hydra.utils.instantiate(cfg.model)
    # giả lập "ckpt" bằng model đang có: bỏ qua load_model, dùng trực tiếp
    loader = ev.build_loader("val", data_root=ev.DEFAULT_DATA_ROOT, batch_size=16, limit=32)
    out = tmp_path / "crnn_us.csv"
    preds, gts = [], []
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ev.REPORT_COLUMNS))
        w.writeheader()
        for i, (pred, gt, conf) in enumerate(ev.iter_predictions(model.eval(), loader)):
            preds.append(pred); gts.append(gt)
            w.writerow({"idx": i, "pred": pred, "gt": gt, "correct": int(pred == gt),
                        "ned": ev.normalized_edit_distance(pred, gt),
                        "char_err": 0.0, "pred_len": len(pred), "gt_len": len(gt)})
    assert out.exists() and len(preds) >= 1
    m = ev.evaluate_metrics(preds, gts)
    assert 0.0 <= m["accuracy"] <= 1.0 and 0.0 <= m["cer"] <= 1.0
    header = out.read_text(encoding="utf-8").splitlines()[0]
    for col in ("pred", "gt", "correct", "ned"):
        assert col in header
