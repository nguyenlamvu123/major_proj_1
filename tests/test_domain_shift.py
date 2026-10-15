"""Test offline cho nhánh 12 — bảng sụt giảm US→VN (không cần dữ liệu thật)."""

from __future__ import annotations

import csv

import pytest

from trademark_ocr import eval_domain_shift as ds

REPORT_HEADER = ["idx", "pred", "gt", "correct", "ned", "char_err", "pred_len", "gt_len"]


def _write_report(path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(REPORT_HEADER)
        for i, (pred, gt) in enumerate(pairs):
            w.writerow([i, pred, gt, int(pred == gt), 0, 0, len(pred), len(gt)])


def test_read_accuracy_uses_correct_column(tmp_path):
    p = tmp_path / "x.csv"
    _write_report(p, [("abc", "abc"), ("abd", "abc"), ("x", "y"), ("z", "z")])
    acc, n = ds.read_accuracy(p)
    assert n == 4 and acc == pytest.approx(2 / 4)


def test_read_accuracy_falls_back_to_pred_gt(tmp_path):
    p = tmp_path / "x.csv"
    with p.open("w", encoding="utf-8", newline="") as fh:
        fh.write("idx,pred,gt\n")
        fh.write("0,a,a\n1,b,c\n")
    acc, n = ds.read_accuracy(p)
    assert n == 2 and acc == pytest.approx(0.5)


def test_discover_ignores_other_files(tmp_path):
    _write_report(tmp_path / "crnn_us.csv", [("a", "a")])
    _write_report(tmp_path / "crnn_vn.csv", [("a", "b")])
    (tmp_path / "domain_shift.csv").write_text("model,acc_us,acc_vn,delta\n")
    (tmp_path / "random.csv").write_text("x\n")
    (tmp_path / "crnn_us.csv.summary.json").write_text("{}")
    found = ds.discover(tmp_path)
    assert set(found) == {"crnn"}
    assert set(found["crnn"]) == {"us", "vn"}


def test_build_rows_delta_and_missing_domain(tmp_path):
    # crnn: US 100% (3/3), VN 1/3 -> delta = -2/3
    _write_report(tmp_path / "crnn_us.csv", [("a", "a"), ("b", "b"), ("c", "c")])
    _write_report(tmp_path / "crnn_vn.csv", [("a", "a"), ("x", "b"), ("y", "c")])
    # trba chỉ có US -> bị bỏ
    _write_report(tmp_path / "trba_us.csv", [("a", "a")])
    rows = ds.build_rows(tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["model"] == "crnn"
    assert r["acc_us"] == pytest.approx(1.0)
    assert r["acc_vn"] == pytest.approx(1 / 3)
    assert r["delta"] == pytest.approx(-2 / 3)
    assert r["n_us"] == 3 and r["n_vn"] == 3


def test_run_writes_required_columns_sorted(tmp_path):
    # bad model sụt nhiều hơn good -> đứng trước khi sort theo delta tăng dần
    _write_report(tmp_path / "good_us.csv", [("a", "a"), ("b", "b")])
    _write_report(tmp_path / "good_vn.csv", [("a", "a"), ("b", "b")])
    _write_report(tmp_path / "bad_us.csv", [("a", "a"), ("b", "b")])
    _write_report(tmp_path / "bad_vn.csv", [("a", "z"), ("b", "z")])
    out = tmp_path / "out" / "domain_shift.csv"
    ds.run(reports_dir=tmp_path, out_path=out)
    with out.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert set(["model", "acc_us", "acc_vn", "delta"]) <= set(reader.fieldnames)
        rows = list(reader)
    assert [r["model"] for r in rows] == ["bad", "good"]
    assert float(rows[0]["delta"]) == pytest.approx(-1.0)
    assert float(rows[1]["delta"]) == pytest.approx(0.0)


def test_run_empty_reports_dir(tmp_path):
    out = tmp_path / "domain_shift.csv"
    res = ds.run(reports_dir=tmp_path, out_path=out)
    assert res["rows"] == [] and out.exists()
    assert out.read_text(encoding="utf-8").splitlines() == [
        ",".join(ds.COLUMNS)]


def test_cli_help_exits_zero():
    with pytest.raises(SystemExit) as e:
        ds.main(["--help"])
    assert e.value.code == 0
