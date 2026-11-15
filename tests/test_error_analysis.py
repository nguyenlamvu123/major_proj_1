"""Test offline cho nhánh 13 — phân tích lỗi (không mạng, không notebook)."""

from __future__ import annotations

import csv

import pytest

from trademark_ocr import error_analysis as ea

HEADER = ["idx", "pred", "gt", "correct", "ned", "char_err", "pred_len", "gt_len"]


def _report(path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, (pred, gt) in enumerate(pairs):
            w.writerow([i, pred, gt, int(pred == gt), 0, 0, len(pred), len(gt)])


def test_discover_and_load(tmp_path):
    _report(tmp_path / "crnn_us.csv", [("abc", "abc"), ("abd", "abc")])
    _report(tmp_path / "crnn_vn.csv", [("x", "y")])
    (tmp_path / "domain_shift.csv").write_text("model\n")
    (tmp_path / "notes.txt").write_text("ignore")
    paths = ea.discover_reports(tmp_path)
    assert [p.name for p in paths] == ["crnn_us.csv", "crnn_vn.csv"]
    pairs = ea.load_pairs(paths)
    assert pairs == [("abc", "abc"), ("abd", "abc"), ("x", "y")]


def test_by_length_buckets():
    pairs = [("a", "a"), ("ab", "ab"), ("xyz", "abc"),
             ("abcd", "abcdefgh"), ("x" * 12, "x" * 12)]
    stats = ea.by_length(pairs, bins=(0, 3, 10, 100))
    buckets = {s["bucket"]: s for s in stats}
    assert set(buckets) == {"0-2", "3-9", "10-99"}
    assert buckets["0-2"]["n"] == 2 and buckets["0-2"]["acc"] == 1.0
    assert buckets["3-9"]["n"] == 2
    assert buckets["3-9"]["acc"] == pytest.approx(0.0)   # cả 2 sai
    assert buckets["10-99"]["n"] == 1


def test_top_errors_sorted_and_filters_correct():
    pairs = [("abc", "abc"), ("ax", "abc"), ("a", "abcd"), ("zzzz", "abcd")]
    # edits: 0, 2, 3, 4
    tops = ea.top_errors(pairs, k=2)
    assert [t["dist"] for t in tops] == [4, 3]
    assert tops[0]["gt"] == "abcd" and tops[0]["pred"] == "zzzz"
    assert all(t["dist"] > 0 for t in tops)


def test_char_confusions_counts():
    pairs = [("ABCD", "ABCD".replace("A", "A")), ("XBCD", "ABCD"), ("XYCD", "ABCD")]
    conf = ea.char_confusions(pairs)
    got = {(g, p): n for g, p, n in conf}
    assert got.get(("A", "X")) == 2   # A->X xuất hiện 2 lần
    assert got.get(("B", "Y")) == 1


def test_analyze_writes_figures(tmp_path):
    _report(tmp_path / "crnn_us.csv",
            [("ABC HOLDING", "ABC HOLDING"), ("ABD", "ABC"), ("XY", "ABCD"), ("", "LONG LABEL")])
    out = tmp_path / "figs"
    res = ea.analyze(reports_dir=tmp_path, out_dir=out, top_k=3, log=lambda *_: None)
    assert res["n"] == 4
    figs = res["figures"]
    for key, name in (("by_length", "err_by_length.png"),
                      ("top_errors", "err_top_samples.png"),
                      ("confusions", "err_char_confusions.png")):
        assert key in figs and (out / name).exists()
    assert res["top_errors"][0]["dist"] >= 1


def test_analyze_empty_reports_dir(tmp_path):
    res = ea.analyze(reports_dir=tmp_path, out_dir=tmp_path / "figs", log=lambda *_: None)
    assert res == {"n": 0, "figures": {}, "reports": []}


def test_error_analysis_cli_help_exits_zero():
    with pytest.raises(SystemExit) as e:
        ea.main(["--help"])
    assert e.value.code == 0
