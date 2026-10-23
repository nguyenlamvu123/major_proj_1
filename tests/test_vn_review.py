"""Test offline cho ``trademark_ocr.vn_review`` — vòng duyệt tay nhãn VN (REVIEW.md §4.7)."""

from __future__ import annotations

import csv
import json

import pytest

from trademark_ocr import vn_review as vr

SPLIT = [
    ("VN1", "ATLANTIS"),
    ("VN2", "VT VẠN TÍN"),
    ("VN3", "KY OTO GENIUNE PARTS"),
    ("VN4", "Hoff <b>&</b> Co"),
]


@pytest.fixture()
def data_root(tmp_path):
    path = tmp_path / "processed" / "split_test_vn.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["serial_number", "image_path", "word_mark", "n_chars"])
        for serial, label in SPLIT:
            w.writerow([serial, f"raw/images_vn/{serial}.jpg", label, len(label)])
    return tmp_path


def _review_rows(root):
    with (root / "processed" / "review_vn.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_make_prefills_visible_text_and_writes_escaped_html(data_root):
    res = vr.make(data_root=data_root, n=0, log=lambda *_: None)
    rows = _review_rows(data_root)
    assert res["n"] == 4 and [r["visible_text"] for r in rows] == [lab for _, lab in SPLIT]
    page = (data_root / "processed" / "review_vn.html").read_text(encoding="utf-8")
    assert 'src="../raw/images_vn/VN1.jpg"' in page
    assert "Hoff &lt;b&gt;&amp;&lt;/b&gt; Co" in page          # nhãn không chèn được HTML


def test_make_samples_deterministically_and_refuses_to_overwrite(data_root):
    vr.make(data_root=data_root, n=2, seed=7, log=lambda *_: None)
    first = [r["serial_number"] for r in _review_rows(data_root)]
    with pytest.raises(SystemExit, match="overwrite"):
        vr.make(data_root=data_root, n=2, seed=7, log=lambda *_: None)
    vr.make(data_root=data_root, n=2, seed=7, overwrite=True, log=lambda *_: None)
    assert [r["serial_number"] for r in _review_rows(data_root)] == first and len(first) == 2


def test_apply_reports_noise_rate_and_writes_verified_split(data_root):
    vr.make(data_root=data_root, n=0, log=lambda *_: None)
    rows = _review_rows(data_root)
    rows[1]["visible_text"] = "VẠN TÍN"            # ảnh không có "VT"
    rows[2]["visible_text"] = "KY OTO GENUINE PARTS"
    rows[3]["visible_text"] = "  "                  # không dùng được -> bỏ
    with (data_root / "processed" / "review_vn.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(vr.REVIEW_COLUMNS))
        w.writeheader()
        w.writerows(rows)

    rep = vr.apply(data_root=data_root, log=lambda *_: None)
    assert (rep["n_same"], rep["n_fixed"], rep["n_dropped"]) == (1, 2, 1)
    assert rep["label_noise_rate"] == pytest.approx(3 / 4)
    with (data_root / "processed" / "split_test_vn_verified.csv").open(encoding="utf-8") as fh:
        verified = list(csv.DictReader(fh))
    assert [r["word_mark"] for r in verified] == ["ATLANTIS", "VẠN TÍN", "KY OTO GENUINE PARTS"]
    assert verified[1]["n_chars"] == "7"
    saved = json.loads((data_root / "processed" / "review_vn_report.json").read_text("utf-8"))
    assert saved["n_reviewed"] == 4


def test_cli_help_exits_zero():
    with pytest.raises(SystemExit) as e:
        vr.main(["--help"])
    assert e.value.code == 0
