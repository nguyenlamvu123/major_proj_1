"""Test offline cho nhánh 11 — parser WIPO Publish + ghi sản phẩm (không mạng)."""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

import pytest

from trademark_ocr import fetch_vn as fv

FIXTURE = Path(__file__).parent / "data" / "vn_panel_sample.html"


@pytest.fixture()
def panel() -> str:
    return FIXTURE.read_text(encoding="utf-8")


# ------------------------------ parse_form ------------------------------

def test_parse_form_variable_ids():
    html = ('<script>Wicket.Ajax.ajax({"f":"id6","u":"./trademarks;jsessionid=AB?'
            '0-1.IBehaviorListener.0-body-basicSearchTab-searchInputPanel-searchForm-'
            'searchSubmitLink&query=*:*","e":"click","c":"buttonSearchBasic"});</script>'
            '<script>Wicket.Ajax.baseUrl="public/trademarks?0&amp;query=*:*";</script>')
    fid, submit, base = fv.parse_form(html)
    assert fid == "id6"
    assert "searchSubmitLink" in submit and submit.startswith("./trademarks;jsessionid=AB?")
    assert base == "public/trademarks?0&query=*:*"


def test_parse_form_missing_raises():
    with pytest.raises(RuntimeError):
        fv.parse_form("<html>không có gì</html>")


# ------------------------------ parse_rows ------------------------------

def test_parse_rows_from_real_fixture(panel):
    rows = fv.parse_rows(panel)
    assert len(rows) == 3
    first = rows[0]
    assert first["serial_number"] == "VN4202645978"
    assert first["word_mark"] == "ATLANTIS"
    assert first["n_chars"] == "8"
    assert first["application_no"] == "VN-4-2026-45978"
    assert first["image_url"].startswith("https://wipopublish.ipvietnam.gov.vn/")
    assert "/service/trademarks/application/VN4202645978/thumbnail" in first["image_url"]
    ids = [r["serial_number"] for r in rows]
    assert ids == ["VN4202645978", "VN4202645648", "VN4202645228"]
    assert rows[1]["word_mark"] == "KP-Food TINH HOA TỪ BIỂN - TRỌN VỊ QUÊ HƯƠNG"


def test_next_url_from_fixture(panel):
    nxt = fv.next_url(panel)
    assert nxt and "navigator-next" in nxt and nxt.startswith("./trademarks?")


def test_usable_filters_image_only_marks():
    assert fv.usable({"word_mark": "ACME", "image_url": "http://x/a.jpg"})
    assert not fv.usable({"word_mark": "", "image_url": "http://x/a.jpg"})
    assert not fv.usable({"word_mark": "X", "image_url": ""})


# -------------------------------- labels ------------------------------

def test_to_label_nfc_and_whitespace():
    decomposed = "A\u0301"                      # A + combining acute
    assert unicodedata.is_normalized("NFC", fv.to_label(decomposed))
    assert fv.to_label("A\u0301") == "Á"
    assert fv.to_label("  VĂN   QUÂN  ") == "VĂN QUÂN"
    assert fv.usable({"word_mark": fv.to_label("ĐÔNG Á"), "image_url": "x"})
    assert fv.to_label("A &amp; B") == "A & B"


# ------------------------------ write_outputs ------------------------------

def test_write_outputs_schema(tmp_path, panel):
    rows = [r for r in fv.parse_rows(panel) if fv.usable(r)]
    for r in rows:
        rows_img = tmp_path / "raw" / "images_vn"
        rows_img.mkdir(parents=True, exist_ok=True)
        p = rows_img / f"{r['serial_number']}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0junk")
        r["image_path"] = str(p.relative_to(tmp_path))
    rep = fv.write_outputs(rows, data_root=tmp_path)
    assert rep["n"] == len(rows)
    split = (tmp_path / "processed" / "split_test_vn.csv")
    with split.open(encoding="utf-8", newline="") as fh:
        got = list(csv.DictReader(fh))
    assert got and set(got[0]) >= {"image_path", "word_mark", "n_chars", "serial_number"}
    labels = (tmp_path / "processed" / "test_vn.txt").read_text(encoding="utf-8").splitlines()
    assert labels == [r["word_mark"] for r in rows]
    assert unicodedata.is_normalized("NFC", "\n".join(labels)) if labels else True
    assert (tmp_path / "raw" / "vn_marks.csv").exists()
    assert (tmp_path / "raw" / "vn_report.json").exists()


# ---------------------------- verify_images ----------------------------

def test_verify_images_ratio(tmp_path):
    from PIL import Image
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(3):
        Image.new("RGB", (20, 10), "white").save(d / f"{i}.jpg")
    (d / "broken.jpg").write_bytes(b"not an image")
    res = fv.verify_images(d)
    assert res["total"] == 4 and res["ok"] == 3
    assert res["ratio"] == pytest.approx(0.75)
    assert res["bad"] == ["broken.jpg"]


def test_verify_images_empty_dir(tmp_path):
    res = fv.verify_images(tmp_path)
    assert res == {"total": 0, "ok": 0, "bad": [], "ratio": 0.0}


# -------------------------------- CLI --------------------------------

def test_fetch_vn_cli_help_exits_zero():
    with pytest.raises(SystemExit) as e:
        fv.main(["--help"])
    assert e.value.code == 0
