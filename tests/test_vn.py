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


# ------------------------- VnClient (luồng phân trang) -------------------------

_SEARCH_HTML = (
    '<script>Wicket.Ajax.ajax({"f":"id9","u":"./trademarks;jsessionid=S1?0-1.IBehaviorListener.'
    '0-body-basicSearchTab-searchInputPanel-searchForm-searchSubmitLink&query=*:*"});</script>'
    '<script>Wicket.Ajax.baseUrl="public/trademarks?0&amp;query=*:*";</script>'
)


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text
        self.status_code = 200
        self.content = text.encode()


class _FakeSession:
    """Giả lập WIPO Publish: GET trang tìm kiếm → POST tìm → GET 'navigator-next' (lặp lại panel)."""

    def __init__(self, panel: str) -> None:
        self.panel = panel
        self.headers: dict[str, str] = {}
        self.verify = True
        self.calls: list[tuple[str, str]] = []

    def get(self, url, **_kw):
        self.calls.append(("GET", url))
        return _Resp(self.panel if "navigator-next" in url else _SEARCH_HTML)

    def post(self, url, data=None, headers=None, **_kw):
        self.calls.append(("POST", url))
        assert data.startswith("id9_hf_0=")                # form id đọc từ HTML, không hard-code
        assert headers["Wicket-Ajax"] == "true"
        assert headers["Wicket-Ajax-BaseURL"] == "public/trademarks?0&query=*:*"
        return _Resp(self.panel)


def test_client_iter_rows_respects_limit_without_extra_requests(panel):
    sess = _FakeSession(panel)
    client = fv.VnClient(session=sess, sleep=0)
    rows = list(client.iter_rows(limit=2))
    assert [r["serial_number"] for r in rows] == ["VN4202645978", "VN4202645648"]
    assert sess.verify is False                            # chứng chỉ tự ký
    assert [m for m, _ in sess.calls] == ["GET", "POST"]   # đủ limit → không lật trang
    assert sess.calls[1][1].startswith(fv.VN_BASE + "/public/trademarks;jsessionid=S1?")


def test_client_iter_rows_paginates_and_never_repeats_a_mark(panel):
    sess = _FakeSession(panel)
    rows = list(fv.VnClient(session=sess, sleep=0).iter_rows(limit=0))
    ids = [r["serial_number"] for r in rows]
    assert ids == ["VN4202645978", "VN4202645648", "VN4202645228"]
    assert any("navigator-next" in url for _m, url in sess.calls)   # đã đi trang kế


# ------------------- chất lượng tập VN (REVIEW.md §4.7) -------------------

def _panel(rows, has_next=True):
    """Panel WIPO Publish tổng hợp: rows = [(id, word_mark, có_ảnh, số_đăng_ký)]."""
    trs = []
    for i, (doc_id, mark, with_img, reg) in enumerate(rows):
        img = (f"<img src='https://x/public/service/trademarks/application/{doc_id}/"
               "thumbnail?noLogo=true'/>") if with_img else ""
        trs.append(f'<tr class="{"even" if i % 2 == 0 else "odd"}" id="{doc_id}"><td>{img}'
                   f'<span class="rs-MK">{mark}</span><span class="rs-RENB">{reg}</span></td></tr>')
    nxt = ('"u":"./trademarks?0-1.IBehaviorListener.0-body-navigator-next&amp;query=*:*"'
           if has_next else "")
    return "<ajax-response>" + "".join(trs) + nxt + "</ajax-response>"


class _PagedSession(_FakeSession):
    """Trả lần lượt từng panel trong ``pages`` (POST = trang 1, mỗi GET next = trang kế)."""

    def __init__(self, pages):
        super().__init__(pages[0])
        self.pages = list(pages)

    def get(self, url, **kw):
        if "navigator-next" in url:
            self.calls.append(("GET", url))
            self.pages.pop(0)
            return _Resp(self.pages[0])
        return super().get(url, **kw)

    def post(self, url, data=None, headers=None, **kw):
        self.calls.append(("POST", url))
        return _Resp(self.pages[0])


def test_iter_rows_keeps_going_past_a_page_without_usable_rows():
    # ROOT CAUSE:
    #
    # iter_rows dừng ngay khi MỘT trang không có mẫu dùng được (`if not nxt or new == 0`),
    # mà trang toàn nhãn hình (không chữ) là chuyện bình thường → crawl kết thúc sớm.
    #
    # Đã sửa: chỉ dừng khi hết trang hoặc `max_empty_pages` trang liên tiếp không có mẫu mới.
    pages = [
        _panel([("VN1", "ALPHA", True, "1")]),
        _panel([("VN2", "", True, "2"), ("VN3", "NO IMAGE", False, "3")]),   # 0 mẫu dùng được
        _panel([("VN4", "BETA", True, "4")], has_next=False),
    ]
    rows = list(fv.VnClient(session=_PagedSession(pages), sleep=0).iter_rows(limit=0))
    assert [r["serial_number"] for r in rows] == ["VN1", "VN4"]


def test_iter_rows_stops_after_max_empty_pages():
    pages = [_panel([("VN1", "ALPHA", True, "1")])] + [_panel([])] * 10
    sess = _PagedSession(pages)
    rows = list(fv.VnClient(session=sess, sleep=0).iter_rows(limit=0, max_empty_pages=3))
    assert [r["serial_number"] for r in rows] == ["VN1"]
    assert sum("navigator-next" in url for _m, url in sess.calls) == 3


def test_iter_rows_dedupes_labels_and_can_keep_only_registered():
    pages = [_panel([
        ("VN1", "ENGLISH1984", True, "11"),
        ("VN2", "ENGLISH 1984", True, "12"),   # trùng sau khi bỏ khoảng trắng
        ("VN3", "GASU", True, ""),             # đơn chưa cấp
        ("VN4", "gasu", True, "14"),           # trùng hoa/thường với VN3
        ("VN5", "OMEGA", True, ""),            # đơn chưa cấp
    ], has_next=False)]
    client = fv.VnClient(session=_PagedSession(pages), sleep=0)
    assert [r["serial_number"] for r in client.iter_rows(limit=0)] == ["VN1", "VN3", "VN5"]
    assert client.skipped["duplicate_label"] == 2

    client = fv.VnClient(session=_PagedSession(pages), sleep=0)
    got = list(client.iter_rows(limit=0, registered_only=True))
    assert [r["serial_number"] for r in got] == ["VN1", "VN4"]
    assert client.skipped["unregistered"] == 2


def test_dedupe_key_examples():
    assert fv.dedupe_key("ENGLISH 1984") == fv.dedupe_key("English1984") == "ENGLISH1984"
    assert fv.dedupe_key("Luân  Hồi") == "LUÂNHỒI"


def test_write_outputs_keeps_registration_number(tmp_path, panel):
    # ROOT CAUSE:
    #
    # Cột CSV thô khai "reg_no" nhưng parse_rows lưu khoá "registration_no" → DictWriter
    # (extrasaction="ignore") ghi ô rỗng cho MỌI dòng; mất thông tin đơn đã cấp hay chưa.
    rows = [dict(r, image_path="raw/images_vn/x.jpg", registration_no="4-0123456-000")
            for r in fv.parse_rows(panel)[:1]]
    fv.write_outputs(rows, data_root=tmp_path)
    with (tmp_path / "raw" / "vn_marks.csv").open(encoding="utf-8", newline="") as fh:
        got = next(csv.DictReader(fh))
    assert got["registration_no"] == "4-0123456-000"
