"""Test offline cho nhánh 02 — `trademark_ocr.fetch_meta`.

Không gọi mạng: dùng `FakeSession` mô phỏng EOG REST API và dữ liệu mẫu
từ hai thời kỳ schema khác nhau (2013–2014 không có markDrawingTypeCode;
2015+ có đầy đủ).
"""

from __future__ import annotations

import csv
import json

import pytest

from trademark_ocr.fetch_meta import (
    CSV_COLUMNS,
    classify_drawing,
    image_url,
    is_candidate,
    iter_issue,
    normalize,
    run,
    select_publications,
)

# --------------------------------------------------------------------------
# Fixture: bản ghi thô 2 thời kỳ
# --------------------------------------------------------------------------

REC_2013 = {
    "cls_mult": "N",
    "currentOwner": "HONEYWELL INTERNATIONAL, INC.",
    "docket_no": "H002070",
    "fil_bs_cd": "USECR",
    "filingBasis": "Currently Use",
    "filingDate": "1923-02-01",
    "gsUsClasses": "26",
    "markText": "AQUASTAT ",
    "markTypeCode": "TRDMK",
    "markTypeDescr": "Trademark",
    "publicationDate": "2014-01-07",
    "publicationId": 110,
    "regNum": "188577",
    "registrationDate": "1924-09-02",
    "seriaNum": "71175395",
    "trademarkId": 313849,
}

REC_2020 = {
    "cls_mult": "N",
    "currentOwner": "W. A. HAMMOND DRIERITE COMPANY",
    "filingDate": "2018-05-01",
    "goodsServices": [
        {"classNum": "1", "classType": "INTL", "description": "desiccants"},
        {"classNum": "5", "classType": "INTL", "description": "other"},
        {"classNum": "26", "classType": "US", "description": "old us class"},
    ],
    "markDrawingTypeCode": "3",
    "markDrawingTypeDesc": "Illustration: drawing which also includes word(s)",
    "markText": "DRIERITE  SOLIDS",
    "markTypeCode": "TRDMK",
    "publicationDate": "2020-01-07",
    "publicationId": 900,
    "regNum": "6000000",
    "seriaNum": "71432607",
}

REC_DESIGN_ONLY = {
    "markText": None,
    "markDrawingTypeCode": "2",
    "publicationDate": "2020-01-07",
    "seriaNum": "71400000",
}


# --------------------------------------------------------------------------
# FakeSession
# --------------------------------------------------------------------------

class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    """Mô phỏng EOG: GET /publications và POST /search/all có phân trang."""

    def __init__(self, publications, records_by_date, page_size_cap=100):
        self.publications = publications
        self.records_by_date = records_by_date
        self.page_size_cap = page_size_cap
        self.post_calls = 0

    def get(self, url, timeout=None):  # noqa: ARG002
        assert url.endswith("/external/search/publications")
        return _Resp(self.publications)

    def post(self, url, json=None, timeout=None):  # noqa: ARG002
        assert url.endswith("/external/search/all")
        self.post_calls += 1
        date = json["publicationDates"][0]
        rows = self.records_by_date.get(date, [])
        offset = int(json.get("offset") or 0)
        limit = min(int(json.get("limit") or 100), self.page_size_cap)
        page = rows[offset:offset + limit]
        return _Resp({"totalCount": len(rows), "officialGazetteTrademark": page})


PUBS = [
    {"publicationDate": "2026-10-13", "publicationId": 12134},
    {"publicationDate": "2020-01-07", "publicationId": 900},
    {"publicationDate": "2014-01-07", "publicationId": 110},
]


# --------------------------------------------------------------------------
# 1. Chuẩn hoá
# --------------------------------------------------------------------------

def test_normalize_strips_and_nfc_2013_era():
    row = normalize(REC_2013, "2014-01-07", 110)
    assert row["word_mark"] == "AQUASTAT"
    assert row["serial_number"] == "71175395"
    assert row["mark_drawing_cd"] == ""
    assert row["drawing_kind"] == "unknown"
    assert row["us_class"] == "26"
    assert row["intl_class"] == ""
    assert row["image_status"] == "pending"
    assert row["image_url"] == "https://cdn.uspto.gov/img/71175395/2014-01-07/original"
    assert row["source"] == "tmog"


def test_normalize_collapses_ws_and_extracts_classes():
    row = normalize(REC_2020)
    assert row["word_mark"] == "DRIERITE SOLIDS"
    assert row["us_class"] == "1|5|26"
    assert row["intl_class"] == "1|5"
    assert row["drawing_kind"] == "design_with_words"
    assert row["publication_id"] == "900"


def test_normalize_never_raises_on_missing_fields():
    row = normalize({"seriaNum": "123"}, "2020-01-07")
    assert set(CSV_COLUMNS) == set(row.keys())


def test_classify_drawing_mapping():
    assert classify_drawing("2") == "design_only"
    assert classify_drawing("4") == "standard_character"
    assert classify_drawing("5") == "stylized_words"
    assert classify_drawing(None) == "unknown"
    assert classify_drawing("9") == "other"


def test_image_url_sizes():
    assert image_url("1", "2020-01-07") .endswith("/1/2020-01-07/original")
    assert image_url("1", "2020-01-07", "px300x200").endswith("/px300x200")


# --------------------------------------------------------------------------
# 2. Bộ lọc ứng viên
# --------------------------------------------------------------------------

def _ok_row(**kw):
    base = normalize(REC_2020)
    base.update(kw)
    return base


@pytest.mark.parametrize(
    "row,reason",
    [
        (_ok_row(word_mark=""), "no_text"),
        (_ok_row(mark_drawing_cd="2", drawing_kind="design_only"), "design_without_word"),
        (_ok_row(mark_drawing_cd="6", drawing_kind="no_drawing"), "design_without_word"),
        (_ok_row(mark_drawing_cd="9"), "drawing_excluded"),
        (_ok_row(word_mark="A"), "too_short"),
        (_ok_row(word_mark="A" * 40), "too_long"),
        (_ok_row(word_mark="ABC™"), "unsupported_char"),
    ],
)
def test_is_candidate_rejects_with_reason(row, reason):
    ok, why = is_candidate(row)
    assert (ok, why) == (False, reason)


def test_is_candidate_accepts_vietnamese_diacritics():
    row = _ok_row(word_mark="ĐÔNG Á")
    ok, why = is_candidate(row)
    assert ok and why == ""


def test_is_candidate_accepts_unknown_drawing_code_with_text():
    row = _ok_row(mark_drawing_cd="", drawing_kind="unknown")
    assert is_candidate(row)[0] is True


# --------------------------------------------------------------------------
# 3. Phân trang
# --------------------------------------------------------------------------

def test_iter_issue_paginates_and_stops():
    recs = [normalize({"seriaNum": str(1000 + i), "markText": "AB",
                       "publicationDate": "2026-10-13"}) for i in range(5)]
    raw = [{"seriaNum": str(1000 + i), "markText": "AB",
            "publicationDate": "2026-10-13"} for i in range(5)]
    sess = FakeSession(PUBS, {"2026-10-13": raw})
    got = list(iter_issue(sess, "2026-10-13", page_size=2))
    assert len(got) == 5
    assert sess.post_calls == 3  # 2 + 2 + 1, rồi dừng vì offset >= total
    assert len(recs) == 5


def test_iter_issue_respects_per_issue_limit():
    raw = [{"seriaNum": str(2000 + i), "markText": "AB",
            "publicationDate": "2026-10-13"} for i in range(10)]
    sess = FakeSession(PUBS, {"2026-10-13": raw})
    got = list(iter_issue(sess, "2026-10-13", page_size=3, per_issue_limit=4))
    assert len(got) == 6  # dừng sau khi vượt 4 trong 1 trang


def test_iter_issue_handles_empty_issue():
    sess = FakeSession(PUBS, {})
    assert list(iter_issue(sess, "2026-10-13", page_size=10)) == []


# --------------------------------------------------------------------------
# 4. Chọn issue
# --------------------------------------------------------------------------

def test_select_publications_filters_and_sorts():
    out = select_publications(PUBS, since="2015-01-01", until="2025-01-01")
    assert [p["publicationDate"] for p in out] == ["2020-01-07"]
    out_all = select_publications(PUBS, max_issues=2)
    assert [p["publicationDate"] for p in out_all] == ["2026-10-13", "2020-01-07"]


# --------------------------------------------------------------------------
# 5. run() end-to-end
# --------------------------------------------------------------------------

def _write_and_read(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_run_writes_csv_and_stats(tmp_path):
    out = tmp_path / "us_marks.csv"
    sess = FakeSession(
        PUBS,
        {
            "2026-10-13": [
                {"seriaNum": "50000007", "markText": "X 17 V 27",
                 "markDrawingTypeCode": "3", "publicationDate": "2026-10-13"},
                {"seriaNum": "50000055", "markText": "GABY SKILL",
                 "markDrawingTypeCode": "4", "publicationDate": "2026-10-13"},
                {"seriaNum": "50000099", "markText": None,
                 "markDrawingTypeCode": "2", "publicationDate": "2026-10-13"},
            ],
            "2014-01-07": [REC_2013],
        },
    )
    stats = run(out, session=sess, publications=PUBS, log=lambda *_: None)

    rows = _write_and_read(out)
    serials = {r["serial_number"] for r in rows}
    assert serials == {"50000007", "50000055", "71175395"}
    assert all(r["word_mark"] for r in rows)  # ACCEPTANCE: không rỗng label
    assert stats["rows"] == 3
    assert stats["drop_reasons"]["design_without_word"] == 1
    assert stats["by_drawing_kind"]["standard_character"] == 1
    assert stats["issues_processed"] == 3

    stats_path = out.with_suffix(out.suffix + ".stats.json")
    saved = json.loads(stats_path.read_text(encoding="utf-8"))
    assert saved["rows"] == 3
    assert list(rows[0].keys()) == CSV_COLUMNS


def test_run_limit_stops_early(tmp_path):
    out = tmp_path / "us_marks.csv"
    recs = [{"seriaNum": str(60000000 + i), "markText": f"MARK{i}",
             "markDrawingTypeCode": "5", "publicationDate": "2026-10-13"}
            for i in range(30)]
    sess = FakeSession(PUBS, {"2026-10-13": recs})
    stats = run(out, limit=5, session=sess, publications=PUBS,
                page_size=10, log=lambda *_: None)
    assert stats["rows"] == 5
    assert len(_write_and_read(out)) == 5
    assert stats["issues_processed"] == 1


def test_run_resume_skips_existing_serials(tmp_path):
    out = tmp_path / "us_marks.csv"
    first = FakeSession(PUBS, {"2026-10-13": [
        {"seriaNum": "70000001", "markText": "AAAA", "markDrawingTypeCode": "5",
         "publicationDate": "2026-10-13"},
    ]})
    run(out, session=first, publications=PUBS, log=lambda *_: None)

    second = FakeSession(PUBS, {"2026-10-13": [
        {"seriaNum": "70000001", "markText": "AAAA", "markDrawingTypeCode": "5",
         "publicationDate": "2026-10-13"},
        {"seriaNum": "70000002", "markText": "BBBB", "markDrawingTypeCode": "4",
         "publicationDate": "2026-10-13"},
    ]})
    stats = run(out, session=second, publications=PUBS, resume=True,
                log=lambda *_: None)
    serials = {r["serial_number"] for r in _write_and_read(out)}
    assert serials == {"70000001", "70000002"}
    assert stats["rows"] == 2


def test_run_shuffle_is_deterministic(tmp_path):
    recs = [{"seriaNum": str(80000000 + i), "markText": f"M{i}",
             "markDrawingTypeCode": "5", "publicationDate": "2026-10-13"}
            for i in range(10)]
    sess_a = FakeSession(PUBS, {"2026-10-13": recs})
    sess_b = FakeSession(PUBS, {"2026-10-13": recs})
    out_a, out_b = tmp_path / "a.csv", tmp_path / "b.csv"
    run(out_a, session=sess_a, publications=PUBS, shuffle=True, seed=7,
        log=lambda *_: None)
    run(out_b, session=sess_b, publications=PUBS, shuffle=True, seed=7,
        log=lambda *_: None)
    assert _write_and_read(out_a) == _write_and_read(out_b)
