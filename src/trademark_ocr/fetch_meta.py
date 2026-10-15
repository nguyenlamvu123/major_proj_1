"""Nhánh 02 — tải metadata wordmark US từ USPTO (TMOG / EOG REST API).

Nguồn dữ liệu (đã dò & kiểm chứng, KHÔNG cần API key)
-----------------------------------------------------
USPTO vận hành **Trademark Official Gazette (TMOG)** dưới dạng SPA ở
<https://tmog.uspto.gov>. SPA này gọi một REST API công khai:

    GET  {API}/external/search/publications      -> danh sách số gazette (issue)
    POST {API}/external/search/all               -> các nhãn hiệu trong 1 issue
    GET  {API}/external/single-reg/{regNum}

với ``API = https://tm-eog-service.uspto.gov/eog-rest-service/api``.
Kết quả trả về có ``seriaNum`` (serial), ``markText`` (word mark), ``regNum``,
``publicationDate``... Ảnh nhãn hiệu nằm ở CDN công khai
``https://cdn.uspto.gov/img/{seriaNum}/{publicationDate}/{size}``
(xem `trademark_ocr.fetch_images`).

Vì sao chọn nguồn này
---------------------
- **Miễn phí, không API key** (ODP/`api.uspto.gov` hiện yêu cầu key; `bulkdata.uspto.gov`
  đã ngừng tồn tại — NXDOMAIN).
- Ảnh và nhãn lấy từ **cùng một issue** → nhãn khớp ảnh (self-consistent).
- Phủ ~688 issue, 2013-08-13 → nay (ảnh CDN có từ ~2014), đủ lớn để train.

Schema thay đổi theo thời kỳ (đã kiểm chứng)
-------------------------------------------
- 2013–2014: **không** có ``markDrawingTypeCode``/``goodsServices``; có ``gsUsClasses``.
- 2015 trở đi: có ``markDrawingTypeCode`` + ``markDrawingTypeDesc`` + ``goodsServices``.
→ Mọi truy cập field phải dùng ``.get()``.

Mã "mark drawing" (``lookups/drawing_codes.csv`` của USPTO):
1 = typeset (cũ), 2 = design-only (không chữ), 3 = design + chữ,
4 = standard character (chữ đánh máy), 5 = chữ cách điệu, 6 = không có hình (vd âm thanh).
Ta giữ {1,3,4,5} (có chữ) và loại {2,6}.

CLI
---
    python -m trademark_ocr.fetch_meta --limit 5000 \
        --out $DATA_ROOT/raw/meta/us_marks.csv

Sinh ra:
- ``us_marks.csv``            : bảng nhãn hiệu ứng viên (wordmark).
- ``us_marks.stats.json``     : thống kê (số issue, lý do rớt, phân bố loại hình).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import requests

from trademark_ocr.charset import (
    CHARSET_VI,
    collapse_ws,
    contains_only_supported,
    to_nfc,
)

# --------------------------------------------------------------------------
# Hằng số nguồn dữ liệu
# --------------------------------------------------------------------------

TMOG_API: str = "https://tm-eog-service.uspto.gov/eog-rest-service/api"
IMG_CDN: str = "https://cdn.uspto.gov"

PUBLICATIONS_ENDPOINT: str = f"{TMOG_API}/external/search/publications"
SEARCH_ENDPOINT: str = f"{TMOG_API}/external/search/all"

#: API chấp nhận tối đa 100 bản ghi / trang (đã kiểm chứng: limit=250 -> rỗng).
MAX_PAGE_SIZE: int = 100

DEFAULT_DATA_ROOT: str = os.environ.get(
    "DATA_ROOT", "/media/zaibachkhoa/AC84-F995/trademark_data"
)

#: Cột CSV — ACCEPTANCE nhánh 02 yêu cầu tối thiểu
#: ``serial_number, word_mark, image_status``; phần còn lại phục vụ nhánh sau.
CSV_COLUMNS: list[str] = [
    "serial_number",
    "word_mark",
    "image_status",
    "mark_drawing_cd",
    "drawing_kind",
    "mark_type",
    "publication_date",
    "publication_id",
    "reg_number",
    "filing_date",
    "registration_date",
    "owner",
    "us_class",
    "intl_class",
    "filing_basis",
    "pub_reason",
    "image_url",
    "source",
]

#: Mã drawing chứa chữ (giữ) — loại 2 (không chữ) & 6 (không hình).
WORD_BEARING_CODES: frozenset[str] = frozenset({"1", "3", "4", "5"})

_IMAGE_STATUS_PENDING: str = "pending"


# --------------------------------------------------------------------------
# 1. Chuẩn hoá bản ghi
# --------------------------------------------------------------------------

def classify_drawing(code: Any) -> str:
    """Ánh xạ ``markDrawingTypeCode`` -> nhãn loại hình dễ đọc."""
    c = "" if code is None else str(code).strip()
    return {
        "1": "typeset_old",
        "2": "design_only",
        "3": "design_with_words",
        "4": "standard_character",
        "5": "stylized_words",
        "6": "no_drawing",
        "": "unknown",
    }.get(c, "other")


def image_url(serial: str, publication_date: str, size: str = "original") -> str:
    """URL ảnh nhãn hiệu trên CDN công khai của USPTO.

    ``size`` ∈ {``px160x130``, ``px300x200``, ``original``}.
    """
    return f"{IMG_CDN}/img/{serial}/{publication_date}/{size}"


def _s(value: Any) -> str:
    """Ép về chuỗi đã strip (``None`` -> "")."""
    return "" if value is None else str(value).strip()


def _get_classes(rec: dict[str, Any]) -> tuple[str, str]:
    """Trả (us_class, intl_class) — schema đổi tên theo thời kỳ."""
    us = _s(rec.get("gsUsClasses"))
    if not us:
        gs = rec.get("goodsServices") or []
        us = "|".join(_s(g.get("classNum")) for g in gs if _s(g.get("classNum")))
    intl = _s(rec.get("gsIntlClasses"))
    if not intl:
        gs = rec.get("goodsServices") or []
        intl = "|".join(
            _s(g.get("classNum"))
            for g in gs
            if _s(g.get("classType")).upper() == "INTL" and _s(g.get("classNum"))
        )
    return us, intl


def normalize(rec: dict[str, Any], publication_date: str | None = None,
              publication_id: Any = None) -> dict[str, str]:
    """Chuẩn hoá 1 bản ghi TMOG thành dict phẳng đúng ``CSV_COLUMNS``.

    - ``word_mark`` được NFC + gộp khoảng trắng + strip (TMOG hay trả 'GYV ').
    - Không bao giờ ném lỗi khi thiếu field (schema theo thời kỳ).
    """
    serial = _s(rec.get("seriaNum"))
    pub = _s(rec.get("publicationDate")) or _s(publication_date)
    code = _s(rec.get("markDrawingTypeCode"))
    us_cls, intl_cls = _get_classes(rec)
    pid = rec.get("publicationId", publication_id)
    return {
        "serial_number": serial,
        "word_mark": collapse_ws(to_nfc(_s(rec.get("markText")))),
        "image_status": _IMAGE_STATUS_PENDING,
        "mark_drawing_cd": code,
        "drawing_kind": classify_drawing(code),
        "mark_type": _s(rec.get("markTypeCode")),
        "publication_date": pub,
        "publication_id": _s(pid),
        "reg_number": _s(rec.get("regNum")),
        "filing_date": _s(rec.get("filingDate")),
        "registration_date": _s(rec.get("registrationDate")),
        "owner": _s(rec.get("currentOwner")),
        "us_class": us_cls,
        "intl_class": intl_cls,
        "filing_basis": _s(rec.get("fil_bs_cd")) or _s(rec.get("filingBasis")),
        "pub_reason": _s(rec.get("pubReasonCode")) or _s(rec.get("pubReason")),
        "image_url": image_url(serial, pub) if serial and pub else "",
        "source": "tmog",
    }


# --------------------------------------------------------------------------
# 2. Bộ lọc ứng viên
# --------------------------------------------------------------------------

def is_candidate(
    row: dict[str, str],
    charset: str = CHARSET_VI,
    min_len: int = 2,
    max_len: int = 25,
    allowed_codes: Iterable[str] = WORD_BEARING_CODES,
) -> tuple[bool, str]:
    """Quyết định 1 dòng có dùng làm mẫu huấn luyện được không.

    Trả ``(ok, reason)``; ``reason`` là "" khi ``ok``.
    """
    word = row.get("word_mark", "")
    code = row.get("mark_drawing_cd", "")
    allowed = set(allowed_codes)
    if code and code in {"2", "6"}:
        return False, "design_without_word"
    if code and code not in allowed:
        return False, "drawing_excluded"
    if not word:
        return False, "no_text"
    if len(word) < min_len:
        return False, "too_short"
    if len(word) > max_len:
        return False, "too_long"
    if not contains_only_supported(word, charset):
        return False, "unsupported_char"
    return True, ""


# --------------------------------------------------------------------------
# 3. Truy cập API
# --------------------------------------------------------------------------

def fetch_publications(
    session: requests.Session | Any,
    *,
    timeout: float = 30.0,
    endpoint: str = PUBLICATIONS_ENDPOINT,
) -> list[dict[str, Any]]:
    """Danh sách các số gazette (mới nhất trước)."""
    resp = session.get(endpoint, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"publications: phản hồi không phải list: {type(data)}")
    return [p for p in data if p.get("publicationDate")]


def iter_issue(
    session: requests.Session | Any,
    publication_date: str,
    *,
    publication_id: Any = None,
    page_size: int = MAX_PAGE_SIZE,
    timeout: float = 30.0,
    per_issue_limit: int | None = None,
    sleep: float = 0.0,
    endpoint: str = SEARCH_ENDPOINT,
) -> Iterator[dict[str, Any]]:
    """Sinh ra từng bản ghi thô của một issue (phân trang bằng ``offset``)."""
    page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
    offset = 0
    total: int | None = None
    yielded = 0
    while True:
        body: dict[str, Any] = {
            "publicationDates": [publication_date],
            "offset": offset,
            "limit": page_size,
        }
        if publication_id is not None:
            body["publicationIds"] = [publication_id]
        resp = session.post(endpoint, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if total is None:
            total = int(data.get("totalCount") or 0)
        rows = data.get("officialGazetteTrademark") or []
        if not rows:
            break
        for rec in rows:
            yield rec
        yielded += len(rows)
        offset += len(rows)
        if total and offset >= total:
            break
        if per_issue_limit and yielded >= per_issue_limit:
            break
        if sleep:
            time.sleep(sleep)


# --------------------------------------------------------------------------
# 4. Vòng chạy chính
# --------------------------------------------------------------------------

def _iso(d: str | None) -> str:
    return (d or "").strip()


def select_publications(
    pubs: Sequence[dict[str, Any]],
    *,
    since: str | None = None,
    until: str | None = None,
    max_issues: int | None = None,
) -> list[dict[str, Any]]:
    """Lọc issue theo khoảng ngày (bao gồm) và giới hạn số issue; mới nhất trước."""
    since, until = _iso(since), _iso(until)
    out = [
        p for p in pubs
        if (not since or p["publicationDate"] >= since)
        and (not until or p["publicationDate"] <= until)
    ]
    out.sort(key=lambda p: p["publicationDate"], reverse=True)
    if max_issues:
        out = out[: int(max_issues)]
    return out


def write_csv(rows: Sequence[dict[str, str]], path: Path | str,
              columns: Sequence[str] = CSV_COLUMNS) -> Path:
    """Ghi CSV (UTF-8, LF) theo cách nguyên tử (tmp -> replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    tmp.replace(path)
    return path


def load_existing_rows(path: Path | str) -> list[dict[str, str]]:
    """Đọc toàn bộ CSV cũ (phục vụ ``--resume``)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def run(
    out: Path | str,
    *,
    limit: int = 0,
    since: str | None = None,
    until: str | None = None,
    max_issues: int | None = None,
    per_issue_limit: int | None = None,
    page_size: int = MAX_PAGE_SIZE,
    charset: str = CHARSET_VI,
    min_len: int = 2,
    max_len: int = 25,
    allowed_codes: Iterable[str] = WORD_BEARING_CODES,
    sleep: float = 0.0,
    timeout: float = 30.0,
    shuffle: bool = False,
    seed: int = 1111,
    resume: bool = False,
    session: requests.Session | Any | None = None,
    publications: Sequence[dict[str, Any]] | None = None,
    on_issue: Callable[[dict[str, Any], int], None] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Thu thập metadata ứng viên -> CSV + stats. Trả về dict thống kê."""
    t0 = time.time()
    out = Path(out)
    session = session if session is not None else requests.Session()

    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    by_kind: dict[str, int] = {}
    if resume:
        rows = load_existing_rows(out)
        seen = {r["serial_number"] for r in rows if r.get("serial_number")}
        for r in rows:
            k = r.get("drawing_kind") or "unknown"
            by_kind[k] = by_kind.get(k, 0) + 1
        if rows:
            log(f"[meta] resume: đã có {len(rows)} dòng trong {out.name}")

    pubs = list(publications) if publications is not None else fetch_publications(
        session, timeout=timeout
    )
    pubs = select_publications(pubs, since=since, until=until, max_issues=max_issues)
    log(f"[meta] {len(pubs)} issue sẽ xử lý (mới nhất: "
        f"{pubs[0]['publicationDate'] if pubs else 'n/a'})")

    drop: dict[str, int] = {}
    issues_done = 0
    for pub in pubs:
        if limit and len(rows) >= limit:
            break
        pdate = pub["publicationDate"]
        pid = pub.get("publicationId")
        for rec in iter_issue(
            session, pdate, publication_id=pid, page_size=page_size,
            timeout=timeout, per_issue_limit=per_issue_limit, sleep=sleep,
        ):
            row = normalize(rec, pdate, pid)
            serial = row["serial_number"]
            if not serial or serial in seen:
                continue
            ok, reason = is_candidate(row, charset, min_len, max_len, allowed_codes)
            if not ok:
                drop[reason] = drop.get(reason, 0) + 1
                continue
            seen.add(serial)
            rows.append(row)
            by_kind[row["drawing_kind"]] = by_kind.get(row["drawing_kind"], 0) + 1
            if limit and len(rows) >= limit:
                break
        issues_done += 1
        if on_issue:
            on_issue(pub, len(rows))
        log(f"[meta] issue {pdate}: tổng {len(rows)} dòng")

    if shuffle:
        random.Random(seed).shuffle(rows)
    write_csv(rows, out)

    stats: dict[str, Any] = {
        "out": str(out),
        "rows": len(rows),
        "issues_processed": issues_done,
        "drop_reasons": dict(sorted(drop.items(), key=lambda kv: -kv[1])),
        "by_drawing_kind": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])),
        "limit": limit,
        "since": since,
        "until": until,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    stats_path = out.with_suffix(out.suffix + ".stats.json")
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"[meta] wrote {out} ({len(rows)} dòng) + {stats_path.name}")
    return stats


# --------------------------------------------------------------------------
# 5. CLI
# --------------------------------------------------------------------------

def _parse_codes(s: str) -> list[str]:
    return [c.strip() for c in s.split(",") if c.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Tải metadata wordmark US từ USPTO TMOG (EOG REST API).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--out", default=str(Path(DEFAULT_DATA_ROOT) / "raw" / "meta" / "us_marks.csv"),
                    help="đường dẫn CSV đầu ra")
    ap.add_argument("--limit", type=int, default=0,
                    help="số dòng tối đa (0 = không giới hạn)")
    ap.add_argument("--since", default=None, help="chỉ lấy issue từ ngày YYYY-MM-DD")
    ap.add_argument("--until", default=None, help="chỉ lấy issue tới ngày YYYY-MM-DD")
    ap.add_argument("--max-issues", type=int, default=None, help="số issue tối đa")
    ap.add_argument("--per-issue-limit", type=int, default=None,
                    help="số bản ghi thô tối đa mỗi issue")
    ap.add_argument("--page-size", type=int, default=MAX_PAGE_SIZE,
                    help=f"kích thước trang API (tối đa {MAX_PAGE_SIZE})")
    ap.add_argument("--drawing-codes", default=",".join(sorted(WORD_BEARING_CODES)),
                    help="các mã drawing được giữ")
    ap.add_argument("--min-len", type=int, default=2, help="độ dài nhãn tối thiểu")
    ap.add_argument("--max-len", type=int, default=25, help="độ dài nhãn tối đa")
    ap.add_argument("--sleep", type=float, default=0.0, help="nghỉ giữa các trang (giây)")
    ap.add_argument("--timeout", type=float, default=30.0, help="timeout HTTP (giây)")
    ap.add_argument("--shuffle", action="store_true", help="trộn dòng ngẫu nhiên (theo --seed)")
    ap.add_argument("--seed", type=int, default=1111, help="seed cho --shuffle")
    ap.add_argument("--resume", action="store_true",
                    help="bỏ qua serial đã có trong CSV cũ")
    ap.add_argument("--list-issues", action="store_true",
                    help="chỉ in danh sách issue rồi thoát")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_issues:
        pubs = select_publications(
            fetch_publications(requests.Session(), timeout=args.timeout),
            since=args.since, until=args.until, max_issues=args.max_issues,
        )
        for p in pubs:
            print(p["publicationDate"], p.get("publicationId"))
        print(f"# {len(pubs)} issue")
        return 0

    run(
        args.out,
        limit=args.limit,
        since=args.since,
        until=args.until,
        max_issues=args.max_issues,
        per_issue_limit=args.per_issue_limit,
        page_size=args.page_size,
        min_len=args.min_len,
        max_len=args.max_len,
        allowed_codes=_parse_codes(args.drawing_codes),
        sleep=args.sleep,
        timeout=args.timeout,
        shuffle=args.shuffle,
        seed=args.seed,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
