"""Nhánh 11 — thu nhãn hiệu Việt Nam từ WIPO Publish (IP Vietnam).

Nguồn: ``https://wipopublish.ipvietnam.gov.vn/wopublish-search`` (Wicket/AJAX).
Trang công khai của IP Vietnam là **WIPO Publish**: kết quả không nằm trong HTML
ban đầu mà được trả về qua một lời gọi AJAX của Wicket.

Quy trình (đã kiểm chứng thực tế):

1. ``GET /public/trademarks?query=*:*`` → lấy phiên + đọc trong HTML:
   - id form (``"f":"id6"`` — **thay đổi theo phiên**, không hard-code),
   - URL hành vi tìm kiếm (``"u":"...searchSubmitLink..."``),
   - ``Wicket.Ajax.baseUrl`` (để gửi header ``Wicket-Ajax-BaseURL``).
2. ``POST`` hành vi đó (header ``Wicket-Ajax: true``) với body
   ``<fid>_hf_0=&intelligentSearchInput=&searchSubmitLink=Tra+cuu``
   → XML ``<ajax-response>`` chứa panel kết quả (**50 hàng/trang**).
3. Phân trang: trong panel có ``navigator-next`` (cả ``href`` lẫn behavior ``"u"``)
   → GET để lấy panel trang kế.
4. Ảnh: ``/public/service/trademarks/application/<id>/thumbnail?noLogo=true&disclaimer=Combined``
   (JPEG).  Trường ``rs-MK`` là **nhãn wordmark** cần cho OCR.

Sản phẩm (dưới ``DATA_ROOT``):
- ``raw/vn_marks.csv``       : bản ghi thô (id, word_mark, số đơn, ngày, chủ đơn...).
- ``raw/images_vn/*.jpg``    : ảnh nhãn hiệu + ``raw/images_vn/index.csv``.
- ``processed/split_test_vn.csv`` : cột ``image_path,word_mark`` (khớp ``build_lmdb``).
- ``processed/test_vn.txt``  : một nhãn mỗi dòng (NFC) — dùng cho nghiệm thu.
- ``raw/vn_report.json``     : thống kê.

CLI
---
    python3 -m trademark_ocr.fetch_vn --limit 50
    python3 -m trademark_ocr.fetch_vn --verify

Lưu ý: server dùng chứng chỉ tự ký → phải ``verify=False`` (tương tự ``curl -k``).
"""

from __future__ import annotations

import argparse
import csv
import html as _html
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urljoin

DEFAULT_DATA_ROOT: str = os.environ.get(
    "DATA_ROOT", "/media/zaibachkhoa/AC84-F995/trademark_data"
)
VN_BASE = "https://wipopublish.ipvietnam.gov.vn/wopublish-search"
SEARCH_PATH = "/public/trademarks?query=*:*"

#: cột CSV split (khớp ``clean.CLEAN_COLUMNS`` một phần; build_lmdb cần image_path+word_mark)
SPLIT_COLUMNS: tuple[str, ...] = (
    "serial_number", "image_path", "word_mark", "n_chars",
)

_ROW_RE = re.compile(r'<tr class="(?:even|odd)" id="([^"]+)">(.*?)</tr>', re.S)
_FORM_RE = re.compile(r'\{"f":"(id[0-9a-z]+)","u":"([^"]*searchSubmitLink[^"]*)"')
_BASEURL_RE = re.compile(r'Wicket\.Ajax\.baseUrl="([^"]+)"')
_NEXT_HREF_RE = re.compile(r'href="([^"]*navigator-next[^"]*)"')
_NEXT_BEH_RE = re.compile(r'"u":"([^"]*navigator-next[^"]*)"')

_FIELDS = {
    "word_mark": "rs-MK",
    "application_no": "rs-AFNB_ORI",
    "filing_date": "rs-AFDT",
    "publication_date": "rs-PBDT",
    "registration_no": "rs-RENB",
    "applicant": "rs-APNA",
    "nice_class": "rs-NCL",
    "status": "rs-STLB",
}


def to_label(text: str) -> str:
    """Chuẩn hoá nhãn: bỏ HTML entity, gộp khoảng trắng, **NFC** (giữ dấu tiếng Việt)."""
    import html as _h
    import unicodedata as _u

    s = _h.unescape(text or "")
    s = re.sub(r"<[^>]+>", " ", s)
    s = " ".join(s.split())
    return unicodedata.normalize("NFC", s)


def _cell(body: str, cls: str) -> str:
    m = re.search(rf'class="[^"]*\b{re.escape(cls)}\b[^"]*">(.*?)</span>', body, re.S)
    if not m:
        return ""
    return _field_text(m.group(1))


def _field_text(text: str) -> str:
    import html as _h
    return _h.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def parse_rows(html: str) -> list[dict[str, str]]:
    """Tách các hàng kết quả từ HTML/XML panel hoặc trang kết quả."""
    rows: list[dict[str, str]] = []
    for doc_id, body in _ROW_RE.findall(html):
        row: dict[str, str] = {"serial_number": doc_id}
        img = re.search(r"src='([^']*thumbnail[^']*)'", body)
        if not img:
            img = re.search(r'src="([^"]*thumbnail[^"]*)"', body)
        for key, cls in _FIELDS.items():
            m = re.search(rf'class="{cls}">(.*?)</span>', body, re.S)
            row[key] = _field_text(m.group(1)) if m else ""
        row["image_url"] = img.group(1).replace("&amp;", "&") if img else ""
        row["word_mark"] = to_label(row.get("word_mark", ""))
        row["n_chars"] = str(len(row["word_mark"]))
        rows.append(row)
    return rows


def usable(row: dict[str, str]) -> bool:
    """Chỉ nhận nhãn hiệu có **chữ** (wordmark) + có ảnh."""
    return bool(row.get("word_mark")) and bool(row.get("image_url"))


def next_url(html: str) -> str | None:
    """URL hành vi 'trang kế' trong panel kết quả (None nếu hết trang)."""
    m = _NEXT_BEH_RE.search(html) or _NEXT_HREF_RE.search(html)
    return m.group(1).replace("&amp;", "&") if m else None


def parse_form(html: str) -> tuple[str, str, str]:
    """Đọc ``(form_id, submit_url, base_url)`` từ HTML trang tìm kiếm."""
    m = _FORM_RE.search(html)
    if not m:
        raise RuntimeError("không tìm thấy form tìm kiếm trong HTML (site đổi cấu trúc?)")
    b = _BASEURL_RE.search(html)
    base = b.group(1).replace("&amp;", "&") if b else "public/trademarks?0&query=*:*"
    return m.group(1), m.group(2).replace("&amp;", "&"), base


class VnClient:
    """Phiên làm việc với WIPO Publish của IP Vietnam (``verify=False``)."""

    def __init__(self, base: str = VN_BASE, timeout: int = 60,
                 session: object | None = None, sleep: float = 0.2):
        import requests
        import urllib3
        urllib3.disable_warnings()
        self.base = base.rstrip("/")
        self.public = f"{self.base}/public/"
        self.timeout = timeout
        self.sleep = sleep
        self.s = session or requests.Session()
        self.s.verify = False
        self.s.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
        self.form_id: str | None = None
        self.submit_path: str | None = None
        self.base_url: str | None = None

    def _abs(self, rel: str) -> str:
        return rel if rel.startswith("http") else self.public + rel.lstrip("./")

    def warmup(self, query: str = "*:*") -> str:
        html = self.s.get(f"{self.base}/public/trademarks?query={query}",
                          timeout=self.timeout).text
        self.form_id, self.submit_path, self.base_url = parse_form(html)
        return html

    def _ajax_headers(self) -> dict[str, str]:
        return {
            "Wicket-Ajax": "true",
            "Wicket-Ajax-BaseURL": self.base_url or "public/trademarks?0&query=*:*",
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Referer": f"{self.base}/public/trademarks",
        }

    def search(self, query: str = "*:*") -> str:
        """Gửi tìm kiếm → HTML/XML panel trang 1."""
        if self.form_id is None:
            self.warmup(query)
        body = (f"{self.form_id}_hf_0=&intelligentSearchInput=&"
                f"searchSubmitLink=Tra+cuu")
        r = self.s.post(self._abs(self.submit_path or ""), data=body,
                        headers=self._ajax_headers(), timeout=self.timeout)
        return r.text

    def fetch_page(self, rel_url: str) -> str:
        if self.sleep:
            import time
            time.sleep(self.sleep)
        return self.s.get(self._abs(rel_url), headers=self._ajax_headers(),
                          timeout=self.timeout).text

    def download(self, url: str, dest: "Path") -> int:
        import time
        r = self.s.get(url, timeout=self.timeout)
        if r.status_code != 200 or not r.content:
            raise RuntimeError(f"tải ảnh lỗi {r.status_code}: {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        if self.sleep:
            time.sleep(self.sleep)
        return len(r.content)

    def iter_rows(self, limit: int, query: str = "*:*",
                  max_pages: int = 200,
                  log=lambda *_: None):
        """Sinh các hàng *usable* (có chữ + ảnh) tới khi đủ ``limit``."""
        panel = self.search(query)
        seen: set[str] = set()
        got = 0
        page = 0
        while panel and page < max_pages:
            page += 1
            rows = parse_rows(panel)
            new = 0
            for row in rows:
                if not usable(row) or row["serial_number"] in seen:
                    continue
                seen.add(row["serial_number"])
                new += 1
                yield row
                got += 1
                if limit and got >= limit:
                    return
            log(f"[vn] trang {page}: {len(rows)} hàng ({new} dùng được, tổng {got})")
            nxt = next_url(panel)
            if not nxt or new == 0:
                return
            panel = self.fetch_page(nxt)


def verify_images(images_dir: "Path") -> dict[str, object]:
    """Mở lại mọi ảnh bằng PIL, trả tỉ lệ mở được."""
    from PIL import Image
    files = sorted(p for p in Path(images_dir).glob("*") if p.suffix.lower() in
                   {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    ok, bad = 0, []
    for p in files:
        try:
            with Image.open(p) as im:
                im.verify()
            ok += 1
        except Exception:
            bad.append(p.name)
    total = len(files)
    return {"total": total, "ok": ok, "bad": bad,
            "ratio": (ok / total) if total else 0.0}


def write_outputs(rows: list[dict[str, str]], *, data_root: "Path",
                  log=lambda *_: None) -> dict[str, object]:
    """Ghi CSV thô, CSV split, txt nhãn, JSON báo cáo."""
    root = Path(data_root)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "processed").mkdir(parents=True, exist_ok=True)

    marks = root / "raw" / "vn_marks.csv"
    cols = ["serial_number", "word_mark", "n_chars", "application_no", "filing_date",
            "publication_date", "reg_no", "applicant", "nice_class", "status",
            "image_url", "image_path"]
    with marks.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    split = root / "processed" / "split_test_vn.csv"
    with split.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(SPLIT_COLUMNS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    labels = root / "processed" / "test_vn.txt"
    labels.write_text("\n".join(r["word_mark"] for r in rows) + "\n", encoding="utf-8")

    report = {
        "n": len(rows),
        "marks_csv": str(marks),
        "split_csv": str(split),
        "labels_txt": str(labels),
    }
    (root / "raw" / "vn_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"[vn] ghi {len(rows)} nhãn -> {split} | {labels}")
    return report


def run(*, limit: int = 300, query: str = "*:*", data_root: "Path | str" = None,
        out_dir: "Path | str | None" = None, device: str = "cpu",
        verify_only: bool = False, log=print) -> dict[str, object]:
    root = Path(data_root) if data_root else Path(DEFAULT_DATA_ROOT)
    images = Path(out_dir) if out_dir else root / "raw" / "images_vn"

    if verify_only:
        res = verify_images(images)
        log(f"[vn] verify: {res['ok']}/{res['total']} ảnh mở được "
            f"({res['ratio']*100:.1f}%)")
        if res["bad"]:
            log(f"[vn] ảnh lỗi: {res['bad'][:10]}")
        return res

    client = VnClient(sleep=0.15)
    rows: list[dict[str, str]] = []
    for row in client.iter_rows(limit, query=query, log=log):
        dest = images / f"{row['serial_number']}.jpg"
        try:
            client.download(row["image_url"], dest)
        except Exception as e:  # pragma: no cover - phụ thuộc mạng
            log(f"[vn] bỏ {row['serial_number']}: {e}")
            continue
        row["image_path"] = str(dest.relative_to(root)) \
            if str(dest).startswith(str(root)) else str(dest)
        rows.append(row)
    report = write_outputs(rows, data_root=root, log=log)
    report.update(verify_images(images))
    log(f"[vn] verify: {report['ok']}/{report['total']} ảnh mở được "
        f"({report['ratio']*100:.1f}%)")
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Thu nhãn hiệu Việt Nam (WIPO Publish) — nhánh 11.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--limit", type=int, default=300, help="số nhãn hiệu cần lấy")
    ap.add_argument("--query", default="*:*")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--out-dir", default=None, help="thư mục ảnh (mặc định raw/images_vn)")
    ap.add_argument("--verify", action="store_true", help="chỉ kiểm tra ảnh đã tải")
    return ap


def main(argv: "list[str] | None" = None) -> int:
    a = build_parser().parse_args(argv)
    res = run(limit=a.limit, query=a.query, data_root=a.data_root,
              out_dir=a.out_dir, verify_only=a.verify)
    if a.verify and res.get("total") and res["ratio"] < 0.95:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
