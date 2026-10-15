"""Nhánh 03 — tải ảnh nhãn hiệu US từ CDN công khai của USPTO.

Ảnh nằm ở ``https://cdn.uspto.gov/img/{serial}/{publicationDate}/{size}``
(``size`` ∈ {``px160x130``, ``px300x200``, ``original``}) — URL đã có sẵn trong cột
``image_url`` của ``us_marks.csv`` (nhánh 02). Ảnh có từ ~2014 trở đi.

Đầu ra
------
- ``$DATA_ROOT/raw/images_us/{serial}.png``       : ảnh gốc.
- ``$DATA_ROOT/raw/images_us/index.csv``          : sổ đăng ký ảnh (trạng thái, kích
  thước, sha1…) — **nguồn sự thật** cho các nhánh sau.
- (tuỳ chọn) cập nhật cột ``image_status`` (``ok``/``missing``) trong ``us_marks.csv``.

Đặc tính
--------
- **Resume**: file đã có & ``status=ok`` → không tải lại (``downloaded: 0 new``).
- **Song song** có kiểm soát (thread pool), ghi nguyên tử (tmp → replace).
- **Verify** mở lại ảnh bằng PIL, đếm tỉ lệ mở được (ACCEPTANCE: ≥ 95%).

CLI
---
    bash scripts/02_get_images.sh --limit 200
    python3 -m trademark_ocr.fetch_images --verify --limit 200
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import requests
from PIL import Image

DEFAULT_DATA_ROOT: str = os.environ.get(
    "DATA_ROOT", "/media/zaibachkhoa/AC84-F995/trademark_data"
)

#: Dưới ngưỡng này coi như file lỗi (ảnh hợp lệ nhỏ nhất thực tế vài trăm byte).
MIN_BYTES: int = 100

#: Tỉ lệ mở-được tối thiểu để ACCEPTANCE nhánh 03 pass.
VERIFY_MIN_RATIO: float = 0.95

INDEX_COLUMNS: list[str] = [
    "serial_number",
    "file_name",
    "word_mark",
    "mark_drawing_cd",
    "drawing_kind",
    "publication_date",
    "status",
    "http_status",
    "width",
    "height",
    "bytes",
    "sha1",
    "reason",
    "url",
]

_TLS = threading.local()


# --------------------------------------------------------------------------
# 1. Tiện ích ảnh
# --------------------------------------------------------------------------

def probe_image(path: Path | str) -> tuple[bool, int, int, str]:
    """Mở & kiểm tra ảnh bằng PIL. Trả ``(ok, width, height, reason)``."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
        return True, int(w), int(h), ""
    except Exception as exc:  # noqa: BLE001
        return False, 0, 0, type(exc).__name__


def sha1_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def default_session_factory() -> requests.Session:
    """1 session cho mỗi luồng (``requests.Session`` không thread-safe)."""
    sess = getattr(_TLS, "session", None)
    if sess is None:
        sess = requests.Session()
        _TLS.session = sess
    return sess


# --------------------------------------------------------------------------
# 2. Tải 1 ảnh
# --------------------------------------------------------------------------

def download_one(
    url: str,
    dest: Path | str,
    *,
    session: requests.Session | Any | None = None,
    timeout: float = 30.0,
    retries: int = 2,
    backoff: float = 1.0,
    min_bytes: int = MIN_BYTES,
) -> dict[str, Any]:
    """Tải 1 ảnh về ``dest`` (tmp → replace). Không ném lỗi; trả dict trạng thái."""
    dest = Path(dest)
    session = session if session is not None else default_session_factory()
    last_reason = "error:Unknown"
    for attempt in range(max(0, retries) + 1):
        try:
            resp = session.get(url, timeout=timeout)
            code = getattr(resp, "status_code", 200)
            if code != 200:
                return {"status": "missing", "http_status": code,
                        "reason": f"http_{code}"}
            data = resp.content
            ctype = (resp.headers or {}).get("Content-Type", "")
            if len(data) < min_bytes:
                return {"status": "missing", "http_status": 200,
                        "reason": "too_small"}
            if "image" not in ctype.lower():
                return {"status": "missing", "http_status": 200,
                        "reason": f"not_image:{ctype[:40]}"}
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            ok, w, h, reason = probe_image(tmp)
            if not ok:
                tmp.unlink(missing_ok=True)
                return {"status": "missing", "http_status": 200,
                        "reason": f"corrupt:{reason}"}
            tmp.replace(dest)
            return {"status": "ok", "http_status": 200, "width": w, "height": h,
                    "bytes": len(data), "sha1": sha1_file(dest), "reason": ""}
        except Exception as exc:  # noqa: BLE001
            last_reason = f"error:{type(exc).__name__}"
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    return {"status": "missing", "http_status": 0, "reason": last_reason}


# --------------------------------------------------------------------------
# 3. Index CSV
# --------------------------------------------------------------------------

def read_index(path: Path | str) -> dict[str, dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        return {r["serial_number"]: dict(r) for r in csv.DictReader(fh)
                if r.get("serial_number")}


def write_index(rows: Iterable[dict[str, str]], path: Path | str,
                columns: Sequence[str] = INDEX_COLUMNS) -> Path:
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


def read_meta(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def update_meta_status(
    meta_path: Path | str,
    status_by_serial: dict[str, str],
    *,
    columns: Sequence[str] | None = None,
) -> int:
    """Cập nhật cột ``image_status`` trong ``us_marks.csv`` (ghi nguyên tử)."""
    meta_path = Path(meta_path)
    rows = read_meta(meta_path)
    cols = list(columns) if columns else (list(rows[0].keys()) if rows else [])
    if "image_status" not in cols:
        cols.append("image_status")
    changed = 0
    for r in rows:
        st = status_by_serial.get(r.get("serial_number", ""))
        if st and r.get("image_status") != st:
            r["image_status"] = st
            changed += 1
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(meta_path)
    return changed


# --------------------------------------------------------------------------
# 4. Vòng chạy chính
# --------------------------------------------------------------------------

def _file_name(serial: str, size: str) -> str:
    """Ảnh 'original' lưu ``{serial}.png``; size khác lưu kèm hậu tố."""
    return f"{serial}.png" if size == "original" else f"{serial}__{size}.png"


def run(
    *,
    meta_path: Path | str | None = None,
    img_dir: Path | str | None = None,
    index_path: Path | str | None = None,
    limit: int = 0,
    workers: int = 8,
    timeout: float = 30.0,
    retries: int = 2,
    sleep: float = 0.0,
    size: str = "original",
    retry_missing: bool = False,
    update_meta: bool = True,
    session_factory: Callable[[], Any] | None = None,
    fetch_fn: Callable[..., dict[str, Any]] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Tải ảnh cho các dòng trong metadata. Trả về dict thống kê."""
    t0 = time.time()
    root = Path(DEFAULT_DATA_ROOT)
    meta_path = Path(meta_path) if meta_path else root / "raw" / "meta" / "us_marks.csv"
    img_dir = Path(img_dir) if img_dir else root / "raw" / "images_us"
    index_path = Path(index_path) if index_path else img_dir / "index.csv"

    meta = read_meta(meta_path)
    if limit:
        meta = meta[:limit]

    index = read_index(index_path)
    factory = session_factory or default_session_factory
    fetcher = fetch_fn or download_one

    pending: list[dict[str, str]] = []
    skipped = 0
    for row in meta:
        serial = row.get("serial_number", "")
        url = row.get("image_url", "")
        if not serial or not url:
            continue
        prev = index.get(serial)
        dest = img_dir / _file_name(serial, size)
        if prev and prev.get("status") == "ok" and dest.exists():
            skipped += 1
            continue
        if prev and prev.get("status") != "ok" and not retry_missing:
            skipped += 1
            continue
        pending.append(row)

    log(f"[images] {len(pending)} ảnh cần tải, bỏ qua {skipped} (tổng {len(meta)})")

    def work(row: dict[str, str]) -> dict[str, str]:
        serial = row["serial_number"]
        url = row["image_url"] if size == "original" else row["image_url"].rsplit("/", 1)[0] + f"/{size}"
        dest = img_dir / _file_name(serial, size)
        res = fetcher(url, dest, session=factory(), timeout=timeout,
                      retries=retries)
        if sleep:
            time.sleep(sleep)
        base = {
            "serial_number": serial,
            "file_name": dest.name if res.get("status") == "ok" else "",
            "word_mark": row.get("word_mark", ""),
            "mark_drawing_cd": row.get("mark_drawing_cd", ""),
            "drawing_kind": row.get("drawing_kind", ""),
            "publication_date": row.get("publication_date", ""),
            "url": url,
        }
        base.update({k: str(v) for k, v in res.items()})
        return base

    results: list[dict[str, str]] = []
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            results = list(pool.map(work, pending))

    for r in results:
        index[r["serial_number"]] = r

    write_index([index[k] for k in sorted(index)], index_path)

    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_missing = len(results) - n_ok
    updated = 0
    if update_meta:
        status = {r["serial_number"]: r["status"] for r in results}
        if status:
            updated = update_meta_status(meta_path, status)
            log(f"[images] cập nhật image_status cho {updated} dòng trong {meta_path.name}")

    stats: dict[str, Any] = {
        "requested": len(meta),
        "downloaded": n_ok,
        "missing": n_missing,
        "skipped": skipped,
        "index": str(index_path),
        "img_dir": str(img_dir),
        "size": size,
        "updated_meta_rows": updated,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    log(f"[images] downloaded: {n_ok} new, skipped: {skipped} existing, "
        f"missing: {n_missing} (of {len(meta)})")
    stats_path = index_path.with_suffix(index_path.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    return stats


# --------------------------------------------------------------------------
# 5. Verify
# --------------------------------------------------------------------------

def verify(
    index_path: Path | str | None = None,
    img_dir: Path | str | None = None,
    *,
    limit: int = 0,
    min_ratio: float = VERIFY_MIN_RATIO,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Mở lại ảnh đã tải bằng PIL, tính tỉ lệ mở được."""
    root = Path(DEFAULT_DATA_ROOT)
    img_dir = Path(img_dir) if img_dir else root / "raw" / "images_us"
    index_path = Path(index_path) if index_path else img_dir / "index.csv"

    index = read_index(index_path)
    rows = [r for r in (index[k] for k in sorted(index)) if r.get("status") == "ok"]
    if limit:
        rows = rows[:limit]

    ok = 0
    broken: list[str] = []
    for r in rows:
        path = img_dir / r["file_name"] if r.get("file_name") else None
        if not path or not path.exists():
            broken.append(r["serial_number"])
            continue
        good, _w, _h, _reason = probe_image(path)
        if good:
            ok += 1
        else:
            broken.append(r["serial_number"])

    total = len(rows)
    ratio = (ok / total) if total else 0.0
    verdict = "VERIFY OK" if (total and ratio >= min_ratio) else "VERIFY FAIL"
    log(f"[verify] opened ok: {ok}/{total} ({ratio * 100:.1f}%) — yêu cầu ≥ "
        f"{min_ratio * 100:.0f}% -> {verdict}")
    if broken:
        log(f"[verify] ảnh lỗi (tối đa 10): {broken[:10]}")
    return {"total": total, "ok": ok, "broken": broken, "ratio": ratio,
            "min_ratio": min_ratio, "verdict": verdict}


# --------------------------------------------------------------------------
# 6. CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    root = Path(DEFAULT_DATA_ROOT)
    ap = argparse.ArgumentParser(
        description="Tải ảnh nhãn hiệu US từ CDN USPTO (kèm resume & verify).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--meta", default=str(root / "raw" / "meta" / "us_marks.csv"))
    ap.add_argument("--img-dir", default=str(root / "raw" / "images_us"))
    ap.add_argument("--index", default=None, help="mặc định: <img-dir>/index.csv")
    ap.add_argument("--limit", type=int, default=0, help="số dòng metadata xét (0=∞)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=0.0, help="nghỉ giữa các ảnh (giây)")
    ap.add_argument("--size", default="original",
                    choices=["original", "px300x200", "px160x130"])
    ap.add_argument("--retry-missing", action="store_true",
                    help="tải lại cả các serial từng lỗi")
    ap.add_argument("--no-update-meta", dest="update_meta", action="store_false",
                    help="không cập nhật cột image_status trong us_marks.csv")
    ap.add_argument("--verify", action="store_true",
                    help="chỉ kiểm tra ảnh đã tải (không tải mới)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify:
        res = verify(args.index, args.img_dir, limit=args.limit)
        return 0 if res["verdict"] == "VERIFY OK" else 1
    run(
        meta_path=args.meta,
        img_dir=args.img_dir,
        index_path=args.index,
        limit=args.limit,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
        sleep=args.sleep,
        size=args.size,
        retry_missing=args.retry_missing,
        update_meta=args.update_meta,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
