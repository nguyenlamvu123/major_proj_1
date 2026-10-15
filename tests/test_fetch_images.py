"""Test offline cho nhánh 03 — `trademark_ocr.fetch_images`.

Dùng ảnh PNG nhỏ sinh bằng Pillow và `fetch_fn` giả; không gọi mạng.
"""

from __future__ import annotations

import csv
import io

import pytest
from PIL import Image

from trademark_ocr.fetch_images import (
    INDEX_COLUMNS,
    download_one,
    probe_image,
    read_index,
    run,
    update_meta_status,
    verify,
    write_index,
)

META_COLUMNS = ["serial_number", "word_mark", "image_status", "mark_drawing_cd",
                "drawing_kind", "publication_date", "image_url"]


def _png_bytes(size=(64, 64)) -> bytes:
    """PNG nhiễu (khó nén) để vượt ngưỡng MIN_BYTES khi test."""
    import os as _os
    w, h = size
    img = Image.frombytes("RGB", (w, h), _os.urandom(w * h * 3))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _write_meta(path, n=5):
    rows = []
    for i in range(n):
        rows.append({
            "serial_number": f"{10000000 + i}",
            "word_mark": f"MARK{i}",
            "image_status": "pending",
            "mark_drawing_cd": "5",
            "drawing_kind": "stylized_words",
            "publication_date": "2026-10-13",
            "image_url": f"https://cdn.uspto.gov/img/{10000000 + i}/2026-10-13/original",
        })
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=META_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows


# --------------------------------------------------------------------------
# 1. probe_image
# --------------------------------------------------------------------------

def test_probe_image_valid(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(_png_bytes((10, 4)))
    ok, w, h, reason = probe_image(p)
    assert ok and (w, h) == (10, 4) and reason == ""


def test_probe_image_broken(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nKHONG-PHAI-ANH")
    ok, _w, _h, reason = probe_image(p)
    assert not ok and reason


# --------------------------------------------------------------------------
# 2. download_one (session giả)
# --------------------------------------------------------------------------

class _Resp:
    def __init__(self, status_code=200, content=b"", content_type="image/png"):
        self.status_code = status_code
        self.content = content
        self.headers = {"Content-Type": content_type}


class _Session:
    def __init__(self, resp):
        self._resp = resp

    def get(self, url, timeout=None):  # noqa: ARG002
        return self._resp


def test_download_one_ok(tmp_path):
    dest = tmp_path / "1.png"
    res = download_one("http://x/1", dest, session=_Session(_Resp(content=_png_bytes())),
                       retries=0)
    assert res["status"] == "ok" and res["width"] == 64
    assert dest.exists() and res["sha1"] and not dest.with_suffix(".png.part").exists()


def test_download_one_http_404(tmp_path):
    res = download_one("http://x/1", tmp_path / "1.png",
                       session=_Session(_Resp(status_code=404, content=b"")), retries=0)
    assert res == {"status": "missing", "http_status": 404, "reason": "http_404"}


def test_download_one_too_small_and_not_image(tmp_path):
    small = download_one("u", tmp_path / "a.png",
                         session=_Session(_Resp(content=b"x")), retries=0)
    assert small["reason"] == "too_small"
    notimg = download_one("u", tmp_path / "b.png",
                          session=_Session(_Resp(content=b"x" * 200, content_type="text/html")),
                          retries=0)
    assert notimg["reason"].startswith("not_image")


def test_download_one_corrupt_leaves_no_file(tmp_path):
    dest = tmp_path / "c.png"
    res = download_one("u", dest,
                       session=_Session(_Resp(content=b"\x89PNG" + b"0" * 500)),
                       retries=0)
    assert res["status"] == "missing" and res["reason"].startswith("corrupt")
    assert not dest.exists()


# --------------------------------------------------------------------------
# 3. run() với fetch_fn giả
# --------------------------------------------------------------------------

def _fake_fetch_ok(url, dest, **kw):
    from pathlib import Path
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_png_bytes())
    return {"status": "ok", "http_status": 200, "width": 64, "height": 64,
            "bytes": len(_png_bytes()), "sha1": "deadbeef", "reason": ""}


def test_run_downloads_and_writes_index(tmp_path):
    meta = tmp_path / "us_marks.csv"
    _write_meta(meta, 5)
    img_dir = tmp_path / "images_us"
    stats = run(meta_path=meta, img_dir=img_dir, fetch_fn=_fake_fetch_ok,
                workers=2, log=lambda *_: None)
    assert stats["downloaded"] == 5 and stats["missing"] == 0
    idx = read_index(img_dir / "index.csv")
    assert len(idx) == 5
    assert all(r["status"] == "ok" for r in idx.values())
    assert list(next(iter(idx.values())).keys()) == INDEX_COLUMNS
    updated = list(csv.DictReader(open(meta, encoding="utf-8")))
    assert all(r["image_status"] == "ok" for r in updated)


def test_run_resume_does_not_redownload(tmp_path):
    meta = tmp_path / "us_marks.csv"
    _write_meta(meta, 4)
    img_dir = tmp_path / "images_us"
    run(meta_path=meta, img_dir=img_dir, fetch_fn=_fake_fetch_ok, log=lambda *_: None)

    calls = {"n": 0}

    def counting_fetch(url, dest, **kw):
        calls["n"] += 1
        return _fake_fetch_ok(url, dest, **kw)

    stats = run(meta_path=meta, img_dir=img_dir, fetch_fn=counting_fetch,
                log=lambda *_: None)
    assert calls["n"] == 0
    assert stats["downloaded"] == 0 and stats["skipped"] == 4


def test_run_records_missing_and_respects_retry_flag(tmp_path):
    meta = tmp_path / "us_marks.csv"
    _write_meta(meta, 3)

    def failing_fetch(url, dest, **kw):
        return {"status": "missing", "http_status": 404, "reason": "http_404"}

    img_dir = tmp_path / "images_us"
    s1 = run(meta_path=meta, img_dir=img_dir, fetch_fn=failing_fetch, log=lambda *_: None)
    assert s1["downloaded"] == 0 and s1["missing"] == 3

    calls = {"n": 0}

    def counting(url, dest, **kw):
        calls["n"] += 1
        return {"status": "missing", "http_status": 404, "reason": "http_404"}

    run(meta_path=meta, img_dir=img_dir, fetch_fn=counting, log=lambda *_: None)
    assert calls["n"] == 0  # không retry
    run(meta_path=meta, img_dir=img_dir, fetch_fn=counting, retry_missing=True,
        log=lambda *_: None)
    assert calls["n"] == 3  # retry khi bật cờ


def test_run_limit(tmp_path):
    meta = tmp_path / "us_marks.csv"
    _write_meta(meta, 10)
    stats = run(meta_path=meta, img_dir=tmp_path / "imgs", limit=3,
                fetch_fn=_fake_fetch_ok, log=lambda *_: None)
    assert stats["downloaded"] == 3 and stats["requested"] == 3


def test_run_size_variant_updates_url_and_name(tmp_path):
    meta = tmp_path / "us_marks.csv"
    _write_meta(meta, 2)
    seen = {}

    def spy_fetch(url, dest, **kw):
        seen[url] = url
        return _fake_fetch_ok(url, dest, **kw)

    run(meta_path=meta, img_dir=tmp_path / "imgs", size="px300x200",
        fetch_fn=spy_fetch, log=lambda *_: None)
    assert all(u.endswith("/px300x200") for u in seen)
    idx = read_index(tmp_path / "imgs" / "index.csv")
    assert all(r["file_name"].endswith("__px300x200.png") for r in idx.values())


# --------------------------------------------------------------------------
# 4. verify
# --------------------------------------------------------------------------

def test_verify_ratio_ok(tmp_path):
    meta = tmp_path / "us_marks.csv"
    _write_meta(meta, 10)
    img_dir = tmp_path / "imgs"
    run(meta_path=meta, img_dir=img_dir, fetch_fn=_fake_fetch_ok, log=lambda *_: None)
    res = verify(img_dir / "index.csv", img_dir, log=lambda *_: None)
    assert res["verdict"] == "VERIFY OK" and res["ok"] == 10


def test_verify_detects_broken_file(tmp_path):
    meta = tmp_path / "us_marks.csv"
    _write_meta(meta, 10)
    img_dir = tmp_path / "imgs"
    run(meta_path=meta, img_dir=img_dir, fetch_fn=_fake_fetch_ok, log=lambda *_: None)
    # phá 1 ảnh (10% lỗi > 5%) -> FAIL
    victim = sorted(p for p in img_dir.glob("*.png"))[0]
    victim.write_bytes(b"khong-phai-anh")
    res = verify(img_dir / "index.csv", img_dir, log=lambda *_: None)
    assert res["verdict"] == "VERIFY FAIL" and res["ok"] == 9 and len(res["broken"]) == 1


def test_verify_without_index_is_safe(tmp_path):
    res = verify(tmp_path / "missing.csv", tmp_path, log=lambda *_: None)
    assert res["total"] == 0 and res["verdict"] == "VERIFY FAIL"


# --------------------------------------------------------------------------
# 5. update_meta_status
# --------------------------------------------------------------------------

def test_update_meta_status(tmp_path):
    meta = tmp_path / "us_marks.csv"
    _write_meta(meta, 3)
    n = update_meta_status(meta, {"10000000": "ok", "10000001": "missing"})
    rows = {r["serial_number"]: r["image_status"]
            for r in csv.DictReader(open(meta, encoding="utf-8"))}
    assert rows["10000000"] == "ok" and rows["10000001"] == "missing"
    assert rows["10000002"] == "pending" and n == 2


def test_write_index_roundtrip(tmp_path):
    p = tmp_path / "index.csv"
    write_index([{"serial_number": "1", "status": "ok", "file_name": "1.png"}], p)
    assert read_index(p)["1"]["file_name"] == "1.png"
