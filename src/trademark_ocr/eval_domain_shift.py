"""Nhánh 12 — bảng sụt giảm độ chính xác US→VN (domain shift).

Đọc các báo cáo do nhánh 10 sinh ra: ``reports/<model>_<domain>.csv`` với
``domain ∈ {us, vn}`` (vd ``crnn_us.csv``, ``crnn_vn.csv``), ghép theo ``<model>``
rồi tính ``delta = acc_vn − acc_us`` (**âm = sụt độ chính xác** khi sang domain VN).

Độ chính xác tuyệt đối lấy từ cột ``correct`` (nhánh 10 ghi 0/1); nếu thiếu thì tự tính
lại ``pred == gt`` để tương thích CSV khác.

Tách nguồn gốc của độ sụt (nhánh 16, REVIEW.md §4.2)
----------------------------------------------------
Model chỉ học nhãn US (VIẾT HOA, gần như không dấu) nên ``delta`` tuyệt đối trộn ba nguồn.
Các cột thêm (tính lại từ ``pred/gt``, mức so khớp xem ``metrics.MATCH_LEVELS``):

| cột | loại bỏ | còn lại |
|---|---|---|
| ``delta`` | — | quy ước ghi nhãn + ký tự chưa học + hình ảnh |
| ``delta_nocase`` | hoa/thường | ký tự chưa học + hình ảnh |
| ``delta_base`` | hoa/thường + dấu | hình ảnh (nhưng chấm dễ cho chữ có dấu) |
| ``delta_plain_nocase`` | chỉ VN **không dấu**, bỏ hoa/thường | hình ảnh — ước lượng sạch nhất |

``*_lo/*_hi``: khoảng tin cậy 95% bootstrap của delta (:func:`bootstrap_delta_ci`).
Nhóm rỗng ghi ô trống.

CLI
---
    python3 -m trademark_ocr.eval_domain_shift --reports reports/ --out reports/domain_shift.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Callable, Sequence

from trademark_ocr.metrics import MATCH_LEVELS, has_diacritics

#: 6 cột đầu bắt buộc theo ACCEPTANCE nhánh 12; phần sau thêm ở nhánh 16 (xem docstring)
COLUMNS: tuple[str, ...] = (
    "model", "acc_us", "acc_vn", "delta", "n_us", "n_vn",
    "acc_us_nocase", "acc_vn_nocase", "delta_nocase", "delta_nocase_lo", "delta_nocase_hi",
    "acc_us_base", "acc_vn_base", "delta_base",
    "acc_vn_plain_nocase", "n_vn_plain",
    "delta_plain_nocase", "delta_plain_nocase_lo", "delta_plain_nocase_hi",
    "acc_vn_diac_nocase", "n_vn_diac",
)

_REPORT_RE = re.compile(r"^(?P<model>.+)_(?P<domain>us|vn)\.csv$")
_KNOWN_EXTRA = {"domain_shift"}


def discover(reports_dir: "Path | str") -> dict[str, dict[str, Path]]:
    """``{model: {'us': path, 'vn': path}}`` từ các tệp ``<model>_<domain>.csv``."""
    found: dict[str, dict[str, Path]] = {}
    for p in sorted(Path(reports_dir).glob("*.csv")):
        m = _REPORT_RE.match(p.name)
        if not m or m.group("model") in _KNOWN_EXTRA:
            continue
        found.setdefault(m.group("model"), {})[m.group("domain")] = p
    return found


def read_accuracy(path: "Path | str") -> tuple[float, int]:
    """``(accuracy, n)`` — ưu tiên cột ``correct``, nếu không thì suy từ ``pred/gt``."""
    correct = 0
    n = 0
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            n += 1
            if row.get("correct", "") != "":
                correct += int(float(row["correct"]) != 0)
            else:
                correct += int(row.get("pred", "") == row.get("gt", ""))
    return (correct / n if n else 0.0), n


def read_pairs(path: "Path | str") -> list[tuple[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return [(row.get("pred", ""), row.get("gt", "")) for row in csv.DictReader(fh)]


def bootstrap_delta_ci(*, k_us: int, n_us: int, k_vn: int, n_vn: int,
                       n_boot: int = 2000, seed: int = 1111,
                       alpha: float = 0.05) -> "tuple[float, float] | None":
    """Khoảng tin cậy ``1 − alpha`` (percentile bootstrap) của ``acc_vn − acc_us``.

    ``k`` = số mẫu đúng trên ``n``. ``None`` khi một phía rỗng. Tất định theo ``seed``.
    """
    if not n_us or not n_vn:
        return None
    import numpy as np

    rng = np.random.default_rng(seed)
    # Bootstrap phi tham số của trung bình một vector 0/1: lấy lại n mẫu có hoàn lại thì số
    # mẫu đúng ~ Binomial(n, k/n). Lấy mẫu nhị thức trực tiếp cho đúng phân phối đó mà không
    # phải lặp qua từng mẫu (US ~7k + VN ~5k mẫu × 2000 lần).
    acc_us = rng.binomial(n_us, k_us / n_us, n_boot) / n_us
    acc_vn = rng.binomial(n_vn, k_vn / n_vn, n_boot) / n_vn
    lo, hi = np.quantile(acc_vn - acc_us, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _n_correct(pairs: Sequence[tuple[str, str]], fold: Callable[[str], str]) -> int:
    return sum(fold(p) == fold(g) for p, g in pairs)


def _shift_columns(us: Sequence[tuple[str, str]],
                   vn: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Các cột nhánh 16; ``None`` (ô trống trong CSV) cho nhóm không có mẫu."""
    nocase, base = MATCH_LEVELS["nocase"], MATCH_LEVELS["base"]
    plain = [(p, g) for p, g in vn if not has_diacritics(g)]
    diac = [(p, g) for p, g in vn if has_diacritics(g)]

    def acc(pairs, fold):
        return _n_correct(pairs, fold) / len(pairs) if pairs else None

    def delta_with_ci(vn_pairs):
        a_us, a_vn = acc(us, nocase), acc(vn_pairs, nocase)
        if a_us is None or a_vn is None:
            return None, None, None
        ci = bootstrap_delta_ci(k_us=_n_correct(us, nocase), n_us=len(us),
                                k_vn=_n_correct(vn_pairs, nocase), n_vn=len(vn_pairs))
        return a_vn - a_us, ci[0], ci[1]

    d_nc, d_nc_lo, d_nc_hi = delta_with_ci(vn)
    d_pl, d_pl_lo, d_pl_hi = delta_with_ci(plain)
    acc_us_base, acc_vn_base = acc(us, base), acc(vn, base)
    return {
        "acc_us_nocase": acc(us, nocase), "acc_vn_nocase": acc(vn, nocase),
        "delta_nocase": d_nc, "delta_nocase_lo": d_nc_lo, "delta_nocase_hi": d_nc_hi,
        "acc_us_base": acc_us_base, "acc_vn_base": acc_vn_base,
        "delta_base": (acc_vn_base - acc_us_base
                       if acc_us_base is not None and acc_vn_base is not None else None),
        "acc_vn_plain_nocase": acc(plain, nocase), "n_vn_plain": len(plain),
        "delta_plain_nocase": d_pl, "delta_plain_nocase_lo": d_pl_lo,
        "delta_plain_nocase_hi": d_pl_hi,
        "acc_vn_diac_nocase": acc(diac, nocase), "n_vn_diac": len(diac),
    }


def _fmt(value: "float | None") -> str:
    return "–" if value is None else f"{value:+.4f}"


def build_rows(reports_dir: "Path | str") -> list[dict[str, Any]]:
    """Ghép cặp US↔VN thành bảng; model thiếu 1 domain sẽ bị bỏ qua."""
    rows: list[dict[str, Any]] = []
    for model, dom in sorted(discover(reports_dir).items()):
        if "us" not in dom or "vn" not in dom:
            continue
        acc_us, n_us = read_accuracy(dom["us"])
        acc_vn, n_vn = read_accuracy(dom["vn"])
        rows.append({"model": model, "acc_us": acc_us, "acc_vn": acc_vn,
                     "delta": acc_vn - acc_us, "n_us": n_us, "n_vn": n_vn,
                     **_shift_columns(read_pairs(dom["us"]), read_pairs(dom["vn"]))})
    return rows


def run(*, reports_dir: "Path | str" = "reports/",
        out_path: "Path | str" = "reports/domain_shift.csv",
        log=print) -> dict[str, Any]:
    rows = build_rows(reports_dir)
    for r in rows:
        log(f"[shift] {r['model']}: acc_us={r['acc_us']:.4f} "
            f"acc_vn={r['acc_vn']:.4f} delta={r['delta']:+.4f} "
            f"delta_nocase={_fmt(r['delta_nocase'])} (n={r['n_us']}/{r['n_vn']})")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        w.writeheader()
        for row in sorted(rows, key=lambda r: r["delta"]):
            w.writerow(row)
    print(f"[shift] {len(rows)} model -> {out}")
    return {"out": str(out), "rows": rows}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Bảng sụt giảm US→VN từ reports/ (nhánh 12).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--reports", default="reports/",
                    help="thư mục chứa <model>_us.csv / <model>_vn.csv")
    ap.add_argument("--out", default="reports/domain_shift.csv")
    return ap


def main(argv: "list[str] | None" = None) -> int:
    a = build_parser().parse_args(argv)
    run(reports_dir=a.reports, out_path=a.out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
