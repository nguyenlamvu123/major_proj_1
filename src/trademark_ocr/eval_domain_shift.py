"""Nhánh 12 — bảng sụt giảm độ chính xác US→VN (domain shift).

Đọc các báo cáo do nhánh 10 sinh ra: ``reports/<model>_<domain>.csv`` với
``domain ∈ {us, vn}`` (vd ``crnn_us.csv``, ``crnn_vn.csv``), ghép theo ``<model>``
rồi tính ``delta = acc_vn − acc_us`` (**âm = sụt độ chính xác** khi sang domain VN).

Độ chính xác lấy từ cột ``correct`` (nhánh 10 ghi 0/1); nếu thiếu thì tự tính lại
``pred == gt`` để tương thích CSV khác.

CLI
---
    python3 -m trademark_ocr.eval_domain_shift --reports reports/ --out reports/domain_shift.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

#: cột bắt buộc theo ACCEPTANCE (model,acc_us,acc_vn,delta) + số mẫu để tiện kiểm tra
COLUMNS: tuple[str, ...] = ("model", "acc_us", "acc_vn", "delta", "n_us", "n_vn")

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


def build_rows(reports_dir: "Path | str") -> list[dict[str, Any]]:
    """Ghép cặp US↔VN thành bảng; model thiếu 1 domain sẽ bị bỏ qua."""
    rows: list[dict[str, Any]] = []
    for model, dom in sorted(discover(reports_dir).items()):
        if "us" not in dom or "vn" not in dom:
            continue
        acc_us, n_us = read_accuracy(dom["us"])
        acc_vn, n_vn = read_accuracy(dom["vn"])
        rows.append({"model": model, "acc_us": acc_us, "acc_vn": acc_vn,
                     "delta": acc_vn - acc_us, "n_us": n_us, "n_vn": n_vn})
    return rows


def run(*, reports_dir: "Path | str" = "reports/",
        out_path: "Path | str" = "reports/domain_shift.csv",
        log=print) -> dict[str, Any]:
    rows = build_rows(reports_dir)
    for r in rows:
        log(f"[shift] {r['model']}: acc_us={r['acc_us']:.4f} "
            f"acc_vn={r['acc_vn']:.4f} delta={r['delta']:+.4f} "
            f"(n={r['n_us']}/{r['n_vn']})")
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
