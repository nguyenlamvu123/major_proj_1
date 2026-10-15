"""Nhánh 14 (phần 2) — tổng hợp báo cáo cuối cùng (markdown).

Gom ``reports/<model>_<domain>.csv`` (nhánh 10) + ``reports/domain_shift.csv`` (nhánh 12)
+ danh sách hình (nhánh 06/13) thành ``reports/final_summary.md`` với:
- bảng **model × dataset** (n, accuracy, 1−NED, char-accuracy),
- bảng sụt giảm US→VN,
- danh sách hình minh hoạ.

CLI
---
    python3 -m trademark_ocr.report --reports reports/ --out reports/final_summary.md
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trademark_ocr.metrics import evaluate_metrics

DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", "/media/zaibachkhoa/AC84-F995/trademark_data")
_REPORT_RE = re.compile(r"^(?P<model>.+)_(?P<domain>us|vn)\.csv$")


def collect(reports_dir: "Path | str") -> list[dict[str, Any]]:
    """Đọc mọi ``<model>_<domain>.csv`` -> list dòng tổng hợp."""
    rows: list[dict[str, Any]] = []
    for p in sorted(Path(reports_dir).glob("*.csv")):
        m = _REPORT_RE.match(p.name)
        if not m or m.group("model") == "domain_shift":
            continue
        preds: list[str] = []
        gts: list[str] = []
        with p.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                preds.append(row.get("pred", ""))
                gts.append(row.get("gt", ""))
        met = evaluate_metrics(preds, gts)
        rows.append({
            "model": m.group("model"), "dataset": m.group("domain").upper(),
            "n": len(preds), "accuracy": met["accuracy"],
            "one_minus_ned": met["one_minus_ned"], "char_accuracy": met["char_accuracy"],
            "report": p.name,
        })
    return rows


def read_shift(reports_dir: "Path | str") -> list[dict[str, str]]:
    p = Path(reports_dir) / "domain_shift.csv"
    if not p.exists():
        return []
    with p.open(encoding="utf-8", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def list_figures(figures_dir: "Path | str") -> list[str]:
    d = Path(figures_dir)
    if not d.exists():
        return []
    pats = ("fig_*.png", "err_*.png")
    out: list[str] = []
    for pat in pats:
        out += [str(p) for p in sorted(d.glob(pat))]
    return out


def render_markdown(rows: list[dict[str, Any]], shift: list[dict[str, str]],
                    figures: list[str]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append("# Tổng hợp kết quả — nhận dạng chữ nhãn hiệu")
    lines.append("")
    lines.append(f"_Sinh lúc: {ts}_")
    lines.append("")

    lines.append("## Bảng tổng hợp (model × dataset)")
    lines.append("")
    if rows:
        lines.append("| model | dataset | n | accuracy | 1−NED | char-accuracy |")
        lines.append("|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda x: (x["model"], x["dataset"])):
            lines.append(f"| {r['model']} | {r['dataset']} | {r['n']} | "
                         f"{r['accuracy']:.4f} | {r['one_minus_ned']:.4f} | "
                         f"{r['char_accuracy']:.4f} |")
    else:
        lines.append("_Chưa có báo cáo nào trong `reports/` (chạy nhánh 10 trước)._")
    lines.append("")

    lines.append("## Sụt giảm US → VN (domain shift)")
    lines.append("")
    if shift:
        lines.append("| model | acc_us | acc_vn | delta | n_us | n_vn |")
        lines.append("|---|---|---|---|---|---|")
        for r in shift:
            lines.append(f"| {r.get('model','')} | {r.get('acc_us','')} | "
                         f"{r.get('acc_vn','')} | {r.get('delta','')} | "
                         f"{r.get('n_us','')} | {r.get('n_vn','')} |")
        lines.append("")
        lines.append("_delta âm = độ chính xác giảm khi chuyển sang nhãn hiệu Việt Nam._")
    else:
        lines.append("_Chưa có `domain_shift.csv` (chạy nhánh 12 trước)._")
    lines.append("")

    lines.append("## Hình minh hoạ")
    lines.append("")
    if figures:
        for f in figures:
            lines.append(f"- `{f}`")
    else:
        lines.append("_Không tìm thấy hình (`processed/eda_figures`)._")
    lines.append("")
    return "\n".join(lines)


def run(*, reports_dir: "Path | str" = "reports/",
        out_path: "Path | str" = "reports/final_summary.md",
        figures_dir: "Path | str | None" = None, log=print) -> dict[str, Any]:
    figs_dir = figures_dir if figures_dir is not None \
        else Path(DEFAULT_DATA_ROOT) / "processed" / "eda_figures"
    rows = collect(reports_dir)
    shift = read_shift(reports_dir)
    figures = list_figures(figs_dir)
    md = render_markdown(rows, shift, figures)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    log(f"[report] {len(rows)} dòng báo cáo, {len(shift)} dòng shift, "
        f"{len(figures)} hình -> {out}")
    return {"out": str(out), "rows": rows, "shift": shift, "figures": figures}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Gộp báo cáo thành markdown (nhánh 14).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--reports", default="reports/")
    ap.add_argument("--out", default="reports/final_summary.md")
    ap.add_argument("--figures-dir", default=None,
                    help="mặc định $DATA_ROOT/processed/eda_figures")
    return ap


def main(argv: "list[str] | None" = None) -> int:
    a = build_parser().parse_args(argv)
    run(reports_dir=a.reports, out_path=a.out, figures_dir=a.figures_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
