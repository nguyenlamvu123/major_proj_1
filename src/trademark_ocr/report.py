"""Nhánh 14 (phần 2) — tổng hợp báo cáo cuối cùng (markdown).

Gom ``reports/<model>_<domain>.csv`` (nhánh 10) + ``reports/domain_shift.csv`` (nhánh 12)
+ danh sách hình (nhánh 06/13) thành ``reports/final_summary.md`` với:
- bảng **model × dataset** (n, accuracy, 1−NED, char-accuracy),
- bảng **ba mức so khớp** + độ bao phủ (nhánh 16),
- bảng sụt giảm US→VN + bảng **phân tách nguồn gốc độ sụt** kèm CI 95% (nhánh 16),
- bảng theo **loại nhãn hiệu** (``drawing_kind``, nhánh 16),
- danh sách hình minh hoạ.

CSV cũ (trước nhánh 16, thiếu ``in_scope``/``drawing_kind``) vẫn đọc được: ô tương ứng ghi "–".

CLI
---
    python3 -m trademark_ocr.report --reports reports/ --out reports/final_summary.md
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trademark_ocr import DEFAULT_DATA_ROOT
from trademark_ocr.metrics import evaluate_metrics

_REPORT_RE = re.compile(r"^(?P<model>.+)_(?P<domain>us|vn)\.csv$")


def collect(reports_dir: "Path | str") -> list[dict[str, Any]]:
    """Đọc mọi ``<model>_<domain>.csv`` -> list dòng tổng hợp (metric tính lại từ pred/gt)."""
    rows: list[dict[str, Any]] = []
    for p in sorted(Path(reports_dir).glob("*.csv")):
        m = _REPORT_RE.match(p.name)
        if not m or m.group("model") == "domain_shift":
            continue
        with p.open(encoding="utf-8", newline="") as fh:
            samples = list(csv.DictReader(fh))
        preds = [r.get("pred", "") for r in samples]
        gts = [r.get("gt", "") for r in samples]
        met = evaluate_metrics(preds, gts)
        # in_scope chỉ có từ nhánh 16; CSV cũ -> None (không biết), KHÔNG coi là 100%.
        scope = [r["in_scope"] for r in samples if r.get("in_scope", "") != ""]
        coverage = None
        if scope and len(scope) == len(samples):
            coverage = sum(int(v) for v in scope) / len(scope)
        by_kind: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for r in samples:
            if r.get("drawing_kind"):
                by_kind[r["drawing_kind"]].append((r.get("pred", ""), r.get("gt", "")))
        rows.append({
            "model": m.group("model"), "dataset": m.group("domain").upper(),
            "n": len(preds), "accuracy": met["accuracy"],
            "one_minus_ned": met["one_minus_ned"], "char_accuracy": met["char_accuracy"],
            "accuracy_nocase": met["accuracy_nocase"], "accuracy_base": met["accuracy_base"],
            "coverage": coverage,
            "by_kind": {
                kind: {"n": len(pairs),
                       "accuracy_nocase": evaluate_metrics([a for a, _ in pairs],
                                                           [b for _, b in pairs])["accuracy_nocase"]}
                for kind, pairs in sorted(by_kind.items())
            },
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

    lines.append("## Ba mức so khớp + độ bao phủ")
    lines.append("")
    if rows:
        lines.append("| model | dataset | n | coverage | accuracy | acc (bỏ hoa/thường) | acc (bỏ cả dấu) |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda x: (x["model"], x["dataset"])):
            cov = "–" if r["coverage"] is None else f"{r['coverage']:.3f}"
            lines.append(f"| {r['model']} | {r['dataset']} | {r['n']} | {cov} | "
                         f"{r['accuracy']:.4f} | {r['accuracy_nocase']:.4f} | "
                         f"{r['accuracy_base']:.4f} |")
        lines.append("")
        lines.append("_coverage = tỉ lệ mẫu model CÓ THỂ đúng (nhãn ≤ max_label_length, mọi ký tự "
                     "thuộc charset); mẫu ngoài phạm vi vẫn tính là sai. Nhãn USPTO luôn VIẾT HOA "
                     "và gần như không dấu, nên so sánh US↔VN nên đọc ở cột bỏ hoa/thường._")
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
        if "delta_nocase" in shift[0]:
            lines.append("")
            lines.append("### Phân tách nguồn gốc độ sụt (CI 95% bootstrap)")
            lines.append("")
            lines.append("| model | Δ tuyệt đối | Δ bỏ hoa/thường | Δ bỏ cả dấu | "
                         "Δ VN không dấu, bỏ hoa/thường | n VN không dấu / có dấu |")
            lines.append("|---|---|---|---|---|---|")
            for r in shift:
                lines.append(
                    f"| {r['model']} | {_signed(r['delta'])} | "
                    f"{_signed(r['delta_nocase'])} {_ci(r, 'delta_nocase')} | "
                    f"{_signed(r['delta_base'])} | "
                    f"{_signed(r['delta_plain_nocase'])} {_ci(r, 'delta_plain_nocase')} | "
                    f"{r['n_vn_plain']} / {r['n_vn_diac']} |")
            lines.append("")
            lines.append("_Δ bỏ hoa/thường loại phần sai do quy ước ghi nhãn; Δ bỏ cả dấu loại thêm "
                         "phần sai do chưa học chữ có dấu; Δ trên nhóm VN không dấu là ước lượng "
                         "sạch nhất của khác biệt **hình ảnh**. CI rộng = n quá nhỏ để kết luận._")
    else:
        lines.append("_Chưa có `domain_shift.csv` (chạy nhánh 12 trước)._")
    lines.append("")

    lines.append("## Theo loại nhãn hiệu (drawing_kind)")
    lines.append("")
    kind_rows = [(r, k, v) for r in sorted(rows, key=lambda x: (x["model"], x["dataset"]))
                 for k, v in r["by_kind"].items()]
    if kind_rows:
        lines.append("| model | dataset | drawing_kind | n | acc (bỏ hoa/thường) |")
        lines.append("|---|---|---|---|---|")
        for r, kind, v in kind_rows:
            lines.append(f"| {r['model']} | {r['dataset']} | {kind} | {v['n']} | "
                         f"{v['accuracy_nocase']:.4f} |")
        lines.append("")
        lines.append("_`standard_character` là chữ đánh máy font chuẩn do USPTO dựng — dễ hơn "
                     "nhiều so với `stylized_words`/`design_with_words`; accuracy US gộp chung "
                     "phụ thuộc mạnh vào tỉ lệ loại này._")
    else:
        lines.append("_Báo cáo chưa có cột `drawing_kind` (build lại LMDB + evaluate từ nhánh 16)._")
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


def _signed(value: str) -> str:
    return f"{float(value):+.4f}" if value not in ("", None) else "–"


def _ci(row: dict[str, str], key: str) -> str:
    lo, hi = row.get(f"{key}_lo", ""), row.get(f"{key}_hi", "")
    return f"[{float(lo):+.3f}, {float(hi):+.3f}]" if lo and hi else ""


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
