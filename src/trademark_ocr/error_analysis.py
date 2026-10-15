"""Nhánh 13 — phân tích lỗi từ báo cáo từng mẫu của nhánh 10.

Đầu vào: ``reports/<model>_<domain>.csv`` (cột ``pred,gt,correct,ned,char_err``).
Đầu ra: các hình PNG ``err_*.png`` (mặc định ``processed/eda_figures``) + dict tổng hợp.

Hàm thuần (test offline được):
- ``load_pairs(paths)``     : đọc CSV -> list[(pred, gt)]
- ``by_length(pairs, bins)``: độ chính xác & char-accuracy theo độ dài nhãn thật
- ``top_errors(pairs, k)``  : k mẫu lỗi nặng nhất (edit distance giảm dần)
- ``char_confusions(pairs)``: các cặp ký tự hay bị thay thế (gt -> pred)

Vẽ hình dùng backend ``Agg`` (headless), không phụ thuộc seaborn.

CLI
---
    python3 -m trademark_ocr.error_analysis --reports reports/ --top-k 12
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from trademark_ocr.metrics import edit_distance, evaluate_metrics

DEFAULT_BINS: tuple[int, ...] = (0, 5, 8, 12, 16, 20, 25, 30, 10 ** 9)
_REPORT_RE = re.compile(r"^(?P<model>.+)_(?P<domain>us|vn)\.csv$")


def discover_reports(reports_dir: "Path | str") -> list[Path]:
    """Các CSV báo cáo ``<model>_<domain>.csv`` (bỏ ``domain_shift.csv``)."""
    out: list[Path] = []
    for p in sorted(Path(reports_dir).glob("*.csv")):
        m = _REPORT_RE.match(p.name)
        if m and m.group("model") != "domain_shift":
            out.append(p)
    return out


def load_pairs(paths: "Iterable[Path | str]") -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for p in paths:
        with Path(p).open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                pairs.append((row.get("pred", ""), row.get("gt", "")))
    return pairs


def by_length(pairs: Sequence[tuple[str, str]],
              bins: Sequence[int] = DEFAULT_BINS) -> list[dict[str, Any]]:
    """Gom theo độ dài *nhãn thật* (gt): n, acc, char_acc."""
    out: list[dict[str, Any]] = []
    for lo, hi in zip(bins, bins[1:]):
        grp = [(p, g) for p, g in pairs if lo <= len(g) < hi]
        if not grp:
            continue
        m = evaluate_metrics([p for p, _ in grp], [g for _, g in grp])
        out.append({"bucket": f"{lo}-{hi - 1}" if hi < 10 ** 9 else f"{lo}+",
                    "lo": lo, "hi": hi, "n": len(grp),
                    "acc": m["accuracy"], "char_acc": m["char_accuracy"]})
    return out


def top_errors(pairs: Sequence[tuple[str, str]], k: int = 10) -> list[dict[str, Any]]:
    """k mẫu lỗi nặng nhất (edit distance giảm dần; chỉ lấy mẫu sai)."""
    scored: list[dict[str, Any]] = []
    for pred, gt in pairs:
        d = edit_distance(pred, gt)
        if d > 0:
            denom = max(len(pred), len(gt))
            scored.append({"pred": pred, "gt": gt, "dist": d,
                           "ned": d / denom if denom else 0.0})
    scored.sort(key=lambda r: (r["dist"], r["ned"]), reverse=True)
    return scored[:k]


def char_confusions(pairs: Sequence[tuple[str, str]],
                    top: int = 15) -> list[tuple[str, str, int]]:
    """Cặp thay thế ``(gt_char, pred_char)`` hay gặp nhất (chỉ op 'replace')."""
    import difflib
    c: Counter[tuple[str, str]] = Counter()
    for pred, gt in pairs:
        sm = difflib.SequenceMatcher(a=gt, b=pred, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "replace" and (i2 - i1) == (j2 - j1):
                for gc, pc in zip(gt[i1:i2], pred[j1:j2]):
                    if gc != pc:
                        c[(gc, pc)] += 1
    return [(g, p, n) for (g, p), n in c.most_common(top)]


# ------------------------------- vẽ hình -------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_by_length(stats: Sequence[dict[str, Any]], out_path: "Path | str") -> Path:
    plt = _plt()
    labels = [s["bucket"] for s in stats]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - 0.2 for i in x], [s["acc"] * 100 for s in stats], width=0.4, label="accuracy")
    ax.bar([i + 0.2 for i in x], [s["char_acc"] * 100 for s in stats], width=0.4,
           label="char-accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("độ dài nhãn (gt)")
    ax.set_ylabel("%")
    ax.set_title("Độ chính xác theo độ dài nhãn")
    ax.legend()
    for i, s in enumerate(stats):
        ax.text(i, 2, f"n={int(s['n'])}", ha="center", fontsize=7)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_top_errors(rows: Sequence[dict[str, Any]], out_path: "Path | str",
                    title: str = "Top lỗi nặng nhất (gt vs pred)") -> Path:
    plt = _plt()

    def trunc(s: object, n: int = 34) -> str:
        s = str(s)
        return s if len(s) <= n else s[: n - 1] + "…"

    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 1.2))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=8)
    ax.text(0.02, 0.99, "gt", fontsize=9, weight="bold", va="top")
    ax.text(0.52, 0.99, "pred", fontsize=9, weight="bold", va="top")
    ax.text(0.98, 0.99, "d", fontsize=9, weight="bold", va="top", ha="right")
    y = 0.93
    step = 1.0 / (len(rows) + 2)
    for r in rows:
        ax.text(0.02, y, trunc(r["gt"]), fontsize=8.5, va="top")
        ax.text(0.52, y, trunc(r["pred"]), fontsize=8.5, va="top", color="#b00020")
        ax.text(0.98, y, str(r["dist"]), fontsize=8.5, va="top", ha="right")
        y -= step
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_confusions(conf: Sequence[tuple[str, str, int]], out_path: "Path | str") -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if conf:
        labels = [f"{g}→{p}" for g, p, _ in conf][::-1]
        vals = [n for _, _, n in conf][::-1]
        ax.barh(labels, vals, color="#444")
        ax.set_xlabel("số lần")
        ax.set_title("Thay thế ký tự hay gặp (gt → pred)")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "không có dữ liệu", ha="center")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def analyze(*, reports_dir: "Path | str" = "reports/",
            out_dir: "Path | str | None" = None, top_k: int = 10,
            log=print) -> dict[str, Any]:
    """Sinh các hình ``err_*.png`` + trả dict tổng hợp (dùng cho notebook/test)."""
    fig_dir = Path(out_dir) if out_dir else Path("processed") / "eda_figures"
    paths = discover_reports(reports_dir)
    pairs = load_pairs(paths)
    if not pairs:
        log("[err] không thấy báo cáo nào -> không sinh hình")
        return {"n": 0, "figures": {}, "reports": []}

    stats = by_length(pairs)
    tops = top_errors(pairs, top_k)
    conf = char_confusions(pairs)
    m = evaluate_metrics([p for p, _ in pairs], [g for _, g in pairs])

    figs = {
        "by_length": str(plot_by_length(stats, fig_dir / "err_by_length.png")),
        "top_errors": str(plot_top_errors(tops, fig_dir / "err_top_samples.png")),
        "confusions": str(plot_confusions(conf, fig_dir / "err_char_confusions.png")),
    }
    log(f"[err] n={m['n']:.0f} acc={m['accuracy']:.4f} CER={m['cer']:.4f} | "
        f"{len(tops)} top lỗi, {len(conf)} cặp thay thế")
    for k, v in figs.items():
        log(f"[err] {k}: {v}")
    return {"n": int(m["n"]), "metrics": m, "by_length": stats, "top_errors": tops,
            "confusions": conf, "figures": figs, "reports": [str(p) for p in paths]}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Phân tích lỗi từ reports/ (nhánh 13).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--reports", default="reports/")
    ap.add_argument("--out-dir", default=None, help="mặc định processed/eda_figures")
    ap.add_argument("--top-k", type=int, default=10)
    return ap


def main(argv: "list[str] | None" = None) -> int:
    a = build_parser().parse_args(argv)
    analyze(reports_dir=a.reports, out_dir=a.out_dir, top_k=a.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
