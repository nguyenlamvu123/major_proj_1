"""Đường cong huấn luyện (loss train / val_loss / val_accuracy theo epoch) để xem có overfit.

Không cần sửa code train: ``third_party/parseq/train.py`` đã ghi log TensorBoard vào chính thư
mục run (``runs/<model>/<giờ>/events.out.tfevents.*``):

- ``loss``          loss train MỖI BATCH (``log_every_n_steps: 1``), trên ảnh đã augment, có dropout;
- ``val_loss``, ``val_accuracy`` (%), ``val_NED`` (% = 100·(1 − NED)): mỗi epoch, trên tập val;
- ``epoch``         chỉ số epoch (0-based) ở từng bước — dùng để gom loss train theo epoch.

Accuracy trên tập TRAIN không được đo (parseq chỉ ghi loss train), nên đồ thị so loss train với
val_loss, cạnh val_accuracy. Dấu hiệu overfit = loss train vẫn giảm trong khi val_loss quay đầu
tăng (:func:`overfit_check`).

Các trường hợp đặc biệt trong một run (DESIGN.md 16.11):

- **SWA**: model có BatchNorm (CRNN) — Lightning chạy thêm một lượt sau epoch cuối để tính lại
  BatchNorm cho trọng số trung bình rồi validate lần nữa, ghi ở ``epoch == max_epochs``. Điểm đó
  là model SWA, không phải một epoch train: đánh dấu ``swa``, không tính vào "tốt nhất"/overfit.
- **Train tiếp sau khi bị cắt** (cùng thư mục run, ``ckpt_path=``): thêm một file events mới bắt
  đầu ở ``global_step`` của checkpoint; các điểm của file cũ từ bước đó trở đi là nhánh đã bỏ.
- **Train tiếp một run đã xong** (``09_full_pipeline_cpu.sh --continue``): run MỚI, nạp trọng số
  từ ``last.ckpt`` của run cũ, ghi ``continue_from``/``init_epoch``/``epoch_offset`` vào
  ``pipeline_train.env``. Đồ thị nối run cũ (tới epoch của checkpoint đã nạp) với run mới.

Dùng:
    python -m trademark_ocr.train_curves                         # mọi model trong $DATA_ROOT/runs
    python -m trademark_ocr.train_curves --models crnn_ft --out-dir reports/figures

Ra (trong ``--out-dir``): ``train_curves_<model>.png``, ``train_curves.json`` (số theo epoch,
kết luận overfit) và ``train_curves.md`` (hình + bảng, dán được vào báo cáo).
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import re
from pathlib import Path
from typing import Any

from trademark_ocr import DEFAULT_DATA_ROOT

#: val_loss cuối cao hơn đáy bao nhiêu (tương đối) thì coi là "quay đầu tăng"
OVERFIT_RISE = 0.05
#: số epoch tối thiểu SAU đáy val_loss để kết luận overfit (một epoch nhích lên là nhiễu)
OVERFIT_MIN_EPOCHS_AFTER = 2

TAGS = ("loss", "epoch", "val_loss", "val_accuracy", "val_NED")

# Màu theo vai trò (bảng categorical mặc định của skill dataviz, đã kiểm CVD/tương phản):
# train = slot 1, val = slot 2 — cùng một màu cho "val" ở cả hai ô.
COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"

STATUS_TEXT = {"DONE": "train xong", "UNDERTRAINED": "train xong nhưng chưa học được",
               "chưa xong": "chưa xong hoặc đang train"}


# ---------------------------------------------------------------------------
# đọc log
# ---------------------------------------------------------------------------

def _event_files(run_dir: Path) -> list[Path]:
    """File events theo thứ tự ghi: tên có dạng ``events.out.tfevents.<unix giây>.<máy>.<pid>.<n>``."""
    def key(p: Path) -> tuple[int, str]:
        parts = p.name.split(".")
        return (int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0, p.name)
    return sorted(run_dir.glob("events.out.tfevents.*"), key=key)


def _read_file(path: Path) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})   # 0 = giữ mọi điểm
    ea.Reload()
    have = set(ea.Tags().get("scalars", []))
    return {t: [(e.step, float(e.value)) for e in ea.Scalars(t)] for t in TAGS if t in have}


def read_run_scalars(run_dir: Path | str) -> dict[str, list[tuple[int, float]]]:
    """Gộp mọi file events của một run thành ``{tag: [(step, value)]}`` theo step tăng dần.

    Run bị cắt rồi train tiếp từ ``last.ckpt`` có thêm file events bắt đầu ở ``global_step`` của
    checkpoint. ``last.ckpt`` có thể cũ hơn epoch cuối đã chạy (chỉ ghi khi lọt top-3), nên file cũ
    còn điểm ở các bước SAU đó — nhánh đã bỏ. File sau thắng: bỏ mọi điểm của file trước có
    ``step >=`` bước nhỏ nhất của file sau.
    """
    merged: dict[str, list[tuple[int, float]]] = {}
    for path in _event_files(Path(run_dir)):
        scalars = _read_file(path)
        steps = [s for pts in scalars.values() for s, _ in pts]
        if not steps:
            continue
        cut = min(steps)
        for tag in merged:
            merged[tag] = [(s, v) for s, v in merged[tag] if s < cut]
        for tag, pts in scalars.items():
            merged.setdefault(tag, []).extend(pts)
    return {t: sorted(pts) for t, pts in merged.items()}


def _read_env(run_dir: Path) -> dict[str, str]:
    env = {}
    path = run_dir / "pipeline_train.env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, val = line.partition("=")
            if sep:
                env[key.strip()] = val.strip()
    return env


def _max_epochs(run_dir: Path, env: dict[str, str]) -> int | None:
    """``trainer.max_epochs`` thật của run (config hydra đã lưu), không có thì từ pipeline_train.env."""
    cfg = run_dir / "config" / "config.yaml"
    if cfg.exists():
        import yaml
        try:
            val = (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("trainer", {}).get("max_epochs")
            if isinstance(val, int):
                return val
        except yaml.YAMLError:
            pass
    return int(env["max_epochs"]) if env.get("max_epochs", "").isdigit() else None


def run_status(run_dir: Path) -> str:
    if (run_dir / "UNDERTRAINED").exists():
        return "UNDERTRAINED"
    if (run_dir / "DONE").exists():
        return "DONE"
    return "chưa xong"


# ---------------------------------------------------------------------------
# gom theo epoch
# ---------------------------------------------------------------------------

def epoch_rows(scalars: dict[str, list[tuple[int, float]]], max_epochs: int | None) -> list[dict[str, Any]]:
    """Một dòng mỗi lần validate: ``local_epoch`` (0-based trong run), loss train trung bình
    của epoch đó, val_loss, val_accuracy, val_NED, ``swa``.

    Epoch của một bước = giá trị ``epoch`` ở bước gần nhất không lớn hơn nó (Lightning ghi
    ``epoch`` cùng mọi lần log, nên thường trùng đúng bước).
    """
    ep = scalars.get("epoch", [])
    ep_steps = [s for s, _ in ep]

    def epoch_at(step: int) -> int | None:
        i = bisect.bisect_right(ep_steps, step) - 1
        return int(round(ep[i][1])) if i >= 0 else None

    loss_by_epoch: dict[int, list[float]] = {}
    for s, v in scalars.get("loss", []):
        e = epoch_at(s)
        if e is not None:
            loss_by_epoch.setdefault(e, []).append(v)

    val = {t: dict(scalars.get(t, [])) for t in ("val_loss", "val_accuracy", "val_NED")}
    rows = []
    for s, acc in scalars.get("val_accuracy", []):
        e = epoch_at(s)
        if e is None:
            continue
        # Lượt SWA (tính lại BatchNorm) mang chỉ số epoch = max_epochs: không phải epoch train,
        # loss "train" của nó là một bước forward lẻ — bỏ.
        swa = max_epochs is not None and e >= max_epochs
        losses = loss_by_epoch.get(e, [])
        rows.append({
            "local_epoch": e,
            "train_loss": None if swa or not losses else round(sum(losses) / len(losses), 4),
            "val_loss": _r(val["val_loss"].get(s)),
            "val_accuracy": _r(acc),
            "val_NED": _r(val["val_NED"].get(s)),
            "swa": swa,
        })
    return rows


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


def chain(run_dir: Path, runs_root: Path) -> list[Path]:
    """[run gốc, …, ``run_dir``] theo ``continue_from`` (đường dẫn tương đối ``runs_root``)."""
    out = [run_dir]
    seen = {run_dir.resolve()}
    while True:
        parent = _read_env(out[0]).get("continue_from")
        if not parent:
            return out
        p = (runs_root / parent)
        if not p.is_dir() or p.resolve() in seen:   # run gốc không mang theo / vòng lặp
            return out
        seen.add(p.resolve())
        out.insert(0, p)


def model_curve(run_dir: Path, runs_root: Path) -> dict[str, Any]:
    """Số liệu theo epoch của ``run_dir`` nối với các run nó train tiếp từ đó.

    ``epoch`` trong kết quả là số thứ tự 1-based tính dồn từ run gốc (epoch 1 = epoch đầu tiên của
    run gốc). Run cha chỉ giữ tới epoch của checkpoint mà run con đã nạp (``init_epoch``): các epoch
    sau đó của run cha (vd lượt SWA khi checkpoint là epoch cuối thật) không nằm trên đường trọng
    số được train tiếp.
    """
    runs = chain(run_dir, runs_root)
    epochs: list[dict[str, Any]] = []
    segments = []
    for i, rd in enumerate(runs):
        env = _read_env(rd)
        offset = int(env.get("epoch_offset", "0") or 0)
        rows = epoch_rows(read_run_scalars(rd), _max_epochs(rd, env))
        if i + 1 < len(runs):
            # Chỉ giữ tới epoch của checkpoint run con đã nạp. init_epoch = max_epochs nghĩa là
            # last.ckpt là model sau lượt SWA: giữ cả điểm SWA, vì run con bắt đầu từ chính nó.
            child_env = _read_env(runs[i + 1])
            keep = int(child_env.get("init_epoch", "-1") or -1)
            rows = [r for r in rows if r["local_epoch"] <= keep]
        rel = _rel(rd, runs_root)
        for r in rows:
            # Điểm SWA nằm cùng hoành độ với epoch cuối (đo sau khi train xong epoch đó).
            r["epoch"] = offset + r["local_epoch"] + (0 if r["swa"] else 1)
            r["run"] = rel
        if rows:
            segments.append({"run": rel, "first_epoch": min(r["epoch"] for r in rows),
                             "last_epoch": max(r["epoch"] for r in rows),
                             "continued_from": env.get("continue_from") or None})
        epochs.extend(rows)
    trained = [r for r in epochs if not r["swa"] and r["val_accuracy"] is not None]
    best = max(trained, key=lambda r: r["val_accuracy"]) if trained else None
    if best is not None and best["val_accuracy"] <= 0:   # 0% mọi epoch: không có "tốt nhất"
        best = None
    return {
        "run": _rel(run_dir, runs_root),
        "status": run_status(run_dir),
        "segments": segments,
        "epochs": [{k: r[k] for k in ("epoch", "run", "train_loss", "val_loss", "val_accuracy",
                                      "val_NED", "swa")} for r in epochs],
        "best": {"epoch": best["epoch"], "val_accuracy": best["val_accuracy"]} if best else None,
        "overfit": overfit_check(epochs),
    }


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(p)


def overfit_check(epochs: list[dict[str, Any]]) -> dict[str, Any]:
    """Kết luận từ val_loss / loss train (bỏ điểm SWA).

    - ``chua_du``    < 3 epoch có val_loss: chưa đủ để nói gì;
    - ``co_dau_hieu`` val_loss thấp nhất ở epoch k, sau đó >= 2 epoch và epoch cuối cao hơn đáy
      > 5% trong khi loss train epoch cuối thấp hơn ở epoch k — model học thuộc tập train;
    - ``chua_thay``  val_loss thấp nhất nằm ở epoch cuối hoặc kế cuối: vẫn đang tốt lên;
    - ``di_ngang``   còn lại (đáy sớm hơn nhưng chưa tăng rõ): sắp bão hoà, train thêm ít lợi.
    """
    rows = [r for r in epochs if not r["swa"] and r["val_loss"] is not None]
    if len(rows) < 3:
        return {"verdict": "chua_du", "text": f"mới có {len(rows)} epoch có val_loss — chưa đủ để kết luận"}
    k = min(range(len(rows)), key=lambda i: rows[i]["val_loss"])
    low, last = rows[k], rows[-1]
    after = len(rows) - 1 - k
    rise = (last["val_loss"] - low["val_loss"]) / low["val_loss"] if low["val_loss"] else 0.0
    train_fell = (last["train_loss"] is not None and low["train_loss"] is not None
                  and last["train_loss"] < low["train_loss"])
    if after >= OVERFIT_MIN_EPOCHS_AFTER and rise > OVERFIT_RISE and train_fell:
        return {"verdict": "co_dau_hieu", "best_val_loss_epoch": low["epoch"],
                "text": (f"có dấu hiệu overfit: val_loss thấp nhất ở epoch {low['epoch']} "
                         f"({_fmt(low['val_loss'], 4)}) rồi tăng {rise:.0%} tới epoch {last['epoch']} "
                         f"({_fmt(last['val_loss'], 4)}) trong khi loss train vẫn giảm "
                         f"({_fmt(low['train_loss'], 4)} → {_fmt(last['train_loss'], 4)})")}
    if after <= 1:
        return {"verdict": "chua_thay", "best_val_loss_epoch": low["epoch"],
                "text": (f"chưa thấy overfit: val_loss vẫn giảm tới epoch {low['epoch']} "
                         f"({_fmt(low['val_loss'], 4)}); model còn học thêm được")}
    return {"verdict": "di_ngang", "best_val_loss_epoch": low["epoch"],
            "text": (f"val_loss thấp nhất ở epoch {low['epoch']} ({_fmt(low['val_loss'], 4)}), "
                     f"{after} epoch sau đó đi ngang (epoch cuối {_fmt(last['val_loss'], 4)}, "
                     f"{rise:+.0%}) — gần bão hoà, chưa rõ overfit")}


# ---------------------------------------------------------------------------
# chọn run + vẽ
# ---------------------------------------------------------------------------

def pick_run(model_dir: Path) -> Path | None:
    """Run mới nhất (theo tên thư mục ``<YYYY-MM-DD_HH-MM-SS>``) có ít nhất một lần validate."""
    for rd in sorted((p for p in model_dir.iterdir() if p.is_dir() and p.name != "multirun"),
                     key=lambda p: p.name, reverse=True):
        if any(t == "val_accuracy" for t in read_run_scalars(rd)):
            return rd
    return None


def _fmt(v: float, nd: int = 1) -> str:
    return f"{v:.{nd}f}".replace(".", ",")


def plot_curve(model: str, curve: dict[str, Any], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter

    rows = curve["epochs"]
    trained = [r for r in rows if not r["swa"]]
    swa = [r for r in rows if r["swa"]]

    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150, facecolor=SURFACE)
    for ax in (ax_l, ax_a):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_MUTED)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("epoch", color=INK_2, fontsize=9)

    def series(ax, key, color, label, rows_):
        pts = [(r["epoch"], r[key]) for r in rows_ if r[key] is not None]
        if not pts:
            return
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=1.6, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1, label=label)
        # nhãn trực tiếp ở cuối đường, chữ màu mực (không dùng màu series)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)

    # --- loss: thang log, vì epoch đầu (lớp ra khởi tạo lại) lớn hơn hẳn các epoch sau
    series(ax_l, "train_loss", COLOR_TRAIN, "train", trained)
    series(ax_l, "val_loss", COLOR_VAL, "val", trained)
    if any(r["val_loss"] for r in swa):
        ax_l.plot([r["epoch"] for r in swa], [r["val_loss"] for r in swa], linestyle="none",
                  marker="D", markersize=6, markerfacecolor=SURFACE, markeredgecolor=COLOR_VAL,
                  markeredgewidth=1.5, label="val (sau SWA)")
    # Thang log khi loss trải rộng (epoch đầu: lớp ra vừa khởi tạo lại, loss gấp hơn 10 lần các
    # epoch sau, thang thường ép phần cần xem — các epoch cuối — thành một đường phẳng).
    positive = [r[k] for r in rows for k in ("train_loss", "val_loss") if r[k] and r[k] > 0]
    log_scale = bool(positive) and max(positive) / min(positive) > 3
    if log_scale:
        ax_l.set_yscale("log")
        ax_l.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax_l.yaxis.set_minor_formatter(NullFormatter())
    ax_l.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax_l.set_title("Loss (thang log)" if log_scale else "Loss", loc="left", color=INK, fontsize=10.5)
    ax_l.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")

    # --- accuracy: chỉ có val (parseq không đo accuracy trên tập train)
    series(ax_a, "val_accuracy", COLOR_VAL, "val", trained)
    if swa:
        ax_a.plot([r["epoch"] for r in swa], [r["val_accuracy"] for r in swa], linestyle="none",
                  marker="D", markersize=6, markerfacecolor=SURFACE, markeredgecolor=COLOR_VAL,
                  markeredgewidth=1.5, label="val (sau SWA)")
        ax_a.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    best = curve.get("best")
    if best:
        ax_a.annotate(f"tốt nhất {_fmt(best['val_accuracy'])}% (epoch {best['epoch']})",
                      (best["epoch"], best["val_accuracy"]), xytext=(0, 10),
                      textcoords="offset points", ha="center", fontsize=8.5, color=INK_2)
    top = max([r["val_accuracy"] for r in rows if r["val_accuracy"] is not None] or [1.0])
    ax_a.set_ylim(0, min(100, top + 12))
    ax_a.set_title("Accuracy trên tập val (%)", loc="left", color=INK, fontsize=10.5)

    # --- ranh giới "train tiếp" (run con nạp trọng số run cha, lịch LR khởi động lại)
    for seg in curve["segments"][1:]:
        x = seg["first_epoch"] - 0.5
        for ax in (ax_l, ax_a):
            ax.axvline(x, color=INK_MUTED, linestyle="--", linewidth=1)
        ax_a.annotate("train tiếp", (x, 0), xytext=(4, 4), textcoords="offset points",
                      fontsize=8, color=INK_MUTED)

    run_label = curve["run"].split("/", 1)[-1]
    status = STATUS_TEXT.get(curve["status"], curve["status"])
    fig.suptitle(f"{model} — run {run_label} ({status})", x=0.01, ha="left",
                 color=INK, fontsize=11.5)
    fig.text(0.01, 0.01, "loss train = trung bình các batch trong epoch, tính trên ảnh đã "
             "augment (có dropout); accuracy trên tập train không được ghi lại.",
             fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)


def render_markdown(curves: dict[str, dict[str, Any]], skipped: dict[str, str]) -> str:
    lines = ["# Đường cong huấn luyện", "",
             "Sinh tự động bởi `python -m trademark_ocr.train_curves` từ log TensorBoard của từng "
             "run. `epoch` đánh số từ 1, tính dồn qua các lần train tiếp. Loss train tính trên ảnh "
             "đã augment nên thường cao hơn val_loss — so **xu hướng** hai đường, không so độ lớn.", ""]
    for model, c in curves.items():
        status = STATUS_TEXT.get(c["status"], c["status"])
        lines += [f"## {model}", "", f"Run `{c['run']}` — **{status}** (`{c['status']}`).", ""]
        if len(c["segments"]) > 1:
            parts = [f"`{s['run']}` (epoch {s['first_epoch']}–{s['last_epoch']})" for s in c["segments"]]
            lines += ["Nối các lần train: " + " → ".join(parts) + ".", ""]
        lines += [f"![{model}]({c['figure']})", "",
                  "| epoch | loss train | val_loss | val_accuracy (%) | val 1−NED (%) | ghi chú |",
                  "|---|---|---|---|---|---|"]
        for r in c["epochs"]:
            note = "model sau SWA" if r["swa"] else ""
            if c["best"] and r["epoch"] == c["best"]["epoch"] and not r["swa"]:
                note = "tốt nhất"
            cells = [str(r["epoch"])] + ["—" if r[k] is None else _fmt(r[k], 4 if "loss" in k else 2)
                                        for k in ("train_loss", "val_loss", "val_accuracy", "val_NED")]
            lines.append("| " + " | ".join(cells + [note]) + " |")
        lines += ["", f"**Kết luận:** {c['overfit']['text']}.", ""]
    if skipped:
        lines += ["## Không vẽ được", ""] + [f"- `{m}`: {why}" for m, why in skipped.items()] + [""]
    return "\n".join(lines)


def build(runs_root: Path | str, out_dir: Path | str, models: list[str] | None = None,
          plot: bool = True) -> dict[str, Any]:
    """Vẽ + ghi ``train_curves.json``/``.md`` cho mọi model (hoặc ``models``) trong ``runs_root``."""
    runs_root, out_dir = Path(runs_root), Path(out_dir)
    names = models if models else sorted(p.name for p in runs_root.iterdir() if p.is_dir()) \
        if runs_root.is_dir() else []
    curves: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for m in names:
        md = runs_root / m
        rd = pick_run(md) if md.is_dir() else None
        if rd is None:
            skipped[m] = ("không có thư mục run" if not md.is_dir()
                          else "không run nào có log TensorBoard (events.out.tfevents.*) với val_accuracy")
            continue
        c = model_curve(rd, runs_root)
        c["figure"] = f"train_curves_{m}.png"
        curves[m] = c
    out_dir.mkdir(parents=True, exist_ok=True)
    if plot:
        for m, c in curves.items():
            plot_curve(m, c, out_dir / c["figure"])
    result = {"models": curves, "skipped": skipped}
    (out_dir / "train_curves.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    (out_dir / "train_curves.md").write_text(render_markdown(curves, skipped), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default=None, help="thư mục runs (mặc định $DATA_ROOT/runs)")
    ap.add_argument("--out-dir", default="reports/figures")
    ap.add_argument("--models", default="", help="vd crnn_ft,parseq_ft (mặc định: mọi model có run)")
    args = ap.parse_args(argv)
    runs = Path(args.runs) if args.runs else Path(os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT)) / "runs"
    models = [m for m in re.split(r"[,\s]+", args.models) if m]
    # Tên ở --models giống scripts/05_train.sh; svtr ghi run vào runs/svtrctc/.
    models = ["svtrctc" if m == "svtr" else m for m in models]
    res = build(runs, args.out_dir, models or None)
    for m, c in res["models"].items():
        best = c["best"]
        print(f"[train_curves] {m}: {c['run']} ({c['status']}), {len(c['epochs'])} điểm, "
              f"tốt nhất {_fmt(best['val_accuracy'], 2) + '%' if best else '—'} — {c['overfit']['text']}")
    for m, why in res["skipped"].items():
        print(f"[train_curves] bỏ qua {m}: {why}")
    print(f"[train_curves] -> {Path(args.out_dir) / 'train_curves.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
