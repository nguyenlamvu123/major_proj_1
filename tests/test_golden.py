"""Golden-file test — testcase là DỮ LIỆU, file này không chứa testcase nào.

    tests/input/testcaseN/mota.txt      mô tả tiếng Việt: mục đích, đầu vào, kỳ vọng, vì sao quan trọng
    tests/input/testcaseN/params.json   {"kind", "desc", ...} — xem từng kind bên dưới
    tests/expect/testcaseN/*.json       kết quả kỳ vọng
    tests/actual/testcaseN/             sinh ra mỗi lần chạy rồi đem so với expect/ (gitignore)

kind = "plan"     scripts/09_full_pipeline_cpu.sh --dry-run trên một DATA_ROOT/venv dựng từ
                  params["state"]; params["args"] = cờ thêm.
                  -> plan.json (các dòng [plan], đường dẫn tạm thay bằng <DATA_ROOT>/<VENV>),
                     exit.json {"code"}, error.json {"code", "stderr_contains"} nếu kỳ vọng lỗi,
                     forbidden.json (lệnh git/curl/wget/python bị gọi — dry-run phải là []).
       state = {"stamps": {bước: [nội dung, tuổi_giây]},   -> <DATA_ROOT>/.pipeline/<bước>
                "files": {đường dẫn tương đối DATA_ROOT: tuổi_giây},
                "venv": {"python": true, "stamp": [nội dung, tuổi_giây]},
                "runs": [{"model", "ts", "ckpt_age", "done", "env": {...},
                          "files": {tên file trong run dir: nội dung}}],   vd UNDERTRAINED
                "reports": {tên file: tuổi_giây},  -> <WORK>/reports (REPORTS_DIR)
                "fake_gpu": true}   -> có nvidia-smi giả trong PATH
kind = "budget"   trademark_ocr.train_utils.epochs_for_budget(**params["inputs"]) -> result.json
kind = "eval_sh"  scripts/06_eval.sh với PYTHON giả ghi lại lệnh; params["state"] như plan
                  (runs + files + reports), params["args"] -> calls.json, exit.json,
                  forbidden.json, reports_after.json (file trong REPORTS_DIR sau khi chạy,
                  thư mục archive/<giờ> thay bằng archive/<TS>)
kind = "train_sh" scripts/05_train.sh với PYTHON giả ghi lại lệnh, DATA_ROOT=<WORK>/data dựng từ
                  params["state"]["files"] (đường dẫn có thể ra ngoài DATA_ROOT bằng "../"),
                  params["args"] -> calls.json, exit.json, forbidden.json, log.json (các dòng
                  "[05_train]", python giả thay bằng <PYTHON>), error.json nếu thoát khác 0
kind = "verdict"  trademark_ocr.train_utils.train_verdict trên một run dir có các checkpoint
                  tên params["ckpts"] (file rỗng); params["kwargs"] -> result.json
kind = "metrics"  trademark_ocr.metrics.evaluate_metrics(params["preds"], params["gts"]),
                  làm tròn 6 chữ số -> result.json
kind = "curves"   trademark_ocr.train_curves.build trên <WORK>/runs dựng từ params["runs"]:
                  [{"model", "ts", "env": {...}, "files": {tên: nội dung}, "max_epochs": N
                    (-> config/config.yaml), "events": [{"name": tên file events,
                    "steps_per_epoch": N, "epochs": [{"e": epoch 0-based, "loss": [mỗi bước],
                    "val": [val_loss, val_accuracy, val_NED]}]}]}]
                  bước của loss thứ i trong epoch e = e·N + i; val ghi ở bước e·N + N − 1 (như
                  Lightning: validate ở bước cuối epoch). params["models"], params["plot"].
                  -> result.json, files.json (file trong out-dir), md.json (các dòng train_curves.md)

MỌI lệnh bash chạy với REPORTS_DIR=<WORK>/reports: 06_eval.sh chuyển báo cáo cũ vào archive,
để mặc định thì test sẽ dọn reports/ THẬT của repo (đã xảy ra một lần khi viết tính năng này).

Thêm testcase = thêm thư mục, không sửa file này. Chạy:
    python -m pytest -q tests/test_golden.py
    python -m unittest tests.test_golden -v
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
sys.path.insert(0, str(REPO / "src"))

INPUT, EXPECT, ACTUAL = BASE / "input", BASE / "expect", BASE / "actual"

#: lệnh mà testcase bash KHÔNG được gọi tới (mạng hoặc chạy thật); stub ghi lại rồi thoát 97
FORBIDDEN = ("git", "curl", "wget", "python", "python3", "pip", "systemd-inhibit")


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def touch_age(path: Path, age: float, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    t = time.time() - age
    os.utime(path, (t, t))


def make_stubs(out: Path, state: dict) -> Path:
    """Thư mục bin/ đứng đầu PATH: stub cho lệnh cấm + nvidia-smi giả (nếu state cần)."""
    bin_dir = out / "bin"
    bin_dir.mkdir(parents=True)
    log = out / "forbidden.txt"
    for name in FORBIDDEN:
        stub = bin_dir / name
        stub.write_text(f'#!/bin/sh\necho "{name} $*" >> "{log}"\nexit 97\n', encoding="utf-8")
        stub.chmod(0o755)
    if state.get("fake_gpu"):
        gpu = bin_dir / "nvidia-smi"
        gpu.write_text("#!/bin/sh\necho 'GPU 0: fake'\n", encoding="utf-8")
        gpu.chmod(0o755)
    return bin_dir


def build_state(state: dict, data_root: Path, venv: Path) -> None:
    for step, (content, age) in state.get("stamps", {}).items():
        touch_age(data_root / ".pipeline" / step, age, content + "\n")
    for rel, age in state.get("files", {}).items():
        touch_age(data_root / rel, age, "x")
    if "venv" in state:
        if state["venv"].get("python"):
            touch_age(venv / "bin" / "python", 0, "#!/bin/sh\n")
            (venv / "bin" / "python").chmod(0o755)
        if "stamp" in state["venv"]:
            content, age = state["venv"]["stamp"]
            touch_age(venv / ".pipeline_setup", age, content + "\n")
    for run in state.get("runs", []):
        rd = data_root / "runs" / run["model"] / run["ts"]
        rd.mkdir(parents=True, exist_ok=True)       # run dir có thể rỗng (hydra tạo, chưa có ckpt)
        if run.get("ckpt_age") is not None:
            touch_age(rd / "checkpoints" / "last.ckpt", run["ckpt_age"], "ckpt")
        if run.get("done"):
            touch_age(rd / "DONE", 0)
        if run.get("env"):
            touch_age(rd / "pipeline_train.env", 0,
                      "".join(f"{k}={v}\n" for k, v in run["env"].items()))
        for name, content in run.get("files", {}).items():
            touch_age(rd / name, 0, content)
    for name, age in state.get("reports", {}).items():
        touch_age(data_root.parent / "reports" / name, age, "x")


def normalize(text: str, mapping: dict[str, str]) -> str:
    for real, placeholder in mapping.items():
        text = text.replace(real, placeholder)
    return text


def run_bash(script: str, args: list[str], *, out: Path, state: dict, extra_env: dict) -> tuple:
    work = out / "work"
    data_root, venv = work / "data", work / "venv"
    build_state(state, data_root, venv)
    bin_dir = make_stubs(out, state)
    env = {k: v for k, v in os.environ.items() if k not in ("DATA_ROOT", "PYTHON")}
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '/usr/bin:/bin')}"
    env["REPORTS_DIR"] = str(work / "reports")
    env.update({k: v.replace("<WORK>", str(work)) for k, v in extra_env.items()})
    args = [a.replace("<WORK>", str(work)) for a in args]
    proc = subprocess.run(["bash", str(REPO / "scripts" / script), *args], env=env,
                          capture_output=True, text=True, timeout=60)
    mapping = {str(data_root): "<DATA_ROOT>", str(venv): "<VENV>", str(work): "<WORK>",
               str(REPO): "<REPO>"}
    forbidden_log = out / "forbidden.txt"
    forbidden = forbidden_log.read_text(encoding="utf-8").splitlines() if forbidden_log.exists() else []
    return proc, mapping, [normalize(f, mapping) for f in forbidden], data_root, venv


def run_case(name: str) -> Path:
    inp, out = INPUT / name, ACTUAL / name
    params = json.loads((inp / "params.json").read_text(encoding="utf-8"))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    kind = params["kind"]

    if kind == "budget":
        from trademark_ocr.train_utils import epochs_for_budget
        dump(out / "result.json", epochs_for_budget(**params["inputs"]))

    elif kind == "plan":
        proc, mapping, forbidden, data_root, venv = run_bash(
            "09_full_pipeline_cpu.sh",
            ["--dry-run", "--data-root", "<WORK>/data", "--venv", "<WORK>/venv", *params.get("args", [])],
            out=out, state=params.get("state", {}), extra_env={})
        lines = [normalize(l, mapping) for l in proc.stdout.splitlines() if l.startswith("[plan]")]
        dump(out / "plan.json", lines)
        dump(out / "exit.json", {"code": proc.returncode})
        dump(out / "forbidden.json", forbidden)
        if proc.returncode != 0:
            dump(out / "error.json", {"code": proc.returncode, "stderr": normalize(proc.stderr, mapping)})
        # dry-run không được tạo gì trong DATA_ROOT ngoài trạng thái testcase tự dựng
        work = data_root.parent
        created = sorted(str(p.relative_to(work)) for p in work.rglob("*") if p.is_file()) \
            if work.exists() else []
        dump(out / "files_after.json", created)

    elif kind == "eval_sh":
        calls = out / "python_calls.txt"
        fake_py = out / "fake_python"
        fake_py.write_text(f'#!/bin/sh\necho "$*" >> "{calls}"\n', encoding="utf-8")
        fake_py.chmod(0o755)
        proc, mapping, forbidden, *_ = run_bash(
            "06_eval.sh", params.get("args", []), out=out, state=params.get("state", {}),
            extra_env={"DATA_ROOT": "<WORK>/data", "PYTHON": str(fake_py)})
        lines = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
        dump(out / "calls.json", [normalize(l, mapping) for l in lines])
        dump(out / "exit.json", {"code": proc.returncode})
        dump(out / "forbidden.json", forbidden)
        skipped = [normalize(l, mapping) for l in proc.stdout.splitlines() if "bỏ qua" in l]
        dump(out / "skipped.json", skipped)
        reports = out / "work" / "reports"
        after = sorted(re.sub(r"archive/\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d", "archive/<TS>",
                              str(f.relative_to(reports)))
                       for f in reports.rglob("*") if f.is_file()) if reports.exists() else []
        dump(out / "reports_after.json", after)
        if proc.returncode != 0:
            dump(out / "error.json", {"code": proc.returncode,
                                      "stderr": normalize(proc.stderr, mapping)})
    elif kind == "train_sh":
        calls = out / "python_calls.txt"
        fake_py = out / "fake_python"
        fake_py.write_text(f'#!/bin/sh\necho "$*" >> "{calls}"\n', encoding="utf-8")
        fake_py.chmod(0o755)
        proc, mapping, forbidden, *_ = run_bash(
            "05_train.sh", params.get("args", []), out=out, state=params.get("state", {}),
            extra_env={"DATA_ROOT": "<WORK>/data", "PYTHON": str(fake_py)})
        # python giả nằm dưới tests/actual/ (trong <REPO>) -> thay TRƯỚC khi thay <REPO>
        mapping = {str(fake_py): "<PYTHON>", **mapping}
        lines = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
        dump(out / "calls.json", [normalize(l, mapping) for l in lines])
        dump(out / "exit.json", {"code": proc.returncode})
        dump(out / "forbidden.json", forbidden)
        dump(out / "log.json", [normalize(l, mapping) for l in proc.stdout.splitlines()
                                if l.startswith("[05_train]")])
        if proc.returncode != 0:
            dump(out / "error.json", {"code": proc.returncode,
                                      "stderr": normalize(proc.stderr, mapping)})

    elif kind == "verdict":
        from trademark_ocr.train_utils import train_verdict
        run_dir = out / "work" / "run"
        (run_dir / "checkpoints").mkdir(parents=True)
        for ck in params["ckpts"]:
            (run_dir / "checkpoints" / ck).write_bytes(b"")
        dump(out / "result.json", train_verdict(run_dir, **params.get("kwargs", {})))

    elif kind == "metrics":
        from trademark_ocr.metrics import evaluate_metrics
        res = evaluate_metrics(params["preds"], params["gts"])
        dump(out / "result.json", {k: round(v, 6) for k, v in res.items()})

    elif kind == "curves":
        from trademark_ocr.train_curves import build
        runs_root = out / "work" / "runs"
        for run in params["runs"]:
            rd = runs_root / run["model"] / run["ts"]
            rd.mkdir(parents=True, exist_ok=True)
            if run.get("env"):
                (rd / "pipeline_train.env").write_text(
                    "".join(f"{k}={v}\n" for k, v in run["env"].items()), encoding="utf-8")
            for fname, content in run.get("files", {}).items():
                (rd / fname).parent.mkdir(parents=True, exist_ok=True)
                (rd / fname).write_text(content, encoding="utf-8")
            if run.get("max_epochs") is not None:
                (rd / "config").mkdir(exist_ok=True)
                (rd / "config" / "config.yaml").write_text(
                    f"trainer:\n  max_epochs: {run['max_epochs']}\n", encoding="utf-8")
            for ev in run.get("events", []):
                write_events(rd, ev)
        out_dir = out / "work" / "figures"
        res = build(runs_root, out_dir, params.get("models"), plot=params.get("plot", False))
        dump(out / "result.json", res)
        dump(out / "files.json", sorted(p.name for p in out_dir.iterdir()))
        dump(out / "md.json", (out_dir / "train_curves.md").read_text(encoding="utf-8").splitlines())

    else:
        raise ValueError(f"{name}: kind lạ {kind!r}")
    return out


def write_events(run_dir: Path, ev: dict) -> None:
    """Một file events TensorBoard đúng tên ``ev["name"]`` (thứ tự file theo tên quyết định
    run nào "train tiếp" sau), cùng bố cục bước như parseq/Lightning ghi thật."""
    import tempfile
    from torch.utils.tensorboard import SummaryWriter

    n = ev["steps_per_epoch"]
    with tempfile.TemporaryDirectory(dir=run_dir) as tmp:
        w = SummaryWriter(log_dir=tmp)
        for row in ev["epochs"]:
            e = row["e"]
            for i, loss in enumerate(row.get("loss", [])):
                w.add_scalar("loss", loss, e * n + i)
                w.add_scalar("epoch", e, e * n + i)
            if row.get("val"):
                step = e * n + n - 1
                for tag, v in zip(("val_loss", "val_accuracy", "val_NED"), row["val"]):
                    w.add_scalar(tag, v, step)
                w.add_scalar("epoch", e, step)
        w.close()
        (written,) = Path(tmp).glob("events.out.tfevents.*")
        written.rename(run_dir / ev["name"])


def compare(tc: unittest.TestCase, name: str, exp: Path, act: Path) -> None:
    files = sorted(p for p in exp.rglob("*") if p.is_file())
    # Thiếu expect/ thì vòng so không chạy lần nào và testcase xanh mà không kiểm gì.
    tc.assertTrue(files, f"{name}: expect/ rỗng hoặc không có — testcase không kiểm gì")
    if not (exp / "error.json").exists() and (act / "error.json").exists():
        err = json.loads((act / "error.json").read_text(encoding="utf-8"))
        tc.fail(f"{name}: không kỳ vọng lỗi nhưng exit {err['code']}: {err['stderr'][-500:]}")
    for exp_file in files:
        rel = exp_file.relative_to(exp)
        act_file = act / rel
        tc.assertTrue(act_file.exists(), f"{name}: actual/ thiếu {rel}")
        want = json.loads(exp_file.read_text(encoding="utf-8"))
        got = json.loads(act_file.read_text(encoding="utf-8"))
        if rel.name == "error.json":
            tc.assertEqual(want["code"], got["code"], f"{name}: sai mã thoát")
            tc.assertIn(want["stderr_contains"], got["stderr"], f"{name}: stderr thiếu thông báo")
        else:
            tc.assertEqual(want, got, f"{name}: {rel} khác expect")


class TestGolden(unittest.TestCase):
    maxDiff = None


def make_test(name: str):
    def test(self):
        compare(self, name, EXPECT / name, run_case(name))
    desc = json.loads((INPUT / name / "params.json").read_text(encoding="utf-8")).get("desc", "")
    test.__doc__ = f"{name}: {desc}"
    return test


def natural_key(path: Path) -> int:
    return int(re.search(r"\d+", path.name).group())


if not INPUT.is_dir():
    raise RuntimeError(f"không thấy thư mục testcase: {INPUT}")

for _dir in sorted(INPUT.glob("testcase*"), key=natural_key):
    setattr(TestGolden, f"test_{_dir.name}", make_test(_dir.name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
