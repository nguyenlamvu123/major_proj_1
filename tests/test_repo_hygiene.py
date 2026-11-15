"""Kiểm tra cấu hình repo / script không lệch nhau (REVIEW.md §4.6, §4.8)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PARSEQ = REPO / "third_party" / "parseq"


def test_no_machine_specific_data_root_left():
    """Đường dẫn ổ đĩa của một máy cụ thể không được làm mặc định trong code/config/script."""
    offenders = [
        str(p.relative_to(REPO))
        for d in ("src", "scripts", "configs", "docs")
        for p in (REPO / d).rglob("*")
        if p.is_file() and p.suffix in {".py", ".sh", ".yaml", ".ipynb"}
        and "/media/zaibachkhoa" in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == []


def test_default_data_root_is_repo_data_unless_env_set(tmp_path):
    code = "import trademark_ocr; print(trademark_ocr.DEFAULT_DATA_ROOT)"
    env = {k: v for k, v in os.environ.items() if k != "DATA_ROOT"}
    env["PYTHONPATH"] = str(REPO / "src")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, check=True).stdout.strip()
    assert Path(out) == REPO / "data"
    env["DATA_ROOT"] = str(tmp_path)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, check=True).stdout.strip()
    assert Path(out) == tmp_path


@pytest.mark.parametrize("alias", ["svtr", "svtrctc", "parseq_ft"])
def test_train_script_accepts_documented_model_names(alias, tmp_path):
    # ROOT CAUSE:
    #
    # README hướng dẫn `bash scripts/05_train.sh svtrctc` nhưng script chỉ nhận `svtr`
    # → "unknown model". Đã thêm `svtrctc` (= tên run dir/model.name).
    #
    # DATA_ROOT rỗng → script phải đi QUA bước chọn config rồi mới dừng ở "thiếu LMDB".
    res = subprocess.run(["bash", str(REPO / "scripts" / "05_train.sh"), alias],
                         env={**os.environ, "DATA_ROOT": str(tmp_path)},
                         capture_output=True, text=True)
    assert res.returncode == 1
    assert "unknown model" not in res.stderr
    assert "thiếu LMDB" in res.stderr


def test_readme_train_commands_use_names_the_script_accepts():
    script = (REPO / "scripts" / "05_train.sh").read_text(encoding="utf-8")
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for name in re.findall(r"bash scripts/05_train\.sh (\w+)", readme):
        assert re.search(rf"(^|\|)\s*{name}\s*(\||\))", script, re.M), name


def test_parseq_clone_matches_pinned_commit():
    pinned = re.search(r'^PARSEQ_COMMIT="([0-9a-f]{40})"',
                       (REPO / "scripts" / "00_get_parseq.sh").read_text(encoding="utf-8"), re.M)
    assert pinned, "scripts/00_get_parseq.sh phải ghim SHA đầy đủ"
    if not (PARSEQ / ".git").exists():
        pytest.skip("third_party/parseq không phải git clone")
    head = subprocess.run(["git", "-C", str(PARSEQ), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == pinned.group(1)


def test_requirements_have_no_unused_async_client():
    reqs = (REPO / "requirements.txt").read_text(encoding="utf-8")
    assert not re.search(r"^aiohttp", reqs, re.M)


@pytest.mark.parametrize("name", ["00_setup", "01_download_data", "02_train", "03_eval"])
def test_colab_notebooks_exist_and_reference_real_scripts(name):
    nb = json.loads((REPO / "docs" / "colab" / f"{name}.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and nb["cells"]
    source = "".join("".join(c["source"]) for c in nb["cells"])
    for script in re.findall(r"scripts/\w+\.sh", source):
        assert (REPO / script).exists(), script
    for module in re.findall(r"-m (trademark_ocr\.\w+)", source):
        assert (REPO / "src" / Path(*module.split("."))).with_suffix(".py").exists(), module
