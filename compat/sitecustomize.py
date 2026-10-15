"""Tự động áp shim NumPy-2 trước khi Python chạy `third_party/parseq/train.py`.

`site` import module tên `sitecustomize` từ `sys.path` **ngay lúc khởi động**, nên
`scripts/05_train.sh` chỉ cần:

    export PYTHONPATH="$REPO_ROOT/compat:$REPO_ROOT/src:$REPO_ROOT/third_party/parseq:$PYTHONPATH"

Không dùng `scripts/05_train.sh` thì shim này không chạy → `augment=true` sẽ lỗi
`np.sctypes` (xem `trademark_ocr/_numpy2_compat.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from trademark_ocr._numpy2_compat import apply_numpy2_compat

    apply_numpy2_compat()
except Exception as exc:  # tuyệt đối không được chặn Python khởi động
    sys.stderr.write(f"[compat/sitecustomize] warning: {exc}\n")
