"""Shim tương thích NumPy 2.x cho `imgaug` (RandAugment của parseq).

Vấn đề: `imgaug==0.4.0` (2020) dùng `np.sctypes` đã bị **xóa ở NumPy 2.0**.
- Không sửa được `third_party/parseq` (quy ước: không fork/patch parseq).
- Không hạ `numpy<2` (vỡ torch 2.14 / pyarrow / pandas đã cài).
→ Khôi phục biến thể đó **trong process của mình** trước khi import `imgaug`.

Nên chạy ở **đầu process**:
* train thật: `scripts/05_train.sh` đặt `PYTHONPATH` gồm `compat/` → `compat/sitecustomize.py`
  được `site` import lúc khởi động Python và gọi `apply_numpy2_compat()` trước mọi module.
* dry-run: `train_utils` tự gọi khi import.
"""

from __future__ import annotations

import warnings

_PATCHED = False


def apply_numpy2_compat() -> bool:
    """Khôi phục `np.sctypes` (và vài alias đã xóa) nếu đang chạy NumPy 2. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return False
    import numpy as np

    changed = False
    if not hasattr(np, "sctypes"):
        np.sctypes = {                       # type: ignore[attr-defined]
            "float": [np.float16, np.float32, np.float64, np.longdouble],
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "others": [bool, object, bytes, str, np.void],
        }
        changed = True
    _PATCHED = True
    if changed:
        warnings.filterwarnings("ignore", message=r".*np\.sctypes.*")
    return changed


# Gọi ngay khi import để mọi nơi dùng `trademark_ocr` đều sẵn sàng.
apply_numpy2_compat()
