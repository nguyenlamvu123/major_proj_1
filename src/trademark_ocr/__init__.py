"""trademark_ocr — Nhận diện chữ trong nhãn hiệu sản phẩm (wordmark STR).

Toàn bộ code của đồ án nằm trong package này; khung huấn luyện/mô hình
được mượn từ ``third_party/parseq`` (github.com/baudm/parseq) mà không sửa.
"""

import os
from pathlib import Path

__version__ = "0.1.0"

#: Gốc dữ liệu (raw/, processed/, lmdb/, runs/ — DESIGN.md §2). Biến môi trường ``DATA_ROOT``
#: thắng; không đặt thì dùng ``<repo>/data`` (đã gitignore). Mọi module và CLI lấy giá trị
#: mặc định từ đây, đọc MỘT lần lúc import — đổi ``DATA_ROOT`` sau đó phải truyền
#: ``--data-root`` hoặc khởi động lại tiến trình.
DEFAULT_DATA_ROOT: str = os.environ.get(
    "DATA_ROOT", str(Path(__file__).resolve().parents[2] / "data")
)
