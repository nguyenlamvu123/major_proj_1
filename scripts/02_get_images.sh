#!/usr/bin/env bash
# Nhánh 03 — tải ảnh nhãn hiệu US từ CDN USPTO (kèm resume & verify).
# Dùng: bash scripts/02_get_images.sh --limit 200
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATA_ROOT="${DATA_ROOT:-/media/zaibachkhoa/AC84-F995/trademark_data}"
PYTHON_BIN="${PYTHON:-$(command -v python3 || command -v python)}"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m trademark_ocr.fetch_images "$@"
