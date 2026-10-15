#!/usr/bin/env bash
# Nhánh 11 — thu ảnh + nhãn nhãn hiệu Việt Nam từ WIPO Publish (IP Vietnam).
# Dùng: bash scripts/03_get_vn.sh --limit 50
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATA_ROOT="${DATA_ROOT:-/media/zaibachkhoa/AC84-F995/trademark_data}"
PYTHON_BIN="${PYTHON:-$(command -v python3 || command -v python)}"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m trademark_ocr.fetch_vn "$@"
