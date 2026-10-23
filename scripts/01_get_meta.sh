#!/usr/bin/env bash
# Nhánh 02 — tải metadata wordmark US từ USPTO TMOG (EOG REST API).
# Dùng: bash scripts/01_get_meta.sh --limit 5000
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
PYTHON_BIN="${PYTHON:-$(command -v python3 || command -v python)}"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m trademark_ocr.fetch_meta "$@"
