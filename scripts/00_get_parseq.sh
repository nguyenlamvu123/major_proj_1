#!/usr/bin/env bash
# =====================================================================
# Lấy khung baudm/parseq vào third_party/parseq ở ĐÚNG commit đã kiểm chứng.
#
#   bash scripts/00_get_parseq.sh
#
# Vì sao ghim: đồ án dựa vào API nội bộ của parseq (LmdbDataset, CharsetAdapter,
# _get_model_class, get_pretrained_weights, train.py + hydra) mà không sửa code của họ.
# Clone HEAD mới có thể đổi các API này mà không báo — mọi số liệu trong DESIGN.md đo với
# commit dưới đây. Đổi commit = chạy lại `pytest -q` rồi cập nhật PARSEQ_COMMIT.
# =====================================================================
set -euo pipefail

PARSEQ_COMMIT="1902db043c029a7e03a3818c616c06600af574be"
PARSEQ_URL="https://github.com/baudm/parseq"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/third_party/parseq"

if [ ! -d "$DEST/.git" ]; then
  git clone "$PARSEQ_URL" "$DEST"
fi
# clone cũ có thể là bản --depth 1 ở HEAD khác: lấy đúng commit trước khi checkout
if ! git -C "$DEST" cat-file -e "$PARSEQ_COMMIT^{commit}" 2>/dev/null; then
  git -C "$DEST" fetch --depth 1 origin "$PARSEQ_COMMIT"
fi
git -C "$DEST" -c advice.detachedHead=false checkout "$PARSEQ_COMMIT"
echo "[00_get_parseq] third_party/parseq @ $(git -C "$DEST" rev-parse --short HEAD)"
