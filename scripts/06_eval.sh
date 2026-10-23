#!/usr/bin/env bash
# =====================================================================
# Nhánh 14 — đánh giá end-to-end: mỗi model (crnn/trba/parseq/svtrctc)
# -> test US + test VN -> bảng domain shift -> phân tích lỗi -> báo cáo.
#
#   bash scripts/06_eval.sh                          # tất cả model có ckpt
#   bash scripts/06_eval.sh --only crnn --limit 32   # 1 model, nhanh (CPU)
#   bash scripts/06_eval.sh --device cpu --batch-size 8 --num-workers 0
#
# Sản phẩm: reports/<model>_{us,vn}.csv, reports/domain_shift.csv,
#           reports/final_summary.md, $DATA_ROOT/processed/eda_figures/err_*.png
# =====================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
DATA_ROOT="${DATA_ROOT:-/media/zaibachkhoa/AC84-F995/trademark_data}"
export DATA_ROOT
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/parseq:${PYTHONPATH:-}"

MODELS=(crnn trba parseq svtrctc crnn_ft trba_ft parseq_ft)
LIMIT=0; DEVICE=cpu; BATCH=8; WORKERS=0; ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)        ONLY="$2"; shift 2 ;;
    --limit)       LIMIT="$2"; shift 2 ;;
    --device)      DEVICE="$2"; shift 2 ;;
    --batch-size)  BATCH="$2"; shift 2 ;;
    --num-workers) WORKERS="$2"; shift 2 ;;
    *) echo "tham số lạ: $1" >&2; exit 2 ;;
  esac
done

IFS=, read -ra SEL <<< "${ONLY:-$(IFS=,; echo "${MODELS[*]}")}"
COMMON=(--device "$DEVICE" --batch-size "$BATCH" --num-workers "$WORKERS")
[[ "$LIMIT" -gt 0 ]] && COMMON+=(--limit "$LIMIT")

mkdir -p "$REPO_ROOT/reports"
found=0
for m in "${SEL[@]}"; do
  ckpt="$(ls -dt "$DATA_ROOT/runs/$m"/*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)"
  if [[ -z "$ckpt" ]]; then
    echo "[06_eval] bỏ qua $m (chưa có checkpoint trong $DATA_ROOT/runs/$m/)"
    continue
  fi
  found=1
  echo "[06_eval] $m -> $ckpt"
  for sub in test_us test_vn; do
    dom="${sub#test_}"   # us | vn
    echo "[06_eval]   $m / $sub"
    "$PYTHON" -m trademark_ocr.evaluate --ckpt "$ckpt" --subset "$sub" \
      --out "$REPO_ROOT/reports/${m}_${dom}.csv" "${COMMON[@]}"
  done
done

if [[ "$found" -eq 0 ]]; then
  echo "[06_eval] không tìm thấy checkpoint nào — train trước (scripts/05_train.sh)." >&2
  exit 1
fi

echo "[06_eval] bảng domain shift"
"$PYTHON" -m trademark_ocr.eval_domain_shift --reports "$REPO_ROOT/reports/" \
  --out "$REPO_ROOT/reports/domain_shift.csv"

echo "[06_eval] phân tích lỗi -> $DATA_ROOT/processed/eda_figures/"
"$PYTHON" -m trademark_ocr.error_analysis --reports "$REPO_ROOT/reports/" \
  --out-dir "$DATA_ROOT/processed/eda_figures"

echo "[06_eval] báo cáo tổng hợp"
"$PYTHON" -m trademark_ocr.report --reports "$REPO_ROOT/reports/" \
  --out "$REPO_ROOT/reports/final_summary.md"

echo "[06_eval] xong. Xem $REPO_ROOT/reports/final_summary.md"
