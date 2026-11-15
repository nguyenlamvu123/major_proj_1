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
#           reports/final_summary.md, $DATA_ROOT/processed/eda_figures/err_*.png,
#           reports/figures/train_curves_<model>.png + train_curves.{md,json} (đồ thị train)
#
# Báo cáo cũ được CHUYỂN vào reports/archive/<giờ>/ trước khi đánh giá (report.py gom mọi
# reports/*.csv, nên file của lần chạy khác — model nay không còn checkpoint — sẽ lẫn vào bảng):
#   không --only: chuyển mọi file báo cáo -> bộ báo cáo mới hoàn toàn từ checkpoint hiện có;
#   có --only:    chỉ chuyển file của các model được đánh giá lại, giữ nguyên model khác.
# Run có file UNDERTRAINED (train xong mà không học được, xem 09_full_pipeline_cpu.sh) bị bỏ
# qua; dùng checkpoint của run cũ hơn nếu có.
# REPORTS_DIR (mặc định <repo>/reports) đổi được thư mục báo cáo — test dùng để khỏi đụng repo.
# =====================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
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

REPORTS="${REPORTS_DIR:-$REPO_ROOT/reports}"
mkdir -p "$REPORTS"

# Chọn checkpoint TRƯỚC khi đụng vào báo cáo: không có model nào thì thoát mà báo cáo cũ còn nguyên.
declare -A CKPT=()
EVAL_MODELS=()
for m in "${SEL[@]}"; do
  ckpt=""; skipped=""
  # mới nhất trước; run UNDERTRAINED thì lùi về run cũ hơn
  while IFS= read -r c; do
    [[ -n "$c" ]] || continue
    if [[ -f "${c%/checkpoints/last.ckpt}/UNDERTRAINED" ]]; then skipped="${skipped:-${c%/checkpoints/last.ckpt}}"; continue; fi
    ckpt="$c"; break
  done < <(ls -dt "$DATA_ROOT/runs/$m"/*/checkpoints/last.ckpt 2>/dev/null || true)
  if [[ -z "$ckpt" ]]; then
    if [[ -n "$skipped" ]]; then
      echo "[06_eval] bỏ qua $m (train xong nhưng chưa học được: $skipped/UNDERTRAINED)"
    else
      echo "[06_eval] bỏ qua $m (chưa có checkpoint trong $DATA_ROOT/runs/$m/)"
    fi
    continue
  fi
  CKPT[$m]="$ckpt"; EVAL_MODELS+=("$m")
done

if [[ ${#EVAL_MODELS[@]} -eq 0 ]]; then
  echo "[06_eval] không tìm thấy checkpoint nào — train trước (scripts/05_train.sh)." >&2
  exit 1
fi

shopt -s nullglob
old=()
if [[ -z "$ONLY" ]]; then
  for f in "$REPORTS"/*; do [[ -f "$f" ]] && old+=("$f"); done     # glob bỏ qua .gitkeep, archive/
else
  for m in "${EVAL_MODELS[@]}"; do old+=("$REPORTS/${m}"_us.* "$REPORTS/${m}"_vn.*); done
fi
shopt -u nullglob
if [[ ${#old[@]} -gt 0 ]]; then
  arch="$REPORTS/archive/$(date +%Y-%m-%d_%H-%M-%S)"
  mkdir -p "$arch"
  mv "${old[@]}" "$arch/"
  echo "[06_eval] chuyển ${#old[@]} file báo cáo cũ -> $arch/"
fi

for m in "${EVAL_MODELS[@]}"; do
  ckpt="${CKPT[$m]}"
  echo "[06_eval] $m -> $ckpt"
  for sub in test_us test_vn; do
    dom="${sub#test_}"   # us | vn
    # thiếu LMDB một tập (vd crawl VN lỗi) thì bỏ tập đó: domain_shift/report tự bỏ model thiếu domain
    if [[ ! -f "$DATA_ROOT/lmdb/test/$sub/data.mdb" ]]; then
      echo "[06_eval]   bỏ qua $m / $sub (không có $DATA_ROOT/lmdb/test/$sub/data.mdb)"
      continue
    fi
    echo "[06_eval]   $m / $sub"
    "$PYTHON" -m trademark_ocr.evaluate --ckpt "$ckpt" --subset "$sub" \
      --out "$REPORTS/${m}_${dom}.csv" "${COMMON[@]}"
  done
done

echo "[06_eval] bảng domain shift"
"$PYTHON" -m trademark_ocr.eval_domain_shift --reports "$REPORTS/" \
  --out "$REPORTS/domain_shift.csv"

echo "[06_eval] phân tích lỗi -> $DATA_ROOT/processed/eda_figures/"
"$PYTHON" -m trademark_ocr.error_analysis --reports "$REPORTS/" \
  --out-dir "$DATA_ROOT/processed/eda_figures"

echo "[06_eval] báo cáo tổng hợp"
"$PYTHON" -m trademark_ocr.report --reports "$REPORTS/" \
  --out "$REPORTS/final_summary.md"

# Đồ thị train (loss train / val_loss / val_accuracy theo epoch) từ log TensorBoard của run mới
# nhất mỗi model. reports/figures/ là thư mục con nên bước chuyển báo cáo cũ vào archive/ ở trên
# không đụng tới; BAO_CAO_TONG_KET_DAT_LIEU_HUAN_LUYEN.md trỏ thẳng vào các PNG ở đây.
# Lỗi vẽ không được làm hỏng một lần eval đã xong.
echo "[06_eval] đồ thị train -> $REPORTS/figures/"
"$PYTHON" -m trademark_ocr.train_curves --runs "$DATA_ROOT/runs" --out-dir "$REPORTS/figures" \
  || echo "[06_eval] [WARN] không vẽ được đồ thị train (xem lỗi ở trên); báo cáo eval vẫn dùng được"

echo "[06_eval] xong. Xem $REPORTS/final_summary.md và $REPORTS/figures/train_curves.md"
