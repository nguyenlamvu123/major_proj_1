#!/usr/bin/env bash
set -euo pipefail

BRANCH="16-fix-review"
REPO_URL="git@github.com:nguyenlamvu123/major_proj_1.git"
DATA_ROOT="${DATA_ROOT:-/data/trademark_data}"
WORK_DIR="${WORK_DIR:-$HOME/projects}"
REPO_NAME="major_proj_1"
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"

MODELS=(crnn trba parseq svtrctc)
ACCELERATOR="cpu"
DEVICES=1
BATCH_SIZE=8
NUM_WORKERS=0
MAX_EPOCHS=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch) BRANCH="$2"; shift 2 ;;
    --repo-url) REPO_URL="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --models) IFS=, read -ra MODELS <<< "$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --skip-clone) SKIP_CLONE=1; shift 1 ;;
    --skip-fetch) SKIP_FETCH=1; shift 1 ;;
    --skip-build-lmdb) SKIP_BUILD_LMDB=1; shift 1 ;;
    --skip-train) SKIP_TRAIN=1; shift 1 ;;
    --skip-eval) SKIP_EVAL=1; shift 1 ;;
    *) echo "Tham số không biết: $1"; exit 2 ;;
  esac
done

log() { echo -e "\n[PIPELINE] $(date '+%Y-%m-%d %H:%M:%S') $*"; }
ok()  { echo "[OK] $*"; }
warn(){ echo "[WARN] $*"; }
err() { echo "[ERR] $*"; exit 1; }

log "Kiểm tra môi trường"
command -v git >/dev/null || err "Cần git"
command -v "$PYTHON" >/dev/null || err "Cần python3"
mkdir -p "$WORK_DIR" "$DATA_ROOT"
ok "OK"

REPO_PATH="$WORK_DIR/$REPO_NAME"
log "Clone repo branch $BRANCH"
if [[ -n "${SKIP_CLONE:-}" ]] && [[ -d "$REPO_PATH/.git" ]]; then
  warn "Bỏ qua clone"
else
  if [[ -d "$REPO_PATH/.git" ]]; then
    cd "$REPO_PATH"; git fetch --all --prune; git checkout "$BRANCH" || git checkout -b "$BRANCH" origin/"$BRANCH"; git pull --rebase 2>/dev/null || true
  else
    git clone "$REPO_URL" "$REPO_PATH"; cd "$REPO_PATH"; git checkout "$BRANCH" || git checkout -b "$BRANCH" origin/"$BRANCH"
  fi
fi
ok "Repo: $REPO_PATH"

cd "$REPO_PATH"
log "Clone parseq"
[[ -d third_party/parseq/strhub ]] || git clone https://github.com/baudm/parseq third_party/parseq

export DATA_ROOT="$DATA_ROOT"
export PYTHONPATH="$REPO_PATH/compat:$REPO_PATH/src:$REPO_PATH/third_party/parseq"

log "Cài đặt"
$PYTHON -m pip install -r requirements.txt >/dev/null 2>&1 || true
$PYTHON -m pip install -e . >/dev/null 2>&1 || true

mkdir -p "$DATA_ROOT"/{raw/{meta,images_us,images_vn},processed,lmdb/{train/trademark,val,test/{test_us,test_vn}},runs,reports}

log "Tải dữ liệu USPTO"
if [[ -z "${SKIP_FETCH:-}" ]]; then
  [[ -f "$DATA_ROOT/raw/meta/us_marks.csv" ]] || bash scripts/01_get_meta.sh
  [[ -n "$(ls -A $DATA_ROOT/raw/images_us 2>/dev/null)" ]] || bash scripts/02_get_images.sh
else
  warn "skip-fetch"
fi

log "Clean & split"
if [[ ! -f "$DATA_ROOT/processed/clean.csv" ]]; then
  $PYTHON -m trademark_ocr.clean
fi
if [[ ! -f "$DATA_ROOT/processed/split_train.csv" ]]; then
  $PYTHON -m trademark_ocr.split
fi

log "Build LMDB"
if [[ -z "${SKIP_BUILD_LMDB:-}" ]] && [[ ! -f "$DATA_ROOT/lmdb/train/trademark/data.mdb" ]]; then
  $PYTHON -m trademark_ocr.build_lmdb
fi

log "Tải dữ liệu VN"
if [[ -z "${SKIP_FETCH:-}" ]] && [[ ! -f "$DATA_ROOT/raw/vn_marks.csv" ]]; then
  bash scripts/03_get_vn.sh --limit 500 || true
fi
[[ -f "$DATA_ROOT/lmdb/test/test_vn/data.mdb" ]] || $PYTHON -m trademark_ocr.build_lmdb --subset test_vn 2>/dev/null || true

log "Train"
if [[ -z "${SKIP_TRAIN:-}" ]]; then
  for m in "${MODELS[@]}"; do
    if ls "$DATA_ROOT/runs/$m"/*/checkpoints/last.ckpt >/dev/null 2>&1; then
      warn "Bỏ qua $m (đã có checkpoint)"
      continue
    fi
    log "Train $m"
    bash scripts/05_train.sh "$m" --accelerator "$ACCELERATOR" --devices "$DEVICES" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --max-epochs "$MAX_EPOCHS"
  done
else
  warn "skip-train"
fi

log "Eval"
if [[ -z "${SKIP_EVAL:-}" ]]; then
  bash scripts/06_eval.sh --device "$ACCELERATOR" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
fi

log "Report"
$PYTHON -m trademark_ocr.report --reports reports/ --out reports/final_summary.md 2>/dev/null || true

ok "HOÀN THÀNH"
echo "Repo: $REPO_PATH"
echo "Data: $DATA_ROOT"
echo "Báo cáo: $REPO_PATH/reports/final_summary.md"
