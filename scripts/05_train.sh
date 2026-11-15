#!/usr/bin/env bash
# =====================================================================
# Nhánh 07 — train trên tập nhãn hiệu qua `third_party/parseq/train.py`
# (KHÔNG sửa parseq; chỉ thêm config + shim + tiện ích).
#
#   bash scripts/05_train.sh crnn --limit-train-batches 2 --max-epochs 1 --accelerator cpu
#   bash scripts/05_train.sh parseq --max-epochs 20
#
# Mô hình: crnn | trba | parseq | svtr
# Checkpoint: $DATA_ROOT/runs/<model>/<timestamp>/*.ckpt
# =====================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
MODEL="${1:-crnn}"
if [ $# -gt 0 ]; then shift; fi

case "$MODEL" in
  crnn)      CONFIG="tm_crnn" ;;
  trba)      CONFIG="tm_trba" ;;
  parseq)    CONFIG="tm_parseq" ;;
  svtr)      CONFIG="tm_svtrctc" ;;
  tm_*)      CONFIG="$MODEL" ;;
  *) echo "unknown model '$MODEL' (mô hình: crnn | trba | parseq | svtr)" >&2; exit 1 ;;
esac

DATA_ROOT="${DATA_ROOT:-/media/zaibachkhoa/AC84-F995/trademark_data}"
export DATA_ROOT
# compat/ -> sitecustomize tự áp shim NumPy-2 cho imgaug (RandAugment)
export PYTHONPATH="$REPO_ROOT/compat:$REPO_ROOT/src:$REPO_ROOT/third_party/parseq${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -d "$REPO_ROOT/third_party/parseq/strhub" ]; then
  echo "thiếu third_party/parseq — hãy: git clone https://github.com/baudm/parseq $REPO_ROOT/third_party/parseq" >&2
  exit 1
fi
if [ ! -d "$DATA_ROOT/lmdb/train" ]; then
  echo "thiếu LMDB ở $DATA_ROOT/lmdb — hãy chạy nhánh 05 trước" >&2
  exit 1
fi

# Dịch flag kiểu argparse -> override hydra (của parseq)
overrides=()
while [ $# -gt 0 ]; do
  case "$1" in
    --limit-train-batches)  overrides+=("trainer.limit_train_batches=$2");  shift 2 ;;
    --limit-val-batches)    overrides+=("trainer.limit_val_batches=$2");    shift 2 ;;
    --max-epochs)           overrides+=("trainer.max_epochs=$2");           shift 2 ;;
    --max-steps)            overrides+=("trainer.max_steps=$2");            shift 2 ;;
    --accelerator)          overrides+=("trainer.accelerator=$2");          shift 2 ;;
    --devices)              overrides+=("trainer.devices=$2");              shift 2 ;;
    --batch-size)           overrides+=("model.batch_size=$2");             shift 2 ;;
    --lr)                   overrides+=("model.lr=$2");                     shift 2 ;;
    --num-workers)          overrides+=("data.num_workers=$2");             shift 2 ;;
    -*)                     echo "flag lạ: $1" >&2; exit 1 ;;
    *)                      overrides+=("$1");                              shift ;;
  esac
done

echo "[05_train] model=$MODEL config=$CONFIG overrides: ${overrides[*]:-<none>}"
echo "[05_train] PYTHON=$PYTHON  DATA_ROOT=$DATA_ROOT"
echo "[05_train] checkpoint -> $DATA_ROOT/runs/${CONFIG#tm_}/<timestamp>/"

exec "$PYTHON" "$REPO_ROOT/third_party/parseq/train.py" \
  --config-dir "$REPO_ROOT/configs/hydra" \
  --config-name "$CONFIG" \
  ${overrides[@]+"${overrides[@]}"}
