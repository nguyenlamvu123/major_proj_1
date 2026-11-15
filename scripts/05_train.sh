#!/usr/bin/env bash
# =====================================================================
# Nhánh 07 — train trên tập nhãn hiệu qua `third_party/parseq/train.py`
# (KHÔNG sửa parseq; chỉ thêm config + shim + tiện ích).
#
#   bash scripts/05_train.sh crnn --limit-train-batches 2 --max-epochs 1 --accelerator cpu
#   bash scripts/05_train.sh parseq --max-epochs 20
#
# Mô hình: crnn | trba | parseq | svtr (= svtrctc)  (train từ đầu)
#          crnn_ft | trba_ft | parseq_ft  (fine-tune từ trọng số pretrained của parseq, DESIGN 16.6)
# Checkpoint: $DATA_ROOT/runs/<model>/<timestamp>/*.ckpt   (hoặc hydra.run.dir=<thư mục>)
# LMDB: $DATA_ROOT/lmdb/{train,val}, hoặc data.root_dir=<thư mục chứa train/ và val/> — vd Colab
#   giải nén LMDB ra đĩa máy ảo, còn DATA_ROOT (checkpoint) vẫn trên Drive:
#   bash scripts/05_train.sh trba_ft --max-epochs 20 data.root_dir=/content/lmdb_root/lmdb
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
  crnn_ft|trba_ft|parseq_ft) CONFIG="tm_$MODEL" ;;
  svtr|svtrctc) CONFIG="tm_svtrctc" ;;
  tm_*)      CONFIG="$MODEL" ;;
  *) echo "unknown model '$MODEL' (mô hình: crnn | trba | parseq | svtr | crnn_ft | trba_ft | parseq_ft)" >&2; exit 1 ;;
esac

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
export DATA_ROOT
# compat/ -> sitecustomize tự áp shim NumPy-2 cho imgaug (RandAugment)
export PYTHONPATH="$REPO_ROOT/compat:$REPO_ROOT/src:$REPO_ROOT/third_party/parseq${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -d "$REPO_ROOT/third_party/parseq/strhub" ]; then
  echo "thiếu third_party/parseq — hãy chạy: bash scripts/00_get_parseq.sh" >&2
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

# Kiểm thư mục LMDB mà train.py SẼ đọc, không phải một đường dẫn cố định: override
# data.root_dir= thắng mặc định của config (${oc.env:DATA_ROOT}/lmdb). Bản cũ luôn kiểm
# $DATA_ROOT/lmdb/train nên chặn nhầm lần chạy Colab đúng (LMDB ở /content, DATA_ROOT trên Drive)
# và để lọt lần chạy trỏ data.root_dir= vào thư mục chưa giải nén (lỗi muộn, sau khi nạp model).
lmdb_root="$DATA_ROOT/lmdb"
run_dir="$DATA_ROOT/runs/${CONFIG#tm_}/<timestamp>"
for o in ${overrides[@]+"${overrides[@]}"}; do
  case "$o" in
    data.root_dir=*|+data.root_dir=*|++data.root_dir=*) lmdb_root="${o#*data.root_dir=}" ;;
    hydra.run.dir=*) run_dir="${o#hydra.run.dir=}" ;;
  esac
done
for sub in train val; do
  if [ ! -d "$lmdb_root/$sub" ]; then
    echo "thiếu LMDB ở $lmdb_root/$sub — build LMDB trước (python -m trademark_ocr.build_lmdb)," \
         "hoặc trỏ data.root_dir= tới thư mục chứa train/ và val/" >&2
    exit 1
  fi
done

echo "[05_train] model=$MODEL config=$CONFIG overrides: ${overrides[*]:-<none>}"
echo "[05_train] PYTHON=$PYTHON  DATA_ROOT=$DATA_ROOT  LMDB=$lmdb_root"
echo "[05_train] checkpoint -> $run_dir/"

exec "$PYTHON" "$REPO_ROOT/third_party/parseq/train.py" \
  --config-dir "$REPO_ROOT/configs/hydra" \
  --config-name "$CONFIG" \
  ${overrides[@]+"${overrides[@]}"}
