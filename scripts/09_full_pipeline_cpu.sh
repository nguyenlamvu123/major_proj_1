#!/usr/bin/env bash
# =====================================================================
# Chạy TOÀN BỘ đồ án trên một máy, từ đầu tới cuối:
#   setup → us_meta → us_images → clean → split → lmdb → vn → vn_lmdb → train → eval
#
#   bash scripts/09_full_pipeline_cpu.sh --dry-run          # chỉ in kế hoạch, không chạy gì
#   bash scripts/09_full_pipeline_cpu.sh                    # chạy thật (log: $DATA_ROOT/logs/)
#   bash scripts/09_full_pipeline_cpu.sh --from train --models crnn_ft --max-hours 6
#
# Chạy lại an toàn: mỗi bước dữ liệu ghi một dấu (stamp) vào $DATA_ROOT/.pipeline/ khi xong;
# bước nào đã có dấu, cùng tham số, và mọi bước nó phụ thuộc không mới hơn thì bỏ qua. Model
# train dở (có last.ckpt, chưa có DONE) thì train tiếp từ last.ckpt.
#
# Trạng thái một run (thư mục runs/<model>/<giờ>/), do script này ghi sau khi train:
#   DONE          train đủ epoch và học được (val_accuracy tốt nhất >= 1%)   -> bỏ qua
#   UNDERTRAINED  train đủ epoch mà val_accuracy vẫn ~0 (ngân sách quá ít)   -> bỏ qua, eval
#                 cũng bỏ; chỉ train lại khi có --retrain (thường kèm --max-hours lớn hơn)
#   (không có)    bị cắt vì trần thời gian hoặc lỗi giữa chừng               -> train tiếp
#
# Train THÊM epoch cho model đã DONE/UNDERTRAINED (vd crnn_ft 8 -> 20 epoch):
#   bash scripts/09_full_pipeline_cpu.sh --from train --models crnn_ft --continue --max-epochs 20
# --max-epochs là TỔNG số epoch muốn đạt (tính dồn từ run gốc). Tạo run MỚI, nạp toàn bộ trọng số
# từ last.ckpt của run cũ, lịch LR OneCycle mới cho phần còn thiếu (không thể nối lịch cũ: nó đã
# chạy hết và đã chuyển sang SWA). Chạy lại đúng lệnh đó: đủ tổng thì bỏ qua, run mới bị cắt thì
# train tiếp run mới. pipeline_train.env của run mới ghi continue_from / init_epoch / epoch_offset
# để trademark_ocr.train_curves nối đồ thị hai run (DESIGN.md 16.12).
#
# Tên giữ "_cpu" như bản đầu, nhưng --accelerator auto dùng GPU nếu máy có nvidia-smi.
# =====================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIG_ARGS=("$@")         # vòng parse bên dưới shift hết "$@"; cần để tự chạy lại dưới systemd-inhibit
STEPS=(setup us_meta us_images clean split lmdb vn vn_lmdb train eval)

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
VENV="$REPO_ROOT/.venv"
PYTHON_ARG=""
US_ROWS=60000            # dòng metadata US; lọc xong còn ~97% -> mục tiêu >= 50k mẫu sạch
VN_LIMIT=2000            # nhãn hiệu VN (có chữ + ảnh)
IMAGE_WORKERS=16
MODELS="crnn_ft,parseq_ft"
MAX_EPOCHS=10
MAX_HOURS=""             # trần thời gian MỖI model (Lightning max_time); rỗng = không giới hạn
BATCH_SIZE=64
NUM_WORKERS=4
ACCELERATOR=auto
FROM=""; ONLY=""; SKIP=""
RETRAIN=0; CONTINUE=0; DRY_RUN=0

usage() {
  cat <<'EOF'
Cờ:
  --data-root DIR      gốc dữ liệu (mặc định $DATA_ROOT hoặc <repo>/data)
  --venv DIR           môi trường Python tự dựng (mặc định <repo>/.venv)
  --python PATH        dùng Python có sẵn (bỏ bước setup)
  --us-rows N          số dòng metadata US (60000)
  --vn-limit N         số nhãn hiệu VN (2000)
  --image-workers N    luồng tải ảnh US (16)
  --models a,b         model cần train (crnn_ft,parseq_ft); tên như scripts/05_train.sh
  --max-epochs N       (10)        --batch-size N (64)     --num-workers N (4)
  --max-hours H        trần thời gian mỗi model, vd 4 hoặc 2.5 (không giới hạn)
  --accelerator X      auto | cpu | gpu (auto)
  --from STEP          bỏ qua các bước trước STEP
  --only a,b           chỉ chạy các bước này (BẮT BUỘC chạy, kể cả khi đã xong)
  --skip a,b           bỏ qua các bước này
  --retrain            train lại model kể cả khi đã có DONE
  --continue           train THÊM từ run đã DONE/UNDERTRAINED (nạp trọng số last.ckpt) tới tổng
                       --max-epochs epoch; run mới, lịch LR mới
  --dry-run            chỉ in kế hoạch
Bước: setup us_meta us_images clean split lmdb vn vn_lmdb train eval
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)     DATA_ROOT="$2"; shift 2 ;;
    --venv)          VENV="$2"; shift 2 ;;
    --python)        PYTHON_ARG="$2"; shift 2 ;;
    --us-rows)       US_ROWS="$2"; shift 2 ;;
    --vn-limit)      VN_LIMIT="$2"; shift 2 ;;
    --image-workers) IMAGE_WORKERS="$2"; shift 2 ;;
    --models)        MODELS="$2"; shift 2 ;;
    --max-epochs)    MAX_EPOCHS="$2"; shift 2 ;;
    --max-hours)     MAX_HOURS="$2"; shift 2 ;;
    --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
    --num-workers)   NUM_WORKERS="$2"; shift 2 ;;
    --accelerator)   ACCELERATOR="$2"; shift 2 ;;
    --from)          FROM="$2"; shift 2 ;;
    --only)          ONLY="$2"; shift 2 ;;
    --skip)          SKIP="$2"; shift 2 ;;
    --retrain)       RETRAIN=1; shift ;;
    --continue)      CONTINUE=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "[ERR] cờ lạ: $1" >&2; usage >&2; exit 2 ;;
  esac
done

die() { echo "[ERR] $*" >&2; exit 2; }
in_list() { [[ ",$2," == *",$1,"* ]]; }

# ---- kiểm tham số TRƯỚC khi làm bất cứ gì (dry-run cũng kiểm) ----
for s in ${FROM:+$FROM} ${ONLY//,/ } ${SKIP//,/ }; do
  in_list "$s" "$(IFS=,; echo "${STEPS[*]}")" || die "bước lạ: $s"
done
[[ -z "$FROM" || -z "$ONLY" ]] || die "--from và --only không dùng cùng nhau"
[[ $CONTINUE -eq 0 || $RETRAIN -eq 0 ]] || die "--continue (train thêm từ run cũ) và --retrain (train lại từ đầu) không dùng cùng nhau"
for m in ${MODELS//,/ }; do
  in_list "$m" "crnn,trba,parseq,svtr,svtrctc,crnn_ft,trba_ft,parseq_ft" || die "model lạ: $m"
done
for v in US_ROWS VN_LIMIT IMAGE_WORKERS MAX_EPOCHS BATCH_SIZE NUM_WORKERS; do
  [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v phải là số nguyên: ${!v}"
done
[[ -z "$MAX_HOURS" || "$MAX_HOURS" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--max-hours phải là số: $MAX_HOURS"
case "$ACCELERATOR" in
  auto) if command -v nvidia-smi >/dev/null 2>&1; then ACCELERATOR=gpu; else ACCELERATOR=cpu; fi ;;
  cpu|gpu) ;;
  *) die "--accelerator phải là auto|cpu|gpu" ;;
esac
DEVICE=$([[ "$ACCELERATOR" == gpu ]] && echo cuda || echo cpu)

if [[ -n "$PYTHON_ARG" ]]; then PY="$PYTHON_ARG"; else PY="$VENV/bin/python"; fi
STAMPS="$DATA_ROOT/.pipeline"
SETUP_STAMP="$VENV/.pipeline_setup"

# ---- chạy thật: chặn ngủ + ghi log TRƯỚC khi in kế hoạch (để kế hoạch có trong log) ----
if [[ $DRY_RUN -eq 0 ]]; then
  # Máy để bàn có thể tự ngủ giữa đêm: chạy lại chính script dưới khoá chặn ngủ nếu systemd cho phép.
  if [[ -z "${PIPELINE_INHIBITED:-}" ]] && command -v systemd-inhibit >/dev/null 2>&1 \
     && systemd-inhibit --what=sleep:idle --who=trademark_ocr --why=test true 2>/dev/null; then
    export PIPELINE_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle --who=trademark_ocr --why="pipeline đồ án đang chạy" \
      bash "$0" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}
  fi
  mkdir -p "$DATA_ROOT/logs" "$DATA_ROOT/.pipeline"
  LOG="$DATA_ROOT/logs/pipeline_$(date +%Y-%m-%d_%H-%M-%S).log"
  exec > >(tee -a "$LOG") 2>&1
  echo "[PIPELINE] $(date '+%Y-%m-%d %H:%M:%S') log -> $LOG"
  echo "[PIPELINE] lệnh: $0 ${ORIG_ARGS[*]:-}"
fi

# ---- chọn bước theo --from/--only/--skip ----
selected() {
  local step="$1" seen_from=0 s
  # --python = môi trường do người dùng tự lo: không bao giờ dựng venv, kể cả với --only setup
  [[ "$step" == setup && -n "$PYTHON_ARG" ]] && return 1
  if [[ -n "$ONLY" ]]; then in_list "$step" "$ONLY"; return; fi
  in_list "$step" "$SKIP" && return 1
  [[ -z "$FROM" ]] && return 0
  for s in "${STEPS[@]}"; do
    [[ "$s" == "$FROM" ]] && seen_from=1
    [[ "$s" == "$step" ]] && { [[ $seen_from -eq 1 ]]; return; }
  done
}

declare -A WILL_RUN=()
# Cần chạy lại khi: bị ép (--only), chưa có dấu, tham số đổi, hoặc một bước phụ thuộc mới
# hơn dấu / sẽ chạy lại trong lần này.
stale() {
  local step="$1" stamp="$2" params="$3"; shift 3
  local dep
  [[ -n "$ONLY" ]] && return 0
  [[ -f "$stamp" ]] || return 0
  [[ "$(cat "$stamp")" == "$params" ]] || return 0
  for dep in "$@"; do
    [[ -n "${WILL_RUN[$dep]:-}" ]] && return 0
    [[ -f "$STAMPS/$dep" && "$STAMPS/$dep" -nt "$stamp" ]] && return 0
  done
  return 1
}

plan() { printf '[plan] %-16s | %-6s | %s\n' "$1" "$2" "$3"; }

# ---- kế hoạch: quyết định từng bước chỉ dựa trên trạng thái file ----
PLAN=()
add() { PLAN+=("$1"); WILL_RUN[$1]=1; }

decide() {
  local step="$1" stamp="$2" params="$3" detail="$4"; shift 4
  if ! selected "$step"; then plan "$step" skip "không chọn"; return; fi
  if stale "$step" "$stamp" "$params" "$@"; then add "$step"; plan "$step" run "$detail"
  else plan "$step" skip "đã xong ($params)"; fi
}

if selected setup; then
  if [[ -n "$ONLY" || ! -x "$PY" || ! -f "$SETUP_STAMP" || "$REPO_ROOT/requirements.txt" -nt "$SETUP_STAMP" \
        || "$(cat "$SETUP_STAMP" 2>/dev/null)" != "accelerator=$ACCELERATOR" ]]; then
    add setup; plan setup run "venv $VENV + torch ($ACCELERATOR) + requirements + parseq ghim commit"
  else
    plan setup skip "đã xong (accelerator=$ACCELERATOR)"
  fi
else
  plan setup skip "$([[ -n "$PYTHON_ARG" ]] && echo "dùng --python $PYTHON_ARG" || echo "không chọn")"
fi
decide us_meta   "$STAMPS/us_meta"   "rows=$US_ROWS"  "01_get_meta.sh --limit $US_ROWS --resume"
decide us_images "$STAMPS/us_images" "-"              "02_get_images.sh --workers $IMAGE_WORKERS (tải tiếp ảnh còn thiếu)" us_meta
decide clean     "$STAMPS/clean"     "-"              "trademark_ocr.clean" us_images
decide split     "$STAMPS/split"     "seed=1111"      "trademark_ocr.split --seed 1111" clean
decide lmdb      "$STAMPS/lmdb"      "-"              "build_lmdb train, val, test_us" split
decide vn        "$STAMPS/vn"        "limit=$VN_LIMIT" "03_get_vn.sh --limit $VN_LIMIT"

VERIFIED="$DATA_ROOT/processed/split_test_vn_verified.csv"
if [[ -f "$VERIFIED" ]]; then VN_SPLIT="$VERIFIED"; VN_SRC="verified"; else VN_SPLIT=""; VN_SRC="unverified"; fi
if selected vn_lmdb; then
  if stale vn_lmdb "$STAMPS/vn_lmdb" "split=$VN_SRC" vn \
     || [[ "$VN_SRC" == verified && "$VERIFIED" -nt "$STAMPS/vn_lmdb" ]]; then
    add vn_lmdb
    plan vn_lmdb run "build_lmdb test_vn ($([[ $VN_SRC == verified ]] && echo "đã duyệt tay: split_test_vn_verified.csv" || echo "CHƯA duyệt tay: split_test_vn.csv"))"
  else
    plan vn_lmdb skip "đã xong (split=$VN_SRC)"
  fi
else
  plan vn_lmdb skip "không chọn"
fi

# train: mỗi model một quyết định, theo run mới nhất có last.ckpt trong $DATA_ROOT/runs/<name>/.
# Run do script này tạo có <run>/pipeline_train.env (max_epochs, batch_size): train tiếp PHẢI giữ
# đúng hai số đó, vì lịch LR OneCycle khôi phục từ checkpoint đã cố định tổng số bước.
model_dir() { case "$1" in svtr|svtrctc) echo svtrctc ;; *) echo "$1" ;; esac; }
model_config() { case "$1" in svtr|svtrctc) echo tm_svtrctc ;; *) echo "tm_$1" ;; esac; }
# Lớp system của từng model (giống model._target_/model.target trong configs/hydra/tm_*.yaml).
model_class() {
  case "$1" in
    crnn|crnn_ft)     echo strhub.models.crnn.system.CRNN ;;
    trba|trba_ft)     echo strhub.models.trba.system.TRBA ;;
    parseq|parseq_ft) echo strhub.models.parseq.system.PARSeq ;;
    svtr|svtrctc)     echo trademark_ocr.svtr_ctc.SVTRCTC ;;
  esac
}
latest_ckpt() { ls -dt "$DATA_ROOT/runs/$(model_dir "$1")"/*/checkpoints/last.ckpt 2>/dev/null | head -1 || true; }
# Thiếu file (run cũ) phải trả rỗng: để sed lỗi thì pipefail + set -e giết script không một lời.
env_val() { [[ -f "$1/pipeline_train.env" ]] || return 0; sed -n "s/^$2=//p" "$1/pipeline_train.env" | head -1; }
# Epoch (0-based) của last.ckpt, đọc từ TÊN file (dry-run không được gọi python): Lightning chỉ
# ghi last.ckpt cùng lúc với một checkpoint top-k "epoch=E-step=…ckpt", nên last.ckpt là bản
# top-k mới nhất = E lớn nhất còn trong checkpoints/ (bản top-k cũ hơn bị xoá khi rớt hạng).
ckpt_epoch() { ls "$1/checkpoints" 2>/dev/null | sed -n 's/^epoch=\([0-9][0-9]*\)-.*\.ckpt$/\1/p' | sort -n | tail -1 || true; }
declare -A TRAIN_MODE=() CONT_CK=() CONT_EPOCH=() CONT_TOTAL=()
for m in ${MODELS//,/ }; do
  if ! selected train; then plan "train:$m" skip "không chọn"; continue; fi
  ck="$(latest_ckpt "$m")"
  run_dir="${ck%/checkpoints/last.ckpt}"
  # --continue chỉ đổi cách xử lý run ĐÃ XONG (DONE/UNDERTRAINED). Run mới nhất bị cắt giữa chừng
  # (kể cả chính run --continue lần trước) rơi xuống nhánh "train tiếp" bên dưới như thường.
  if [[ $CONTINUE -eq 1 && -z "$ck" ]]; then
    plan "train:$m" skip "chưa có run nào có checkpoint để train tiếp — bỏ --continue để train từ đầu"
    continue
  fi
  if [[ $CONTINUE -eq 1 && ( -f "$run_dir/DONE" || -f "$run_dir/UNDERTRAINED" ) ]]; then
    e="$(ckpt_epoch "$run_dir")"
    if [[ -z "$e" ]]; then
      plan "train:$m" skip "không biết last.ckpt ở epoch nào (${run_dir#"$DATA_ROOT"/}/checkpoints/ không có file epoch=…ckpt)"
      continue
    fi
    # Tổng epoch đã train trên ĐƯỜNG TRỌNG SỐ này = epoch của các run trước (epoch_offset, nếu run
    # cũ cũng là run --continue) + epoch 0..e của run cũ. Model có BatchNorm (CRNN): Lightning
    # chạy thêm một lượt SWA ở epoch = max_epochs (chỉ tính lại BatchNorm, không train); lượt đó
    # lọt top-3 thì last.ckpt mang epoch = max_epochs — vẫn chỉ là max_epochs epoch đã train.
    off="$(env_val "$run_dir" epoch_offset)"; me="$(env_val "$run_dir" max_epochs)"
    n=$(( e + 1 )); [[ -n "$me" ]] && (( n > me )) && n=$me
    total=$(( ${off:-0} + n ))
    if (( total >= MAX_EPOCHS )); then
      plan "train:$m" skip "đã train $total epoch (${run_dir#"$DATA_ROOT"/}) >= --max-epochs $MAX_EPOCHS — tăng --max-epochs để train thêm"
      continue
    fi
    TRAIN_MODE[$m]=continue; CONT_CK[$m]="$ck"; CONT_EPOCH[$m]="$e"; CONT_TOTAL[$m]="$total"
    add "train:$m"
    if [[ -n "$MAX_HOURS" ]]; then more="ngân sách ${MAX_HOURS}h: đo tốc độ rồi chọn ≤ $((MAX_EPOCHS - total)) epoch"
    else more="thêm $((MAX_EPOCHS - total)) epoch"; fi
    plan "train:$m" continue "run MỚI nạp trọng số ${ck#"$DATA_ROOT"/} (đã train $total epoch), $more (tổng ≤ $MAX_EPOCHS), lịch LR mới, batch $BATCH_SIZE, $ACCELERATOR"
    continue
  fi
  if [[ -n "$ck" && -f "$run_dir/DONE" && $RETRAIN -eq 0 ]]; then
    plan "train:$m" skip "đã xong (${run_dir#"$DATA_ROOT"/})"
  elif [[ -n "$ck" && -f "$run_dir/UNDERTRAINED" && $RETRAIN -eq 0 ]]; then
    # Không tự train lại: cùng ngân sách thì lại ra đúng kết quả đó, mất thêm một đêm.
    plan "train:$m" skip "train xong nhưng CHƯA HỌC ĐƯỢC ($(head -1 "$run_dir/UNDERTRAINED"), ${run_dir#"$DATA_ROOT"/}) — thêm --continue để train thêm từ run này, hoặc --retrain để train lại từ đầu, kèm --max-hours lớn hơn"
  elif [[ -n "$ck" && $RETRAIN -eq 0 ]]; then
    TRAIN_MODE[$m]="$ck"; add "train:$m"
    e="$(env_val "$run_dir" max_epochs)"; b="$(env_val "$run_dir" batch_size)"
    plan "train:$m" resume "tiếp từ ${ck#"$DATA_ROOT"/}, $([[ -n "$e" ]] && echo "giữ $e epoch, batch $b" || echo "run cũ không có pipeline_train.env: dùng $MAX_EPOCHS epoch, batch $BATCH_SIZE")${MAX_HOURS:+, thêm tối đa ${MAX_HOURS}h}"
  else
    TRAIN_MODE[$m]=""; add "train:$m"
    if [[ -n "$MAX_HOURS" ]]; then
      plan "train:$m" run "từ đầu, ngân sách ${MAX_HOURS}h: đo tốc độ rồi chọn ≤ $MAX_EPOCHS epoch, batch $BATCH_SIZE, $ACCELERATOR"
    else
      plan "train:$m" run "từ đầu, $MAX_EPOCHS epoch, batch $BATCH_SIZE, $ACCELERATOR"
    fi
  fi
done
if selected eval; then add eval; plan eval run "06_eval.sh --device $DEVICE (mọi model có checkpoint) + báo cáo"
else plan eval skip "không chọn"; fi

[[ $DRY_RUN -eq 1 ]] && exit 0

# =====================================================================
# Chạy thật
# =====================================================================
export DATA_ROOT
export PYTHON="$PY"
# stdout của Python đi qua tee (pipe) thì bị gom theo khối: lần 04/11 log đứng im 13 phút ở
# us_meta dù vẫn đang tải. Tắt đệm để log hiện tiến độ ngay.
export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
stamp() { echo "$2" > "$STAMPS/$1"; }
step() { echo; echo "[PIPELINE] $(date '+%Y-%m-%d %H:%M:%S') ===== $* ====="; }
will() { [[ -n "${WILL_RUN[$1]:-}" ]]; }

if will setup; then
  step "setup: $VENV"
  [[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"
  "$PY" -m pip install -q --upgrade pip
  if [[ "$ACCELERATOR" == cpu ]]; then
    # bản CPU: tránh tải ~2.5 GB thư viện CUDA không dùng tới
    "$PY" -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision
  fi
  "$PY" -m pip install -q -r requirements.txt
  "$PY" -m pip install -q -e .
  echo "accelerator=$ACCELERATOR" > "$SETUP_STAMP"
fi
bash scripts/00_get_parseq.sh
"$PY" -c "import torch, trademark_ocr; print('[PIPELINE] torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

if will us_meta; then
  step "us_meta"
  bash scripts/01_get_meta.sh --limit "$US_ROWS" --resume
  stamp us_meta "rows=$US_ROWS"
fi
if will us_images; then
  step "us_images"
  bash scripts/02_get_images.sh --workers "$IMAGE_WORKERS"
  "$PY" -m trademark_ocr.fetch_images --verify
  stamp us_images "-"
fi
if will clean; then step "clean"; "$PY" -m trademark_ocr.clean; stamp clean "-"; fi
if will split; then step "split"; "$PY" -m trademark_ocr.split --seed 1111; stamp split "seed=1111"; fi
if will lmdb; then
  step "lmdb"
  for s in train val test_us; do "$PY" -m trademark_ocr.build_lmdb --subset "$s"; done
  for d in train/trademark val test/test_us; do
    "$PY" -c "import lmdb,sys;n=lmdb.open(sys.argv[1],readonly=True,lock=False).begin().get(b'num-samples');print('[PIPELINE]',sys.argv[1],'num-samples =',n);sys.exit(n is None)" \
      "$DATA_ROOT/lmdb/$d"
  done
  stamp lmdb "-"
fi
# Tập VN chỉ cần lúc eval: crawl WIPO lỗi (mạng, site đổi cấu trúc) KHÔNG được chặn train.
VN_OK=1
if will vn; then
  step "vn"
  if bash scripts/03_get_vn.sh --limit "$VN_LIMIT"; then stamp vn "limit=$VN_LIMIT"
  else echo "[PIPELINE] [WARN] crawl VN lỗi — bỏ qua, train vẫn chạy; eval sẽ chỉ có test_us"; VN_OK=0; fi
fi
if will vn_lmdb; then
  step "vn_lmdb ($VN_SRC)"
  if [[ $VN_OK -eq 1 ]] && "$PY" -m trademark_ocr.build_lmdb --subset test_vn ${VN_SPLIT:+--split-csv "$VN_SPLIT"}; then
    stamp vn_lmdb "split=$VN_SRC"
  else
    echo "[PIPELINE] [WARN] không build được LMDB test_vn — eval sẽ chỉ có test_us"; VN_OK=0
  fi
fi

json_get() { "$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
# Trần max_time của Lightning dạng DD:HH:MM:SS = đã train ($1 giây) + ngân sách lần này. Ngân
# sách nới 15% vì số epoch đã chọn để vừa 85% — trần chỉ cắt khi máy chậm bất thường.
CAP_ADD=0
[[ -z "$MAX_HOURS" ]] || CAP_ADD=$(awk -v h="$MAX_HOURS" 'BEGIN{printf "%d", h*3600*1.15}')
max_time_arg() {
  local s; s=$(awk -v d="$1" -v a="$CAP_ADD" 'BEGIN{printf "%d", d + a}')
  printf "+trainer.max_time='%02d:%02d:%02d:%02d'" $((s/86400)) $((s%86400/3600)) $((s%3600/60)) $((s%60))
}

# Lỗi train một model (vd OOM) không dừng cả đêm: ghi lại, train model kế, vẫn eval.
FAILED=(); PARTIAL=(); FEW_EPOCHS=()
for m in ${MODELS//,/ }; do
  will "train:$m" || continue
  step "train $m"
  mode="${TRAIN_MODE[$m]}"
  extra=()
  if [[ -n "$mode" && "$mode" != continue ]]; then
    ck="$mode"
    run="${ck%/checkpoints/last.ckpt}"
    epochs="$(env_val "$run" max_epochs)"; bs="$(env_val "$run" batch_size)"
    epochs="${epochs:-$MAX_EPOCHS}"; bs="${bs:-$BATCH_SIZE}"
    # Đường dẫn trong override Hydra luôn bọc "…": có dấu cách (vd "GenAI 2025") là vỡ grammar của Hydra.
    extra+=("ckpt_path=\"$ck\"")
    if [[ -n "$MAX_HOURS" ]]; then
      # Timer của Lightning cộng dồn thời gian đã train từ checkpoint -> trần = đã train + ngân sách mới
      done_s=0
      if "$PY" -m trademark_ocr.train_utils --ckpt-info "$ck" --json-out "$run/ckpt_info.json" >/dev/null; then
        done_s="$(json_get "$run/ckpt_info.json" train_seconds)"
      fi
      extra+=("$(max_time_arg "$done_s")")
    fi
  else
    run="$DATA_ROOT/runs/$(model_dir "$m")/$(date +%Y-%m-%d_%H-%M-%S)"
    mkdir -p "$run"
    # Run --continue: chỉ train phần còn thiếu tới tổng --max-epochs.
    want="$MAX_EPOCHS"
    [[ "$mode" == continue ]] && want=$(( MAX_EPOCHS - ${CONT_TOTAL[$m]} ))
    epochs="$want"; bs="$BATCH_SIZE"
    if [[ -n "$MAX_HOURS" ]]; then
      if "$PY" -m trademark_ocr.train_utils --config "$(model_config "$m")" --benchmark \
           --budget-hours "$MAX_HOURS" --max-epochs "$want" --json-out "$run/benchmark.json" \
           -o model.batch_size="$bs" -o data.num_workers="$NUM_WORKERS" -o trainer.accelerator="$ACCELERATOR"; then
        epochs="$(json_get "$run/benchmark.json" epochs)"
        echo "[PIPELINE] $m: $(json_get "$run/benchmark.json" epoch_minutes) phút/epoch -> train $epochs epoch (~$(json_get "$run/benchmark.json" estimated_hours)h)"
        if [[ "$(json_get "$run/benchmark.json" too_few_epochs)" == True ]]; then
          echo "[PIPELINE] [WARN] ngân sách ${MAX_HOURS}h chỉ đủ $epochs epoch cho $m — có thể chưa học được gì; tăng --max-hours hoặc train trên GPU"
          FEW_EPOCHS+=("$m($epochs epoch)")
        fi
      else
        echo "[PIPELINE] [WARN] đo tốc độ $m lỗi — giữ $epochs epoch, trần max_time vẫn cắt theo ngân sách"
      fi
      extra+=("$(max_time_arg 0)")
    fi
    printf 'max_epochs=%s\nbatch_size=%s\n' "$epochs" "$bs" > "$run/pipeline_train.env"
    if [[ "$mode" == continue ]]; then
      base_ck="${CONT_CK[$m]}"; base="${base_ck%/checkpoints/last.ckpt}"
      # continue_from tương đối $DATA_ROOT/runs: mang cả hai run sang máy khác vẫn nối được đồ thị.
      printf 'continue_from=%s\ninit_epoch=%s\nepoch_offset=%s\ninit_ckpt=%s\n' \
        "${base#"$DATA_ROOT/runs/"}" "${CONT_EPOCH[$m]}" "${CONT_TOTAL[$m]}" "$base_ck" >> "$run/pipeline_train.env"
      # Nạp trọng số qua factory trademark_ocr.pretrained.finetune (nạp mọi tensor trùng tên +
      # kích thước — cùng kiến trúc, cùng charset nên là toàn bộ). Config *_ft đã dùng factory, chỉ
      # đổi init_from; config train từ đầu thì chuyển _target_ sang factory, lớp thật vào model.target
      # (evaluate.py đọc model.target khi _target_ là factory).
      if [[ "$m" == *_ft ]]; then
        extra+=("model.init_from=\"$base_ck\"")
      else
        extra+=("model._target_=trademark_ocr.pretrained.finetune" "+model.target=$(model_class "$m")"
                "+model.init_from=\"$base_ck\"")
      fi
      echo "[PIPELINE] $m: train tiếp từ $base_ck (đã train ${CONT_TOTAL[$m]} epoch) thêm $epochs epoch"
    fi
  fi
  # hydra.run.dir cố định = run đã chọn: train tiếp ghi vào ĐÚNG thư mục cũ, không đẻ run rỗng.
  # Đường dẫn có khoảng trắng (vd "GenAI 2025") làm vỡ Hydra override grammar → quote.
  extra+=("hydra.run.dir=\"$run\"")
  t0=$(date +%s)
  if bash scripts/05_train.sh "$m" --accelerator "$ACCELERATOR" --devices 1 \
       --batch-size "$bs" --num-workers "$NUM_WORKERS" --max-epochs "$epochs" "${extra[@]}"; then
    # Thoát 0 cũng có thể là do chạm trần max_time. KHÔNG dựa vào epoch trong last.ckpt: Lightning
    # 2.6 chỉ ghi last.ckpt khi có lưu top-k (parseq save_top_k=3), nên nó có thể đứng ở epoch cũ dù
    # đã train đủ (tests/test_train_utils.py::test_last_ckpt_does_not_advance_without_a_topk_save).
    # Tiến trình chạy tới >= 97% trần (tính cả khởi động) = bị cắt; kết thúc sớm hơn = đủ epoch.
    took=$(( $(date +%s) - t0 ))
    if [[ -n "$MAX_HOURS" ]] && (( took * 100 >= CAP_ADD * 97 )); then
      echo "[PIPELINE] [WARN] $m chạm trần thời gian sau $((took/60)) phút — chạy lại script để train tiếp"
      PARTIAL+=("$m")
    else
      echo "[PIPELINE] $m: xong $epochs epoch sau $((took/60)) phút"
      # Đủ epoch chưa chắc đã học được (04/11: parseq_ft 2 epoch, val_accuracy 0%). Không đọc
      # được kết luận thì coi như xong — giữ hành vi cũ thay vì chặn model đang tốt.
      verdict=unknown
      if "$PY" -m trademark_ocr.train_utils --train-verdict "$run" --json-out "$run/verdict.json" >/dev/null; then
        verdict="$(json_get "$run/verdict.json" verdict)"
      fi
      if [[ "$verdict" == undertrained ]]; then
        best="$(json_get "$run/verdict.json" best_val_accuracy)"
        echo "val_accuracy tốt nhất ${best}% sau $epochs epoch" > "$run/UNDERTRAINED"
        echo "[PIPELINE] [WARN] $m CHƯA HỌC ĐƯỢC: val_accuracy tốt nhất ${best}% — ghi UNDERTRAINED, eval sẽ bỏ qua"
      else
        touch "$run/DONE"
      fi
    fi
  else
    echo "[PIPELINE] [ERR] train $m lỗi — xem log; chạy lại script sẽ train tiếp từ last.ckpt (nếu đã có)"
    FAILED+=("$m")
  fi
done

if will eval; then
  step "eval + báo cáo"
  bash scripts/06_eval.sh --device "$DEVICE" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
fi

echo
echo "[PIPELINE] $(date '+%Y-%m-%d %H:%M:%S') HẾT. Báo cáo: $REPO_ROOT/reports/final_summary.md | log: $LOG"
[[ $VN_OK -eq 1 ]] || echo "[PIPELINE] [WARN] không có tập test VN lần này (xem log bước vn)"
if [[ "$VN_SRC" == unverified ]]; then
  echo "[PIPELINE] [WARN] tập VN CHƯA duyệt tay: chạy 'python -m trademark_ocr.vn_review make' / 'apply' rồi chạy lại script"
fi
[[ ${#PARTIAL[@]} -eq 0 ]] || echo "[PIPELINE] [WARN] model chưa đủ epoch: ${PARTIAL[*]}"
[[ ${#FEW_EPOCHS[@]} -eq 0 ]] || echo "[PIPELINE] [WARN] ngân sách chỉ đủ rất ít epoch: ${FEW_EPOCHS[*]}"
# Theo trạng thái file, không theo lần chạy này: model bị bỏ qua vì UNDERTRAINED từ lần trước
# cũng phải được nhắc, nếu không báo cáo thiếu model mà không ai biết vì sao.
for m in ${MODELS//,/ }; do
  ck="$(latest_ckpt "$m")"; rd="${ck%/checkpoints/last.ckpt}"
  if [[ -n "$ck" && -f "$rd/UNDERTRAINED" ]]; then
    echo "[PIPELINE] [WARN] $m: lần train mới nhất chưa học được ($(head -1 "$rd/UNDERTRAINED")) — báo cáo dùng run cũ hơn nếu có, không thì thiếu model này; chạy lại với --models $m --continue --max-epochs <tổng lớn hơn> (train thêm từ run này) hoặc --retrain --max-hours <lớn hơn>, hoặc train trên GPU"
  fi
done
if [[ ${#FAILED[@]} -gt 0 ]]; then echo "[PIPELINE] [ERR] model lỗi: ${FAILED[*]}"; exit 1; fi
