# ACCEPTANCE.md — Lệnh nghiệm thu 15 nhánh

> **Cách dùng**: sau khi merge nhánh NN vào `main` (hoặc đang ở nhánh đó),
> chạy đúng lệnh ở mục **NN**. Nhánh *pass* khi lệnh chạy không lỗi
> và điều kiện in ra thỏa mãn.
> Mọi lệnh chạy được trên **CPU, máy local** (không cần GPU), trừ khi ghi rõ.

**Cài đặt 1 lần (nhánh 00):**

```bash
cd /media/zaibachkhoa/education/GenAI\ 2025/đồ\ án\ chuyên\ ngành\ 1
export DATA_ROOT=/media/zaibachkhoa/AC84-F995/trademark_data
pip install -r requirements.txt
pip install -e .
```

---

## 00-init — scaffold

```bash
pytest -q                      # ≥1 test pass (test_smoke)
python -c "import trademark_ocr, strhub; print(trademark_ocr.__version__)"
test -f DESIGN.md && test -f ACCEPTANCE.md && echo DOCS_OK
```
**Pass khi**: `pytest` xanh, in ra `DOCS_OK`.

## 01-charset-vi — bảng ký tự tiếng Việt NFC

```bash
python -m trademark_ocr.charset --check          # thống kê + xác nhận unique/NFC
python -m trademark_ocr.charset --emit-yaml      # -> configs/hydra/charset/charset_vi.yaml
python -m trademark_ocr.charset --emit-txt       # -> configs/charset_vi.txt
pytest tests/test_charset.py -q
```
**Pass khi**: test xanh; `--check` in `charset OK (unique + NFC)` với
`n_classes_ctc = 218`; file `configs/hydra/charset/charset_vi.yaml` tồn tại và
có `charset_train == charset_test` (217 ký tự).

## 02-fetch-uspto-meta — metadata wordmark US

```bash
bash scripts/01_get_meta.sh --limit 5000        # smoke: 5k dòng (~40s, cần mạng)
wc -l $DATA_ROOT/raw/meta/us_marks.csv          # ≥ 5000 (khi chạy đủ: 50k–150k)
head -3 $DATA_ROOT/raw/meta/us_marks.csv
```
**Pass khi**: CSV có đủ cột `serial_number,word_mark,image_status`(tên cột chi tiết
xem DESIGN.md nhánh 02), không có dòng rỗng label.
> `wc -l` gồm 1 dòng header nên in ra `5001`. Nguồn: USPTO TMOG (EOG REST API), không
> cần API key; thống kê phụ ở `$DATA_ROOT/raw/meta/us_marks.csv.stats.json`.

## 03-fetch-uspto-images — ảnh USPTO

```bash
bash scripts/02_get_images.sh --limit 200       # smoke 200 ảnh (~20s)
python -m trademark_ocr.fetch_images --verify --limit 200
```
**Pass khi**: ≥ 95% ảnh trong index mở được bằng PIL; chạy lại lệnh → **không tải lại**
(resume, `downloaded: 0 new`).
> Ảnh lưu ở `$DATA_ROOT/raw/images_us/{serial}.png`, sổ đăng ký `.../index.csv`.
> Nguồn: CDN công khai USPTO (không cần API key).

## 04-clean-label — lọc & chuẩn hoá

```bash
python -m trademark_ocr.clean --limit 5000
python -c "import json,os;r=json.load(open(os.environ['DATA_ROOT']+'/processed/clean_report.json'));print(r['keep_ratio'])"
```
**Pass khi**: `clean_report.json` tồn tại, in ra `keep_ratio` (0<x≤1),
report liệt kê lý do rớt (no_text / too_long / unsupported_char / missing_image).
> Lọc theo thứ tự: no_text → too_short → too_long → unsupported_char → missing_image →
> bad_image → duplicate_label; report luôn có đủ 7 khoá. Output `processed/clean.csv`
> (`image_path` tương đối so với `$DATA_ROOT`).

## 05-split-build_data — split + LMDB

```bash
python -m trademark_ocr.split --seed 1111
python -m trademark_ocr.build_lmdb --subset train --limit 500
python -m trademark_ocr.build_lmdb --subset val   --limit 200
python -m trademark_ocr.build_lmdb --subset test_us --limit 200
pytest tests/test_lmdb.py -q
```
**Pass khi**: test LMDB pass (đọc được `num-samples`, label NFC, ảnh decode được),
không trùng `serial`/label giữa train và val/test (in ra số 0 trùng).
> LMDB ở `$DATA_ROOT/lmdb/{train/<name>, val, test/test_us}` đúng layout parseq
> (`root_dir=$DATA_ROOT/lmdb`, `train_dir=trademark`). Khi chạy tay mọi thứ dùng `strhub`
> cần `export PYTHONPATH=$PWD/third_party/parseq`.

## 06-eda — notebook EDA

```bash
jupyter nbconvert --to notebook --execute notebooks/01_eda_us.ipynb --output /tmp/eda_out.ipynb
ls $DATA_ROOT/processed/eda_figures/*.png | head
```
**Pass khi**: notebook chạy hết cell không lỗi, sinh ≥ 4 figure.
> Notebook dùng `DATA_ROOT` (env). Sinh 5 hình: `fig_len_hist, fig_drawing_kind, fig_nwords,
> fig_img_size, fig_char_freq`. Đã chạy: 4840 dòng, 5 hình, ~10 s. Notebook đã commit ở dạng
> **đã execute** (không lỗi).

## 07-train-crnn — baseline CRNN+CTC

*(Train thật chạy trên Colab — cell trong `docs/colab/02_train.ipynb`. Nghiệm thu local:)*

```bash
python -m trademark_ocr.train_utils --config tm_crnn --dry-run   # hydra compose OK, 1 batch forward CPU
bash scripts/05_train.sh crnn --limit-train-batches 2 --max-epochs 1 --accelerator cpu   # dry 2 batch
pytest tests/test_train_utils.py -q
```
**Pass khi**: in ra summary mô hình (số tham số), loss giảm sau 2 batch (≥0),
file `runs/crnn/**/last.ckpt` được tạo (hoặc dry-run báo OK nếu `--dry-run`).

> **Đã nghiệm thu**: dry-run `tm_crnn` = **8.424.026** tham số, loss 17.37, grad OK;
> `tm_trba` = 49.883.236 (loss 5.39); `tm_parseq` = 23.927.258 (loss 5.45). Smoke train
> 2 batch: TB ghi `loss 20.29 → 19.50 → 19.25` và tạo
> `$DATA_ROOT/runs/crnn/<ts>/checkpoints/last.ckpt`. 7 test pass.
>
> - Config ở `configs/hydra/tm_*.yaml` chạy qua entrypoint của parseq
>   (`third_party/parseq/train.py --config-dir configs/hydra --config-name ...`) → **không sửa** parseq.
> - `augment=true` cần shim NumPy-2 (`np.sctypes` đã bị xóa) → `scripts/05_train.sh` tự đặt
>   `PYTHONPATH=.../compat` để `sitecustomize.py` chạy; chạy tay `parseq/train.py` phải tự set `PYTHONPATH`.
> - Dry-run phải ở `model.eval()` (TRBA `forward` chỉ hợp lệ khi eval — xem DESIGN mục 07).

## 08-train-parseq — TRBA + PARSeq

```bash
python -m trademark_ocr.train_utils --config tm_trba  --dry-run
python -m trademark_ocr.train_utils --config tm_parseq --dry-run
pytest tests/test_forward.py -q
```
**Pass khi**: 2 lệnh trên in `[dry-run] OK` (compose config + instantiate model + 1 forward),
`test_forward.py` pass (shape `(N,L,C)`, loss ≥ 0, ký tự decode nằm trong charset).

> **Đã nghiệm thu**: `tm_trba` = 49.883.236 tham số (loss 5.39), `tm_parseq` = 23.927.258
> (loss 5.45); 11 test forward pass. Smoke train 2 batch CPU **exit 0** cho cả 3 model
> → `runs/{crnn,trba,parseq}/<ts>/checkpoints/last.ckpt`.
>
> ⚠️ Local (7.6 GB RAM): smoke phải kèm `--batch-size 8 --num-workers 0`; **batch 64 bị
> OOM-kill (exit 137)**. Colab batch 64 OK (nếu OOM → hạ `model.batch_size`).

## 09-train-svtrv2 — SVTR-CTC

```bash
python -m trademark_ocr.train_utils --config tm_svtrctc --dry-run
pytest tests/test_svtr.py -q
```
**Pass khi**: test forward shape `(N, L, C)` đúng, dry-run OK.

> **Đã nghiệm thu**: `SVTRCTC` = **2.330.042** tham số, `seq_len=64 ≥ 25`, dry-run loss 35.25,
> `[dry-run] OK`; 9 test pass (gồm test overfit: loss giảm >30%). Smoke train 2 batch CPU
> **exit 0** → `runs/svtrctc/<ts>/checkpoints/last.ckpt`.
>
> - Đây là bản **tự viết** (SVTR-Lite) đóng vai `CTCSystem` → dùng chung metric/decode của parseq.
> - `SVTRCTC` phải tự có `training_step` (CTCSystem không có) — nếu thiếu, Lightning báo
>   "No `training_step()` method defined".

## 10-eval-metrics — metrics + evaluate

```bash
pytest tests/test_metrics.py -q
python -m trademark_ocr.evaluate --help >/dev/null && echo CLI_OK
# (sau khi có ckpt trên Colab)
python -m trademark_ocr.evaluate --ckpt runs/crnn/.../last.ckpt --subset test_us --out reports/crnn_us.csv
```
**Pass khi**: `test_metrics.py` pass (Acc/1−NED/char-acc đúng ví dụ chuẩn),
`CLI_OK` in ra; chạy thật sinh CSV có cột `pred,gt,correct,ned`.

> **Đã nghiệm thu**: 19 test pass (gồm ví dụ chuẩn `kitten→sitting=3`, `ĐÔNG Á→DONG A=3`,
> và chạy thật trên `lmdb/val`). `CLI_OK` in ra. Chạy với ckpt crnn (mới 2 batch) →
> `n=200 acc=0.0000 1-NED=0.0233 char_acc=-0.1266 CER=1.1266`, CSV có
> `idx,pred,gt,correct,ned,char_err,pred_len,gt_len`.
> - Không cần `python-Levenshtein` (edit distance viết tay).
> - Đặt tên báo cáo `reports/<model>_<domain>.csv` để nhánh 12 ghép cặp US↔VN.
> - `char_accuracy` có thể âm với model chưa train (CER > 1) — đúng về toán.

## 11-fetch-vn — nhãn hiệu Việt Nam

```bash
bash scripts/03_get_vn.sh --limit 50        # smoke 50 ảnh
python -m trademark_ocr.fetch_vn --verify
```
**Pass khi**: ≥95% ảnh mở được, `processed/test_vn.txt` ≥ 48/50 dòng, label NFC.

> **Đã nghiệm thu**: `bash scripts/03_get_vn.sh --limit 50` → 50 nhãn, `--verify` báo
> **50/50 ảnh mở được (100%)**, `all NFC: True`. `build_lmdb --subset test_vn` → 50 mẫu.
> - Nguồn là WIPO Publish (Wicket) → `verify=False`; form id **đổi theo phiên** nên được
>   parse tự động (không hard-code).
> - `--limit N` lấy đúng N nhãn *có chữ + có ảnh* (bỏ qua nhãn chỉ có hình).
> - `processed/test_vn.txt` mỗi dòng 1 nhãn (dấu tiếng Việt giữ nguyên, NFC).

## 12-eval_domain_shift — bảng sụt giảm US→VN

```bash
python -m trademark_ocr.eval_domain_shift --reports reports/ --out reports/domain_shift.csv
```
**Pass khi**: `domain_shift.csv` có cột `model,acc_us,acc_vn,delta`
(hiệu số âm = sụt độ chính xác khi sang domain VN).

> **Đã nghiệm thu**: 7 test pass. Chạy thật với `reports/crnn_us.csv` + `reports/crnn_vn.csv`
> → `crnn,0.0,0.0,0.0,200,36`. Cột gồm cả `n_us,n_vn`; bảng **sort theo `delta` tăng dần**.
> - Model chỉ có 1 domain bị bỏ qua (không sinh dòng rác).
> - `n_vn=36` (không phải 50) vì nhãn VN dài/có dấu bị `max_label_length=25` lọc — hợp lệ.

## 13-error-analysis — phân tích lỗi

```bash
jupyter nbconvert --to notebook --execute notebooks/02_error_analysis.ipynb --output /tmp/err_out.ipynb
ls $DATA_ROOT/processed/eda_figures/err_*.png | head
```
**Pass khi**: notebook chạy hết, sinh figure lỗi theo độ dài + top-N lỗi mẫu.

> **Đã nghiệm thu**: 7 test pass; notebook đã execute & commit kèm output (9 cell, **0 lỗi**),
> sinh 3 hình `err_by_length.png`, `err_top_samples.png`, `err_char_confusions.png`
> (`acc=0.0, CER=1.12` trên 236 mẫu US+VN với ckpt chưa train).
> - Không cần GPU; đọc trực tiếp `reports/<model>_<domain>.csv`.
> - Nếu `reports/` rỗng, notebook vẫn chạy hết và in cảnh báo (không crash).

## 14-demo-report — demo + báo cáo

```bash
python -m trademark_ocr.demo_app --help >/dev/null && echo CLI_OK
python -m trademark_ocr.demo_app --image demo_images/*.png --ckpt <ckpt>   # khi có ckpt
python -m trademark_ocr.report --reports reports/ --out reports/final_summary.md
```
**Pass khi**: `CLI_OK`; `reports/final_summary.md` sinh ra (có bảng tổng hợp model × dataset).

> **Đã nghiệm thu**: 8 test pass (stub model, offline). `--help` → `CLI_OK`;
> `--ckpt <parseq> --image demo_images/*.png` chạy OK (ckpt chưa train → chuỗi rác, đúng dự kiến);
> `report` → `final_summary.md` có bảng `model × dataset` + bảng US→VN + danh sách hình.
> - `demo_images/` đã commit 2 ảnh mẫu (`demo_us.png`, `demo_vn.png`).
> - Demo không cần GPU (`--device cpu` mặc định); `--serve` mở Gradio nếu muốn.
> - `bash scripts/06_eval.sh` chạy end-to-end (evaluate US+VN → domain shift → phân tích lỗi → báo cáo).
> - Chỉ số trong `final_summary.md` hiện là của checkpoint **chưa train**; train trên Colab mới có ý nghĩa.

## 15-review-tests — review + test khoá phần đã chạy đúng

```bash
test -f REVIEW.md && echo REVIEW_OK
pytest -q tests/test_pipeline_e2e.py tests/test_parseq_contracts.py
pytest -q
```
**Pass khi**: in `REVIEW_OK`; hai file test mới xanh; toàn bộ `pytest -q` xanh.

> **Đã nghiệm thu**: `pytest -q` → 177 pass, 1 skip (test cần LMDB thật), trên venv
> torch 2.14.1+cpu / PL 2.6.6 / timm 1.0.30. Không cần mạng, không cần `DATA_ROOT`.

## 16-fix-review — xử lý các vấn đề trong REVIEW.md

### 16.1 LMDB ≥ 1.000 mẫu

```bash
pytest -q tests/test_lmdb.py
python -m trademark_ocr.build_lmdb --subset train          # toàn bộ split, không --limit
python -c "import lmdb,os;e=lmdb.open(os.environ['DATA_ROOT']+'/lmdb/train/trademark',readonly=True,lock=False);print(e.begin().get(b'num-samples'))"
```
**Pass khi**: test xanh; `num-samples` in ra đúng bằng số dòng `split_train.csv` (không phải `None`).

### 16.2 SVTR-CTC: trục thời gian theo chiều ngang

```bash
pytest -q tests/test_svtr.py
python -m trademark_ocr.train_utils --config tm_svtrctc --dry-run
```
**Pass khi**: test xanh (`seq_len == 32` @ 32×128, 32×64 bị từ chối); dry-run in `[dry-run] OK`.
> Thay cho số liệu nhánh 09 (`seq_len=64`). Checkpoint SVTR cũ phải train lại.

### 16.3 Đánh giá trung thực

```bash
pytest -q tests/test_evaluate.py tests/test_metrics.py
python -m trademark_ocr.build_lmdb --subset test_vn          # build lại để có khoá meta-*
python -m trademark_ocr.evaluate --ckpt $DATA_ROOT/runs/<model>/<ts>/checkpoints/last.ckpt \
    --subset test_vn --out reports/<model>_vn.csv
python -c "import json;s=json.load(open('reports/<model>_vn.summary.json'));print(s['n'],s['coverage'],s['accuracy'],s['accuracy_nocase'],s['accuracy_base'])"
```
**Pass khi**: test xanh; số dòng CSV = `num-samples` của LMDB (không bị lọc); cột `gt` giữ
nguyên nhãn gốc; summary có `coverage`, `accuracy_nocase`, `accuracy_base`.
> LMDB build trước nhánh 16 vẫn chấm được nhưng cột `serial_number`/`drawing_kind` rỗng.

### 16.4 Phân tách độ sụt US→VN + báo cáo theo loại nhãn hiệu

```bash
pytest -q tests/test_domain_shift.py tests/test_demo_report.py
python -m trademark_ocr.eval_domain_shift --reports reports/ --out reports/domain_shift.csv
python -m trademark_ocr.report --reports reports/ --out reports/final_summary.md
```
**Pass khi**: test xanh; `domain_shift.csv` có thêm `delta_nocase`, `delta_base`,
`delta_plain_nocase` và cột `_lo/_hi`; `final_summary.md` có các mục "Ba mức so khớp",
"Phân tách nguồn gốc độ sụt", "Theo loại nhãn hiệu".

### 16.5 Chất lượng tập VN

```bash
pytest -q tests/test_vn.py tests/test_vn_review.py
bash scripts/03_get_vn.sh --limit 2000                 # (cần mạng) thêm --registered-only nếu muốn
python -c "import json,os;r=json.load(open(os.environ['DATA_ROOT']+'/raw/vn_report.json'));print(r['n'],r['n_registered'],r['skipped'])"
python -m trademark_ocr.vn_review make --n 300         # rồi sửa processed/review_vn.csv bằng tay
python -m trademark_ocr.vn_review apply
python -m trademark_ocr.build_lmdb --subset test_vn --split-csv $DATA_ROOT/processed/split_test_vn_verified.csv
```
**Pass khi**: test xanh; `vn_report.json` có `skipped` (lý do bỏ) và `n_registered`;
`review_vn_report.json` có `label_noise_rate` — con số này phải nêu trong báo cáo đồ án.

### 16.6 Fine-tune từ trọng số pretrained

```bash
pytest -q tests/test_pretrained.py
python -m trademark_ocr.train_utils --config tm_parseq_ft --dry-run -o model.init_from=null
bash scripts/05_train.sh parseq_ft --limit-train-batches 2 --max-epochs 1 --accelerator cpu --batch-size 8 --num-workers 0
```
**Pass khi**: test xanh; log train có dòng
`[pretrained] parseq: nạp 172 tensor, bỏ 3 lệch kích thước [...]`; tạo `runs/parseq_ft/<ts>/checkpoints/last.ckpt`.
> Lần đầu cần mạng (~95 MB parseq, ~33 MB crnn, ~200 MB trba vào cache `torch.hub`).

---

## Ghi chú chung
- Nếu máy không có lệnh `python` (chỉ có `python3`): thay `python` → `python3` trong
  các lệnh dưới (script trong `scripts/` đã tự dò `python3`/`python`).
- `<ckpt>`: đường dẫn checkpoint trong `$DATA_ROOT/runs/...` (Colab tải về Drive).
- Nếu thiếu data thật ở bước smoke → chạy `scripts/01..04` trước,
  hoặc dùng data giả đã tạo trong `tests/fixtures` (pytest tự tạo).
- Lệnh Colab chi tiết: `docs/colab/00_setup.ipynb` → `03_eval.ipynb`.
