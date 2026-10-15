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

## 06-eda — notebook EDA

```bash
jupyter nbconvert --to notebook --execute notebooks/01_eda_us.ipynb --output /tmp/eda_out.ipynb
ls $DATA_ROOT/processed/eda_figures/*.png | head
```
**Pass khi**: notebook chạy hết cell không lỗi, sinh ≥ 4 figure.

## 07-train-crnn — baseline CRNN+CTC

*(Train thật chạy trên Colab — cell trong `docs/colab/02_train.ipynb`. Nghiệm thu local:)*

```bash
python -m trademark_ocr.train_utils --config tm_crnn --dry-run   # hydra compose OK, 1 batch forward CPU
bash scripts/05_train.sh crnn --limit-train-batches 2 --max-epochs 1 --accelerator cpu   # dry 2 batch
```
**Pass khi**: in ra summary mô hình (số tham số), loss giảm sau 2 batch (≥0),
file `runs/crnn/**/last.ckpt` được tạo (hoặc dry-run báo OK nếu `--dry-run`).

## 08-train-parseq — TRBA + PARSeq

```bash
python -m trademark_ocr.train_utils --config tm_trba  --dry-run
python -m trademark_ocr.train_utils --config tm_parseq --dry-run
```
**Pass khi**: 2 lệnh trên in `DRY-RUN OK` (compose config + instantiate model + 1 forward).

## 09-train-svtrv2 — SVTR-CTC

```bash
python -m trademark_ocr.train_utils --config tm_svtrctc --dry-run
pytest tests/test_svtr.py -q
```
**Pass khi**: test forward shape `(N, L, C)` đúng, dry-run OK.

## 10-eval-metrics — metrics + evaluate

```bash
pytest tests/test_metrics.py -q
python -m trademark_ocr.evaluate --help >/dev/null && echo CLI_OK
# (sau khi có ckpt trên Colab)
python -m trademark_ocr.evaluate --ckpt runs/crnn/.../last.ckpt --subset test_us --out reports/crnn_us.csv
```
**Pass khi**: `test_metrics.py` pass (Acc/1−NED/char-acc đúng ví dụ chuẩn),
`CLI_OK` in ra; chạy thật sinh CSV có cột `pred,gt,correct,ned`.

## 11-fetch-vn — nhãn hiệu Việt Nam

```bash
bash scripts/03_get_vn.sh --limit 50        # smoke 50 ảnh
python -m trademark_ocr.fetch_vn --verify
```
**Pass khi**: ≥95% ảnh mở được, `processed/test_vn.txt` ≥ 48/50 dòng, label NFC.

## 12-eval_domain_shift — bảng sụt giảm US→VN

```bash
python -m trademark_ocr.eval_domain_shift --reports reports/ --out reports/domain_shift.csv
```
**Pass khi**: `domain_shift.csv` có cột `model,acc_us,acc_vn,delta`
(hiệu số âm = sụt độ chính xác khi sang domain VN).

## 13-error-analysis — phân tích lỗi

```bash
jupyter nbconvert --to notebook --execute notebooks/02_error_analysis.ipynb --output /tmp/err_out.ipynb
ls $DATA_ROOT/processed/eda_figures/err_*.png | head
```
**Pass khi**: notebook chạy hết, sinh figure lỗi theo độ dài + top-N lỗi mẫu.

## 14-demo-report — demo + báo cáo

```bash
python -m trademark_ocr.demo_app --help >/dev/null && echo CLI_OK
python -m trademark_ocr.demo_app --image demo_images/*.png --ckpt <ckpt>   # khi có ckpt
python -m trademark_ocr.report --reports reports/ --out reports/final_summary.md
```
**Pass khi**: `CLI_OK`; `reports/final_summary.md` sinh ra (có bảng tổng hợp model × dataset).

---

## Ghi chú chung
- Nếu máy không có lệnh `python` (chỉ có `python3`): thay `python` → `python3` trong
  các lệnh dưới (script trong `scripts/` đã tự dò `python3`/`python`).
- `<ckpt>`: đường dẫn checkpoint trong `$DATA_ROOT/runs/...` (Colab tải về Drive).
- Nếu thiếu data thật ở bước smoke → chạy `scripts/01..04` trước,
  hoặc dùng data giả đã tạo trong `tests/fixtures` (pytest tự tạo).
- Lệnh Colab chi tiết: `docs/colab/00_setup.ipynb` → `03_eval.ipynb`.
