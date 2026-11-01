# Tổng kết: Dữ liệu train/test và quá trình huấn luyện

*Tài liệu được sinh tự động từ repo tại nhánh `16-fix-review` (HEAD `077a81b`) và dữ liệu thực tế trên máy.*

## 1. Nguồn dữ liệu (train / test)

### 1.1 Tập huấn luyện & đánh giá - Hoa Kỳ (USPTO)

Dữ liệu được thu thập từ **USPTO EOG/TMOG REST API** (không cần API key) theo pipeline các nhánh `02`, `03`.

| Thành phần | Vị trí (DATA_ROOT = `/media/zaibachkhoa/AC84-F995/trademark_data`) | Ghi chú |
|---|---|---|
| Metadata | `raw/meta/us_marks.csv` | ~6000 dòng (sau thu thập). Thống kê: `raw/meta/us_marks.csv.stats.json` |
| Ảnh gốc | `raw/images_us/` | Ảnh nhãn hiệu từ CDN USPTO (định dạng PNG/JPG) |
| Dữ liệu đã làm sạch | `processed/clean.csv` + `processed/clean_report.json` | **4840** mẫu sau làm sạch (quy trình nhánh 04). Báo cáo lý do loại bỏ: `clean_report.json` |
| Chia tập (theo nhóm nhãn, tránh leakage) | `processed/split_train.csv` (4356 dòng), `processed/split_val.csv` (242 dòng), `processed/split_test_us.csv` (242 dòng) | Nhánh 05. Chia theo nhóm nhãn (label-group split). |
| EDA | `processed/eda_figures/` (`fig_*.png`) | Nhánh 06 |

### 1.2 Tập đánh giá mới - Việt Nam (IP Vietnam / WIPO Publish)

Thu thập bằng scraper **WIPO Publish** (nhánh 11), đã được cải thiện ở nhánh `16-fix-review` (§16.5: crawl qua trang trống, dedupe, lọc `registered-only`, công cụ duyệt tay).

| Thành phần | Vị trí | Ghi chú |
|---|---|---|
| Metadata VN | `raw/vn_marks.csv` + `raw/vn_report.json` | Kết quả crawl từ WIPO Publish. Có fix ở nhánh 16.5: sửa mismatch cột `reg_no` / `registration_no` |
| Ảnh VN | `raw/images_vn/*.jpg` | Ảnh thumbnail từ WIPO Publish |
| Split test VN | `processed/split_test_vn.csv` (50 dòng), `processed/test_vn.txt` | **50 mẫu** hiện tại. Sau khi chuẩn bị đủ dữ liệu, cần tăng lên ~1–2k và duyệt tay bằng `vn_review` |

### 1.3 Định dạng LMDB dùng để huấn luyện/đánh giá

Toàn bộ dữ liệu đưa về định dạng **LMDB** theo layout của parseq (nhánh 05). Đã sửa bug quan trọng ở nhánh `16-fix-review` (16.1): quản lý transaction thủ công, commit đúng batch cuối + ghi `num-samples` (trước đây bị lỗi khi >=1000 mẫu).

| Tập | Đường dẫn LMDB | `num-samples` | Ghi chú |
|---|---|---|---|
| Train | `lmdb/train/trademark/` | `500` | *Lưu ý*: trên máy hiện tại chỉ có **500** mẫu trong LMDB train (trong khi `split_train.csv` có 4356 dòng). Nguyên nhân có thể do build LMDB giới hạn/test, hoặc build trước đó. Cần kiểm tra lại `build_lmdb`. |
| Val | `lmdb/val/` | `200` | So khớp gần đúng với `split_val.csv` (243) |
| Test US | `lmdb/test/test_us/` | `200` | Gần đúng với `split_test_us.csv` (243) |
| Test VN | `lmdb/test/test_vn/` | `50` | Đúng với `split_test_vn.csv` (50) |

> **Quan trọng:** Để train đầy đủ với toàn bộ `train/trademark`, cần build lại LMDB từ `split_train.csv` + ảnh tương ứng theo đúng pipeline (nhánh 05). Trên máy hiện tại, thực tế đang có **train LMDB 500 mẫu**.

## 2. Quá trình huấn luyện

### 2.1 Mô hình sử dụng

Repo tích hợp với **[baudm/parseq](https://github.com/baudm/parseq)** (clone vào `third_party/parseq`, không fork, không sửa). Trên nhánh `16-fix-review`, đã thêm các cấu hình và hỗ trợ fine-tune:

| Mô hình | Config | Kiểu | Ghi chú |
|---|---|---|---|
| **CRNN** | `configs/hydra/tm_crnn.yaml` | CTC baseline (ResNet + BiLSTM) | Train từ đầu |
| **TRBA** | `configs/hydra/tm_trba.yaml` | Attention-based (TPS + ResNet + BiLSTM + Attn) | Train từ đầu |
| **PARSeq** | `configs/hydra/tm_parseq.yaml` | Transformer (Vision Encoder + Permuted Autoregressive Sequence) | Train từ đầu |
| **SVTR-CTC** | `configs/hydra/tm_svtrctc.yaml` | SVTR (Vision Transformer cho STR) | Đã **sửa kiến trúc** ở nhánh 16.2: `merging stride (2,1)`, gộp chiều cao, positional embedding → **32 cột thời gian** thay vì 64 cột sai trước đó. **Checkpoint cũ không tương thích**, cần train lại. |
| **CRNN-FT / TRBA-FT / PARSeq-FT** | `tm_crnn_ft.yaml`, `tm_trba_ft.yaml`, `tm_parseq_ft.yaml` | Fine-tune từ **pretrained weights của PARSeq** (REBU-SynText/COCO-Text) | Thêm ở nhánh 16.6, khớp charset tiếng Việt (NFC, 218 class CTC/charset phù hợp). |

**Bảng ký tự:** Tiếng Việt Unicode NFC, cấu hình tại `configs/charset/charset_vi.yaml` và `src/trademark_ocr/charset.py`. Đã đảm bảo `charset_test == charset_train` và xử lý đúng 2 bẫy parseq: `normalize_unicode: false`, `remove_whitespace: false`.

### 2.2 Cách chạy huấn luyện

Sử dụng script chính: `scripts/05_train.sh`. Script này gọi `third_party/parseq/train.py` qua Hydra (`--config-dir configs/hydra --config-name ...`).

**Chuẩn bị môi trường:**

```bash
export DATA_ROOT=/media/zaibachkhoa/AC84-F995/trademark_data
export PYTHONPATH=$PWD/compat:$PWD/src:$PWD/third_party/parseq
```

**Ví dụ chạy (theo README/ACCEPTANCE):**

```bash
# Train từ đầu
bash scripts/05_train.sh crnn
bash scripts/05_train.sh trba
bash scripts/05_train.sh parseq
bash scripts/05_train.sh svtr   # hoặc svtrctc

# Fine-tune từ pretrained
bash scripts/05_train.sh parseq_ft
bash scripts/05_train.sh crnn_ft
bash scripts/05_train.sh trba_ft

# Smoke test trên CPU (không đủ cho train thật)
bash scripts/05_train.sh crnn --limit-train-batches 2 --max-epochs 1 --accelerator cpu
```

Checkpoint được lưu tại: `$DATA_ROOT/runs/<model>/<YYYY-MM-DD_HH-MM-SS>/checkpoints/last.ckpt` (kèm `config/config.yaml`).

### 2.3 Kết quả huấn luyện hiện tại

Trên máy hiện đang có các checkpoint từ **các lần chạy trước** (ngày 2026-10-16):

| Model | Thư mục run | Checkpoint | Trạng thái |
|---|---|---|---|
| **crnn** | `runs/crnn/2026-10-16_00-43-02/`, `runs/crnn/2026-10-16_00-51-24/` | `checkpoints/last.ckpt` | Đã có (2 phiên) |
| **trba** | `runs/trba/2026-10-16_01-04-44/` | `checkpoints/last.ckpt` | Đã có |
| **parseq** | `runs/parseq/2026-10-16_01-06-47/` | `checkpoints/last.ckpt` | Đã có |
| **svtrctc** | `runs/svtrctc/2026-10-16_01-14-21/` | `checkpoints/last.ckpt` | Đã có (kiến trúc **cũ** trước khi sửa ở nhánh 16.2) |

**Lưu ý quan trọng về kết quả hiện tại:**
- Các checkpoint này được huấn luyện với phiên bản code **trước nhánh `16-fix-review`**. Sau khi sửa SVTR (16.2), **kiến trúc SVTR-CTC đã thay đổi hoàn toàn**, vì vậy checkpoint `svtrctc` cũ **không thể nạp lại** nếu dùng code nhánh 16-fix-review.
- Do LMDB train hiện tại chỉ có **500 mẫu**, các mô hình này chỉ được huấn luyện với **quy mô rất nhỏ** (chứ không phải toàn bộ tập train).
- Hiện tại **chưa có checkpoint fine-tune** (`crnn_ft`/`trba_ft`/`parseq_ft`) trên máy.

### 2.4 Đánh giá hiện tại

Sử dụng `scripts/06_eval.sh` (nhánh 14, đã mở rộng) hoặc trực tiếp `src/trademark_ocr/evaluate.py`. Nhánh `16-fix-review` đã sửa đánh giá để **trung thực hơn** (16.3): chấm **mọi mẫu**, giữ `gt` gốc, thêm `in_scope`/`coverage`, `idx` truy ngược được, và 3 mức đánh giá `strict` / `nocase` / `base` (bỏ dấu).

**Kết quả hiện tại (từ `reports/final_summary.md`):**

| Model | Dataset | n | accuracy (strict) | 1−NED | char-accuracy | Ghi chú |
|---|---|---|---tag|---|---|---|
| crnn | US (test_us) | 200 | 0.0000 | 0.0009 | -0.2315 | Trên checkpoint chưa train đủ / quy mô nhỏ |
| crnn | VN (test_vn) | 36 | 0.0000 | 0.0115 | -0.1906 | Chỉ 36/50 mẫu vào được do giới hạn đánh giá trước đây; nhánh 16 đã sửa để đánh giá toàn bộ |
| **Domain shift (US→VN)** | crnn | acc_us=0.0, acc_vn=0.0, delta=0.0000 | n_us=200, n_vn=36 | Phân tách độ sít US→VN theo hoa/thường/dấu/hình ảnh đã được bổ sung ở nhánh 16.4 (`eval_domain_shift` nâng cấp) |

> **Kết luận về kết quả:** Các con số trên **không phản ánh chất lượng huấn luyện thực tế**. Lý do chính: (1) huấn luyện chỉ với **500/4356** mẫu train, (2) các checkpoint là phiên chạy nhỏ/không đầy đủ, (3) tập VN hiện tại chỉ **50 mẫu** chưa được duyệt tay nhiều. Để có kết quả đúng theo đề tài, cần train với **toàn bộ tập dữ liệu** trên **GPU**.

## 3. Lệnh tổng kết nhanh

```bash
# 1) Thiết lập môi trường
cd "/media/zaibachkhoa/education/GenAI 2025/đồ án chuyên ngành 1"
export DATA_ROOT=/media/zaibachkhoa/AC84-F995/trademark_data
export PYTHONPATH=$PWD/compat:$PWD/src:$PWD/third_party/parseq

# 2) Đánh giá toàn bộ model có checkpoint (CPU)
bash scripts/06_eval.sh --device cpu --batch-size 8 --num-workers 0

# 3) Train (khuyến nghị trên Colab/GPU) - ví dụ
bash scripts/05_train.sh parseq_ft --batch-size 64 --max-epochs 20
# hoặc
bash scripts/05_train.sh svtr --batch-size 64 --max-epochs 20

# 4) Demo suy luận
python -m trademark_ocr.demo_app --ckpt $DATA_ROOT/runs/parseq/<ts>/checkpoints/last.ckpt --image demo_images/*.png
```

**Tài liệu tham khảo:** `README.md`, `DESIGN.md` (mục 01–16), `ACCEPTANCE.md` (mục 01–16), `REVIEW.md`, `docs/colab/*.ipynb`.
