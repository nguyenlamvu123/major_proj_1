# Tổng kết: Dữ liệu train/test và quá trình huấn luyện

*Tài liệu được sinh tự động từ repo tại nhánh `16-fix-review` (HEAD `077a81b`) và dữ liệu thực tế trên máy.*

## 1. Nguồn dữ liệu (train / test)

### 1.1 Tập huấn luyện & đánh giá - Hoa Kỳ (USPTO)

#### Nguồn dữ liệu và link tải

Dữ liệu **không có sẵn một gói tải cố định** mà được thu thập trực tiếp từ **USPTO EOG/TMOG REST API** (công khai, không cần API key) rồi tái lập bằng pipeline các nhánh `02`, `03`, `04`, `05`.

| Thành phần | Link / lệnh tải |
|---|---|
| Mã nguồn + script tái lập dữ liệu | `https://github.com/nguyenlamvu123/major_proj_1.git` (nhánh `16-fix-review`) |
| Notebook tải & chuẩn bị dữ liệu (Colab, CPU) | `docs/colab/01_download_data.ipynb` |
| Metadata USPTO (text, 18 cột) | `python -m trademark_ocr.fetch_meta --limit 6000` → `$DATA_ROOT/raw/meta/us_marks.csv` |
| Ảnh nhãn hiệu USPTO (CDN) | `python -m trademark_ocr.fetch_images` → `$DATA_ROOT/raw/images_us/` |
| Làm sạch + chia tập + đóng gói LMDB | `python -m trademark_ocr.clean && python -m trademark_ocr.split --seed 1111 && python -m trademark_ocr.build_lmdb --subset <train|val|test_us>` |
| API metadata | `https://tm-eog-service.uspto.gov/eog-rest-service/api/external/search/publications` |
| API ảnh (CDN) | `https://cdn.uspto.gov/img/{serial}/{publicationDate}/original` |
| Bản đóng gói sẵn (nếu phát hành) | **TODO: chèn link Drive/HuggingFace của gói `trademark_data`** — dữ liệu lớn nằm ngoài Git |

> `DATA_ROOT = /media/zaibachkhoa/AC84-F995/trademark_data`. Trên Colab, dữ liệu US được tải vào ổ máy ảo `/content/trademark_data` rồi sao lưu `raw/meta/`, `raw/images_us.tar`, `processed/`, `lmdb/` lên `MyDrive/trademark_data/`.

#### Mô tả dữ liệu

- **Bài toán**: nhận dạng **một dòng chữ** trên toàn bộ ảnh nhãn hiệu (chưa có bước phát hiện vùng chữ): đầu vào là ảnh nhãn hiệu, đầu ra là chuỗi `word_mark`.
- Đây là bài toán **phân loại ở cấp ký tự/token** (sequence labeling / structured prediction), không phải hồi quy.
- Số lớp là cấp ký tự, không phải mỗi nhãn hiệu một lớp. Bảng nhãn có **217 ký tự** (`configs/hydra/charset/charset_vi.yaml`: chữ hoa/thường, chữ có dấu, số, dấu câu, khoảng trắng); số đầu ra phụ thuộc tokenizer:
  - Số đầu ra theo từng mô hình:
    - **CTC** (CRNN, SVTR-CTC): **218** = 217 ký tự + `blank`.
    - **PARSeq**: **218** = 217 ký tự + `EOS` (head bỏ `BOS`/`PAD`).
    - **TRBA**: **220** logits = 217 ký tự + `EOS` + `BOS` + `PAD`; chỉ dự đoán ký tự + `EOS` (`PAD` bị ignore, `BOS` chỉ làm input).
- **Metadata**: `raw/meta/us_marks.csv` — **6000 dòng**, 18 cột (chủ yếu `serial_number`, `word_mark`, `mark_drawing_cd`, `drawing_kind`, `publication_date`, `owner`, `us_class`, `image_url`…). Thống kê: `raw/meta/us_marks.csv.stats.json`.
- **Ảnh**: `raw/images_us/{serial_number}.png` — **5002 ảnh** tải từ CDN USPTO, trung bình ~45 KB/ảnh. Sổ đăng ký ảnh: `raw/images_us/index.csv` (trạng thái `ok`/`missing`, kích thước, sha1…).
- **Nhãn chữ**: Unicode **NFC**, giữ nguyên hoa/thường và dấu, độ dài 2–25 ký tự (mean ≈ 10.65), bảng ký tự 217 ký tự (`configs/hydra/charset/charset_vi.yaml`).
- **Phân bố theo loại nhãn hiệu** (trong 4840 mẫu sạch): `standard_character` 3341, `design_with_words` 853, `stylized_words` 410, `typeset_old` 236.
- **Lưu ý**: nhãn USPTO hầu như viết HOA; ảnh nhãn hiệu thường là ảnh nguyên (logo/nhiều dòng), không phải ảnh dòng chữ đã cắt.

#### Các thao tác làm sạch dữ liệu (nhánh `04`)

Đầu vào là 5000 dòng metadata đầu tiên (tham số `limit=5000`) + ảnh tương ứng trong `index.csv`. Thứ tự lọc cố định:

1. **Chuẩn hoá nhãn**: NFC → gộp khoảng trắng liên tiếp → `strip`.
2. **`no_text`**: nhãn rỗng sau chuẩn hoá.
3. **`too_short`**: nhãn < 2 ký tự.
4. **`too_long`**: nhãn > 25 ký tự (vượt `max_label_length` của mô hình).
5. **`unsupported_char`**: chứa ký tự ngoài bảng ký tự 217 ký tự.
6. **`missing_image`**: không có ảnh `status=ok` trong `index.csv`.
7. **`bad_image`**: file ảnh hỏng/không mở được (kiểm bằng PIL `probe_image`).
8. **`duplicate_label`**: trùng chuỗi nhãn (giữ 1 mẫu; có thể tắt bằng `--no-dedupe`).

Kết quả: **giữ 4840/5000** (`keep_ratio = 0.968`), chỉ rớt **160 `duplicate_label`** (các lý do text đã được lọc từ nhánh `02`). Báo cáo đầy đủ: `processed/clean_report.json`.

> Ở bước thu thập metadata (6000 dòng), nhánh `02` đã loại: `too_long` 517, `design_without_word` 305, `too_short` 72, `unsupported_char` 29. Nhánh `04` chỉ xử lý 5000 dòng đầu nên 1000 dòng metadata còn lại **chưa** được đưa vào tập sạch hiện tại.

#### Chia tập train/val/test (nhánh `05`)

- **Cách chia**: gom các dòng theo **nhóm nhãn** (`word_mark`) rồi trộn **nhóm** bằng seed cố định `1111` → tránh cùng một nhãn xuất hiện ở cả train và test (label leakage).
- **Tỉ lệ**: **90% train / 5% val / 5% test US**. Sau khi nhánh `04` dedupe, mỗi nhãn chỉ còn 1 dòng nên số nhóm = số dòng.
- **Kết quả**: train **4356**, val **242**, test US **242**. Kiểm tra chồng lấn: **0** nhãn và **0** serial giữa mọi cặp tập.
- **Tính tất định**: cùng seed `1111` cho cùng kết quả; báo cáo: `processed/split_report.json`.

| Thành phần | Vị trí (dưới `DATA_ROOT`) | Số lượng / ghi chú |
|---|---|---|
| Metadata thô | `raw/meta/us_marks.csv` + `.stats.json` | 6000 dòng, 18 cột |
| Ảnh gốc | `raw/images_us/` + `index.csv` | 5002 ảnh PNG/JPG từ CDN USPTO |
| Dữ liệu đã làm sạch | `processed/clean.csv` + `clean_report.json` | **4840** mẫu (giữ 96.8%) |
| Chia tập | `processed/split_train.csv` (4356), `split_val.csv` (242), `split_test_us.csv` (242) | 90/5/5 theo nhóm nhãn, seed 1111, overlap = 0 |
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
| **CRNN-FT / TRBA-FT / PARSeq-FT** | `tm_crnn_ft.yaml`, `tm_trba_ft.yaml`, `tm_parseq_ft.yaml` | Fine-tune từ **pretrained weights tương ứng của từng kiến trúc** (`crnn` / `trba` / `parseq`, huấn luyện trên REBU-SynText/COCO-Text) | Thêm ở nhánh 16.6, khớp charset tiếng Việt (NFC, 218 class CTC/charset phù hợp). |

**Vai trò của từng thành phần trong đồ án:**

- **CTC (Connectionist Temporal Classification)** — không phải một mô hình riêng mà là **hàm loss + cách giải mã** cho bài toán chuỗi: căn chuỗi đặc trưng ảnh với chuỗi nhãn mà không cần căn chỉnh từng ký tự, dùng token `blank` rồi gộp các bước lặp khi decode. Trong đồ án, CTC là **nhánh baseline/đối chứng**: **CRNN + CTC** (baseline cổ điển) và **SVTR-CTC** (Transformer + CTC, hiện đại hơn). Đầu ra 218 logits = 217 ký tự + `blank`.
- **TRBA (TPS + ResNet + BiLSTM + Attention)** — mô hình **attention-based** đại diện cho hướng decoder tự hồi quy: TPS hiệu chỉnh biến dạng ảnh, ResNet + BiLSTM trích đặc trưng, decoder attention sinh ký tự lần lượt và dùng `EOS` để dừng. Vai trò: **so sánh trực tiếp với nhánh CTC**, train từ đầu trên cùng charset và cùng tập dữ liệu.
- **PARSeq (Permuted Autoregressive Sequence)** — mô hình **Transformer encoder-decoder** đại diện SOTA trong 4 mô hình, dùng permutation language modeling để học ngữ cảnh hai chiều khi train. Vai trò kép: (1) **ứng viên được kỳ vọng** cho kết quả tốt nhất; (2) kho trọng số pretrained của `baudm/parseq` cung cấp bản tương ứng từng kiến trúc (`crnn`, `trba`, `parseq`) để khởi tạo `crnn_ft` / `trba_ft` / `parseq_ft` — chỉ nạp các tensor cùng tên và cùng kích thước, bỏ phần head phụ thuộc charset (DESIGN 16.6).
- **CRNN** là baseline tối giản (CNN + BiLSTM + CTC); **SVTR-CTC** là nhánh CTC hiện đại (Vision Transformer) để kiểm chứng CTC vẫn cạnh tranh khi kiến trúc encoder tốt hơn. Cả 4 mô hình dùng **cùng đầu vào ảnh nhãn hiệu, cùng đầu ra chuỗi `word_mark`, cùng charset 217 ký tự**; khác nhau ở kiến trúc, loss và cách giải mã. Các biến thể `*_ft` không phải kiến trúc mới mà là **cùng mô hình, khởi tạo từ pretrained** để so sánh công bằng giữa "train từ đầu" và "fine-tune".

**Bảng ký tự:** Tiếng Việt Unicode NFC, cấu hình tại `configs/charset/charset_vi.yaml` và `src/trademark_ocr/charset.py`. Đã đảm bảo `charset_test == charset_train` và xử lý đúng 2 bẫy parseq: `normalize_unicode: false`, `remove_whitespace: false`.

### 2.2 Cách chạy huấn luyện

Huấn luyện chạy trực tiếp `third_party/parseq/train.py` qua Hydra (`--config-dir configs/hydra --config-name ...`).

**Chuẩn bị môi trường:**

```bash
export DATA_ROOT=/media/zaibachkhoa/AC84-F995/trademark_data
export PYTHONPATH=$PWD/compat:$PWD/src:$PWD/third_party/parseq
```

**Train từ đầu:**

```bash
# CRNN + CTC
python third_party/parseq/train.py --config-dir configs/hydra --config-name tm_crnn

# TRBA
python third_party/parseq/train.py --config-dir configs/hydra --config-name tm_trba

# PARSeq
python third_party/parseq/train.py --config-dir configs/hydra --config-name tm_parseq

# SVTR-CTC
python third_party/parseq/train.py --config-dir configs/hydra --config-name tm_svtrctc
```

**Fine-tune từ trọng số pretrained:**

```bash
# PARSeq-FT
python third_party/parseq/train.py --config-dir configs/hydra --config-name tm_parseq_ft

# CRNN-FT
python third_party/parseq/train.py --config-dir configs/hydra --config-name tm_crnn_ft

# TRBA-FT
python third_party/parseq/train.py --config-dir configs/hydra --config-name tm_trba_ft
```

**Smoke test trên CPU** (không đủ cho train thật):

```bash
python third_party/parseq/train.py --config-dir configs/hydra --config-name tm_crnn \
  trainer.limit_train_batches=2 trainer.max_epochs=1 trainer.accelerator=cpu
```

Các cờ hay dùng: `--batch-size 64` → `model.batch_size=64`; `--max-epochs 20` → `trainer.max_epochs=20`; `--num-workers 0` → `data.num_workers=0`; `--lr 7e-4` → `model.lr=7e-4`; `--devices 1` → `trainer.devices=1`.

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

Chạy trực tiếp `python -m trademark_ocr.evaluate` (module `src/trademark_ocr/evaluate.py`) cho từng checkpoint, sau đó tổng hợp bằng `trademark_ocr.eval_domain_shift` và `trademark_ocr.report`. Nhánh `16-fix-review` đã sửa đánh giá để **trung thực hơn** (16.3): chấm **mọi mẫu**, giữ `gt` gốc, thêm `in_scope`/`coverage`, `idx` truy ngược được, và 3 mức đánh giá `strict` / `nocase` / `base` (bỏ dấu).

**Kết quả hiện tại (từ `reports/final_summary.md`):**

| Model | Dataset | n | accuracy (strict) | 1−NED | char-accuracy | Ghi chú |
|---|---|---|---|---|---|---|
| crnn | US (test_us) | 200 | 0.0000 | 0.0009 | -0.2315 | Trên checkpoint chưa train đủ / quy mô nhỏ |
| crnn | VN (test_vn) | 36 | 0.0000 | 0.0115 | -0.1906 | Chỉ 36/50 mẫu vào được do giới hạn đánh giá trước đây; nhánh 16 đã sửa để đánh giá toàn bộ |
| **Domain shift (US→VN)** | crnn | acc_us=0.0, acc_vn=0.0, delta=0.0000 | n_us=200, n_vn=36 | Phân tách độ sụt US→VN theo hoa/thường/dấu/hình ảnh đã được bổ sung ở nhánh 16.4 (`eval_domain_shift` nâng cấp) |

> **Kết luận về kết quả:** Các con số trên **không phản ánh chất lượng huấn luyện thực tế**. Lý do chính: (1) huấn luyện chỉ với **500/4356** mẫu train, (2) các checkpoint là phiên chạy nhỏ/không đầy đủ, (3) tập VN hiện tại chỉ **50 mẫu** chưa được duyệt tay nhiều. Để có kết quả đúng theo đề tài, cần train với **toàn bộ tập dữ liệu** trên **GPU**.

## 3. Lệnh tổng kết nhanh

```bash
# 1) Thiết lập môi trường
cd "/media/zaibachkhoa/education/GenAI 2025/đồ án chuyên ngành 1"
export DATA_ROOT=/media/zaibachkhoa/AC84-F995/trademark_data
export PYTHONPATH=$PWD/compat:$PWD/src:$PWD/third_party/parseq

# 2) Đánh giá toàn bộ model có checkpoint (CPU)
for m in crnn trba parseq svtrctc crnn_ft trba_ft parseq_ft; do
  ckpt=$(ls -dt "$DATA_ROOT/runs/$m"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
  [ -z "$ckpt" ] && continue
  python -m trademark_ocr.evaluate --ckpt "$ckpt" --subset test_us \
    --out "reports/${m}_us.csv" --device cpu --batch-size 8 --num-workers 0
  python -m trademark_ocr.evaluate --ckpt "$ckpt" --subset test_vn \
    --out "reports/${m}_vn.csv" --device cpu --batch-size 8 --num-workers 0
done
python -m trademark_ocr.eval_domain_shift --reports reports/ --out reports/domain_shift.csv
python -m trademark_ocr.error_analysis --reports reports/ --out-dir "$DATA_ROOT/processed/eda_figures"
python -m trademark_ocr.report --reports reports/ --out reports/final_summary.md

# 3) Train (khuyến nghị trên Colab/GPU) - ví dụ
python third_party/parseq/train.py --config-dir configs/hydra --config-name tm_parseq_ft \
  model.batch_size=64 trainer.max_epochs=20
# hoặc
python third_party/parseq/train.py --config-dir configs/hydra --config-name tm_svtrctc \
  model.batch_size=64 trainer.max_epochs=20

# 4) Demo suy luận
python -m trademark_ocr.demo_app --ckpt $DATA_ROOT/runs/parseq/<ts>/checkpoints/last.ckpt --image demo_images/*.png
```

**Tài liệu tham khảo:** `README.md`, `DESIGN.md` (mục 01–16), `ACCEPTANCE.md` (mục 01–16), `REVIEW.md`, `docs/colab/*.ipynb`.
