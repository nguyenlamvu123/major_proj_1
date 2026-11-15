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

> **Lưu ý (05/11/2026):** mục 2.3 và 2.4 mô tả các checkpoint cũ ngày 16/10, train trên 500 mẫu.
> Từ 04/11 các model được train lại trên toàn bộ 51.230 mẫu: quá trình train và đồ thị ở **mục 2.5**,
> cách train thêm epoch ở **mục 2.6**, nhật ký từng lần chạy và số liệu test ở
> [`docs/nhat_ky_chay.md`](docs/nhat_ky_chay.md).

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

### 2.5 Đường cong huấn luyện và kiểm tra overfit

**Mục đích.** Xem model còn học thêm được hay đã bắt đầu học thuộc tập train (overfit), để quyết
định có nên train thêm epoch hay không.

**Số liệu lấy từ đâu.** Trong lúc train, `third_party/parseq/train.py` tự ghi log TensorBoard vào
thư mục run (`runs/<model>/<giờ>/events.out.tfevents.*`). Không cần sửa code train để có đồ thị:
module `trademark_ocr.train_curves` đọc log đó rồi vẽ. Log gồm:

| Đại lượng | Đo trên | Khi nào | Ghi chú |
|---|---|---|---|
| loss train | tập train **đã augment** (xoay, nhiễu, đổi màu…), model ở chế độ train (có dropout) | mỗi batch; đồ thị lấy trung bình theo epoch | thường **cao hơn** val_loss vì ảnh train bị làm khó hơn |
| val_loss | tập val (2.846 ảnh US, không augment) | cuối mỗi epoch | |
| val_accuracy | tập val | cuối mỗi epoch | tỉ lệ ảnh đọc đúng **toàn bộ** chuỗi |

Accuracy trên tập **train** không được đo (parseq chỉ ghi loss train). Accuracy trên tập **test**
cố ý không đo theo epoch: tập val đóng vai trò đó, còn tập test chỉ chấm **một lần** ở cuối. Nếu
nhìn kết quả test để quyết định dừng train thì con số test không còn khách quan.

**Cách đọc.** Overfit là khi **loss train vẫn giảm nhưng val_loss quay đầu tăng**: model khớp tập
train ngày càng tốt mà dự đoán trên ảnh chưa thấy lại kém đi. Vì loss train tính trên ảnh đã
augment, chỉ so **xu hướng** của hai đường, không so độ lớn. Module tự kết luận theo luật: val_loss
thấp nhất ở epoch k, sau đó ít nhất 2 epoch, epoch cuối cao hơn mức thấp nhất trên 5%, trong khi
loss train vẫn giảm, thì là "có dấu hiệu overfit".

#### Tổng hợp các model đã train (06/11/2026)

| Model | Train | val tốt nhất | Checkpoint đem chấm test | Test US | Test VN (bỏ hoa/thường) | Đường cong |
|---|---|---|---|---|---|---|
| `parseq_ft` | 7 epoch, CPU, 7,3 giờ | **89,49%** (epoch 5) | epoch 6 (val 89,46%) | **88,7%** | **27,6%** (41,1%) | gần bão hoà, chưa rõ overfit |
| `crnn_ft` | 8 epoch, CPU, 2,5 giờ | 74,07% (epoch 8) | epoch 8 (val 74,07%) | 73,0% | 18,3% (27,0%) | chưa overfit, còn học được |
| `svtrctc` | 10 epoch, CPU, 2,7 giờ, train từ đầu | 71,82% (epoch 10) | model sau SWA (val 70,52%) | 69,4% | 11,4% (16,4%) | chưa overfit, còn học được |
| `trba_ft` | đang train trên Colab GPU (8/20 epoch) | — | — | — | — | — |

Test US: 2.847 ảnh. Test VN: 2.000 ảnh, **chưa duyệt tay** nhãn.

**Checkpoint đem chấm test** là file `checkpoints/last.ckpt` của run. Với Lightning 2.6, file này chỉ
được ghi lại ở epoch có checkpoint lọt **top-3 theo val_accuracy**, nên nó là "epoch gần nhất lọt
top-3", không nhất thiết là epoch cuối hay epoch tốt nhất. Cột "Checkpoint đem chấm test" ghi epoch
thật sự được chấm. Chọn checkpoint chỉ dựa trên tập val, không nhìn tập test.

#### CRNN fine-tune (`crnn_ft`) — 8 epoch trên CPU, 04/11/2026

![Đường cong huấn luyện crnn_ft](reports/figures/train_curves_crnn_ft.png)

| epoch | loss train | val_loss | val_accuracy (%) | val 1−NED (%) | ghi chú |
|---|---|---|---|---|---|
| 1 | 6,8885 | 2,5449 | 0,00 | 5,30 | lớp ra vừa khởi tạo lại cho 217 ký tự |
| 2 | 1,3157 | 0,6365 | 60,93 | 85,74 | |
| 3 | 0,7338 | 0,5189 | 67,64 | 87,49 | |
| 4 | 0,6431 | 0,4765 | 70,48 | 88,59 | |
| 5 | 0,6020 | 0,4575 | 71,61 | 88,86 | |
| 6 | 0,5769 | 0,4456 | 72,63 | 89,08 | |
| 7 | 0,5632 | 0,4354 | 73,05 | 89,35 | |
| 8 | 0,5503 | **0,4254** | **74,07** | 89,52 | tốt nhất, checkpoint dùng để test |
| 8 (SWA) | — | 0,4445 | 72,59 | 89,10 | model sau khi gộp trọng số SWA |

**Nhận xét:**

1. **Chưa overfit.** val_loss giảm liên tục tới epoch cuối (0,4254 là mức thấp nhất) và
   val_accuracy vẫn tăng (73,05% → 74,07% ở epoch cuối). Model **chưa bão hoà**: dừng ở 8 epoch là
   do ngân sách 3 giờ CPU, không phải vì đã học xong. Vì vậy đợt tiếp theo **train thêm** (mục 2.6).
2. Mức tăng mỗi epoch giảm dần (+6,7 → +2,8 → +1,1 → +1,0 → +0,4 → +1,0 điểm %), nên train thêm có
   lẽ chỉ được vài điểm %.
3. Epoch 1 có accuracy 0%: lớp ra được khởi tạo lại cho bảng 217 ký tự tiếng Việt. Từ epoch 2 trở đi
   phần trọng số pretrained mới phát huy tác dụng.
4. Điểm "SWA" (Stochastic Weight Averaging, lấy trung bình trọng số 2 epoch cuối) được đo sau khi
   train xong. Ở đây nó kém epoch 8 (72,59% so với 74,07%). Lightning không ghi checkpoint cho điểm
   này, nên checkpoint dùng để test (`last.ckpt`) là epoch 8.
5. Kết quả test của checkpoint này (`docs/nhat_ky_chay.md`, lần 1): test US **73,0%** (sát val 74,1%, tức val đại diện
   tốt cho test cùng nguồn); test VN 18,3%. Mức sụt US→VN là do khác miền dữ liệu (chữ có dấu,
   nhãn viết thường, kiểu chữ), **không** phải overfit. Overfit chỉ đo được trên dữ liệu cùng phân
   bố với tập train.

#### PARSeq fine-tune (`parseq_ft`) — 7 epoch trên CPU, đêm 05→06/11/2026

![Đường cong huấn luyện parseq_ft](reports/figures/train_curves_parseq_ft.png)

| epoch | loss train | val_loss | val_accuracy (%) | val 1−NED (%) | ghi chú |
|---|---|---|---|---|---|
| 1 | 2,1533 | 0,5365 | 84,26 | 94,47 | lớp ra vừa khởi tạo lại cho 217 ký tự |
| 2 | 0,3033 | 0,4813 | 87,81 | 95,60 | |
| 3 | 0,2492 | 0,4797 | 88,65 | 95,97 | |
| 4 | 0,2224 | 0,4792 | 89,39 | 96,18 | |
| 5 | 0,2041 | **0,4791** | **89,49** | 96,28 | tốt nhất (cả val_loss lẫn val_accuracy) |
| 6 | 0,1963 | 0,4927 | 89,46 | 96,23 | checkpoint dùng để test |
| 7 | 0,1883 | 0,4969 | 89,14 | 96,23 | |

**Nhận xét:**

1. **Gần bão hoà, mới chớm dấu hiệu overfit, chưa đủ để kết luận.**
   - Từ epoch 4, val_accuracy chỉ dao động trong khoảng 89,1–89,5%.
   - val_loss thấp nhất ở epoch 5 (0,4791), rồi tăng 4% trong 2 epoch cuối, trong khi loss train
     vẫn giảm (0,204 → 0,188). Đó đúng là chiều của overfit.
   - Nhưng mức tăng chưa tới ngưỡng 5% của luật, và val_accuracy chỉ giảm 0,35 điểm (khoảng 10
     trên 2.846 ảnh).
   - Kết luận: train thêm với cùng dữ liệu khó cải thiện, nên **không** train thêm parseq_ft.
2. **Học rất nhanh:** đạt 84,3% ngay sau epoch 1, trong khi crnn_ft ở epoch 1 là 0%. Loss train
   epoch 1 (2,15) cao hơn hẳn val_loss vì nó là trung bình cả epoch, gồm cả những bước đầu khi lớp ra
   còn đoán ngẫu nhiên (loss 5,39 = ln 217).
3. **Vì sao lần train trước (2 epoch, 04→05/11) không học được.**
   - Cùng cấu hình, cùng trọng số khởi tạo: lần trước sau 1.602 bước val vẫn 0%, lần này sau 801
     bước đã 84,3%.
   - Trong epoch đầu, hai lần chỉ khác độ dài lịch learning rate OneCycle (tính theo tổng số epoch).
     Lần trước, learning rate lên đỉnh ở bước 120, khi lớp ra còn đoán ngẫu nhiên (loss ~5,3). Lần
     này lên đỉnh ở bước 420, khi loss đã 1,23.
   - Giả thuyết: learning rate cao quá sớm làm hỏng phần trọng số pretrained. Chưa kiểm chứng bằng
     thí nghiệm đối chứng; số liệu từng bước ở
     [`docs/nhat_ky_chay.md`](docs/nhat_ky_chay.md), lần 2.
4. **Checkpoint dùng để test là epoch 6** (89,46%), không phải epoch 7. Epoch 7 (89,14%) không lọt
   top-3 nên Lightning không ghi lại `last.ckpt`. Chênh lệch với epoch tốt nhất (89,49%) không đáng
   kể.
5. **Không có điểm SWA trên đồ thị.** PARSeq không có BatchNorm, nên Lightning không chạy lượt đánh
   giá thêm cho model sau SWA. Model đó cũng không được lưu, và không được đem chấm.
6. **Test:** US 88,7%, sát val 89,5%. VN 27,6%, bỏ hoa/thường 41,1%: tốt nhất trong các model đã
   train. Tuy vậy, trên 558 nhãn VN có dấu chỉ đọc đúng 2 nhãn (0,4%), vì dữ liệu train USPTO gần
   như không có chữ có dấu.

#### SVTR-CTC (`svtrctc`) — 10 epoch trên CPU, train từ đầu, 06/11/2026

![Đường cong huấn luyện svtrctc](reports/figures/train_curves_svtrctc.png)

| epoch | loss train | val_loss | val_accuracy (%) | val 1−NED (%) | ghi chú |
|---|---|---|---|---|---|
| 1 | 6,1960 | 3,4441 | 0,00 | 0,00 | khởi tạo ngẫu nhiên, không có pretrained |
| 2 | 3,3263 | 2,4724 | 0,74 | 36,34 | |
| 3 | 2,0051 | 0,8851 | 50,46 | 78,45 | |
| 4 | 1,3090 | 0,7054 | 61,38 | 82,25 | |
| 5 | 1,1181 | 0,6392 | 64,86 | 83,77 | |
| 6 | 0,9806 | 0,5984 | 68,13 | 84,32 | |
| 7 | 0,8959 | 0,5807 | 69,22 | 84,93 | |
| 8 | 0,8486 | 0,5627 | 69,75 | 85,10 | |
| 9 | 0,8139 | 0,5439 | 71,08 | 86,08 | |
| 10 | 0,7910 | **0,5325** | **71,82** | 85,93 | tốt nhất |
| 10 (SWA) | — | 0,5611 | 70,52 | 85,36 | model sau SWA; checkpoint dùng để test |

**Nhận xét:**

1. **Chưa overfit, chưa bão hoà.** val_loss giảm liên tục tới epoch cuối, và val_accuracy vẫn tăng
   (+1,3 rồi +0,7 điểm ở hai epoch cuối). Dừng ở 10 epoch là do ngân sách 3 giờ CPU còn lại của đêm.
   Có thể train thêm bằng cơ chế ở mục 2.6.
2. **Khởi đầu chậm:** hai epoch đầu gần 0%, epoch 3 mới vọt lên 50%. Đây là model duy nhất train từ
   đầu, không có trọng số pretrained, và loss CTC thường cần một thời gian trước khi canh được vị trí
   ký tự.
3. **Checkpoint dùng để test là model sau SWA** (70,52%), kém epoch 10 khoảng 1,3 điểm val.
   - Sau epoch cuối, Lightning chạy thêm một lượt qua tập train để tính lại BatchNorm cho model SWA.
     Lượt này được ghi thành checkpoint, và lọt top-3 vì hơn epoch 8 (69,75%).
   - Hệ quả: số test của svtrctc có thể thấp hơn một chút so với nếu chấm epoch 10.
4. **Test:** US 69,4%, VN 11,4% (16,4% bỏ hoa/thường): thấp nhất trong ba model. Kém rõ nhất ở nhãn
   có hình vẽ (`design_with_words`, US: 3,4% so với 14,4% của crnn_ft và 40,9% của parseq_ft).

**Vẽ lại / cập nhật đồ thị.** `bash scripts/06_eval.sh` (hoặc bước `eval` của
`09_full_pipeline_cpu.sh`) tự vẽ lại sau mỗi lần đánh giá. Vẽ riêng, không cần đánh giá:

```bash
.venv/bin/python -m trademark_ocr.train_curves --out-dir reports/figures
# chỉ vài model:  --models crnn_ft,parseq_ft     thư mục runs khác:  --runs /đường/dẫn/runs
```

Kết quả: `reports/figures/train_curves_<model>.png`, bảng số theo epoch và kết luận ở
`reports/figures/train_curves.md` (dán thẳng vào báo cáo được), số liệu máy đọc ở `train_curves.json`.

### 2.6 Train thêm epoch từ checkpoint cũ (không train lại từ đầu)

**Vấn đề.** crnn_ft dừng ở 8 epoch khi vẫn đang tốt lên. Train lại từ đầu với 20 epoch sẽ bỏ phí 8
epoch đã có (khoảng 2,3 giờ CPU). Cơ chế "train tiếp" sẵn có của Lightning (`ckpt_path=`) chỉ áp
dụng cho lần train **bị ngắt giữa chừng**. Nó khôi phục cả lịch learning rate OneCycle, mà lịch này
đã chạy hết số bước của 8 epoch, rồi chuyển sang SWA. Kéo dài thêm epoch theo cách đó sẽ báo lỗi.

**Cách làm** (`scripts/09_full_pipeline_cpu.sh --continue`, DESIGN 16.12):

1. Tạo một run **mới**, nạp **toàn bộ** trọng số từ checkpoint cuối của run cũ (cùng kiến trúc, cùng
   bảng ký tự: nạp đủ 46/46 tensor của CRNN).
2. Tạo optimizer và lịch learning rate **mới** cho số epoch còn thiếu. Learning rate tăng lại lên
   đỉnh rồi giảm dần (kiểu *warm restart*, SGDR — Loshchilov & Hutter 2017). Vì vậy 1–2 epoch đầu
   của đợt mới val có thể chững hoặc giảm nhẹ rồi mới tăng tiếp.
3. `--max-epochs` là **tổng** số epoch. crnn_ft đã có 8, nên `--max-epochs 20` train thêm 12. Chạy
   lại đúng lệnh thì an toàn: đủ 20 thì bỏ qua; nếu bị ngắt giữa chừng thì train tiếp chính run mới.
4. Run mới ghi lại nó được train tiếp từ run nào và đã có bao nhiêu epoch trước đó. Nhờ vậy đồ thị ở
   mục 2.5 nối hai run thành một đường epoch 1 → 20, có vạch đứt ở chỗ nối.

```bash
bash scripts/09_full_pipeline_cpu.sh --from train --models crnn_ft --continue --max-epochs 20 --max-hours 8
```

**Kế hoạch:** crnn_ft 8 → 20 epoch, chạy trên một máy Ubuntu khác (yếu hơn). Từng bước, gồm cả
cách mang dữ liệu và checkpoint qua lại giữa hai máy, ở
[`docs/nhat_ky_chay.md`](docs/nhat_ky_chay.md) mục "Việc tiếp theo → 4". Kết quả sẽ được bổ sung vào
mục 2.5 sau khi train xong. svtrctc (10 epoch, chưa bão hoà) cũng có thể train thêm theo cách này.

**Phân biệt với train tiếp một run bị ngắt.** `trba_ft` trên Colab dừng ở epoch 8/20 vì hết hạn mức
GPU, tức run **chưa** chạy hết lịch learning rate. Trường hợp này dùng cơ chế sẵn có
`ckpt_path=<run>/checkpoints/last.ckpt`: Lightning khôi phục trọng số, optimizer, lịch learning rate
và bộ đếm epoch, rồi chạy nốt các epoch còn lại của **chính run đó**. Về nguyên tắc, kết quả tương
đương một lần train liền mạch, chỉ khác thứ tự xáo dữ liệu. `--continue` chỉ dùng cho run đã train
**xong** như crnn_ft. Các bước làm
trên Colab ở [`docs/nhat_ky_chay.md`](docs/nhat_ky_chay.md) mục "Việc tiếp theo → 3a".

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
