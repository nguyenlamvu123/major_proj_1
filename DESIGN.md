# DESIGN.md — Tài liệu thiết kế chi tiết (đọc trước khi sửa code)

> **Đối tượng đọc**: người/agent tiếp tục phát triển đồ án.
> File này mô tả **từng module, interface, dữ liệu vào/ra và lý do thiết kế**.
> Cập nhật file này trong **cùng commit** với mỗi thay đổi code (quy tắc).

---

## 1. Mục tiêu & ràng buộc

| Hạng mục | Quyết định |
|---|---|
| Bài toán | Scene Text Recognition (STR) trên ảnh wordmark nhãn hiệu: image → chuỗi ký tự |
| Supervised / DL | Có giám sát, PyTorch + PyTorch-Lightning |
| Khung mã gốc | `baudm/parseq` — **clone** vào `third_party/parseq`, **không fork, không sửa** |
| Mô hình so sánh (4) | CRNN+CTC (baseline) · TRBA (TPS+BiLSTM+Attention) · PARSeq (ViT+permutation LM) · SVTR-CTC (CTC transformer, code riêng mình) |
| Charset | Unicode có dấu tiếng Việt, chuẩn hoá **NFC** |
| Train data | Nhãn hiệu US (USPTO), 50k–150k ảnh |
| Test mới | Nhãn hiệu VN (IP Vietnam/WIPO Publish), 2k–5k ảnh |
| Nơi train | Google Colab (GPU T4); local chỉ build data + test CPU |
| Git | 15 nhánh local, merge/push do chủ repo |

## 2. Quy ước chung

- **Package**: `src/trademark_ocr/` (cài `pip install -e .`). Import: `from trademark_ocr.charset import ...`.
- **Môi trường**:
  - `DATA_ROOT` — gốc dữ liệu. Local: `/media/zaibachkhoa/AC84-F995/trademark_data`; Colab: `/content/drive/MyDrive/trademark_data`. Mọi script nhận `--data-root` hoặc env `DATA_ROOT` (fallback `./data`).
  - `REPO_ROOT` = thư mục chứa file này (tự resolve từ vị trí file).
- **Layout dữ liệu dưới DATA_ROOT**:

```
$DATA_ROOT/
├── raw/
│   ├── meta/            # metadata USPTO (csv/parquet)  — branch 02
│   ├── images_us/       # ảnh nhãn hiệu US              — branch 03
│   └── images_vn/       # ảnh nhãn hiệu VN              — branch 11
├── processed/           # sau cleaning/split: train/val/test .txt (path\tlabel)
│   ├── train.txt  val.txt  test_us.txt  test_vn.txt
│   └── clean_report.json
├── lmdb/                # LMDB cho parseq                — branch 05
│   ├── train/us_marks/data.mdb
│   ├── val/us_marks/data.mdb
│   └── test/{us_test,vn_test}/data.mdb
└── runs/                # checkpoints, logs tensorboard
```

- **Nhánh git**: branch N kẹt từ branch N−1, commit theo `NN-mô-tả`. **Không merge/push** (việc của chủ repo).
- **Docs**: mọi commit code phải kèm cập nhật `DESIGN.md` (mô tả phần code mới) và `ACCEPTANCE.md` (lệnh nghiệm thu).

## 3. Luồng dữ liệu (pipeline)

```
[02] USPTO case files CSV ──┐
                            ├─► [04] clean (lọc wordmark, NFC, bộ lọc) ─► processed/{train,val,test_us}.txt
[03] USPTO images ──────────┘                                     │
                                                                  ├─► [05] split (anti-leak) + build LMDB ─► $DATA_ROOT/lmdb/
                                                                  │
                                                                  ├─► [07..09] train (hydra + parseq) ─► runs/*.ckpt
                                                                  │
[11] VN images + labels ─► processed/test_vn.txt + lmdb/test/vn_test ─┤
                                                                      └─► [10][12] evaluate ─► reports/*.csv
                                                                             │
                                                                        [13] error analysis ─► notebooks/02
                                                                             │
                                                                        [14] demo (CLI/Gradio)
```

## 4. Cách tích hợp `baudm/parseq` (quan trọng — đọc kỹ)

### 4.1 Không sửa code của họ
Toàn bộ tích hợp xảy ra ở **3 điểm tiếp xúc**, đều nằm ở code mình:

1. **Hydra config override**: `train.py` của parseq khai báo
   `@hydra.main(config_path='configs', config_name='main')`.
   Hydra cho phép **thêm config dir** bằng cờ CLI `--config-dir`.
   → Mình để `third_party/parseq/configs/` nguyên, và viết config riêng tại
   `configs/hydra/` của repo mình:

   ```bash
   python third_party/parseq/train.py \
       --config-dir "$REPO/configs/hydra" \
       --config-name tm_crnn        # file configs/hydra/tm_crnn.yaml
   ```

2. **Charset**: parseq nạp charset từ field `model.charset_train` (chuỗi) trong yaml
   (xem `configs/charset/*.yaml` — chỉ là chuỗi 1 dòng). → Mình **tự sinh**
   `configs/charset_vi.yaml` (nhánh 01) từ `trademark_ocr.charset` và trỏ config huấn luyện tới nó.

3. **Dataset**: parseq đọc LMDB theo layout
   `root/{train,val}/<subset>/data.mdb` và `root/test/<subset>/data.mdb`
   (xem `strhub/data/module.py::SceneTextDataModule` + `strhub/data/dataset.py`).
   → Mình build LMDB đúng format đó (nhánh 05), **không** cần viết datamodule mới.

### 4.2 Hai *cạm bẫy* của parseq với tiếng Việt (ghi lại để không quên)

| Cạm bẫy | Nơi xảy ra | Cách mình xử lý |
|---|---|---|
| `LmdbDataset` mặc định **NFKD → ASCII** khi `normalize_unicode=true` → **mất dấu tiếng Việt** | `strhub/data/dataset.py` `_preprocess_labels` | Config của mình đặt `data.normalize_unicode: false`; label đã NFC từ khâu cleaning |
| `remove_whitespace=true` mặc định → xoá khoảng trắng trong nhãn hiệu có space | `SceneTextDataModule` | Config của mình đặt `data.remove_whitespace: false`, charset **có ký tự space** |
| `charset_test` mặc định 36 ký tự lowercase → `CharsetAdapter` sẽ lower + strip ký tự lạ khi eval | `main.yaml` + `strhub/models/base.py` | `model.charset_test` = cùng charset VN đầy đủ (case-sensitive) |

### 4.3 Đường dẫn dữ liệu
`train.py` của parseq gọi `hydra.utils.to_absolute_path(config.data.root_dir)` →
config của mình đặt `data.root_dir` là **đường dẫn tuyệt đối**指向 `$DATA_ROOT/lmdb`
(tự resolve bằng env trong wrapper shell `scripts/05_train.sh`), tránh phụ thuộc cwd của Hydra.

### 4.4 Load checkpoint để eval/demo
Mô hình parseq là `LightningModule` đã `save_hyperparameters()` → load được bằng:

```python
from strhub.models.crnn.system import CRNN
model = CRNN.load_from_checkpoint(path)   # hparams (charset…) nằm trong ckpt
model.eval()
```
Mô hình SVTR-CTC của mình cũng tuân theo pattern này (nhánh 09).

## 5. Bảng module (file → trách nhiệm → input/output)

| File | Nhánh | Trách nhiệm | Input | Output |
|---|---|---|---|---|
| `src/trademark_ocr/__init__.py` | 00 | package marker, version | – | – |
| `src/trademark_ocr/charset.py` | 01 | bảng ký tự VN NFC, encode/decode, sinh yaml hydra | – | `configs/charset_vi.yaml`, `Charset` object |
| `src/trademark_ocr/fetch_meta.py` | 02 | tải metadata wordmark USPTO | URL/`--out` | `$DATA_ROOT/raw/meta/*.csv` |
| `src/trademark_ocr/fetch_images.py` | 03 | tải ảnh theo serial, resume, thread pool | meta csv | `$DATA_ROOT/raw/images_us/*.jpg` + index |
| `src/trademark_ocr/clean.py` | 04 | lọc wordmark, NFC, filter nhãn | meta+images | `processed/*.txt`, `clean_report.json` |
| `src/trademark_ocr/split.py` | 05 | split 80/10/10 chống leak | processed txt | `processed/{train,val,test_us}.txt` |
| `src/trademark_ocr/build_lmdb.py` | 05 | convert txt → LMDB parseq | txt + images | `$DATA_ROOT/lmdb/**` |
| `src/trademark_ocr/train_utils.py` | 07 | helpers cho train (resolve config, sanity check) | – | – |
| `src/trademark_ocr/models_svtr.py` | 09 | `SVTRCTC` (Lightning, CTCSystem) | – | ckpt |
| `src/trademark_ocr/metrics.py` | 10 | accuracy, 1−NED, char accuracy | pred/gt lists | dict/DataFrame |
| `src/trademark_ocr/evaluate.py` | 10 | inference hàng loạt trên LMDB/txt | ckpt + dataset | `reports/*.csv` |
| `src/trademark_ocr/fetch_vn.py` | 11 | crawl nhãn hiệu VN (WIPO Publish) | query/pages | `raw/images_vn/`, `processed/test_vn.txt` |
| `src/trademark_ocr/eval_domain_shift.py` | 12 | bảng so sánh US-test vs VN-test | reports csv | `reports/domain_shift.csv` |
| `src/trademark_ocr/report.py` | 13 | figures + bảng lỗi top-N | reports csv | `reports/figures/*` |
| `src/trademark_ocr/demo_app.py` | 14 | CLI + Gradio demo | ảnh + ckpt | chuỗi + confidence |

*(Các dòng trên được **bổ sung chi tiết** (interface hàm, hàm chính) ở từng nhánh — tìm theo tên nhánh `## Nhánh NN`).*

## 6. Quy ước test & nghiệm thu

- `pytest -q` chạy được trên **CPU, không cần GPU, không cần data thật** (test dùng fixture giả/taruffle nhỏ).
- Mỗi nhánh phải có: (a) code, (b) mục `## Nhánh NN` trong DESIGN.md, (c) lệnh trong `ACCEPTANCE.md`.
- Test dữ liệu thật (smoke 50–200 mẫu) tách riêng trong ACCEPTANCE (chạy tay, không bắt buộc trong CI).

## 7. Model zoo & lý do chọn

| Model | Nhóm | Vì sao có mặt | Nguồn |
|---|---|---|---|
| CRNN | CTC cổ điển | baseline tối giản, tiêu chuẩn STR 2016 | `strhub.models.crnn` (parseq) |
| TRBA | TPS + Attention | đại diện chuỗi attention truyền thống | `strhub.models.trba` |
| PARSeq | Transformer + permutation LM | đại diện SOTA encoder-decoder | `strhub.models.parseq` |
| SVTR-CTC | Transformer + CTC | **mô hình hiện đại** (đối thủ SVTRv2/PP-OCRv4); code riêng | `trademark_ocr.models_svtr` |

*(Nhắc lại phương án mở rộng — nếu sau này cần thêm:*
- ***B** = `clovaai/deep-text-recognition-benchmark` (benchmark chuẩn, data LMDB MJ+ST, so số liệu báo cáo);*
- ***C** = `Topdu/OpenOCR` (SVTRv2 chính thức + pipeline det+rec PPOCR-style, muốn detection end-to-end).)*

## 8. Cấu trúc config huấn luyện (`configs/hydra/`)

Mỗi file `tm_<model>.yaml` là bản sao cấu trúc `main.yaml` của parseq với các sửa đổi:

```yaml
defaults:
  - _self_
  - override /model: crnn        # crnn | trba | parseq   (SVTR-CTC dùng _target_ riêng)
  - override /charset: charset_vi # file configs/charset_vi.yaml (sinh từ charset.py)

model:
  charset_train: ???             # được charset_vi.yaml điền
  charset_test:  ???             # = charset_train (case-sensitive)
  batch_size: 128
data:
  _target_: strhub.data.module.SceneTextDataModule
  root_dir: ???                  # scripts/05_train.sh resolve từ $DATA_ROOT
  train_dir: us_marks
  remove_whitespace: false
  normalize_unicode: false       # ★ bắt buộc với tiếng Việt (xem 4.2)
  augment: true
trainer:
  accelerator: gpu               # Colab; đổi 'cpu' khi test local
  devices: 1
  max_epochs: 20
```

*Lưu ý*: `--config-dir` của Hydra **thay/mở rộng search path** — config của mình
**không** phụ thuộc file nào trong `third_party/parseq/configs` ngoại trừ
`model/*` (nạp qua `override /model` — vẫn đọc từ repo gốc, không sửa).

## 9. Workflow Colab

`docs/colab/` chứa các notebook con (copy-paste từng cell cũng được):

| Notebook | Việc |
|---|---|
| `00_setup.ipynb` | clone repo, mount Drive, `pip install -r requirements.txt && pip install -e .`, set `DATA_ROOT` |
| `01_download_data.ipynb` | tải metadata + ảnh USPTO → Drive (dùng khi local tải chậm) |
| `02_train.ipynb` | build LMDB (nếu cần) + gọi `scripts/05_train.sh <model>` |
| `03_eval.ipynb` | `scripts/06_eval.sh` trên test US + test VN, xuất báo cáo về Drive |

Checkpoints lưu tại `$DATA_ROOT/runs/<model>/` (Drive) → giữ qua các phiên Colab.

---

# Nhật ký chi tiết theo nhánh

*(Mỗi nhánh bổ sung một mục dưới đây mô tả interface hàm cụ thể.)*

## Nhánh 00-init
- `pyproject.toml`: package `trademark_ocr` (`src/`-layout), pytest `pythonpath = [src, third_party/parseq]` để test import được cả `strhub` mà không cài.
- `requirements.txt`: ban đầu pin theo parseq (`pytorch-lightning==2.2.0.post0`, `timm==0.9.16`); sau khi kiểm chứng đã nới pin (xem mục "Kiểm chứng môi trường" ngay dưới).
- `.gitignore`: loại `third_party/`, `*.lmdb`, `runs/`, `*.ckpt` — dữ liệu không bao giờ vào git.
- `tests/test_smoke.py`: kiểm tra package import được và `strhub` tìm thấy từ `third_party/parseq`.

**Kiểm chứng môi trường (ghi lại để tra cứu)**: torch 2.14.0 + pytorch-lightning 2.6.6 +
timm 1.0.30 + hydra 1.3.7 → CRNN/TRBA/PARSeq forward OK (cả train-mode teacher forcing
của TRBA lẫn eval), `Trainer.fit` 1 batch CPU OK. Vì vậy `requirements.txt` đã nới pin
thành `pytorch-lightning>=2.2,<3`, `timm>=0.9`. Nếu gặp lỗi API lạ, hạ về đúng pin trong
`third_party/parseq/requirements/{core,train}.txt` (torch 2.2.1, PL 2.2.0, timm 0.9.16).

## Nhánh 01-charset-vi — `src/trademark_ocr/charset.py`

**Trách nhiệm**: định nghĩa bảng ký tự Unicode tiếng Việt (NFC) + tiện ích
chuẩn hoá/lọc + sinh config hydra. *Không* thay thế tokenizer của parseq; chỉ
cung cấp **chuỗi charset** mà parseq dùng để build tokenizer.

**Cấu trúc charset (217 ký tự → 218 class CTC)**:
| Khối | Nguồn | Ghi chú |
|---|---|---|
| 10 chữ số | `_DIGITS` | |
| 26 Latin thường + 26 Latin hoa | `_ASCII_LOWER/UPPER` | giữ cả 2 case (case-sensitive) |
| 146 chữ Việt (73 lower + 73 hoa) | sinh tự động: 12 nguyên âm (`a ă â e ê i o ô ơ u ư y`) × 6 thanh (`"" \u0301 \u0300 \u0309 \u0303 \u0323`) + `đ` | tổ hợp bằng `unicodedata.normalize('NFC', base+tone)` |
| 21 dấu câu + space | `_PUNCT = " .,-'’&/()!+*@#%:;=?_"` | ⚠️ KHÔNG gồm `™ © ®` (glyph nhỏ, khó đọc → nhiễu) |

**Hàm chính**:
| Hàm | Vào | Ra | Ghi chú |
|---|---|---|---|
| `build_charset()` | – | `str` | deterministic, loại trùng |
| `to_nfc(text)` | chuỗi (có thể NFD) | chuỗi NFC | gọi trong mọi khâu nhãn |
| `collapse_ws(text)` | chuỗi | chuỗi | gộp space, strip |
| `filter_supported(text, charset)` | chuỗi | chuỗi | bỏ ký tự ngoài charset |
| `Charset.encode/decode` | – | list id / chuỗi | tham chiếu cho test & EDA (blank id = 0) |
| `emit_hydra_yaml(path)` | – | `Path` | ghi `configs/hydra/charset/charset_vi.yaml` |
| `emit_txt(path)` | – | `Path` | ghi `configs/charset_vi.txt` |
| `stats()` | – | dict | số class, số chữ Việt, NFC-stable… |

**Tại sao `emit_hydra_yaml` nằm ở `configs/hydra/charset/`**: hydra coi mỗi thư mục
con trong search path là một **config group**; để `defaults: [_self_, charset: charset_vi]`
resolve được thì file phải nằm trong `<search_path>/charset/` (xem §8).

**Test** (`tests/test_charset.py`, 13 test): unique/deterministic, NFC ổn định,
đủ chữ Việt, roundtrip encode↔decode trên câu có dấu, CTC best-path decode
(gộp lặp + bỏ blank), nội dung yaml sinh ra, thống kê hợp lý.

## Nhánh 02-fetch-uspto-meta — `src/trademark_ocr/fetch_meta.py`

**Trách nhiệm**: thu thập **metadata** (nhãn chữ, serial, ngày công bố, loại hình,
class, chủ sở hữu) của các nhãn hiệu Mỹ → `$DATA_ROOT/raw/meta/us_marks.csv`.
Không tải ảnh (nhánh 03).

### Nguồn dữ liệu (đã dò & kiểm chứng bằng curl, **không cần API key**)

`bulkdata.uspto.gov` **đã bị khai tử** (NXDOMAIN, kể cả khi hỏi DNS 8.8.8.8) và
`trademarks.reedtech.com` cũng NXDOMAIN → không dùng được. `api.uspto.gov` (Open Data
Portal) hiện trả `{"message":"Unauthorized"}` — **cần API key**. Nguồn dùng được:

| Nguồn | URL | Ghi chú |
|---|---|---|
| **TMOG EOG search** | `GET  https://tm-eog-service.uspto.gov/eog-rest-service/api/external/search/publications` | list 688 issue gazette, 2013-08-13 → nay |
| | `POST .../api/external/search/all` | body `{publicationDates:[...], offset, limit}`; **limit tối đa 100** (250→rỗng) |
| | `GET  .../api/external/single-reg/{regNum}` | tra 1 đăng ký |
| **Ảnh (CDN)** | `https://cdn.uspto.gov/img/{seriaNum}/{publicationDate}/{size}` | `size ∈ {px160x130, px300x200, original}`; ảnh có từ ~2014 trở đi (2013-08-13 → 404) |
| **Gazette gốc (backup)** | GCS bucket `tmogdocs` **public-read + listable**: `https://www.googleapis.com/storage/v1/b/tmogdocs/o?prefix=TMOGIssue_` | file `TMOGIssue_YYYYMMDD_entire.{xml,pdf,zip}` (XML ~80–105 MB/issue) |

> ⚠️ Vì sao phải đi qua REST API thay vì tải XML gazette: XML 80–105 MB/issue quá nặng
> cho mục đích metadata; API phân trang cho đúng trường cần, nhẹ hơn ~1000×.

### Schema EOG đổi theo thời kỳ (đã kiểm chứng)

- **2013–2014**: KHÔNG có `markDrawingTypeCode`/`markDrawingTypeDesc`/`goodsServices`;
  class nằm ở `gsUsClasses`. Có thêm `docket_no`.
- **2015+**: có `markDrawingTypeCode` + `markDrawingTypeDesc` + `goodsServices` (list
  `{classNum,classType,description}`).
- `markText` thường **có space thừa cuối** (`'GYV '`) → luôn strip + gộp space + NFC.
- ⇒ Mọi truy cập field dùng `.get()`; `_get_classes()` hợp nhất 2 schema.

### Mã "mark drawing" (USPTO) & cách phân loại

`1` typeset (cũ) · `2` design-only (không chữ) · `3` design có chữ · `4` standard
character (chữ đánh máy) · `5` chữ cách điệu · `6` không có hình (âm thanh…).
`classify_drawing()` → `typeset_old / design_only / design_with_words /
standard_character / stylized_words / no_drawing / unknown / other`.
Mặc định giữ mã **{1,3,4,5}** (`WORD_BEARING_CODES`), loại {2,6}.
⚠️ Với task STR ta quan tâm nhất `3` + `5` (chữ cách điệu thật); `4` là ảnh chữ
"đánh máy" (dễ); `2` không có nhãn.

### Cột CSV đầu ra (`CSV_COLUMNS`, 18 cột)

ACCEPTANCE nhánh 02 yêu cầu tối thiểu **`serial_number, word_mark, image_status`**
(3 cột đầu); phần còn lại phục vụ các nhánh sau:

| Cột | Ý nghĩa |
|---|---|
| `serial_number` | số đơn (8 chữ số, khoá join) |
| `word_mark` | nhãn chữ đã NFC + gộp space + strip (**có thể rỗng → bị lọc bỏ**) |
| `image_status` | `pending` khi mới lấy; nhánh 03 đổi thành `ok`/`missing` |
| `mark_drawing_cd`,`drawing_kind` | mã & nhãn loại hình |
| `mark_type` | `TRDMK`/`SRVCK`/… |
| `publication_date`,`publication_id` | issue gazette (dùng cho URL ảnh) |
| `reg_number`,`filing_date`,`registration_date` | số đăng ký, ngày nộp/đăng ký |
| `owner`,`us_class`,`intl_class`,`filing_basis`,`pub_reason` | ngữ cảnh (report/EDA) |
| `image_url` | URL ảnh CDN (`.../original`) |
| `source` | `tmog` |

### Hàm chính

| Hàm | Vào | Ra |
|---|---|---|
| `fetch_publications(session, timeout, endpoint)` | – | `list[dict]` (mới nhất trước) |
| `iter_issue(session, date, page_size=100, per_issue_limit, sleep)` | – | generator bản ghi thô (phân trang `offset`) |
| `normalize(rec, pub_date, publication_id)` | bản ghi thô | dict phẳng đúng `CSV_COLUMNS` |
| `classify_drawing(code)` | mã | nhãn loại hình |
| `image_url(serial, pub_date, size="original")` | – | URL ảnh CDN |
| `is_candidate(row, charset, min_len, max_len, allowed_codes)` | dòng | `(ok, reason)` |
| `select_publications(pubs, since, until, max_issues)` | – | list issue đã lọc/sắp |
| `run(out, ...)` | – | ghi CSV + `.stats.json`, trả stats |
| `write_csv`, `load_existing_rows` | – | I/O (atomic tmp→replace) |

`is_candidate` lọc theo thứ tự & lý do rớt: `design_without_word` (mã 2/6) →
`drawing_excluded` (mã ngoài tập) → `no_text` → `too_short` → `too_long` →
`unsupported_char` (ký tự ngoài charset VN). ⚠️ Mã rỗng (`""`, thời kỳ 2013–2014)
**được chấp nhận** nếu có chữ (không có mã để phán).

### CLI & script

```bash
bash scripts/01_get_meta.sh --limit 5000 --sleep 0.05
# tương đương:
python3 -m trademark_ocr.fetch_meta --out $DATA_ROOT/raw/meta/us_marks.csv --limit 5000
```

Cờ chính: `--limit` (0=∞), `--since/--until` (YYYY-MM-DD), `--max-issues`,
`--per-issue-limit`, `--page-size` (≤100), `--drawing-codes`, `--min-len/--max-len`,
`--shuffle --seed` (trộn tất định), `--resume`, `--list-issues`.
Script dùng `${PYTHON:-$(command -v python3 || command -v python)}` nên chạy được cả
nơi chỉ có `python3`.

**Ngữ nghĩa `--resume`**: nạp **toàn bộ** CSV cũ vào bộ nhớ, bỏ qua serial đã có, rồi
ghi lại cả file (atomic) → CSV không mất dòng cũ, không trùng serial.
Mặc định `resume=False` ⇒ **ghi đè** (idempotent cho ACCEPTANCE smoke).

### File phụ: `us_marks.csv.stats.json`

Lưu `rows, issues_processed, drop_reasons, by_drawing_kind, limit, since, until,
elapsed_sec` — để nhánh 04/06/12 đọc lại mà không cần chạy lại mạng.

### Số liệu thực đo (2026-10-13)

Một issue ~25.5k bản ghi thô; lấy 5000 dòng trong **41 s** (51 trang × 100).
Phân bố mẫu 5000 dòng: `standard_character 3414`, `design_with_words 916`,
`stylized_words 433`, `typeset_old 237`; rớt `too_long 440`, `design_without_word 250`,
`too_short 56`, `unsupported_char 28`. Ảnh CDN: 4/4 loại trả `200 image/png` (đã spot-check).
Toàn bộ 688 issue có thể cho vài triệu dòng; ACCEPTANCE chốt mục tiêu thực dụng 50k–150k.

### Nguồn bổ sung (tuỳ chọn, đã dò)

`illeatmyhat/uspto-trademarks` trên HuggingFace = **USPTO Trademark Case Files**
(14.347.764 đơn, 1870→2026-09-30, parquet, có `serial_no`, `mark_id_char`, `filing_dt`,
lookups `drawing_codes.csv`...). Truy vấn không cần tải: DuckDB + `hf://datasets/...`.
Dùng để **mở rộng phủ lịch sử** hoặc tra thêm trạng thái khi cần (không bắt buộc trong
pipeline vì ảnh trước ~2014 không có trên CDN).

**Test** (`tests/test_fetch_meta.py`, 22 test, offline bằng `FakeSession`): chuẩn hoá 2
thời kỳ schema, không ném lỗi khi thiếu field, mapping mã drawing, mọi nhánh lý do rớt,
phân trang/`per_issue_limit`/issue rỗng, lọc issue, `run()` end-to-end (CSV+stats,
limit, resume, shuffle tất định).

## Nhánh 03-fetch-uspto-images — `src/trademark_ocr/fetch_images.py`

**Trách nhiệm**: tải ảnh nhãn hiệu về đĩa theo cột ``image_url`` của nhánh 02, có
**resume**, **song song**, **kiểm tra ảnh bằng PIL**, và cập nhật trạng thái ảnh.

### Nguồn & phạm vi

URL ảnh: ``https://cdn.uspto.gov/img/{serial}/{publicationDate}/{size}`` — **công khai,
không API key**; ảnh có từ ~2014 (2013-08-13 → 404). ``size`` mặc định ``original``;
``--size px300x200|px160x130`` dùng bản nhỏ (URL đổi hậu tố cuối, tên file thêm hậu tố
``__{size}.png`` để không ghi đè bản gốc).

### Đầu ra

| File | Nội dung |
|---|---|
| ``$DATA_ROOT/raw/images_us/{serial}.png`` | ảnh (bản ``original``) |
| ``$DATA_ROOT/raw/images_us/index.csv`` | **sổ đăng ký ảnh** (nguồn sự thật) |
| ``$DATA_ROOT/raw/images_us/index.csv.stats.json`` | thống kê lần chạy |
| ``$DATA_ROOT/raw/meta/us_marks.csv`` | cột ``image_status`` đổi ``pending``→``ok``/``missing`` |

`index.csv` (14 cột): `serial_number, file_name, word_mark, mark_drawing_cd,
drawing_kind, publication_date, status, http_status, width, height, bytes, sha1,
reason, url`. `status ∈ {ok, missing}`; `reason` ghi mã lỗi
(`http_404`, `too_small`, `not_image:…`, `corrupt:…`, `error:…`).

### Hàm chính

| Hàm | Vào | Ra |
|---|---|---|
| `probe_image(path)` | – | `(ok, w, h, reason)` (PIL `verify()` + mở lại lấy size) |
| `sha1_file(path)` | – | hex sha1 |
| `download_one(url, dest, session, timeout, retries, backoff, min_bytes)` | – | dict trạng thái; ghi tmp→replace; **không ném lỗi** |
| `read_index / write_index` | – | đọc/ghi index (atomic) |
| `update_meta_status(meta_path, status_by_serial)` | map serial→status | số dòng đổi |
| `run(..., limit, workers, size, retry_missing, update_meta, fetch_fn)` | – | ghi ảnh + index + stats |
| `verify(index_path, img_dir, limit, min_ratio=0.95)` | – | `{total, ok, broken, ratio, verdict}` |

**Đặc tính kỹ thuật**:
- Session **thread-local** (`requests.Session` không thread-safe); tải bằng
  `ThreadPoolExecutor(max_workers=workers)`.
- `download_one` coi ảnh không hợp lệ nếu: HTTP≠200, ``< MIN_BYTES`` (100 B), Content-Type
  không chứa ``image``, hoặc PIL không mở/verify được → xoá file ``.part``.
- **Resume**: serial đã ``status=ok`` + file tồn tại → bỏ qua; serial từng lỗi chỉ tải lại
  khi bật ``--retry-missing``.
- ``--verify`` mở lại ảnh đã tải; in ``VERIFY OK``/``VERIFY FAIL`` (ngưỡng ≥95%) và trả
  **exit code 1** khi FAIL.

### CLI & script

```bash
bash scripts/02_get_images.sh --limit 200      # smoke 200 ảnh (~20s)
python3 -m trademark_ocr.fetch_images --verify --limit 200
python3 -m trademark_ocr.fetch_images --limit 200 --retry-missing --workers 16
```

### Số liệu thực đo

200 ảnh / **20 s** (workers=8, ~10 ảnh/s), dung lượng 9 MB → trung bình **~45 KB/ảnh**.
Chạy lại: ``downloaded: 0 new, skipped: 200 existing``. Verify: **200/200 = 100%**.

### Hạn chế

- Không xác nhận được *bằng mắt* nhãn↔ảnh (môi trường agent không đọc ảnh) — dựa trên
  giả định ảnh CDN gắn với đúng bản ghi gazette; **cần** kiểm tra tự động ở nhánh 06 (EDA:
  vài mẫu + nhãn) và nhánh 13 (phân tích lỗi).
- Chỉ có ảnh cho các nhãn đã công bố trong OG (2013+); nhãn cũ hơn không có ảnh CDN.

**Test** (`tests/test_fetch_images.py`, 16 test, offline): probe ảnh hợp lệ/lỗi, các nhánh
`download_one` (200/404/too_small/not_image/corrupt), `run()` ghi index + cập nhật meta,
resume không tải lại, ghi nhận `missing` + cờ `--retry-missing`, `--limit`, biến thể
`--size`, `verify` đạt/không đạt ngưỡng, index rỗng an toàn, `update_meta_status`.

## Nhánh 04-clean-label — `src/trademark_ocr/clean.py`

**Trách nhiệm**: chuẩn hoá nhãn + lọc mẫu không dùng được + **ghép metadata với ảnh**,
sinh tập "sạch" cho nhánh chia split.

### Đầu vào / đầu ra

| | Đường dẫn |
|---|---|
| vào | `$DATA_ROOT/raw/meta/us_marks.csv` (nhánh 02) + `raw/images_us/index.csv` & `{serial}.png` (nhánh 03) |
| ra | `$DATA_ROOT/processed/clean.csv` + `$DATA_ROOT/processed/clean_report.json` |

`clean.csv` (7 cột): `serial_number, image_path, word_mark, n_chars, drawing_kind,
mark_drawing_cd, publication_date`. ⚠️ `image_path` là **tương đối so với DATA_ROOT**
(`raw/images_us/{serial}.png`) để dùng được cả local và Colab (chỉ khác gốc DATA_ROOT).

### Thứ tự lọc & lý do rớt (cố định trong report)

`no_text` → `too_short` → `too_long` → `unsupported_char` → `missing_image` →
`bad_image` → `duplicate_label`. Report **luôn có đủ 7 khoá** (0 nếu không rớt) để
ACCEPTANCE/so sánh ổn định.

- `duplicate_label`: bỏ nhãn trùng (nhiều đơn cùng chữ, ví dụ "APPLE") — **giữ 1 mẫu**.
  Có thể tắt bằng `--no-dedupe`. (Dedup theo *chuỗi nhãn*, không theo serial.)
- `missing_image`/`bad_image`: ảnh chưa tải (ngoài `--limit` của nhánh 03) hoặc file hỏng
  (đã kiểm bằng `probe_image` của nhánh 03).

### Hàm chính

| Hàm | Vai trò |
|---|---|
| `normalize_label(text, charset)` | NFC + gộp space; trả `(label, reason)` |
| `clean_row(row, index, data_root, ...)` | lọc 1 dòng → `(clean_row, reason)` |
| `clean(meta_rows, index, data_root, ...)` | lọc tất cả → `(kept, report)` |
| `write_clean_csv(rows, path)` | ghi CSV nguyên tử |
| `run(meta_path, index_path, out_path, report_path, data_root, ...)` | điều phối + ghi report |

`report` gồm: `total, kept, keep_ratio, drop_reasons{7}, by_drawing_kind, label_len{min,max,mean}, params`.

### CLI

```bash
python3 -m trademark_ocr.clean --limit 5000
python3 -c "import json,os;r=json.load(open(os.environ['DATA_ROOT']+'/processed/clean_report.json'));print(r['keep_ratio'])"
```

### Số liệu thực đo (5000 dòng, 5000 ảnh có sẵn)

`kept 4840/5000`, **`keep_ratio = 0.968`**, chỉ rớt `duplicate_label: 160` (do nhánh 02
đã lọc sẵn các lý do text). Phân bố loại hình: `standard_character 3341`,
`design_with_words 853`, `stylized_words 410`, `typeset_old 236`; độ dài nhãn
min 2 / max 25 / mean 10.65.

**Test** (`tests/test_clean.py`, 11 test, offline): NFC+gộp space, lần lượt từng nhánh lý
do rớt, nhánh `missing_image`/`bad_image` (file hỏng), `require_image=False`,
dedupe bật/tắt, `--limit`, `run()` ghi CSV+report, roundtrip `write_clean_csv`.

## Nhánh 05-split-build_data — `src/trademark_ocr/split.py` + `src/trademark_ocr/build_lmdb.py`

**Trách nhiệm**: chia tập **không rò rỉ nhãn** rồi đóng gói thành LMDB đúng định dạng
`baudm/parseq` để train/eval.

### 5.1 `split.py` — chia theo nhóm nhãn

Vì nhiều đơn có **cùng chữ** (nhiều "APPLE"), chia theo *dòng* sẽ khiến cùng nhãn nằm ở
train lẫn test → điểm test ảo. Nên:

1. Gom dòng theo `word_mark` → "nhóm nhãn".
2. `random.Random(seed).shuffle(labels)` trên danh sách nhãn **đã sort** (ổn định).
3. Cắt `train_ratio` (0.9) / `val_ratio` (0.05) / phần còn lại `test_us`.

⚠️ Thực tế nhánh 04 đã `--dedupe` nên mỗi nhãn chỉ 1 dòng; cơ chế nhóm vẫn đúng nếu chạy
`--no-dedupe` (khi đó `n_labels < counts`).

Xuất: `processed/split_{train,val,test_us}.csv` (giữ nguyên `CLEAN_COLUMNS`) +
`processed/split_report.json` (`counts, n_labels, overlap{...,any_serial}, seed, ratios`).
Report luôn `overlap = 0` theo thiết kế.

### 5.2 `build_lmdb.py` — LMDB đúng chuẩn parseq

Khoá LMDB (khớp `strhub/data/dataset.py::LmdbDataset`):

| Khoá | Giá trị |
|---|---|
| `num-samples` | chuỗi số nguyên |
| `image-{idx:09d}` | bytes ảnh (ghi thẳng PNG gốc) |
| `label-{idx:09d}` | **UTF-8**, giữ NFC (config train đặt `normalize_unicode: false`) |

`idx` từ **1**. `map_size = max(1 GiB, tổng_bytes × 1.3 + 1 MiB)`.

**Ánh xạ đường dẫn** = đúng kỳ vọng của `SceneTextDataModule`:

| subset | thư mục LMDB | được đọc bởi |
|---|---|---|
| `train` | `<DATA_ROOT>/lmdb/train/<name>/data.mdb` (`name=trademark`) | `root_dir/train/<train_dir>` |
| `val` | `<DATA_ROOT>/lmdb/val/data.mdb` | `root_dir/val` (glob `**/data.mdb`) |
| `test_us` | `<DATA_ROOT>/lmdb/test/test_us/data.mdb` | `root_dir/test/<subset>` |
| `test_vn` | `<DATA_ROOT>/lmdb/test/test_vn/data.mdb` | như trên (nhánh 11) |

⇒ Khi train, đặt `root_dir = $DATA_ROOT/lmdb` và `train_dir = trademark`.

**Lưu ý `PYTHONPATH`**: `strhub` KHÔNG được cài như package (chỉ nằm ở
`third_party/parseq`); code chạy thật cần `export PYTHONPATH=$REPO_ROOT/third_party/parseq`
(hoặc `pip install -e third_party/parseq`). pytest tự thêm qua `pythonpath` trong
`pyproject.toml`.

### Hàm chính

| Module | Hàm |
|---|---|
| split | `group_by_label`, `split_rows(rows, ratios, seed)`, `run(...)`, `_overlap` |
| build_lmdb | `lmdb_path`, `split_csv_path`, `create_lmdb(rows, out_dir, data_root, map_size)`, `iter_lmdb`, `run(subset, ...)` |

### CLI

```bash
python3 -m trademark_ocr.split --seed 1111
python3 -m trademark_ocr.build_lmdb --subset train --limit 500
python3 -m trademark_ocr.build_lmdb --subset val   --limit 200
python3 -m trademark_ocr.build_lmdb --subset test_us --limit 200
pytest tests/test_lmdb.py -q
```

### Số liệu thực đo (4840 dòng clean)

`counts={train:4356, val:242, test_us:242}`, `n_labels={4356,242,242}`,
`overlap = 0` (mọi khoá). LMDB: train 500 / val 200 / test_us 200 mẫu; nạp lại bằng
`build_tree_dataset` của parseq → đúng số mẫu, nhãn `'OUSEN HOME'`, ảnh `(600×250)` PIL hợp lệ.

**Test** (`tests/test_lmdb.py`, 8 test): không trùng nhãn/serial, tất định theo seed, ghi
đủ file + report, roundtrip LMDB (NFC giữ dấu, decode ảnh), đúng layout parseq, `--limit`,
bỏ ảnh thiếu, và **tích hợp thật** với `strhub.data.dataset.LmdbDataset`.

## Nhánh 06-eda — `notebooks/01_eda_us.ipynb`

Notebook EDA (đã execute & commit kèm output). Đọc `processed/clean.csv` (nhánh 04) và các
`split_*.csv`, sinh hình vào `$DATA_ROOT/processed/eda_figures/`:

| Hình | Nội dung |
|---|---|
| `fig_len_hist.png` | phân bố `n_chars` |
| `fig_drawing_kind.png` | số mẫu theo `drawing_kind` |
| `fig_nwords.png` | phân bố số từ mỗi nhãn |
| `fig_img_size.png` | histogram width/height (mẫu ≤ 500 ảnh) |
| `fig_char_freq.png` | top 30 ký tự + đếm ký tự có dấu tiếng Việt |

Lưu ý kỹ thuật: dùng backend `Agg` + chỉ `savefig` (môi trường headless). Cấu hình qua env
`DATA_ROOT`. **Không** xác minh nhãn↔ảnh bằng mắt trong notebook (giới hạn môi trường) —
việc này tách sang nhánh 13 (error analysis) + kiểm tra thủ công vài mẫu.

**Số liệu thực đo**: 4840 dòng; `standard_character 3341 / design_with_words 853 /
stylized_words 410 / typeset_old 236`; mean words/label `1.692`; median ảnh `600×250`;
top ký tự `E,A,O,I,' ',R,...`; 1 ký tự có dấu tiếng Việt trong nhãn US.

## Nhánh 07-train-crnn — `train_utils.py` + `configs/hydra/tm_*.yaml` + `scripts/05_train.sh`

**Triết lý**: KHÔNG viết lại vòng train. Dùng đúng `third_party/parseq/train.py`
(`@hydra.main(config_path='configs', config_name='main')`), chỉ cung cấp config riêng qua
`--config-dir configs/hydra --config-name tm_crnn` (đã kiểm chứng CLI của parseq chấp nhận).

### Cấu trúc config (tự chứa, KHÔNG group `model/`)

`configs/hydra/tm_{crnn,trba,parseq,svtrctc}.yaml` (nút `svtr` → `tm_svtrctc`, nhánh 09) — mỗi file có đủ `model` / `data` /
`trainer` / `ckpt_path` / `pretrained` / `hydra`; chỉ dùng group **một** lần:
`defaults: [- _self_, - charset: charset_vi]` (charset group `@package _global_` nạp
`model.charset_train/test` — thứ tự quan trọng, `_self_` đứng trước).

| Khóa | Giá trị & lý do |
|---|---|
| `model.charset_*` | 217 ký tự tiếng Việt NFC (từ `charset/charset_vi.yaml`, nhánh 01) |
| `data.normalize_unicode` | **false** — đúng/NFKD-ASCII thì mất dấu |
| `data.remove_whitespace` | **false** — charset có space |
| `data.root_dir` | `${oc.env:DATA_ROOT,...}/lmdb` |
| `data.train_dir` | `trademark` (== `lmdb/train/trademark`, nhánh 05) |
| `data.augment` | `true` (RandAugment của parseq) |
| `trainer.val_check_interval` | `1.0` — 4356 mẫu / bs 64 ≈ 68 step/epoch (mặc định parseq 1000 → chậm) |
| `trainer.limit_train_batches` / `limit_val_batches` / `max_steps` / `log_every_n_steps` | khai báo sẵn để script **ghi đè không cần `+`**; `log_every_n_steps=1` để loss hiện cả khi smoke-test 2 batch |
| `trainer.accelerator/devices` | `gpu`/`1` (Colab); local đổi bằng `--accelerator cpu` |
| `hydra.run.dir` | `$DATA_ROOT/runs/${model.name}/${now:...}` → checkpoint + tb events + `config/config.yaml` |

### Shim NumPy-2 cho `imgaug` (quan trọng)

`imgaug==0.4.0` dùng `np.sctypes` **bị xóa ở NumPy 2.0** → RandAugment của parseq chết.
Không hạ `numpy<2` (vỡ torch/pyarrow) và không sửa parseq → khôi phục biến thể đó:

- `src/trademark_ocr/_numpy2_compat.py` — `apply_numpy2_compat()` (idempotent).
- `compat/sitecustomize.py` — `site` tự import lúc khởi động **nếu** `compat/` nằm trong
  `PYTHONPATH` → `scripts/05_train.sh` set sẵn; `train_utils` cũng tự gọi khi import.

### `train_utils.py` — compose + dry-run

| Hàm | Việc |
|---|---|
| `compose_config(name, overrides)` | hydra `initialize_config_dir('configs/hydra')` + `compose` |
| `dry_run(cfg, num_batches, num_images, device)` | instantiate model+datamodule, 1 batch, `forward_logits_loss` + `backward`, decode thử |
| `train_command(config, extra)` | in lệnh train thật (parseq entrypoint) |
| `available_configs()` | liệt kê `configs/hydra/tm_*.yaml` |

⚠️ **Dry-run phải chạy ở `model.eval()`**: với TRBA, nhánh `training` của
`trba/prediction.py:Attention.forward` đòi `text` 2D nhưng `TRBA.forward` tạo `text` 1D
→ chỉ hợp lệ khi eval (đúng đường parseq dùng lúc validate). Đây là bẫy có thật đã bắt được.

### `scripts/05_train.sh <model> [flag...]`

Dịch flag argparse → hydra override: `--limit-train-batches/--limit-val-batches/
--max-epochs/--max-steps/--accelerator/--devices/--batch-size/--lr/--num-workers`;
`crnn|trba|parseq|svtr` → config `tm_*`; set `PYTHONPATH=compat:src:third_party/parseq`.

### Số liệu thực đo

| Config | Tham số | dry-run loss (16 ảnh, eval) | train loss 2 batch |
|---|---|---|---|
| `tm_crnn` | 8 424 026 | 17.37 | **20.29 → 19.50 → 19.25** (giảm) |
| `tm_trba` | 49 883 236 | 5.39 | — |
| `tm_parseq` | 23 927 258 | 5.45 | — |

Smoke `bash scripts/05_train.sh crnn --limit-train-batches 2 --max-epochs 1 --accelerator cpu`
→ exit 0, in model summary, chạy val, tạo
`$DATA_ROOT/runs/crnn/<ts>/checkpoints/{last.ckpt, epoch=...-val_accuracy=...ckpt}`
(+ `config/config.yaml`, `events.out.tfevents.*`, `train.log`). TB scalars đọc được:
`loss, val_loss, val_accuracy, val_NED, hp_metric`.

## Nhánh 08-train-parseq — TRBA + PARSeq

Config `tm_trba` / `tm_parseq` đã có từ nhánh 07; nhánh này **nghiệm thu** và khoá lại
bằng test forward thật.

### `tests/test_forward.py` (11 test)

Parametrize theo `tm_crnn`, `tm_trba`, `tm_parseq` (model dựng 1 lần, `scope=module`,
`model.batch_size=2`):

| Test | Kiểm |
|---|---|
| `test_forward_logits_shape` | `(N, L, 3)` với `N=2`, `C > len(CHARSET_VI)` |
| `test_loss_non_negative` | `forward_logits_loss` hữu hạn, `≥ 0`, `numel > 0` |
| `test_decode_returns_labels_in_charset` | ký tự decode **thuộc** `CHARSET_VI`; xác suất ∈ [0,1] |
| `test_decode_length_bounded_by_timesteps` | `len(pred) ≤ L` (số timestep của logits) |

Lưu ý kỹ thuật: `tokenizer.decode` trả `(preds, probs)` với `probs` là **list tensor** (không phải tensor).

### Smoke train thật trên CPU (2 batch)

| model | lệnh thêm | kết quả | peak RSS |
|---|---|---|---|
| `trba` | `--batch-size 8 --num-workers 0` | exit 0, `runs/trba/<ts>/checkpoints/last.ckpt` | 2.88 GB |
| `parseq` | `--batch-size 8 --num-workers 0` | exit 0, `runs/parseq/...` | 2.28 GB |

⚠️ **OOM với `--batch-size 64` trên máy 7.6 GB RAM** (exit 137, bị OOM-killer).
- Local smoke: luôn kèm `--batch-size 8 --num-workers 0` (worker copy thêm RAM).
- Colab (T4, ~12 GB): batch 64 bình thường; nếu OOM → hạ `model.batch_size`.

## Nhánh 09-train-svtrv2 — SVTR-CTC tự viết (`src/trademark_ocr/svtr_ctc.py`)

Model so sánh **thứ 4** — cài đặt như `strhub.models.base.CTCSystem` (không sửa parseq)
nên dùng lại y nguyên vòng train/eval/metric + decode CTC của parseq.

### Kiến trúc `SVTRNet` (bản "SVTR-Lite" tự viết)

| Thành phần | Chi tiết |
|---|---|
| `PatchEmbed` | 2× `Conv3×3 s2 + BN + GELU` → giảm 4× (từ `[32,128]` → `8×32`) |
| `MixingBlock` | `LayerNorm → LocalMixing (depthwise conv 3×3) + GlobalMixing (MultiheadAttention) → LayerNorm → MLP`, có residual + DropPath |
| `Merging` | `Conv3×3 s2 + BN` → giảm 2× (→ `4×16`) |
| Head | `LayerNorm → Linear(dim, num_classes)` theo từng vị trí (CTC) |

Cấu hình: `embed_dim=192`, `depths=[2,2]`, `num_heads=6`, `mlp_ratio=4`,
`drop_path_rate=0.1`, `num_classes = len(tokenizer) = 218`.

**Bất biến sống còn của CTC**: số timestep `L` phải `≥ max_label_length` (25). Với
`img_size=[32,128]` ⇒ `L = 4×16 = 64`. Class **tự kiểm tra** lúc `__init__` bằng 1 forward
dummy và `assert L >= max_label_length` (báo lỗi rõ nếu vi phạm), đồng thời lưu `self.seq_len`.

> ⚠️ Đã sửa ở **16.2**: bản này trải phẳng lưới 4×16 theo hàng nên thực chất chỉ có 16 cột
> < 25 ký tự. Hiện tại `Merging` stride (2,1) + gộp chiều cao → `L = W/4 = 32` cột.

### Bẫy đã bắt được

`CTCSystem`/`BaseSystem` **không** định nghĩa `training_step` (CRNN và TRBA tự khai báo).
Thiếu nó ⇒ Lightning báo *"No `training_step()` method defined"*. → `SVTRCTC.training_step`
được viết giống CRNN: `loss = forward_logits_loss(images, labels)[1]; self.log('loss', loss)`.

### Số liệu thực đo

| | |
|---|---|
| Tham số | **2 330 042** (~2.3 M — nhẹ hơn CRNN 8.4 M) |
| `seq_len` | 64 |
| dry-run loss (16 ảnh, eval) | 35.25, grad OK, decode trả ký tự trong charset |
| smoke train 2 batch CPU | **exit 0**, peak RSS 1.21 GB, ~18 s → `runs/svtrctc/<ts>/checkpoints/last.ckpt` |

### `tests/test_svtr.py` (9 test)

`isinstance(CTCSystem)` + `blank_id` khớp; forward shape `(2, 64, >217)`; `seq_len ≥ max_label_length`;
loss hữu hạn `≥0`, `numel==N`; decode ký tự thuộc charset; số tham số trong khoảng hợp lý;
`training_step` có trên class; và **test overfit** (`SVTRNet` + AdamW 40 step → loss giảm > 30%).

## Nhánh 10-eval-metrics — `src/trademark_ocr/metrics.py` + `evaluate.py`

### Metric (`metrics.py`) — thuần Python, không phụ thuộc ngoài

| Metric | Công thức |
|---|---|
| `accuracy` | tỉ lệ `pred == gt` (phân biệt hoa/thường & dấu) |
| `ned` | trung bình `levenshtein(pred, gt) / max(\|pred\|, \|gt\|)` (chuẩn ICDAR) |
| `one_minus_ned` | `1 − ned` |
| `cer` | mức corpus: `Σ edit / Σ \|gt\|` |
| `char_accuracy` | `1 − cer` |

`edit_distance` viết tay (DP 2 hàng) → **không cần `python-Levenshtein`** (chạy offline/Colab).
Hàm: `evaluate_metrics(preds, gts) -> dict`, `per_sample(pred, gt) -> dict` (khớp
`REPORT_COLUMNS = idx, pred, gt, correct, ned, char_err, pred_len, gt_len`), `format_summary`.
⚠️ CER **không bị chặn** ở 1 → `char_accuracy` có thể **âm** khi model chưa train
(pred dài hơn gt). Đúng về toán, đã thấy thực tế (`char_acc=-0.13` với ckpt 2 batch).

### `evaluate.py`

| Hàm | Việc |
|---|---|
| `lmdb_dir(root, subset)` | `val → lmdb/val`, `test_* → lmdb/test/<subset>` |
| `load_model(ckpt, device)` | parseq `_get_model_class(...).load_from_checkpoint` (crnn/trba/parseq/...) hoặc fallback `SVTRCTC.load_from_checkpoint` (model tự viết) |
| `build_loader(subset, ...)` | `LmdbDataset` + `SceneTextDataModule.get_transform((32,128), augment=False)` (không augment, có tensor) |
| `iter_predictions` | `model(images) → softmax → tokenizer.decode → charset_adapter(pred)` (giống parseq `_eval_step`) |
| `run(...)` | ghi CSV từng mẫu + `<out>.summary.json`, in `format_summary` |

Chuẩn tên báo cáo: `reports/<model>_<domain>.csv` (`crnn_us.csv`, `crnn_vn.csv`, ...) —
nhánh 12 ghép cặp theo `<model>`. `reports/*.csv|*.json` đã **gitignore**.

### Số liệu thực đo (ckpt crnn mới train 2 batch → cố ý kém)

`--ckpt .../crnn/.../last.ckpt --subset test_us --out reports/crnn_us.csv` →
`n=200 acc=0.0000 1-NED=0.0233 char_acc=-0.1266 CER=1.1266`; CSV có đủ cột
`idx,pred,gt,correct,ned,char_err,pred_len,gt_len`. `--help` trả exit 0.

### `tests/test_metrics.py` (19 test)

`edit_distance` 10 ví dụ kinh điển (kể cả `kitten/sitting=3`, `ĐÔNG Á/DONG A=3`, đối xứng);
`ned`/`cer` canonical; aggregate (perfect, known values, rỗng, lệch độ dài → `ValueError`);
`per_sample` khớp cột; CLI `--help` exit 0; mapping `lmdb_dir`; và **end-to-end trên LMDB
thật** (val, 32 mẫu) → CSV đúng cột.

## 11-fetch-vn — thu nhãn hiệu Việt Nam (WIPO Publish / IP Vietnam)

Nguồn: `https://wipopublish.ipvietnam.gov.vn/wopublish-search` — đây là nền **WIPO
Publish** (Wicket/AJAX), chứng chỉ TLS tự ký → **phải `verify=False`** (như `curl -k`).
Kết quả **không** có trong HTML đầu; phải đi qua 4 bước (đã kiểm chứng thực tế):

1. `GET /public/trademarks?query=*:*` → lấy phiên + đọc trong HTML:
   - `id` form: `"f":"id6"` — **đổi theo phiên** (không hard-code `id12`),
   - URL hành vi tìm kiếm: `"u":"./trademarks;jsessionid=...?0-1.IBehaviorListener.0-body-basicSearchTab-searchInputPanel-searchForm-searchSubmitLink&query=*:*"`,
   - `Wicket.Ajax.baseUrl` → dùng cho header `Wicket-Ajax-BaseURL`.
2. `POST` URL hành vi (header `Wicket-Ajax: true`) với body
   `<fid>_hf_0=&intelligentSearchInput=&searchSubmitLink=Tra+cuu`
   → XML `<ajax-response><component id="searchResultPanel">` chứa **50 hàng/trang**.
3. Phân trang: panel có behavior `navigator-next` (`"u":"...navigator-next..."`, GET lại).
4. Ảnh: `/public/service/trademarks/application/<id>/thumbnail?noLogo=true&disclaimer=Combined`
   → **JPEG** (vd `VN4202645978` → 200×47). Field chữ cần cho OCR là **`rs-MK`**.

### Bản đồ field trong 1 hàng `<tr id="VN...">`

`rs-MK` = word_mark · `rs-AFNB_ORI` = số đơn · `rs-AFDT` = ngày nộp · `rs-PBDT` = ngày
công bố · `rs-RENB` = số đăng ký · `rs-APNA` = chủ đơn · `rs-NCL` = nhóm Nice · `rs-STLB` = trạng thái.

### Hàm (offline-testable)

`parse_form(html) -> (fid, submit_url, base_url)`;
`parse_rows(html) -> list[dict]`; `usable(row)` (có chữ **và** ảnh);
`next_url(html)`; `to_label(mk)` (unescape HTML → strip tag → **NFC**);
`VnClient` (`warmup/search/fetch_page/download/iter_rows`); `verify_images(dir)`;
`write_outputs(rows, data_root)`.

### Sản phẩm (dưới `DATA_ROOT`)

- `raw/vn_marks.csv` — bản ghi thô; `raw/images_vn/<id>.jpg` + `raw/vn_report.json`
- `processed/split_test_vn.csv` — cột `serial_number,image_path,word_mark,n_chars`
  (khớp `build_lmdb`, cần `image_path`+`word_mark`)
- `processed/test_vn.txt` — mỗi dòng 1 nhãn NFC (dùng cho ACCEPTANCE)

### Số liệu thực đo

`bash scripts/03_get_vn.sh --limit 50` → 50 nhãn (trang 1 có 46 hàng usable, lấy thêm
trang 2), **50/50 ảnh mở được (100%)**, `all NFC: True`, mean label len 19.58.
`python3 -m trademark_ocr.build_lmdb --subset test_vn` → `num_samples=50`.
⚠️ Ghi chú: `run()` mặc định `log=print` (từng để no-op khiến CLI im lặng — đã sửa).

### `tests/test_vn.py` (11 test, offline với fixture thật `tests/data/vn_panel_sample.html`)

`parse_form` id thay đổi theo phiên + thiếu form → `RuntimeError`; `parse_rows` trên
fixture (3 hàng, id/word_mark/afnb/image_url); `next_url`; `usable` lọc ảnh-không-chữ;
`to_label` NFC (`A\u0301`→`Á`); `write_outputs` (đúng cột split, txt khớp, report);
`verify_images` (3 ảnh + 1 file hỏng → ratio 0.75); CLI `--help` exit 0.
⚠️ `.gitignore`: `data/` khớp `tests/data/` → đã thêm `!tests/data/**`.

## 12-eval_domain_shift — bảng sụt giảm US→VN

`src/trademark_ocr/eval_domain_shift.py` đọc `reports/<model>_<domain>.csv`
(do nhánh 10 ghi; `domain ∈ {us, vn}`), ghép theo `<model>` và tính
``delta = acc_vn − acc_us`` (**âm = sụt độ chính xác**). Độ chính xác lấy từ cột
``correct``; nếu thiếu thì suy từ ``pred == gt``.

`discover(dir)` → `{model: {domain: path}}` (bỏ `domain_shift.csv`, `*.summary.json`,
file lạ). `read_accuracy(path) -> (acc, n)`. `build_rows(dir)` bỏ model thiếu 1 domain.
`run(reports_dir, out_path)` ghi CSV cột `model,acc_us,acc_vn,delta,n_us,n_vn`, **sort
theo delta tăng dần** (sụt nặng nhất lên đầu).

### Số liệu thực đo (ckpt crnn chưa train)

`crnn` trên `test_us` (n=200, acc 0) và `test_vn` (n=36, acc 0) → `delta=0.0`.
⚠️ **test_vn: LMDB 50 mẫu nhưng chỉ 36 vào được dataloader** — vì nhãn VN dài hơn
(mean 19.58 ký tự; vd `KP-Food TINH HOA TỪ BIỂN - TRỌN VỊ QUÊ HƯƠNG` = 45) nên bị
`max_label_length=25` lọc. Đây là tín hiệu domain shift thật (VN = nhãn dài + có dấu).

### `tests/test_domain_shift.py` (7 test, offline)

`read_accuracy` (ưu tiên `correct`, fallback `pred/gt`); `discover` bỏ file lạ/`*.json`/
`domain_shift.csv`; `build_rows` tính delta + bỏ model thiếu domain; `run` ghi đủ cột +
sort; mọi report rỗng → chỉ header; CLI `--help` exit 0.

## 13-error-analysis — phân tích lỗi

`src/trademark_ocr/error_analysis.py` đọc báo cáo nhánh 10 (`reports/<model>_<domain>.csv`)
và sinh 3 hình vào `$DATA_ROOT/processed/eda_figures/`:

| Hàm thuần | Ý nghĩa |
|---|---|
| `discover_reports(dir)` | các CSV `<model>_<domain>.csv` (bỏ `domain_shift.csv`) |
| `load_pairs(paths)` | list `(pred, gt)` |
| `by_length(pairs, bins)` | `n, acc, char_acc` theo độ dài **gt** |
| `top_errors(pairs, k)` | k mẫu sai nặng nhất (edit distance giảm dần) |
| `char_confusions(pairs)` | cặp `(gt_char, pred_char)` hay thay thế (difflib op `replace`) |

Vẽ hình: `plot_by_length` → `err_by_length.png`; `plot_top_errors` → `err_top_samples.png`;
`plot_confusions` → `err_char_confusions.png`. `analyze(...)` gom tất cả + trả dict.
Backend `Agg` (headless). ⚠️ **Không dùng font monospace** cho text có dấu — DejaVu Sans
Mono thiếu glyph tiếng Việt (warning + ô vuông); DejaVu Sans (mặc định) hiển thị đủ.

`notebooks/02_error_analysis.ipynb` (đã execute & commit kèm output, 9 cell): in danh sách
báo cáo, metrics tổng, bảng theo độ dài / top lỗi / cặp thay thế, và hiển thị 3 hình inline.
Notebook tự thêm `src/` vào `sys.path` và chịu được trường hợp `reports/` rỗng (in cảnh báo).

### Số liệu thực đo (ckpt crnn chưa train, US+vn 236 mẫu)

`acc=0.0000 CER=1.1203`, 12 top lỗi, 15 cặp thay thế; 3 hình PNG (~27–74 KB).

### `tests/test_error_analysis.py` (7 test, offline)

`discover`/`load`; `by_length` chia bucket đúng (acc 0.0 bucket 3–9 vì cả 2 sai);
`top_errors` sort + lọc mẫu đúng; `char_confusions` đếm đúng (`A→X` = 2); `analyze` sinh
đủ 3 file `err_*.png`; thư mục rỗng → `{"n":0,...}`; CLI `--help` exit 0.

## 14-demo-report — demo + báo cáo tổng hợp

### `demo_app.py` (suy luận ảnh → chữ)

| Hàm | Việc |
|---|---|
| `load_transform(img_size=(32,128))` | transform như lúc train (`SceneTextDataModule.get_transform`, không augment) |
| `preprocess(paths)` | list ảnh → tensor `(B,3,H,W)` |
| `decode_predictions(model, logits)` | softmax → `tokenizer.decode` → `charset_adapter` → `[(pred, conf)]` |
| `predict(model, paths, device)` | `model.eval()` + `inference_mode` → `[{image,pred,confidence}]` |
| `run(...)` | nạp ckpt (dùng lại `evaluate.load_model`) rồi `predict` + in kết quả |
| `build_interface(model)` | Gradio `Interface` (`--serve`, `--share`) |

`--help` chạy được **không cần ckpt**; thiếu `--ckpt`/`--image` thì in hướng dẫn (`SystemExit`).

### `report.py` (báo cáo markdown)

`collect(reports_dir)` tính lại metric từ `pred/gt` (không tin cột sẵn) cho mọi
`<model>_<domain>.csv`; `read_shift()` đọc `domain_shift.csv`; `list_figures()` gom
`fig_*.png` + `err_*.png`; `render_markdown()` tạo 3 mục: **bảng model × dataset**
(n, accuracy, 1−NED, char-accuracy), **bảng sụt giảm US→VN**, **danh sách hình**.
`run()` ghi `reports/final_summary.md`. ⚠️ đã sửa lỗi thiếu `import sys` ở `__main__`.

`demo_images/demo_us.png` + `demo_images/demo_vn.png` (ảnh mẫu commit sẵn) để lệnh
ACCEPTANCE `--image demo_images/*.png` chạy ngay.

### `scripts/06_eval.sh` (đánh giá end-to-end)

Tiện ích: với mỗi model có checkpoint trong `$DATA_ROOT/runs/<model>/*/checkpoints/last.ckpt`
→ `evaluate` trên `test_us` + `test_vn` → `eval_domain_shift` → `error_analysis` →
`report`. Cờ: `--only crnn,trba`, `--limit N`, `--device`, `--batch-size`, `--num-workers`
(mặc định CPU + limit 0). Cần `PYTHONPATH=src:third_party/parseq` (script tự export).

    bash scripts/06_eval.sh --only crnn --limit 32 --batch-size 8 --num-workers 0


### Số liệu thực đo

`demo_app --help` → `CLI_OK`; `--ckpt <parseq> --image demo_images/*.png` chạy (ckpt chưa
train → chuỗi rác, đúng như dự kiến). `report` → `final_summary.md` có bảng
`crnn | US | 200 | 0.0000 | 0.0233 | -0.1266` và `crnn | VN | 36 | ...`.

### `tests/test_demo_report.py` (8 test, offline)

`demo_app`: `--help` exit 0; `predict` với **stub model** (2 ảnh PNG thật qua PIL);
`run` thiếu ckpt/ảnh → `SystemExit`; `run` với stub in ra dự đoán.
`report`: `collect`+`render` (đúng bảng, US/VN, dòng shift), thư mục rỗng vẫn ghi md
(có câu "Chưa có báo cáo nào"), liệt kê `fig_*`/`err_*` (bỏ `.txt`), CLI `--help` exit 0.

## 15-review-tests — review + test khoá các phần đã chạy đúng

`REVIEW.md` ghi nhận định review (đối chiếu `detai.txt`): mức hoàn thành, điểm mạnh, 8 nhóm
vấn đề xếp theo mức nghiêm trọng kèm cách tái hiện. Mục §6 của file đó là nơi cập nhật trạng
thái xử lý.

Nhánh này **không sửa code chạy thật**; chỉ thêm test cho những hành vi đã đúng, để các nhánh
sửa sau không làm hỏng chúng:

| File | Khoá điều gì |
|---|---|
| `tests/test_pipeline_e2e.py` | chuỗi `clean.run → split.run → build_lmdb.run → SceneTextDataModule` (config `tm_crnn` thật) trên dữ liệu giả: đếm đúng từng lý do rớt, số mẫu split = số mẫu LMDB, nhãn có dấu/space/dấu câu đi qua cả chuỗi không đổi, batch `(4,3,32,128)` trong `[-1,1]`, và đường dẫn ảnh tương đối vẫn build được sau khi chuyển `DATA_ROOT` |
| `tests/test_parseq_contracts.py` | mọi `tm_*.yaml`: khối `model` và `data` khớp nhau (charset, `max_label_length`, `img_size`, `batch_size`), cả hai cờ tiếng Việt tắt, layout LMDB; chứng minh bằng `LmdbDataset` thật vì sao phải tắt (`remove_whitespace` → `ĐÔNGÁ`, `normalize_unicode` → `ONG A`); `CharsetAdapter` giữ hoa/thường; CRNN có ≥ `max_label_length` bước CTC; shim NumPy-2 cho RandAugment |
| `tests/test_charset.py` (+3) | phủ toàn bộ khối Unicode U+1EA0–U+1EF9 + chữ Việt ở Latin-1/Ext-A; đúng 134 chữ Việt, không có dấu tổ hợp rời; nhãn VN thật chỉ hợp lệ sau NFC |
| `tests/test_metrics.py` (+4) | `edit_distance` khớp định nghĩa đệ quy trên chuỗi ngẫu nhiên (có chữ Việt), là metric (đối xứng, bất đẳng thức tam giác), NED ∈ [0,1]; accuracy hiện so khớp tuyệt đối |
| `tests/test_vn.py` (+2) | `VnClient.iter_rows` với session giả: form id đọc từ HTML, header Wicket, `verify=False`, dừng khi đủ `limit` mà không lật trang thừa, phân trang không lặp mẫu |

Số mẫu trong test e2e cố ý < 1.000 (giới hạn của `create_lmdb`, REVIEW.md §4.1).

## 16-fix-review — xử lý các vấn đề trong REVIEW.md

Mỗi mục dưới đây ứng với một mục của REVIEW.md §4 và một commit trên nhánh này.

### 16.1 `build_lmdb.create_lmdb` — commit theo lô (REVIEW §4.1)

Transaction được quản lý tay: `txn = env.begin(write=True)` → commit mỗi `commit_every`
mẫu → lô cuối ghi `num-samples` rồi commit → `abort()` nếu có ngoại lệ → `sync/close` trong
`finally`. Không dùng `with env.begin(...)`: `with` chỉ đóng transaction ban đầu, nên bản cũ
mất lô cuối + `num-samples` khi có ≥ `commit_every` (1000) mẫu.

`num-samples` cố ý nằm ở lô cuối: dừng giữa chừng thì LMDB không có khoá này và parseq báo
lỗi ngay, thay vì âm thầm đọc một tập thiếu mẫu.

Test hồi quy (`tests/test_lmdb.py`): `commit_every=3` với 7 mẫu (đọc lại đủ, đúng thứ tự)
và 1.500 mẫu với mặc định 1.000, đọc bằng `LmdbDataset` thật của parseq.

### 16.2 `svtr_ctc.SVTRNet` — trục thời gian CTC theo chiều ngang (REVIEW §4.4)

| Thành phần | Trước | Sau |
|---|---|---|
| `Merging` | `Conv3×3 stride 2` (giảm cả rộng) | `Conv3×3 stride (2,1)` — chỉ giảm cao |
| Sau các stage | `view(b, h*w, c)` (trải theo hàng) | `mean(dim=2)` — gộp cao về 1, mỗi cột = 1 bước |
| Vị trí | không có | `pos_embed` học được `(1, C, ceil(H/4), ceil(W/4))` cộng sau stem |
| `L` @ 32×128 | 64 (= 4 lần quét × 16 cột) | **32 cột** |

`L` giờ chỉ phụ thuộc chiều rộng (`ceil(W/4)`): ảnh cao hơn không thêm bước, ảnh rộng gấp đôi
thì gấp đôi bước. `assert L >= max_label_length` vì thế mới thực sự kiểm số cột — `img_size`
32×64 (16 cột) bị từ chối lúc dựng model. Tham số: 2.338.085 (thêm ~49k của `pos_embed`).
Checkpoint SVTR cũ (chỉ có bản smoke 2 batch) **không** nạp được vào kiến trúc mới.

Test hồi quy (`tests/test_svtr.py`): `L` theo chiều rộng ở 3 kích thước, từ chối 32×64,
có `pos_embed`; hai test cũ khoá `seq_len == 64` đổi thành 32.

### 16.3 Đánh giá trung thực: ba mức so khớp, không lọc ngầm, truy ngược được mẫu (REVIEW §4.2, §4.5, §4.8)

**`metrics.py`** — hai mức so khớp thêm cho accuracy và 1 − NED:

| mức | chuẩn hoá áp lên cả pred và gt | khoá |
|---|---|---|
| strict | không | `accuracy`, `one_minus_ned` (như cũ) |
| nocase | `str.upper` | `accuracy_nocase`, `one_minus_ned_nocase` |
| base | `strip_diacritics(...).upper()` (`đ`→`d`) | `accuracy_base`, `one_minus_ned_base` |

Lý do: nhãn USPTO luôn VIẾT HOA còn nhãn VN thì không, và tập US gần như không có chữ có
dấu. `nocase` loại phần sai do quy ước ghi nhãn; `base` loại thêm phần sai do chưa từng học
chữ có dấu — phần còn lại mới là khác biệt hình ảnh. `has_diacritics(gt)` cho phép tách tập
VN thành nhóm có dấu / không dấu.

`REPORT_COLUMNS` (CSV từng mẫu) giờ là:
`idx, serial_number, drawing_kind, pred, gt, correct, correct_nocase, correct_base, ned,
char_err, pred_len, gt_len, gt_has_diacritics, in_scope, confidence`.
`per_sample(pred, gt)` điền mọi cột trừ `SAMPLE_CONTEXT_COLUMNS` (idx, serial, kind,
in_scope, confidence — do `evaluate` điền). Bên đọc (nhánh 12/13/14) đọc theo **tên cột**.

**`build_lmdb.py`** — thêm khoá `meta-{idx:09d}` = JSON `{serial_number, drawing_kind}`
(`META_FIELDS`, chỉ trường có giá trị); `read_meta(txn, idx)` trả `{}` với LMDB cũ. parseq
không đọc khoá này.

**`evaluate.py`** — viết lại phần nạp dữ liệu và nạp model:

| Trước | Sau |
|---|---|
| `LmdbDataset` của parseq: bỏ nhãn > 25 ký tự, xoá ký tự ngoài charset khỏi gt | `LmdbEvalDataset`: **mọi** mẫu, gt gốc (NFC); `in_scope = len(gt) ≤ max_len ∧ set(gt) ⊆ charset` |
| `idx` = thứ tự sau lọc | `idx` = chỉ số LMDB; `serial_number`, `drawing_kind` từ `meta-*` |
| `(32,128)` và `25` viết cứng | `model.hparams.img_size / max_label_length / charset_train` |
| `except Exception` → thử `SVTRCTC`; đoán lớp theo chuỗi con đường dẫn | `resolve_model_class`: `--model-class` → `<run>/config/config.yaml` (`model._target_`) → quy ước tên parseq → `SystemExit` có hướng dẫn |

`summary.json` thêm: `coverage`, `n_too_long`, `n_unsupported_char`, `in_scope{metrics}`,
`model_class`, `img_size`, `max_label_length`, và các khoá `*_nocase`/`*_base`. Metric chính
tính trên **toàn bộ** tập (mẫu ngoài phạm vi tính là sai); `in_scope` để so với số liệu kiểu
parseq.

**`demo_app.py`** — `model_img_size(model)` đọc `hparams.img_size`; bỏ `DEFAULT_IMG_SIZE`.

Test: `tests/test_evaluate.py` (mới, offline: ckpt SVTR nhỏ thật + run dir giả có config
hydra, LMDB VN có nhãn quá dài và ký tự lạ, `img_size` 32×160 để chứng minh đọc hparams),
`tests/test_metrics.py` (+5, bỏ test cũ chỉ chạy với LMDB thật và API cũ), `tests/test_lmdb.py`
(+1 meta), `tests/test_demo_report.py` (+1 resize theo hparams).
