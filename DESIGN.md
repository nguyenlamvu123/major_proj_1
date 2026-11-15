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
