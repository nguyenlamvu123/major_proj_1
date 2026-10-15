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
