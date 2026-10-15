# MAJOR_PROJ_1 — Nhận diện chữ trong nhãn hiệu sản phẩm (Text Recognition)

Đồ án: nhận diện chữ (wordmark) trong nhãn hiệu sản phẩm bằng học sâu có giám sát.
- **Train**: ảnh nhãn hiệu đăng ký tại Mỹ (USPTO) + nhãn chữ tương ứng.
- **Đánh giá mới**: ảnh nhãn hiệu đăng ký tại Việt Nam (IP Vietnam / WIPO Publish).
- **Khung mã**: [baudm/parseq](https://github.com/baudm/parseq) (clone vào `third_party/parseq`, không fork, không sửa).
- **Bảng ký tự**: Unicode có dấu tiếng Việt (NFC).

> Tài liệu kỹ thuật chi tiết (mô tả code cho người/đọc giả/AI): xem **[DESIGN.md](DESIGN.md)**.
> Lệnh nghiệm thu từng nhánh: xem **[ACCEPTANCE.md](ACCEPTANCE.md)**.

## Cấu trúc

```
major_proj_1/
├── DESIGN.md               # mô tả chi tiết code (đọc trước khi sửa)
├── ACCEPTANCE.md           # lệnh nghiệm thu 15 nhánh
├── README.md               # file này — hướng dẫn chạy
├── requirements.txt
├── pyproject.toml          # pip install -e .  → package `trademark_ocr`
├── src/trademark_ocr/      # toàn bộ code của đồ án
├── configs/                # charset + hydra experiment configs
│   ├── charset_vi.yaml     # bảng ký tự VN (tự sinh từ src/trademark_ocr/charset.py)
│   └── hydra/              # config huấn luyện (crnn/trba/parseq/svtrctc)
├── scripts/                # pipeline shell (01..08) + Colab helper
├── notebooks/              # EDA, error analysis
├── docs/colab/             # notebook/cell Colab copy-paste
├── tests/                  # pytest
├── reports/                # bảng biểu kết quả (csv/json/figures)
└── third_party/parseq/     # clone baudm/parseq (gitignored)
```

## Thiết lập môi trường

```bash
# 1) cài package của đồ án + deps
python -m venv .venv && source .venv/bin/activate   # (tuỳ chọn)
pip install -r requirements.txt
pip install -e .

# 2) clone khung parseq (nếu chưa có)
git clone https://github.com/baudm/parseq third_party/parseq

# 3) chạy test nhanh
pytest -q
```

## Nguồn dữ liệu

- **DATA_ROOT** (mặc định ổ đầy đủ dung lượng):
  `export DATA_ROOT=/media/zaibachkhoa/AC84-F995/trademark_data`
  Trên Colab: `DATA_ROOT=/content/drive/MyDrive/trademark_data` (mount Drive).
- Metadata USPTO (text): `scripts/01_get_meta.sh` → `$DATA_ROOT/raw/meta/`
  (nguồn TMOG / EOG REST API công khai, **không cần API key** — chi tiết DESIGN.md nhánh 02)
- Ảnh USPTO: `scripts/02_get_images.sh` → `$DATA_ROOT/raw/images_us/`
- (Hoặc tải bằng Colab: `docs/colab/01_download_data.ipynb`)

## Huấn luyện (Google Colab, GPU T4)

```bash
# clone repo + mount Drive, cài deps  →  docs/colab/00_setup.ipynb
bash scripts/05_train.sh crnn      # baseline CRNN+CTC
bash scripts/05_train.sh trba      # TRBA (attention)
bash scripts/05_train.sh parseq    # PARSeq (transformer)
bash scripts/05_train.sh svtrctc   # SVTR-CTC (CTC transformer)
bash scripts/06_eval.sh            # đánh giá trên test US + test VN
```

## Nhánh Git (15 nhánh, merge tuần tự)

`00-init → 01-charset-vi → 02-fetch-uspto-meta → 03-fetch-uspto-images →
04-clean-label → 05-split-build_data → 06-eda → 07-train-crnn →
08-train-parseq → 09-train-svtrv2 → 10-eval-metrics → 11-fetch-vn →
12-eval_domain_shift → 13-error-analysis → 14-demo-report`

Mỗi nhánh có lệnh nghiệm thu tương ứng trong `ACCEPTANCE.md`.
