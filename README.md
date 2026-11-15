# MAJOR_PROJ_1 — Nhận diện chữ trong nhãn hiệu sản phẩm (Text Recognition)

Đồ án: nhận diện chữ (wordmark) trong nhãn hiệu sản phẩm bằng học sâu có giám sát.
- **Train**: ảnh nhãn hiệu đăng ký tại Mỹ (USPTO) + nhãn chữ tương ứng.
- **Đánh giá mới**: ảnh nhãn hiệu đăng ký tại Việt Nam (IP Vietnam / WIPO Publish).
- **Khung mã**: [baudm/parseq](https://github.com/baudm/parseq) (clone vào `third_party/parseq`, không fork, không sửa).
- **Bảng ký tự**: Unicode có dấu tiếng Việt (NFC).

> Tài liệu kỹ thuật chi tiết (mô tả code cho người/đọc giả/AI): xem **[DESIGN.md](DESIGN.md)**.
> Lệnh nghiệm thu từng nhánh: xem **[ACCEPTANCE.md](ACCEPTANCE.md)**.
> Đánh giá phương án & code (vấn đề đã biết, trạng thái xử lý): xem **[REVIEW.md](REVIEW.md)**.

## Cấu trúc

```
major_proj_1/
├── DESIGN.md               # mô tả chi tiết code (đọc trước khi sửa)
├── ACCEPTANCE.md           # lệnh nghiệm thu từng nhánh
├── REVIEW.md               # đánh giá phương án/code + trạng thái xử lý
├── README.md               # file này — hướng dẫn chạy
├── requirements.txt
├── pyproject.toml          # pip install -e .  → package `trademark_ocr`
├── src/trademark_ocr/      # toàn bộ code của đồ án
├── configs/                # charset + hydra experiment configs
│   ├── charset_vi.txt      # bảng ký tự VN (tự sinh từ src/trademark_ocr/charset.py)
│   └── hydra/              # config huấn luyện (crnn/trba/parseq/svtrctc + *_ft) + charset/charset_vi.yaml
├── scripts/                # pipeline shell (00→06)
├── notebooks/              # EDA, error analysis
├── docs/colab/             # notebook Colab: setup → tải dữ liệu → train → eval
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

# 2) chạy test nhanh
pytest -q
```

| Script | Lệnh tương đương |
|---|---|
| `bash scripts/00_get_parseq.sh` | `git clone https://github.com/baudm/parseq third_party/parseq` (bỏ qua nếu đã có)<br>`git -C third_party/parseq fetch --depth 1 origin 1902db043c029a7e03a3818c616c06600af574be`<br>`git -C third_party/parseq checkout 1902db043c029a7e03a3818c616c06600af574be` |

## Nguồn dữ liệu

**DATA_ROOT**: không đặt thì mặc định `<repo>/data` (gitignore). Đặt sang ổ đủ dung lượng:

```bash
export DATA_ROOT=/media/zaibachkhoa/AC84-F995/trademark_data   # máy chủ repo
export PYTHONPATH=$PWD/compat:$PWD/src:$PWD/third_party/parseq
# Colab: export DATA_ROOT=/content/drive/MyDrive/trademark_data (mount Drive)
```

| Script | Lệnh tương đương | Đầu ra | Ghi chú |
|---|---|---|---|
| `bash scripts/01_get_meta.sh --limit 6000` | `python -m trademark_ocr.fetch_meta --limit 6000` | `$DATA_ROOT/raw/meta/us_marks.csv`, `us_marks.csv.stats.json` | `GET .../search/publications` lấy danh sách issue gazette; mỗi issue `POST .../search/all` phân trang `offset/limit` (≤100/trang) trong `iter_issue()`; lọc `is_candidate` rồi dừng khi đủ `--limit`. |
| `bash scripts/02_get_images.sh` | `python -m trademark_ocr.fetch_images` | `$DATA_ROOT/raw/images_us/{serial}.png`, `raw/images_us/index.csv`; cập nhật `image_status` trong metadata | Đọc `image_url` từ metadata, tải song song bằng `ThreadPoolExecutor`; **resume** (bỏ qua serial đã `ok`), kiểm tra ảnh bằng PIL, ghi `index.csv`; `--verify` mở lại toàn bộ ảnh. |
| `bash scripts/03_get_vn.sh --limit 2000` | `python -m trademark_ocr.fetch_vn --limit 2000` | `raw/vn_marks.csv`, `raw/images_vn/`, `raw/vn_report.json`, `processed/split_test_vn.csv`, `processed/test_vn.txt` | WIPO Publish: `GET` trang tìm kiếm lấy form/session → `POST` AJAX `searchSubmitLink` → panel 50 hàng/trang, phân trang `navigator-next`; lấy `rs-MK` + ảnh thumbnail; dedupe, `--registered-only`, dừng khi đủ `--limit`. |

Metadata USPTO lấy từ TMOG / EOG REST API công khai, **không cần API key** (chi tiết DESIGN.md
nhánh 02); ảnh US lấy từ CDN USPTO, ảnh VN từ WIPO Publish. Kiểm tra tay nhãn VN bằng
`python -m trademark_ocr.vn_review make|apply` (DESIGN.md 16.5). Có thể tải bằng Colab:
`docs/colab/01_download_data.ipynb`.

## Chạy toàn bộ trên một máy (tải dữ liệu → train → eval → báo cáo)

```bash
bash scripts/09_full_pipeline_cpu.sh --dry-run       # xem kế hoạch, không chạy gì
bash scripts/09_full_pipeline_cpu.sh --max-hours 3   # chạy thật (nên trong tmux); log ở data/logs/
```

Script tự dựng `.venv` (torch CPU nếu không có GPU), chạy lại an toàn (bước nào xong thì bỏ qua,
model train dở thì train tiếp) và với `--max-hours` thì đo tốc độ thật để chọn số epoch vừa ngân
sách mỗi model. Model đã xong muốn train thêm epoch (giữ trọng số): `--continue --max-epochs <tổng>`.
Eval vẽ luôn đồ thị train (loss train / val_loss / val_accuracy theo epoch, kết luận overfit) vào
`reports/figures/`. Chi tiết: [HUONG_DAN_PIPELINE_CPU.md](HUONG_DAN_PIPELINE_CPU.md); train trên CPU
nhanh/chậm thế nào: [CACH_CHAY_TRAIN_CPU.md](CACH_CHAY_TRAIN_CPU.md); các lần đã chạy, kết quả và
việc còn lại: [docs/nhat_ky_chay.md](docs/nhat_ky_chay.md).

## Huấn luyện (Google Colab, GPU T4)

```bash
# clone repo + mount Drive, cài deps  →  docs/colab/00_setup.ipynb
export DATA_ROOT=/content/drive/MyDrive/trademark_data
export PYTHONPATH=$PWD/compat:$PWD/src:$PWD/third_party/parseq
```

| Script | Lệnh tương đương |
|---|---|
| `bash scripts/05_train.sh crnn` | `python third_party/parseq/train.py --config-dir configs/hydra --config-name tm_crnn` |
| `bash scripts/05_train.sh trba` | `python third_party/parseq/train.py --config-dir configs/hydra --config-name tm_trba` |
| `bash scripts/05_train.sh parseq` | `python third_party/parseq/train.py --config-dir configs/hydra --config-name tm_parseq` |
| `bash scripts/05_train.sh svtrctc` (hoặc `svtr`) | `python third_party/parseq/train.py --config-dir configs/hydra --config-name tm_svtrctc` |
| `bash scripts/05_train.sh crnn_ft` | `python third_party/parseq/train.py --config-dir configs/hydra --config-name tm_crnn_ft` |
| `bash scripts/05_train.sh trba_ft` | `python third_party/parseq/train.py --config-dir configs/hydra --config-name tm_trba_ft` |
| `bash scripts/05_train.sh parseq_ft` | `python third_party/parseq/train.py --config-dir configs/hydra --config-name tm_parseq_ft` |
| `bash scripts/06_eval.sh` | Xem khối lệnh đánh giá ngay dưới bảng |

Cờ của `05_train.sh` được thay bằng override Hydra: `--batch-size 64` → `model.batch_size=64`,
`--max-epochs 20` → `trainer.max_epochs=20`, `--num-workers 0` → `data.num_workers=0`,
`--limit-train-batches 2` → `trainer.limit_train_batches=2`, `--accelerator cpu` →
`trainer.accelerator=cpu`. Ví dụ:

```bash
python third_party/parseq/train.py --config-dir configs/hydra --config-name tm_parseq_ft \
  model.batch_size=64 trainer.max_epochs=20
```

`bash scripts/06_eval.sh` tương đương:

```bash
for m in crnn trba parseq svtrctc crnn_ft trba_ft parseq_ft; do
  ckpt=$(ls -dt "$DATA_ROOT/runs/$m"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
  [ -z "$ckpt" ] && continue
  python -m trademark_ocr.evaluate --ckpt "$ckpt" --subset test_us --out "reports/${m}_us.csv"
  python -m trademark_ocr.evaluate --ckpt "$ckpt" --subset test_vn --out "reports/${m}_vn.csv"
done
python -m trademark_ocr.eval_domain_shift --reports reports/ --out reports/domain_shift.csv
python -m trademark_ocr.error_analysis --reports reports/ --out-dir "$DATA_ROOT/processed/eda_figures"
python -m trademark_ocr.report --reports reports/ --out reports/final_summary.md
```

Đọc kết quả ở `reports/final_summary.md`: so sánh US↔VN nên đọc ở mức **bỏ hoa/thường**
(nhãn USPTO luôn viết hoa) và ở nhóm VN **không dấu** (model chưa từng học chữ có dấu) —
xem REVIEW.md §4.2 và DESIGN.md 16.4.

## Nhánh Git (merge tuần tự)

`00-init → 01-charset-vi → 02-fetch-uspto-meta → 03-fetch-uspto-images →
04-clean-label → 05-split-build_data → 06-eda → 07-train-crnn →
08-train-parseq → 09-train-svtrv2 → 10-eval-metrics → 11-fetch-vn →
12-eval_domain_shift → 13-error-analysis → 14-demo-report →
15-review-tests → 16-fix-review`

Mỗi nhánh có lệnh nghiệm thu tương ứng trong `ACCEPTANCE.md`.
