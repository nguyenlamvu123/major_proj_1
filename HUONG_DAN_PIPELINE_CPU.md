# Hướng dẫn chạy pipeline train/eval trên máy Ubuntu khác (CPU)

File script: `scripts/09_full_pipeline_cpu.sh`

## 1) Yêu cầu môi trường

- Ubuntu/Linux
- Git, Python 3.10+, pip
- Kết nối mạng
- Dung lượng ổ: ~20–40GB (code + data + checkpoint)
- RAM: khuyến nghị >= 16GB để train, nếu 8–12GB chỉ nên train 1 model với batch nhỏ

## 2) Chuẩn bị

Trên máy mới, mở terminal tại thư mục muốn lưu project:

```bash
# Tạo thư mục làm việc
mkdir -p ~/projects
cd ~/projects
```

Clone repo hoặc copy script này qua máy mới (có thể copy cả repo hiện tại).

## 3) Cách chạy

### Trường hợp A: Chưa có dữ liệu gì cả (tự tải từ đầu)

Script sẽ tự clone repo nhánh `16-fix-review`, cài dependencies, tải meta + ảnh USPTO, clean/split, build LMDB, crawl VN, train, eval, sinh báo cáo.

```bash
cd /path/to/repo  # hoặc chạy từ nơi có script
bash scripts/09_full_pipeline_cpu.sh \
  --repo-url git@github.com:nguyenlamvu123/major_proj_1.git \
  --branch 16-fix-review \
  --data-root /home/$USER/data/trademark_data \
  --work-dir ~/projects \
  --models crnn \
  --batch-size 8 \
  --max-epochs 15
```

> **Lưu ý:** Tải ảnh USPTO có thể mất vài giờ. Trên CPU train cũng rất chậm.

### Trường hợp B: Đã copy dữ liệu từ máy cũ qua

Nếu bạn đã copy toàn bộ `trademark_data/` (hoặc ít nhất `raw/`, `processed/`, `lmdb/`) sang máy mới, bỏ qua bước tải:

```bash
bash scripts/09_full_pipeline_cpu.sh \
  --branch 16-fix-review \
  --data-root /path/to/trademark_data \
  --skip-fetch \
  --models crnn,parseq \
  --batch-size 8 \
  --max-epochs 20
```

### Trường hợp C: Đã có repo + data, chỉ muốn train/eval

```bash
bash scripts/09_full_pipeline_cpu.sh \
  --data-root /path/to/trademark_data \
  --skip-clone --skip-fetch --skip-build-lmdb \
  --models crnn \
  --batch-size 6 \
  --num-workers 0 \
  --max-epochs 10
```

## 4) Các tham số quan trọng

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--branch` | `16-fix-review` | Nhánh code sử dụng |
| `--repo-url` | SSH | URL repo |
| `--data-root` | `/data/trademark_data` | Thư mục chứa dữ liệu |
| `--work-dir` | `$HOME/projects` | Thư mục clone repo |
| `--models` | `crnn,trba,parseq,svtrctc` | Danh sách model train |
| `--batch-size` | `8` | Batch size (giảm xuống 4–6 nếu thiếu RAM) |
| `--num-workers` | `0` | Số worker dataloader (0 an toàn nhất trên CPU/RAM ít) |
| `--max-epochs` | `20` | Số epoch train |
| `--skip-clone` | false | Bỏ qua clone repo |
| `--skip-fetch` | false | Bỏ qua tải meta/ảnh USPTO + VN |
| `--skip-build-lmdb` | false | Bỏ qua build LMDB |
| `--skip-train` | false | Bỏ qua train |
| `--skip-eval` | false | Bỏ qua eval |

## 5) Sau khi chạy xong

Kết quả sẽ nằm ở:

```text
~/projects/major_proj_1/
├── reports/
│   ├── final_summary.md      # Báo cáo tổng hợp
│   ├── <model>_us.csv
│   ├── <model>_vn.csv
│   └── domain_shift.csv
└── runs/                     # Checkpoint
    └── <model>/<timestamp>/checkpoints/last.ckpt
```

Xem báo cáo:

```bash
cat ~/projects/major_proj_1/reports/final_summary.md
```

## 6) Lời khuyên khi train trên CPU

- **Bắt đầu với 1 model**: `--models crnn` thay vì train cả 4 cùng lúc
- **Giảm batch size**: nếu RAM < 12GB dùng `--batch-size 4`
- **Luôn dùng `--num-workers 0`**: tránh tăng RAM
- **Build lại LMDB đầy đủ**: trên máy cũ train chỉ 500 mẫu, trên máy mới nên build lại từ `split_train.csv` (4356 mẫu) để có kết quả đúng
- **Ưu tiên fine-tune**: nếu có pretrained, `parseq_ft` thường hội tụ nhanh hơn trên dữ liệu nhỏ
- **Chạy nền**: có thể dùng `tmux` hoặc `screen` để train không bị ngắt khi mất kết nối SSH
- **Kiểm tra RAM**: `htop` / `free -h` trong lúc train
