# Có thể train trên CPU không? Chỉ chậm thôi đúng không?

**Trả lời ngắn:** Đúng là *"chỉ chậm thôi"* — chạy được, nhưng chậm hơn GPU nhiều lần. Muốn có
kết quả đồ án trong một đêm trên CPU thì phải **giới hạn thời gian mỗi model**:
`bash scripts/09_full_pipeline_cpu.sh --max-hours 3` (xem `HUONG_DAN_PIPELINE_CPU.md` mục 5).

## 1) Cấu hình các máy

| Thông số | Máy cũ (`zaibachkhoa`) | Máy hiện tại (`vunl`) | Nhận xét |
|---|---|---|---|
| CPU | 8 nhân / 16 luồng | Intel i5-10400, 6 nhân / 12 luồng | |
| RAM | 7,6 GiB (lúc kiểm tra trống ~1,1 GiB) | 31 GiB (lúc kiểm tra trống ~8 GiB) | máy cũ dễ OOM với batch 64 |
| GPU | không (`torch.cuda.is_available() = False`) | không (không có `nvidia-smi`) | chỉ dùng CPU |
| Ổ dữ liệu | `/media/zaibachkhoa/AC84-F995/` | `/mnt/...` còn ~730 GB | đủ |
| LMDB train | 500 mẫu (build với `--limit 500`) | 51.230 mẫu (val 2.846, test_us 2.847) | xem mục 3 |

## 2) Thời gian

**Số đo thật trên máy `vunl`** (lần chạy 04/11, batch 64, 4 worker — chi tiết `docs/nhat_ky_chay.md`):

| Mô hình | giây/batch train | phút/epoch | Ngân sách 3h cho | Kết quả |
|---|---|---|---|---|
| `crnn_ft` | 1,25 | 17,1 | 8 epoch | val acc 74,1%, test US 73,0% |
| `parseq_ft` | 4,17 | 57,1 | 2 epoch | **0%** — chưa học được, cần ≥ 8–9 giờ hoặc GPU |
| `trba_ft`, `svtr` | chưa đo | | | |

Bảng dưới là **ước lượng** cũ (không phải số đo), giữ để so với GPU:

| Mô hình | Colab T4 (GPU) | CPU (ước lượng) | Ghi chú |
|---|---|---|---|
| **CRNN** | ~1–3 giờ / 20–50 epoch | 10–30 giờ (hoặc hơn) | ít tham số nhất (8,4 M) |
| **TRBA** | ~2–4 giờ | 20–40 giờ | 50 M tham số |
| **PARSeq** | ~3–5 giờ | 30–60 giờ | Transformer 24 M tham số |
| **SVTR-CTC** | ~2–4 giờ | 20–40 giờ | kiến trúc đã sửa (16.2), checkpoint cũ phải train lại |
| **`*_ft`** | ~1–2 giờ (ít epoch hơn) | 8–20 giờ | hội tụ nhanh hơn train từ đầu |

Thay vì đoán, `--max-hours` **đo tốc độ thật** trên máy (vài batch) rồi chọn số epoch vừa ngân
sách, ghi vào `runs/<model>/<giờ>/benchmark.json` — sau lần chạy đầu sẽ có số đo thật để thay bảng trên.

## 3) Các vấn đề thực tế

| Vấn đề | Mức độ | Giải thích |
|---|---|---|
| **RAM** | Cao (máy cũ) | PyTorch + Lightning + DataLoader nhiều worker dễ OOM (`exit 137`), từng gặp với `batch-size 64` trên máy 7,6 GiB. Giảm `--batch-size` / `--num-workers`. |
| **LMDB train 500 mẫu (máy cũ)** | Cao | Build bằng `--limit 500` nên chỉ train trên 500/4356 mẫu. Script pipeline luôn build toàn bộ split (không `--limit`). |
| **SVTR cũ không tương thích** | Trung bình | Checkpoint `svtrctc` cũ không nạp được sau khi sửa kiến trúc (16.2), phải train lại. |
| **Chạy liên tục nhiều giờ** | Cao | Máy không được ngủ/tắt: script giữ khoá chặn ngủ (`systemd-inhibit`); nên chạy trong `tmux`. |
| **Cạnh tranh tài nguyên** | Trung bình | Đóng bớt ứng dụng nặng trước khi chạy để còn RAM trống. |

## 4) Lệnh smoke-test (kiểm tra pipeline, vài phút)

```bash
export DATA_ROOT=<gốc dữ liệu đã có LMDB>
bash scripts/05_train.sh crnn --accelerator cpu --devices 1 --batch-size 8 --num-workers 0 \
  --limit-train-batches 2 --max-epochs 1
bash scripts/05_train.sh parseq --accelerator cpu --devices 1 --batch-size 8 --num-workers 0 \
  --limit-train-batches 2 --max-epochs 1
```

## 5) Khuyến nghị

- **Chỉ kiểm tra pipeline:** smoke-test như trên.
- **Cần kết quả đồ án trong một đêm trên CPU:** `scripts/09_full_pipeline_cpu.sh --max-hours <giờ>`
  với **một** model nặng mỗi đêm. `crnn_ft` vừa 3h; `parseq_ft` cần cả đêm
  (`--models parseq_ft --retrain --max-hours 9` ≈ 8 epoch); TRBA (50 M tham số) nên train trên GPU.
- **Cần kết quả tốt nhất:** train trên GPU (Colab T4, `docs/colab/02_train.ipynb`).
