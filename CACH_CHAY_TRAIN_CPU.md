# Có thể train trên CPU không? Chỉ chậm thôi đúng không?

**Trả lời ngắn:** Đúng là *"chỉ chậm thôi"*, nhưng với cấu hình máy hiện tại **không thực tế để train đầy đủ**. Có thể chạy smoke-test, nhưng **không nên** chạy train toàn bộ để ra kết quả đồ án.

## 1) Tình trạng máy hiện tại

| Thông số | Giá trị | Nhận xét |
|tag: th, td |---|---|
| CPU | 8 nhân / 16 luồng (xem `nproc`) | Tạm ổn |
| RAM | 7.6 GiB (available ~1.1 GiB lúc kiểm tra) | **Hạn chế nhiều** |
| GPU | Không có (PyTorch `cuda.is_available() = False`) | Chỉ dùng CPU |
| Ổ dữ liệu | `/media/zaibachkhoa/AC84-F995/` (~dung lượng còn trống đủ) | Ổn |
| LMDB train hiện tại | `lmdb/train/trademark/` có **500 mẫu** (`num-samples = b'500'') | Rất nhỏ so với `split_train.csv` (4356 mẫu) |

## 2) So sánh thời gian dự kiến

| Mô hình | Colab T4 (GPU) | Máy local CPU (ước tính) | Ghi chú |
|---|---|---|---|
| **CRNN** | ~1–3 giờ/20–50 epoch | **10–30 giờ** (hoặc hơn) | Ít tham số nhất nhưng vẫn chậm trên CPU |
| **TRBA** | ~2–4 giờ | **20–40 giờ** | Attention nặng hơn |
| **PARSeq** | ~3–5 giờ | **30–60 giờ** | Transformer, chậm nhất |
| **SVTR-CTC** | ~2–4 giờ | **20–40 giờ** | Sau khi sửa kiến trúc (nhánh 16.2) cần train lại từ đầu |
| **Fine-tune `*_ft`** | ~1–2 giờ (ít epoch hơn) | **8–20 giờ** | Nhanh hơn train từ đầu, nhưng vẫn rất dài trên CPU |

> Ước tính trên thực tế rất khó chính xác. Với batch-size nhỏ trên CPU, thời gian có thể dài gấp **10–20 lần** so với GPU.

## 3) Các vấn đề thực tế trên máy này

| Vấn đề | Mức độ | Giải thích |
|---|---|---|
| **RAM 7.6GB** | Cao | PyTorch + Lightning + DataLoader với `num_workers` có thể dễ gây **OOM** (out of memory). Trước đây đã từng gặp `exit 137` (SIGKILL) khi dùng `batch-size 64`. |
| **LMDB train chỉ 500 mẫu** | Cao | Nếu dùng đúng LMDB hiện tại, bạn **chỉ train với 500 mẫu** thay vì 4356 mẫu. Kết quả sẽ không đúng với toàn bộ dữ liệu US đã thu thập. Cần build lại LMDB đầy đủ từ `split_train.csv`. |
| **SVTR cũ không tương thích** | Trung bình | Checkpoint `svtrctc` cũ không load được do thay đổi kiến trúc ở nhánh 16.2. Nếu muốn dùng SVTR, **phải train lại hoàn toàn**. |
| **Thời gian liên tục** | Cao | Để train đủ, máy phải chạy liên tục hàng chục giờ (không tắt máy, không khóa màn hình quá mức). Điện, nhiệt độ máy cũng là vấn đề. |
| **Cạnh tranh tài nguyên** | Trung bình | RAM hiện tại chỉ còn ~1.1GB khả dụng, rất dễ bị OOM khi mở nhiều ứng dụng khác. |

## 2/4? Thêm ví dụ lệnh</## 4) Lệnh có thể chạy (smoke-test, có thể thực tế)

Trên máy này, **an toàn nhất** là chỉ chạy **smoke test** để kiểm chứng pipeline:

```bash
cd "/media/zaibachkhoa/education/GenAI 2025/đồ án chuyên ngành 1"
export DATA_ROOT=/media/zaibachkhoa/AC84-F995/trademark_data
export PYTHONPATH=$PWD/compat:$PWD/src:$PWD/third_party/parseq

# Smoke test an toàn nhất (CPU + batch nhỏ + ít batch/epoch)
bash scripts/05_train.sh crnn \
  --accelerator cpu \
  --devices 1 \
  --batch-size 8 \
  --num-workers 0 \
  --limit-train-batches 2 \
  --max-epochs 1
}

bash scripts/05_train.sh parseq \
  --acceler
...</div>

## 5) Khuyến nghị

- **Nếu chỉ muốn kiểm tra pipeline:** Chạy smoke-test như trên. Hoàn toàn khả thi.
- **Nếu muốn train để có kết quả đồ án thực tế:** **Không khuyến nghị trên máy này**. Tốt nhất là đưa lên **Google Colab (T4)** hoặc máy có GPU.
- **Trước khi train thật:** Phải build lại LMDB train đầy đủ (4356 mẫu) thay vì 500 mẫu hiện tại. Nếu không, mô hình chỉ học trên 1/9 dữ liệu.
