# Hướng dẫn chạy toàn bộ đồ án trên một máy (`scripts/09_full_pipeline_cpu.sh`)

Một lệnh chạy từ đầu tới cuối: dựng môi trường → tải dữ liệu US → làm sạch, chia tập, LMDB →
crawl tập VN → train → đánh giá + báo cáo. Chạy được trên máy chỉ có CPU (tên "_cpu"), và tự
dùng GPU nếu máy có `nvidia-smi` (`--accelerator auto`).

## 1. Yêu cầu

- Linux, `git`, `python3` có module `venv` (Ubuntu: gói `python3-venv`). **Không cần** `pip`
  hệ thống: script tự dựng `<repo>/.venv` và cài torch bản CPU (tránh ~2,5 GB thư viện CUDA).
- Mạng tới USPTO (metadata + CDN ảnh), WIPO Publish (VN), GitHub (parseq + trọng số pretrained).
- Ổ đĩa: dữ liệu ~6 GB (ảnh + LMDB) + checkpoint vài trăm MB mỗi model + venv ~2,2 GB.
- RAM: ≥ 8 GB trống cho batch 64. Thiếu thì `--batch-size 32` (hoặc 16).

## 2. Lấy code

```bash
git clone https://github.com/nguyenlamvu123/major_proj_1.git
cd major_proj_1
git switch 16-fix-review
```

Script chạy ngay trong repo chứa nó — không tự clone thêm bản khác.

## 3. Chạy

```bash
tmux new -s doan                                     # đóng terminal không làm chết tiến trình
bash scripts/09_full_pipeline_cpu.sh --dry-run       # 1) xem kế hoạch, không chạy gì
bash scripts/09_full_pipeline_cpu.sh --max-hours 3   # 2) chạy thật; Ctrl-b d để thoát ra
```

Sáng hôm sau: `tmux attach -t doan`, hoặc xem log `data/logs/pipeline_*.log` và báo cáo
`reports/final_summary.md`.

Script tự giữ khoá chặn máy ngủ (`systemd-inhibit`) trong suốt lần chạy, nếu hệ thống cho phép.

### Chạy lại an toàn

Mỗi bước dữ liệu xong thì ghi một dấu vào `$DATA_ROOT/.pipeline/<bước>`. Chạy lại thì:

- bước đã có dấu, cùng tham số, và các bước nó phụ thuộc không mới hơn → **bỏ qua**;
- đổi tham số (vd `--us-rows`) hoặc bước phía trước chạy lại → bước đó và mọi bước phía sau chạy lại;
- model đã train xong (`runs/<model>/<giờ>/DONE`) → bỏ qua; train dở (có `last.ckpt`, chưa
  `DONE`) → **train tiếp** từ `last.ckpt`, giữ đúng số epoch/batch đã ghi trong
  `runs/<model>/<giờ>/pipeline_train.env`;
- tải ảnh dở → tải tiếp phần còn thiếu.

Lỗi giữa chừng (mất mạng, mất điện) thì cứ chạy lại đúng lệnh cũ.

## 4. Các cờ

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--dry-run` | | chỉ in kế hoạch (không gọi mạng, không tạo file) |
| `--data-root DIR` | `$DATA_ROOT` hoặc `<repo>/data` | gốc dữ liệu |
| `--venv DIR` | `<repo>/.venv` | môi trường Python tự dựng |
| `--python PATH` | | dùng Python có sẵn, bỏ bước dựng venv |
| `--us-rows N` | 60000 | số dòng metadata US (lọc xong còn ~97%) |
| `--vn-limit N` | 2000 | số nhãn hiệu VN |
| `--image-workers N` | 16 | luồng tải ảnh US |
| `--models a,b` | `crnn_ft,parseq_ft` | model cần train (tên như `scripts/05_train.sh`) |
| `--max-epochs N` | 10 | số epoch tối đa |
| `--max-hours H` | không giới hạn | **ngân sách giờ cho MỖI model** (xem mục 5) |
| `--batch-size N` | 64 | |
| `--num-workers N` | 4 | worker dataloader |
| `--accelerator X` | `auto` | `auto` / `cpu` / `gpu` |
| `--from STEP` | | bỏ qua các bước trước STEP |
| `--only a,b` | | chỉ chạy các bước này, ép chạy kể cả khi đã xong |
| `--skip a,b` | | bỏ qua các bước này |
| `--retrain` | | train lại từ đầu kể cả model đã `DONE` |

Các bước: `setup us_meta us_images clean split lmdb vn vn_lmdb train eval`.

## 5. `--max-hours`: train theo ngân sách giờ

Với mỗi model train mới, script **đo tốc độ thật** trên máy (vài batch train + val, đúng
batch/worker/augment), rồi chọn số epoch lớn nhất (≤ `--max-epochs`) vừa 85% ngân sách
(`runs/<model>/<giờ>/benchmark.json`). Nhờ vậy lịch learning rate (OneCycle) chạy trọn, thay vì
bị cắt ngang ở LR còn cao.

Trần cứng `max_time` của Lightning = 115% ngân sách — chỉ cắt khi máy chậm bất thường; khi đó
model được đánh dấu "chưa đủ epoch" và chạy lại script sẽ train tiếp.

Ước lượng tổng thời gian một đêm: dữ liệu ~1,5–2,5 giờ (phần lớn là tải ~60k ảnh) + số model ×
`--max-hours` + eval ~0,5 giờ. Ví dụ 2 model, `--max-hours 3`: khoảng 8–9 giờ.

## 6. Tập VN và duyệt tay

Nhãn VN (`rs-MK` của WIPO Publish) là phần chữ được đăng ký, không phải bản chép lại ảnh. Lần
chạy tự động dùng nhãn **chưa duyệt** (cuối log có cảnh báo). Để có số liệu đúng:

```bash
.venv/bin/python -m trademark_ocr.vn_review make --n 300   # sửa tay data/processed/review_vn.csv
.venv/bin/python -m trademark_ocr.vn_review apply
bash scripts/09_full_pipeline_cpu.sh --from vn_lmdb         # build lại test_vn + eval lại
```

Crawl VN lỗi (mạng, site đổi cấu trúc) **không** dừng train: eval chỉ có `test_us`, cuối log có cảnh báo.

## 7. Kết quả

```text
<repo>/reports/final_summary.md        báo cáo tổng hợp (3 mức so khớp, phân tách độ sụt US→VN, theo drawing_kind)
<repo>/reports/<model>_{us,vn}.csv     kết quả từng mẫu
<DATA_ROOT>/runs/<model>/<giờ>/         checkpoints/last.ckpt, config/, pipeline_train.env, benchmark.json, DONE
<DATA_ROOT>/logs/pipeline_<giờ>.log     log đầy đủ của lần chạy
```

Mang model sang máy khác: chép **cả thư mục run** (cần `config/config.yaml` để `evaluate` biết lớp model).

## 8. Lưu ý

- Train trên CPU chậm hơn GPU nhiều lần; `--max-hours` là cách giữ cả đêm trong tầm kiểm soát.
- `last.ckpt` là checkpoint top-3 (theo `val_accuracy`) **mới nhất**, không nhất thiết là epoch
  cuối: với pytorch-lightning 2.6 + `save_top_k=3` của parseq, epoch không lọt top-3 thì không ghi
  `last.ckpt`. Train tiếp sau khi bị cắt có thể làm lại vài epoch cuối.
- Lỗi train một model (vd hết RAM) không dừng cả đêm: model kế vẫn train, eval vẫn chạy, cuối log
  liệt kê model lỗi và script thoát mã 1.
