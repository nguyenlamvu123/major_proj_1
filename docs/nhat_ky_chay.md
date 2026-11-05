# Nhật ký các lần chạy pipeline

Mỗi lần chạy thật `scripts/09_full_pipeline_cpu.sh` (hoặc train/eval trên Colab) ghi một mục, **mới
nhất ở trên**: lệnh, thời gian từng bước, kết quả, cái gì sai, đã sửa gì, việc tiếp theo. Log đầy đủ
nằm ở `data/logs/pipeline_<giờ>.log` (không đưa vào git).

---

## Lần 1 — 04/11/2026 20:59 → 05/11/2026 02:38 (máy `vunl`, CPU)

### Bối cảnh

- Máy: Intel i5-10400 (6 nhân / 12 luồng), 31 GiB RAM, không GPU. Nhánh `16-fix-review`, commit
  `f848fcf`.
- Lệnh (lịch hẹn 21:00 tự chạy vì chưa chọn phương án 4 model; dùng mặc định 2 model):

  ```bash
  env -u DATA_ROOT tmux new-session -d -s doan "bash scripts/09_full_pipeline_cpu.sh --max-hours 3"
  ```
- Log: `data/logs/pipeline_2026-11-04_20-59-47.log`. `systemd-inhibit` giữ khoá chặn ngủ suốt
  lần chạy (đã kiểm lúc 21:03).

### Thời gian từng bước

| Giờ | Bước | Kết quả |
|---|---|---|
| 20:59–21:13 | `us_meta` | 60.000 dòng metadata USPTO |
| 21:13–21:50 | `us_images` | ảnh US (2,5 GB) |
| 21:50 | `clean`, `split`, `lmdb` | train **51.230** / val **2.846** / test_us **2.847** mẫu (LMDB train 2,1 GB) |
| 21:50–21:58 | `vn`, `vn_lmdb` | 2.000 nhãn hiệu VN, **chưa duyệt tay** |
| 21:58–00:27 | train `crnn_ft` | đo 1,25 s/batch → 17,1 phút/epoch → chọn 8 epoch; xong sau 148 phút |
| 00:27–02:35 | train `parseq_ft` | đo 4,17 s/batch → 57,1 phút/epoch → chọn **2 epoch**; xong sau 126 phút |
| 02:35–02:38 | eval + báo cáo | `reports/final_summary.md` |

Xong sớm hơn ước lượng (5–7 giờ sáng) vì phần dữ liệu chỉ mất ~1 giờ (ước 1,5–2,5 giờ) và mỗi model
dừng ở ~85% ngân sách theo thiết kế.

### Kết quả dùng được: `crnn_ft`

Val accuracy theo epoch: 0 → 60,9 → 67,6 → 70,5 → 71,6 → 72,6 → 73,0 → **74,1%** (epoch 0–7).
Checkpoint đánh giá: `data/runs/crnn_ft/2026-11-04_21-58-41/checkpoints/last.ckpt` (epoch 7).

| Tập | n | accuracy | bỏ hoa/thường | bỏ cả dấu | 1−NED | coverage |
|---|---|---|---|---|---|---|
| test US | 2.847 | **73,0%** | 73,0% | 73,1% | 0,889 | 100% |
| test VN | 2.000 | **18,3%** | 27,0% | 28,5% | 0,380 | 77,8% |
| test VN, chỉ mẫu trong phạm vi | 1.556 | 23,5% | 34,7% | 36,6% | 0,474 | — |

Theo loại nhãn hiệu (US, bỏ hoa/thường): `standard_character` 84,9% (n=2.288), `typeset_old`
96,6% (29), `stylized_words` 42,0% (112), `design_with_words` **14,4%** (418).

Độ sụt US → VN (bỏ hoa/thường, CI 95% bootstrap): **−0,460 [−0,484; −0,434]**; chỉ trên 1.442 nhãn
VN **không dấu**: −0,355 [−0,384; −0,325]. Trên 558 nhãn **có dấu** accuracy = **0%**: dữ liệu
train (USPTO) gần như không có chữ có dấu nên model chưa học được dấu — đây là kết quả cần nêu
trong báo cáo đồ án, không phải lỗi pipeline.

Coverage VN 77,8%: 425 nhãn dài quá 25 ký tự (`max_label_length`), 83 nhãn có ký tự ngoài bảng
217 ký tự — các mẫu này vẫn tính là sai. Nhãn VN trung bình 18,6 ký tự, US 10,7.

### Cái gì sai trong `final_summary.md`

1. **`parseq_ft` = 0%** (US lẫn VN) — model chưa học được, không phải lỗi eval:
   - Ngân sách 3h trên CPU chỉ đủ 2 epoch (1.602 bước).
   - Lớp ra (`head`) và `text_embed` khởi tạo lại theo bảng 217 ký tự, nên loss bắt đầu ở
     5,39 = ln 217 (đoán ngẫu nhiên). Sau 1.602 bước mới xuống 3,27, vẫn đang giảm đều.
   - `val_accuracy` = 0,0000 ở cả hai epoch (val NED 4,2% → 7,0%). Model chưa học ký tự kết
     thúc chuỗi nên luôn sinh đủ 26 ký tự (`mean_pred_len` = 26).
   - Đối chiếu: `crnn_ft` cũng 0% sau epoch đầu, nhưng mỗi epoch chỉ 17 phút.
   - Yếu tố phụ: SWA của `parseq/train.py` bật từ 75% số epoch. Với 2 epoch, nó thay lịch
     OneCycle bằng LR cố định ngay từ epoch thứ hai, nên LR không giảm dần về cuối.
   - **Lỗi của pipeline:** vẫn ghi `DONE` và đưa 0% vào bảng như một kết quả.
2. **Dòng `crnn` 200 mẫu US / 36 mẫu VN, accuracy 0**: file `reports/crnn_*` ngày 16/10 của máy
   cũ (LMDB 500 mẫu). `report.py` gom mọi `reports/*.csv` nên chúng lẫn vào bảng.
3. **`char-accuracy` âm** (parseq_ft −1,26): định nghĩa `1 − CER` mà CER > 1 khi model đoán dài
   hơn nhãn.
4. Phụ: tiêu đề ghi `19:38 UTC` (= 02:38 giờ VN), dễ đọc nhầm.

Quan sát khác, không ảnh hưởng kết quả:
- Log đứng im 13 phút ở `us_meta` dù vẫn đang tải: stdout của Python qua `tee` bị đệm theo khối.
- `DecompressionBombWarning`: vài ảnh USPTO tới 160 MP. Trên 2.000 mẫu ngẫu nhiên, 0,5% ảnh lớn
  hơn 10 MP; trung vị 0,1 MP, 13 KB.

### Đã sửa (commit sau `f848fcf`, DESIGN.md 16.9)

| Sửa | Ở đâu |
|---|---|
| Train xong mà `val_accuracy` tốt nhất < 1% → ghi `UNDERTRAINED` thay vì `DONE`; kế hoạch bỏ qua kèm gợi ý `--retrain` + tăng `--max-hours`; eval bỏ qua (lùi về run cũ hơn nếu có) | `09_full_pipeline_cpu.sh`, `06_eval.sh`, `train_utils --train-verdict` |
| Ngân sách chỉ đủ < 3 epoch → `[WARN]` ngay sau khi đo tốc độ và ở cuối log | `train_utils.epochs_for_budget` (`too_few_epochs`), pipeline |
| Eval đầy đủ chuyển báo cáo cũ vào `reports/archive/<giờ>/` trước khi chạy; `--only` chỉ chuyển file của model được đánh giá lại | `06_eval.sh` (+ `REPORTS_DIR`) |
| `char_accuracy = max(0, 1 − CER)` | `metrics.py` |
| Báo cáo ghi giờ máy kèm độ lệch UTC | `report.py` |
| `PYTHONUNBUFFERED=1` cho log hiện ngay | pipeline |

Test: golden `testcase28..44` (17 testcase mới, tổng 44); toàn bộ `pytest tests/` 284 passed.
Kiểm ngược: trả 3 file về code cũ thì 7 testcase mới (28, 38–43) đỏ.

**Sự cố khi sửa:** một lần chạy test (trước khi runner đặt `REPORTS_DIR` vào thư mục tạm) đã
chuyển 15 file trong `reports/` thật vào `reports/archive/2026-11-05_13-31-04/`. Đã trả lại nguyên
vẹn (cùng ổ đĩa, giữ giờ sửa) và xoá thư mục archive đó; runner nay luôn đặt `REPORTS_DIR`.

**Trạng thái dữ liệu sau khi sửa** (05/11, ~14h):
- `data/runs/parseq_ft/2026-11-05_00-27-27/`: `DONE` đổi thành `UNDERTRAINED` ("val_accuracy tốt
  nhất 0.0% sau 2 epoch") — đúng như code mới lẽ ra đã ghi.
- `data/runs/crnn_ft/2026-11-04_21-58-41/`: giữ `DONE`.
- `reports/` vẫn là bộ của lần 1 (kể cả file `crnn_*` 16/10); lần eval đầy đủ kế tiếp sẽ tự chuyển
  hết vào `reports/archive/`. `reports/final_summary.md` (có trong git) đang sửa đổi bởi lần chạy,
  **không** commit vì chứa số sai.

### Việc tiếp theo

1. Train lại `parseq_ft` đủ epoch — CPU cả một đêm:
   `bash scripts/09_full_pipeline_cpu.sh --models parseq_ft --retrain --max-hours 9`
   (≈ 8 epoch × 57 phút, xong ~7,6 giờ sau khi bắt đầu; eval chạy lại cho mọi model), hoặc GPU
   trên Colab.
2. `trba_ft` (50 M tham số): train trên GPU (Colab). `svtr` (2,3 M, train từ đầu): CPU một đêm.
3. Duyệt tay nhãn VN (`python -m trademark_ocr.vn_review make --n 300` → sửa → `apply`), rồi
   `bash scripts/09_full_pipeline_cpu.sh --from vn_lmdb` để có số liệu VN chính thức.
