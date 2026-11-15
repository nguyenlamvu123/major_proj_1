# Nhật ký các lần chạy pipeline

Mỗi lần chạy thật `scripts/09_full_pipeline_cpu.sh` (hoặc train/eval trên Colab) ghi một mục, **mới
nhất ở trên**: lệnh, thời gian từng bước, kết quả, cái gì sai, đã sửa gì, việc tiếp theo. Log đầy đủ
nằm ở `data/logs/pipeline_<giờ>.log` (không đưa vào git).

## Tình trạng 4 mô hình (cập nhật chiều 05/11)

| Kiến trúc | Biến thể | Trạng thái | Kết quả |
|---|---|---|---|
| CRNN (CTC, 8,4 M) | `crnn_ft` | ✅ xong — CPU, 8 epoch (lần 1) | test US **73,0%**; VN **18,3%** (27,0% bỏ hoa/thường; 0% trên nhãn có dấu) |
| PARSeq (24 M) | `parseq_ft` | ❌ lần 1 chỉ 2 epoch, chưa học được (`UNDERTRAINED`) → ⏳ train lại tối 05/11, CPU 9h | — |
| SVTR-CTC (2,3 M) | `svtr` (từ đầu, không có pretrained) | ⏳ tối 05/11 nếu parseq_ft xong trước ~05:35 | — |
| TRBA (50 M) | `trba_ft` | ⏳ Colab GPU (mục 3 bên dưới) | — |

Bản train từ đầu `crnn`, `trba`, `parseq` chưa lên kế hoạch (chỉ cần nếu muốn so fine-tune với
train từ đầu). `reports/final_summary.md` hiện vẫn là bản sai của lần 1; lần eval kế tiếp sinh lại.

## Việc tiếp theo

### 1. Ban ngày: duyệt tay nhãn VN

Không cần chạy thêm gì, vì tối nay script tự build lại tập test VN từ bản đã duyệt.

```bash
.venv/bin/python -m trademark_ocr.vn_review make --n 300   # rồi sửa data/processed/review_vn.csv
.venv/bin/python -m trademark_ocr.vn_review apply
```

- `make` tạo `data/processed/review_vn.csv` và `review_vn.html` (ảnh cạnh nhãn để xem nhanh).
  Sửa cột `visible_text` thành đúng chữ nhìn thấy trong ảnh (đúng rồi thì giữ nguyên); ảnh không
  dùng được thì để **trống**.
- `apply` phải xong **trước 21:00**: pipeline thấy `split_test_vn_verified.csv` mới hơn thì tự
  build lại `test_vn` trước khi train.
- Sau khi duyệt, tập test VN chỉ còn các mẫu đã duyệt (~300, trừ dòng trống), không còn 2.000:
  số liệu VN đúng hơn nhưng khoảng tin cậy rộng hơn. Tỉ lệ nhãn WIPO sai nằm ở
  `data/processed/review_vn_report.json`.

### 2. Tối 05/11, 21:00 (đã hẹn lịch): `parseq_ft` rồi `svtr`

Lịch trong phiên Claude chạy `data/night_2026-11-05.sh` trong tmux `doan` (không nằm trong git):

1. `09_full_pipeline_cpu.sh --models parseq_ft --retrain --max-hours 9`: khoảng 8 epoch × 57 phút,
   dự kiến xong ~04:45, sau đó eval.
2. Còn ≥ 2 giờ ngân sách trước **08:00 06/11** thì
   `09_full_pipeline_cpu.sh --models svtr --max-hours <phần còn lại> --max-epochs 30`, rồi eval
   lại mọi model. Ngân sách = (08:00 − lúc xong − 10 phút) / 1,15, vì trần cứng là 115% ngân sách.
   Xong lúc 04:45 thì được 2,7h; xong sau ~05:35 thì bỏ qua svtr.
3. Cả chuỗi nằm dưới một khoá `systemd-inhibit`. Script ép `LC_NUMERIC=C`, vì locale `vi_VN` làm
   `awk` in `2,7` mà pipeline từ chối `--max-hours 2,7`.

Log: `data/logs/night_2026-11-05.log` (quyết định của chuỗi) và hai file
`data/logs/pipeline_2026-11-0{5,6}_*.log`. Muốn dừng: `tmux kill-session -t doan`. Lịch chỉ sống
trong phiên Claude đang mở: thoát Claude trước 21:00 thì tự chạy tay
`env -u DATA_ROOT tmux new-session -d -s doan "bash data/night_2026-11-05.sh"`.

Sáng 06/11 đọc `reports/final_summary.md`. Nếu parseq_ft vẫn chưa học được (đủ 8 epoch mà
`val_accuracy` < 1%), log ghi `UNDERTRAINED`, báo cáo bỏ model này; khi đó chuyển parseq_ft sang
Colab như mục 3.

### 3. `trba_ft` trên Colab GPU

Train trên Colab, **eval ở máy local** để mọi model đánh giá trên đúng một bộ test. **Không** tải
lại dữ liệu bằng `01_download_data.ipynb`: danh sách gazette USPTO thay đổi theo ngày, nên tập
train/test sẽ khác lần 1 và không so được với `crnn_ft`/`parseq_ft`. Thay vào đó mang LMDB ở máy
này lên.

**Máy local (một lần):**

```bash
git push origin 16-fix-review                                    # Colab clone từ GitHub
tar -C data -cf data/lmdb_trainval.tar lmdb/train lmdb/val       # ~2,2 GB; ảnh đã nén nên không nén thêm
```

Tải `data/lmdb_trainval.tar` lên Google Drive, vào thư mục `MyDrive/trademark_data/`
(drive.google.com → Mới → Tải tệp lên).

**Colab** (Runtime → Change runtime type → T4 GPU):

1. `docs/colab/00_setup.ipynb`: ở cell 1 đặt `BRANCH = "16-fix-review"`. Nhánh `main` trên GitHub
   chỉ có `detai.txt`, clone nó thì cell 3 báo thiếu `requirements.txt`. Cell 2 bỏ qua clone khi
   thư mục đã có, nên phải clone lại sạch, nhớ **ra khỏi thư mục trước khi xoá**:

   ```python
   %cd /content
   BRANCH = "16-fix-review"
   !rm -rf /content/major_proj_1
   !git clone -b {BRANCH} {REPO_URL} /content/major_proj_1
   %cd /content/major_proj_1
   !git log --oneline -1
   ```
2. Ô train (đặt cuối `00_setup` hoặc trong `02_train`, cùng phiên đã chạy cell 1–3). Bỏ ô `rsync`
   của `02_train`, vì nó chép `trademark_data/lmdb/` cũ trên Drive, là bộ dữ liệu khác:

   ```python
   !mkdir -p /content/lmdb_root && tar -xf /content/drive/MyDrive/trademark_data/lmdb_trainval.tar -C /content/lmdb_root
   !ls /content/lmdb_root/lmdb/train/trademark /content/lmdb_root/lmdb/val    # phải thấy data.mdb lock.mdb
   !bash scripts/05_train.sh trba_ft --max-epochs 20 --num-workers 2 data.root_dir=/content/lmdb_root/lmdb
   ```

   - Cần code từ commit "05_train.sh: check the LMDB dir train.py will read" trở đi (dòng
     `git log --oneline -1` ở cell 2 phải in commit đó). Code cũ kiểm `$DATA_ROOT/lmdb/train` trên
     Drive, bỏ qua `data.root_dir=`, nên báo "thiếu LMDB ở …/trademark_data/lmdb".
   - Đọc LMDB từ đĩa máy ảo, checkpoint ghi vào Drive. Dòng
     `[05_train] checkpoint -> /content/drive/MyDrive/trademark_data/runs/trba_ft/<timestamp>/` là
     nơi lưu thật.
   - Batch giữ 64 như lần chạy CPU.
   - Bị ngắt giữa chừng: máy ảo mới mất `/content/lmdb_root`. Chạy lại dòng `tar`, rồi thêm
     `hydra.run.dir=<thư mục run cũ trên Drive> ckpt_path=<thư mục run cũ>/checkpoints/last.ckpt`
     vào lệnh train.
   - Colab miễn phí có giới hạn phiên, và để tab im lâu cũng bị ngắt. Thời gian 20 epoch trên T4
     chưa đo; bảng ước lượng cũ ghi 2–4 giờ.
3. Xong thì ghi lại `val_accuracy` tốt nhất (nằm trong tên file `checkpoints/epoch=…-val_accuracy=…ckpt`).

**Mang về máy local:**

1. Từ Drive tải `trademark_data/runs/trba_ft/<giờ>/`, chỉ cần `checkpoints/last.ckpt`, `config/`
   và `hparams.yaml` (các checkpoint top-3 khác nặng, không cần), đặt vào
   `data/runs/trba_ft/<giờ>/`.
2. Nếu `val_accuracy` tốt nhất ≥ 1%: `touch data/runs/trba_ft/<giờ>/DONE`. Thiếu `DONE`, lần
   pipeline sau có `trba_ft` trong `--models` sẽ coi là train dở và train tiếp trên CPU.
3. Eval lại mọi model trên bộ test local, vào lúc máy rảnh (không chạy song song với train đêm):
   `bash scripts/09_full_pipeline_cpu.sh --only eval`

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

Xem mục "Việc tiếp theo" ở đầu file (train lại parseq_ft, svtr, trba_ft trên Colab, duyệt tay VN).
