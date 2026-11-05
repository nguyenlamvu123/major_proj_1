# Nhật ký các lần chạy pipeline

Mỗi lần chạy thật `scripts/09_full_pipeline_cpu.sh` (hoặc train/eval trên Colab) ghi một mục, **mới
nhất ở trên**: lệnh, thời gian từng bước, kết quả, cái gì sai, đã sửa gì, việc tiếp theo. Log đầy đủ
nằm ở `data/logs/pipeline_<giờ>.log` (không đưa vào git).

## Tình trạng 4 mô hình (cập nhật chiều 05/11)

| Kiến trúc | Biến thể | Trạng thái | Kết quả |
|---|---|---|---|
| CRNN (CTC, 8,4 M) | `crnn_ft` | ✅ xong — CPU, 8 epoch (lần 1); chưa overfit, còn học thêm được → ⏳ train thêm lên 20 epoch trên máy khác (mục 4) | test US **73,0%**; VN **18,3%** (27,0% bỏ hoa/thường; 0% trên nhãn có dấu) |
| PARSeq (24 M) | `parseq_ft` | ❌ lần 1 chỉ 2 epoch, chưa học được (`UNDERTRAINED`) → ⏳ train lại tối 05/11, CPU 9h | — |
| SVTR-CTC (2,3 M) | `svtr` (từ đầu, không có pretrained) | ⏳ tối 05/11 nếu parseq_ft xong trước ~05:35 | — |
| TRBA (50 M) | `trba_ft` | ⏳ Colab GPU (mục 3 bên dưới) | — |

Bản train từ đầu `crnn`, `trba`, `parseq` chưa lên kế hoạch (chỉ cần nếu muốn so fine-tune với
train từ đầu). `reports/final_summary.md` hiện vẫn là bản sai của lần 1; lần eval kế tiếp sinh lại.

Đồ thị train (loss train / val_loss / val_accuracy theo epoch, kết luận overfit) ở
`reports/figures/`, giải thích trong `BAO_CAO_TONG_KET_DAT_LIEU_HUAN_LUYEN.md` mục 2.5. Mỗi lần
eval vẽ lại; vẽ riêng: `.venv/bin/python -m trademark_ocr.train_curves --out-dir reports/figures`.

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

1. Từ Drive tải `trademark_data/runs/trba_ft/<giờ>/`: `checkpoints/last.ckpt`, `config/`,
   `hparams.yaml`, **`events.out.tfevents.*`** (log để vẽ đồ thị train) và **tên** các file
   `checkpoints/epoch=…ckpt` (cần để `--continue` biết đã train mấy epoch; chép file nặng, hoặc
   `touch` một file rỗng cùng tên). Đặt vào `data/runs/trba_ft/<giờ>/`.
2. Nếu `val_accuracy` tốt nhất ≥ 1%: `touch data/runs/trba_ft/<giờ>/DONE`. Thiếu `DONE`, lần
   pipeline sau có `trba_ft` trong `--models` sẽ coi là train dở và train tiếp trên CPU.
3. Eval lại mọi model trên bộ test local, vào lúc máy rảnh (không chạy song song với train đêm):
   `bash scripts/09_full_pipeline_cpu.sh --only eval`

### 4. Train thêm `crnn_ft` lên 20 epoch trên một máy Ubuntu khác (yếu hơn)

Viết để một người hoặc một AI khác làm theo từng bước, không cần đọc phần còn lại. Nhãn **[vunl]**
là máy hiện tại (có dữ liệu, checkpoint, các model khác); **[máy B]** là máy Ubuntu yếu hơn sẽ train.

**Vì sao làm.** crnn_ft mới train 8 epoch (giới hạn 3 giờ CPU). Đồ thị
(`reports/figures/train_curves_crnn_ft.png`, báo cáo mục 2.5) cho thấy val_loss vẫn giảm tới epoch
cuối và val_accuracy vẫn tăng (74,07% ở epoch 8): **chưa overfit, còn học thêm được**. Train thêm 12
epoch từ checkpoint epoch 8, xem accuracy còn tăng không và có xuất hiện overfit không.

**Cơ chế** (`09_full_pipeline_cpu.sh --continue`, DESIGN.md 16.12):
- Tạo run **mới** `data/runs/crnn_ft/<giờ chạy>/`, nạp toàn bộ trọng số từ
  `data/runs/crnn_ft/2026-11-04_21-58-41/checkpoints/last.ckpt`.
- Lịch learning rate **mới** cho phần còn thiếu; LR khởi động lại nên 1–2 epoch đầu có thể chững.
- `--max-epochs 20` là **tổng**, nên run mới train thêm tối đa 12 epoch.
- Run cũ giữ nguyên, **không xoá**.

#### Bước 1 — [vunl] đóng gói

```bash
cd /mnt/1864514f-3a41-4b8f-9907-2fbcd9d2815b/airi/doanchuyennganh1
git status --short                 # code phải đã commit (trừ reports/…)
git push origin 16-fix-review      # máy B clone code từ GitHub
ls -la data/lmdb_trainval.tar      # đã có từ 05/11 (lmdb/train + lmdb/val, 2,2 GB); thiếu thì:
#   tar -C data -cf data/lmdb_trainval.tar lmdb/train lmdb/val
tar -C data -cf data/crnn_ft_run.tar runs/crnn_ft/2026-11-04_21-58-41     # ~520 MB, CẢ thư mục run
```

Chép hai file `data/lmdb_trainval.tar` và `data/crnn_ft_run.tar` sang máy B: USB, hoặc
`scp data/*.tar <user>@<máy B>:~/`, hoặc qua Google Drive. **Không** tải lại dữ liệu trên máy B
(`01_get_meta`…): danh sách USPTO đổi theo ngày, tập train/val sẽ khác và không so được.

Thư mục run phải mang **đủ**:
- `checkpoints/last.ckpt`: trọng số để nạp.
- `checkpoints/epoch=7-…ckpt`: tên file cho biết đã train mấy epoch.
- `config/`, `pipeline_train.env`, `DONE`.
- `events.out.tfevents.*`: log để vẽ đồ thị nối 8 epoch cũ với các epoch mới.

#### Bước 2 — [máy B] code, dữ liệu, môi trường

Yêu cầu:
- Ubuntu có `git`, `python3`, `python3-venv` (`sudo apt install python3-venv`) và mạng (pip, GitHub).
- Ổ trống ≥ 8 GB: LMDB 2,4 GB, venv ~2,2 GB, mỗi checkpoint ~100 MB.
- Đặt repo ở đường dẫn **không có dấu cách / tiếng Việt có dấu**: đường dẫn checkpoint đi vào
  override Hydra.

```bash
git clone -b 16-fix-review https://github.com/nguyenlamvu123/major_proj_1.git ~/major_proj_1
cd ~/major_proj_1
git log --oneline -3          # phải có commit "train curves + --continue" (hoặc mới hơn)
mkdir -p data
tar -xf ~/lmdb_trainval.tar -C data        # -> data/lmdb/train/trademark, data/lmdb/val
tar -xf ~/crnn_ft_run.tar  -C data         # -> data/runs/crnn_ft/2026-11-04_21-58-41
ls data/lmdb/train/trademark data/lmdb/val                       # thấy data.mdb lock.mdb
ls data/runs/crnn_ft/2026-11-04_21-58-41/checkpoints             # thấy epoch=7-…ckpt và last.ckpt
bash scripts/09_full_pipeline_cpu.sh --only setup                # tạo .venv (torch CPU) + parseq, ~10 phút
.venv/bin/python -m pytest -q tests/test_golden.py               # phải xanh hết
```

`--only setup` chỉ dựng môi trường. Không chạy `09_full_pipeline_cpu.sh` khi **thiếu**
`--from train`: máy B không có dấu `data/.pipeline/`, nên script sẽ tải lại dữ liệu từ đầu.

#### Bước 3 — [máy B] chọn batch theo RAM, xem kế hoạch

```bash
nproc; free -h                      # số luồng CPU, RAM còn trống ("available")
bash scripts/09_full_pipeline_cpu.sh --dry-run --from train --skip eval \
  --models crnn_ft --continue --max-epochs 20 --max-hours 8 --batch-size 64 --num-workers 4
```

Dry-run phải in:

```
[plan] train:crnn_ft    | continue | run MỚI nạp trọng số runs/crnn_ft/2026-11-04_21-58-41/checkpoints/last.ckpt (đã train 8 epoch), ngân sách 8h: đo tốc độ rồi chọn ≤ 12 epoch (tổng ≤ 20), …
[plan] eval             | skip   | không chọn
```

Chọn batch theo RAM còn trống. Các mức dưới đây là ước lượng an toàn, chưa đo trên máy B. Máy
`vunl` chạy batch 64 với 4 worker trên 31 GB; máy 7,6 GB cũ từng bị OOM với batch 64.

| RAM available | Cờ |
|---|---|
| ≥ 12 GB | `--batch-size 64 --num-workers 4` (giống lần 1) |
| 6–12 GB | `--batch-size 32 --num-workers 2` |
| < 6 GB | `--batch-size 16 --num-workers 1` |

Đổi batch khi train tiếp vẫn được, vì lịch LR là lịch mới và parseq tự nhân LR theo `batch/256`.
Tuy vậy, cùng batch 64 thì so với lần 1 sạch hơn: chỉ khác số epoch.

#### Bước 4 — [máy B] train (trong tmux, qua đêm)

```bash
tmux new -s crnn
bash scripts/09_full_pipeline_cpu.sh --from train --skip eval \
  --models crnn_ft --continue --max-epochs 20 --max-hours 8 --batch-size 64 --num-workers 4
# Ctrl-b d để thoát ra; xem lại: tmux attach -t crnn
```

- Đầu log có dòng `[PIPELINE] crnn_ft: <x> phút/epoch -> train <N> epoch`. Trên `vunl` là 17,1
  phút/epoch (i5-10400, batch 64). Máy chậm gấp 2–3 lần thì 8 giờ chỉ đủ khoảng 5–8 epoch, không
  đủ 12. **Không sao:** đêm sau chạy **lại đúng lệnh đó**. Script thấy mới có tổng 8 + N epoch nên
  train thêm từ run vừa xong, cho tới khi đủ 20. Mỗi đêm là một đợt LR mới, và đồ thị có một vạch
  "train tiếp" ở mỗi chỗ nối.
- `--skip eval`: không đánh giá trên máy B. Eval làm ở `vunl` (bước 6), để mọi model chấm trên cùng
  một bộ test với cùng code.
- Theo dõi: `tail -f data/logs/pipeline_*.log`. Checkpoint ở `data/runs/crnn_ft/<giờ chạy>/checkpoints/`;
  tên file có `val_accuracy=…`.
- Xong: log có `crnn_ft: xong N epoch` và thư mục run mới có file `DONE`.

**Sự cố:**

| Triệu chứng | Làm gì |
|---|---|
| Log có `Killed` / mã thoát 137 (hết RAM) **trong epoch đầu** (run mới chưa có `checkpoints/epoch=…ckpt`) | Chạy lại với batch nhỏ hơn một mức. Script tạo một run mới khác từ checkpoint cũ; run hỏng không có checkpoint nên eval và đồ thị tự bỏ qua. Muốn gọn thì xoá run hỏng đó: thư mục có `continue_from=` trong `pipeline_train.env`, **không bao giờ** xoá `2026-11-04_21-58-41` |
| Hết RAM / mất điện / tắt máy **sau** khi đã có checkpoint của run mới | Chạy lại **đúng lệnh cũ, giữ nguyên batch**: plan in `resume`, train tiếp run mới từ `last.ckpt` của nó (lịch LR phụ thuộc batch nên không đổi batch giữa chừng) |
| `[WARN] … chạm trần thời gian` | Như trên: chạy lại đúng lệnh, run train tiếp |
| Plan in `skip … chưa có run nào có checkpoint để train tiếp` | Chưa giải nén `crnn_ft_run.tar` vào `data/` (bước 2) |
| Plan in `skip … không biết last.ckpt ở epoch nào` | Thiếu file `checkpoints/epoch=7-…ckpt` (đã chép thiếu) |
| Plan in `skip … đã train 20 epoch` | Đã xong |

**Không** dùng `--retrain`, vì nó train lại từ trọng số pretrained và bỏ 8 epoch đã có. **Không**
xoá hay sửa thư mục `2026-11-04_21-58-41`.

#### Bước 5 — [máy B] mang kết quả về

```bash
ls -d data/runs/crnn_ft/*/              # run cũ + một (hoặc vài, nếu nhiều đêm) run mới
grep -H continue_from data/runs/crnn_ft/*/pipeline_train.env     # các run mới
.venv/bin/python -m trademark_ocr.train_curves --models crnn_ft --out-dir /tmp/curves   # xem trước, tuỳ chọn
tar -C data -cf ~/crnn_ft_continue_result.tar $(cd data && ls -d runs/crnn_ft/*/ | grep -v 2026-11-04_21-58-41)
```

Chép `~/crnn_ft_continue_result.tar` về `vunl`. Mang **cả thư mục**, không chỉ `last.ckpt`:
eval cần `config/`, đồ thị cần `events.out.tfevents.*` và `pipeline_train.env`.

#### Bước 6 — [vunl] đánh giá, đồ thị, cập nhật báo cáo

Làm khi `vunl` **không** đang train (không chạy song song với chuỗi đêm):
`pgrep -af 09_full_pipeline_cpu` phải rỗng.

```bash
cd /mnt/1864514f-3a41-4b8f-9907-2fbcd9d2815b/airi/doanchuyennganh1
tar -xf ~/crnn_ft_continue_result.tar -C data       # -> data/runs/crnn_ft/<giờ chạy trên máy B>/
bash scripts/09_full_pipeline_cpu.sh --only eval    # eval mọi model + vẽ lại reports/figures/
```

Kiểm:
- `reports/figures/train_curves.md`, mục crnn_ft: dòng "Nối các lần train" có đủ hai (hoặc nhiều)
  run, bảng có epoch 1 → 20.
- `reports/figures/train_curves_crnn_ft.png`: có vạch đứt "train tiếp" ở giữa epoch 8 và 9.
- `reports/final_summary.md` có dòng crnn_ft với kết quả của run mới (đường dẫn checkpoint trong
  `reports/crnn_ft_us.summary.json`).

**Cập nhật báo cáo** `BAO_CAO_TONG_KET_DAT_LIEU_HUAN_LUYEN.md`, viết cho người hướng dẫn đọc:
- **Mục 2.5, phần crnn_ft:**
  - Đổi tiêu đề thành "… 20 epoch (8 epoch ngày 04/11 + 12 epoch train tiếp ngày …)".
  - Thay bảng bằng bảng crnn_ft trong `reports/figures/train_curves.md` (chép nguyên, đã có dấu
    phẩy thập phân).
  - Viết lại phần "Nhận xét" theo số mới:
    1. val_accuracy tốt nhất bao nhiêu, ở epoch nào; tăng bao nhiêu so với 74,07%.
    2. Kết luận overfit: lấy câu "**Kết luận:**" trong `train_curves.md`. Nếu là "có dấu hiệu
       overfit", nêu epoch val_loss thấp nhất và cho biết checkpoint dùng để test là epoch tốt
       nhất theo val_accuracy, không phải epoch cuối.
    3. Các epoch ngay sau chỗ nối (epoch 9–10) có chững hoặc giảm không. Nếu có, giải thích là do
       learning rate khởi động lại (mục 2.6), không phải overfit.
    4. Test US / VN mới (`reports/crnn_ft_us.summary.json`, `crnn_ft_vn.summary.json`, hoặc
       `final_summary.md`) so với 73,0% / 18,3% của bản 8 epoch.
- **Mục 2.6:** thay đoạn "**Kế hoạch:**" bằng một câu đã làm: máy nào (CPU, số luồng, RAM), batch,
  bao nhiêu phút/epoch (`benchmark.json` của run mới), mấy đêm.
- **Không** đổi các số của lần 1 (8 epoch) ở chỗ khác. Đó là lịch sử, giữ để so sánh.

Ghi thêm một mục "Lần 2 — crnn_ft train tiếp trên máy B" ở đầu phần nhật ký bên dưới (trên "Lần 1"),
cùng khuôn: lệnh, máy, thời gian, kết quả, sự cố. Cập nhật bảng "Tình trạng 4 mô hình" ở đầu file.

Commit trên `vunl`:

```bash
git add BAO_CAO_TONG_KET_DAT_LIEU_HUAN_LUYEN.md docs/nhat_ky_chay.md reports/figures/
git commit -m "16-fix-review: crnn_ft train tiếp lên 20 epoch: đồ thị + báo cáo"
```

**Không** commit `data/` (đã `.gitignore`, chứa checkpoint vài trăm MB) và `reports/*.csv`.

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
