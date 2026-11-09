# Nhật ký các lần chạy pipeline

Mỗi lần chạy thật `scripts/09_full_pipeline_cpu.sh` (hoặc train/eval trên Colab) ghi một mục, **mới
nhất ở trên**: lệnh, thời gian từng bước, kết quả, cái gì sai, đã sửa gì, việc tiếp theo. Log đầy đủ
nằm ở `data/logs/pipeline_<giờ>.log` (không đưa vào git).

## Tình trạng 4 mô hình (cập nhật 09/11)

| Kiến trúc | Biến thể | Trạng thái | Kết quả test |
|---|---|---|---|
| CRNN (CTC, 8,4 M) | `crnn_ft` | ✅ train lại **từ đầu 20 epoch** trên máy C (CPU, 07–08/11, lần 4): val **76,35%** (epoch 20), vẫn chưa overfit. ⚠️ **Checkpoint chưa mang về** (`checkpoints/` trống) → chưa eval. Bản 8 epoch (lần 1, val 74,07%) vẫn là bản đang được chấm | (bản 8 epoch) US **73,0%**; VN **18,3%** (27,0% bỏ hoa/thường; 0% trên nhãn có dấu) |
| PARSeq (24 M) | `parseq_ft` | ✅ xong — CPU, 7 epoch (lần 2); val gần bão hoà từ epoch 4–5, train thêm khó được gì | US **88,7%**; VN **27,6%** (41,1% bỏ hoa/thường) |
| SVTR-CTC (2,3 M) | `svtr` (từ đầu, không có pretrained) | ✅ xong — CPU, 10 epoch (lần 2); val_loss vẫn giảm, train thêm được bằng `--continue` | US **69,4%**; VN **11,4%** (16,4% bỏ hoa/thường) |
| TRBA (50 M) | `trba_ft` | ✅ **20/20 epoch** trên Colab T4, 4 phiên (05–07/11, lần 3); val tốt nhất **86,72%** (epoch 19), `last.ckpt` = epoch 20 (86,47%); đã mang về, có `DONE`. Chưa eval | — |

Bản train từ đầu `crnn`, `trba`, `parseq` chưa lên kế hoạch (chỉ cần nếu muốn so fine-tune với
train từ đầu). `reports/final_summary.md` là bản của lần 2 (06/11 07:10): 3 model trên test US
(2.847 mẫu) và test VN (2.000 mẫu **chưa duyệt tay**). Chưa có trba_ft và crnn_ft 20 epoch.

**Việc cần làm tiếp (09/11):**
1. crnn_ft 20 epoch: checkpoint chỉ có ở máy C. Eval ngay trên máy C (dự phòng Colab), mang 4 file
   kết quả và `last.ckpt` (khoảng 135 MB) về, rồi `touch data/runs/crnn_ft/2026-11-07_20-44-52/DONE`.
   Chi tiết ở [`docs/eval_crnn_ft_may_C.md`](eval_crnn_ft_may_C.md).
2. trba_ft (checkpoint đã ở máy này): `bash scripts/06_eval.sh --only trba_ft`, chạy sau bước 1.
   Chưa có `last.ckpt` của run 20 epoch thì **đừng** eval crnn_ft ở máy này, kể cả qua
   `09_full_pipeline_cpu.sh --only eval`: eval lấy run mới nhất có checkpoint, nên sẽ chấm bản 8 epoch
   (04/11) và ghi đè.
3. Cập nhật số test vào `BAO_CAO_TONG_KET_DAT_LIEU_HUAN_LUYEN.md` mục 2.5 (các ô "chờ eval").

Checkpoint được chấm test là `checkpoints/last.ckpt`. Với Lightning 2.6, file này chỉ được ghi lại
ở epoch nào lọt **top-3 val_accuracy**, nên nó là "epoch gần nhất lọt top-3", không nhất thiết là
epoch cuối. Epoch thật sự được chấm của từng model ghi ở "Lần 2 → Checkpoint được chấm".

Đồ thị train (loss train / val_loss / val_accuracy theo epoch, kết luận overfit) ở
`reports/figures/`, giải thích trong `BAO_CAO_TONG_KET_DAT_LIEU_HUAN_LUYEN.md` mục 2.5. Mỗi lần
eval vẽ lại; vẽ riêng: `.venv/bin/python -m trademark_ocr.train_curves --out-dir reports/figures`.

## Việc tiếp theo

### 1. Duyệt tay nhãn VN (chưa làm)

```bash
.venv/bin/python -m trademark_ocr.vn_review make --n 300   # rồi sửa data/processed/review_vn.csv
.venv/bin/python -m trademark_ocr.vn_review apply
bash scripts/09_full_pipeline_cpu.sh --only vn_lmdb,eval      # build lại test VN + chấm lại mọi model
```

- `make` tạo `data/processed/review_vn.csv` và `review_vn.html` (ảnh cạnh nhãn để xem nhanh).
  Sửa cột `visible_text` thành đúng chữ nhìn thấy trong ảnh (đúng rồi thì giữ nguyên); ảnh không
  dùng được thì để **trống**.
- Dòng thứ ba không train gì, chỉ build lại `test_vn` từ bản đã duyệt rồi chấm lại mọi model có
  checkpoint (vài phút). Đừng chạy nó lúc máy đang train. Lần pipeline có train kế tiếp cũng tự
  build lại `test_vn` nếu thấy `split_test_vn_verified.csv` mới hơn.
- Sau khi duyệt, tập test VN chỉ còn các mẫu đã duyệt (~300, trừ dòng trống), không còn 2.000:
  số liệu VN đúng hơn nhưng khoảng tin cậy rộng hơn. Tỉ lệ nhãn WIPO sai nằm ở
  `data/processed/review_vn_report.json`.

### 2. ✅ Tối 05/11: `parseq_ft` rồi `svtr` — đã chạy xong

Kết quả và thời gian ở mục "Lần 2" bên dưới. Lịch hẹn trong phiên Claude chỉ chạy một lần và đã
hết; không còn lịch nào tự chạy. Script chuỗi `data/night_2026-11-05.sh` (không nằm trong git) giữ
lại để tham khảo.

### 3. `trba_ft` trên Colab GPU

**Tình trạng 09/11: ✅ xong 20/20 epoch** (diễn biến ở "Lần 3"). Run đã mang về
`data/runs/trba_ft/2026-11-05_08-28-47/`, có `DONE`. Các mục 3a–3c dưới đây giữ lại làm hướng dẫn
cho lần train trên Colab sau. Rút kinh nghiệm từ lần 3: dùng `--num-workers 0` hoặc `1`, vì với
`--num-workers 2` tiến trình đọc dữ liệu bị kill do hết RAM ở 3/4 phiên.

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
   - Bị ngắt giữa chừng (mất kết nối, hết 12 giờ, hết hạn mức GPU): **train tiếp theo mục 3a**.
     Chạy lại ô train ở trên mà **thiếu** `ckpt_path=` là tạo run mới, train lại từ epoch 1.
   - Colab miễn phí có giới hạn phiên, và để tab im lâu cũng bị ngắt. Lần đầu (tháng 11) được
     8 epoch thì hết hạn mức GPU.
3. Xong thì ghi lại `val_accuracy` tốt nhất (nằm trong tên file `checkpoints/epoch=…-val_accuracy=…ckpt`).

#### 3a. Train tiếp run bị ngắt (cùng tài khoản Google)

**Cơ chế.** Override `ckpt_path=<run>/checkpoints/last.ckpt` khôi phục trọng số, optimizer, lịch
learning rate OneCycle, bộ đếm epoch/bước và danh sách top-3 checkpoint, rồi chạy nốt các epoch còn
lại của **chính run đó**. Run chưa chạy hết 20 epoch nên lịch learning rate nối tiếp đúng chỗ. Đây là
chỗ khác với `--continue` (mục 4), vốn dành cho run đã train **xong**. Đã thử trên máy local với
trba_ft thu nhỏ (8 batch/epoch, 3 epoch): `kill -9` giữa epoch 2 rồi chạy lại với `ckpt_path`.
Lightning in "Restored all states", train nốt epoch 2–3, không lỗi. Đồ thị nối liền qua hai file log.

Trên Colab, sau khi chạy lại 3 ô đầu của `00_setup` (mount Drive, clone `16-fix-review`, cài thư
viện):

**Ô 1 — xem checkpoint trên Drive còn nguyên không.** Máy ảo bị thu hồi đột ngột có thể làm file
đang ghi dở chưa kịp đồng bộ lên Drive.

```python
RUN = "/content/drive/MyDrive/trademark_data/runs/trba_ft/<giờ>"   # thư mục có các file epoch=…ckpt
!ls -la {RUN}/checkpoints
import torch
c = torch.load(f"{RUN}/checkpoints/last.ckpt", map_location="cpu", weights_only=False)
print("last.ckpt: đã xong", c["epoch"] + 1, "epoch | global_step", c["global_step"])
```

- `global_step` = số epoch × 801 (51.230 ảnh / batch 64, làm tròn lên). 8 epoch thì in `8 | 6408`.
- Số epoch in ra có thể **nhỏ hơn** epoch đang chạy lúc bị ngắt. Lightning 2.6 chỉ ghi lại
  `last.ckpt` ở epoch lọt **top-3 val_accuracy**, nên nó là epoch gần nhất lọt top-3. Train tiếp từ
  đó vẫn đúng, chỉ phải chạy lại các epoch sau nó.
- `torch.load` báo lỗi (file ghi dở): thay `last.ckpt` trong ô 2 bằng file `epoch=…ckpt` có số
  epoch lớn nhất. File đó cũng chứa đủ trạng thái để train tiếp.

**Ô 2 — train tiếp.**

```python
!mkdir -p /content/lmdb_root && tar -xf /content/drive/MyDrive/trademark_data/lmdb_trainval.tar -C /content/lmdb_root
!bash scripts/05_train.sh trba_ft --max-epochs 20 --num-workers 2 data.root_dir=/content/lmdb_root/lmdb hydra.run.dir={RUN} ckpt_path={RUN}/checkpoints/last.ckpt
```

Chạy đúng khi đủ cả ba điều sau:

- dòng `[05_train] checkpoint -> …` trỏ đúng `RUN` cũ;
- log có `Restored all states from the checkpoint at …`;
- thanh tiến trình hiện `Epoch <số epoch ô 1 in ra − 1>/20` trong vài phút rồi tới
  `Epoch <số epoch ô 1 in ra>/20`. Thấy `Epoch 0/20` là sai, phải dừng ngay.
  - Lightning đếm epoch từ 0, nên `Epoch 8/20` là epoch thứ 9.
  - Mẫu số là 20 chứ không phải 19: TRBA có BatchNorm nên SWA cộng thêm một lượt (output ở A cũng
    hiện `Epoch 7/20`).
  - Vài phút đầu: Lightning 2.6 chấm val lại đúng bước vừa khôi phục, và ghi lại checkpoint
    `epoch=<N−1>-…ckpt` của bước đó (thấy khi thử trên máy local).

Không được đổi:

- `--max-epochs 20` và batch (mặc định 64; không truyền `--batch-size`). Đổi một trong hai thì số
  bước của lịch OneCycle khác bản đã lưu, dẫn tới lỗi "Tried to step … times" hoặc learning rate
  sai. `--num-workers` đổi thoải mái.
- `hydra.run.dir` và `ckpt_path` phải cùng một run. Thiếu `ckpt_path` là run mới từ epoch 1.

Bị ngắt lần nữa thì chạy lại đúng ô 1 và ô 2 với cùng `RUN`. Mỗi lần chỉ mất phần epoch chưa kịp ghi
checkpoint.

#### 3b. Chuyển run sang một tài khoản Google khác

> **Lưu ý điều khoản.** FAQ của Colab (<https://research.google.com/colaboratory/faq.html>, mục
> các hoạt động bị cấm) ghi rõ: "using multiple accounts to work around access or resource usage
> restrictions". Dùng tài khoản thứ hai **chỉ để né** hạn mức GPU của tài khoản thứ nhất là vi phạm,
> và có thể khiến tài khoản bị hạn chế. Có hai cách không vi phạm, đều train tiếp theo mục 3a, không
> cần chuyển gì:
>
> - chờ hạn mức GPU tự mở lại (Google không công bố bao lâu, vì hạn mức thay đổi theo thời gian);
> - mua compute units (gói Pay As You Go) cho chính tài khoản đang giữ run.
>
> Các bước dưới đây dùng khi chuyển sang một tài khoản được phép dùng, ví dụ tài khoản có gói trả phí.

Checkpoint và dữ liệu nằm trên Drive của tài khoản cũ (gọi là A); Colab chạy bằng tài khoản mới (B)
chỉ mount được Drive của B. Có hai cách. Cả hai đều giữ **đúng đường dẫn**
`/content/drive/MyDrive/trademark_data/runs/trba_ft/<giờ>/` như lúc A train. Lý do: `last.ckpt` lưu
đường dẫn thư mục checkpoint. Nếu đường dẫn khác, Lightning cảnh báo "The dirpath has changed…" và
lập lại danh sách top-3 từ rỗng: các file `epoch=…ckpt` cũ không bao giờ bị dọn, tốn thêm vài GB
Drive. Kết quả train vẫn đúng.

Run `trba_ft` hiện có là `2026-11-05_08-28-47` (giờ UTC). Notebook đã ghi chú sẵn cho cách 1 nằm ở
máy `vunl`: `~/Downloads/00_setup.ipynb`, bản gốc tải từ A là `~/Downloads/00_setup.A_goc.ipynb`.

Dung lượng: mỗi checkpoint TRBA chứa cả trạng thái optimizer (AdamW), **0,8 GB** (đo trên máy local).
Một run gồm `last.ckpt` và 3 file `epoch=…ckpt`, tức khoảng 3,2 GB. Drive miễn phí có 15 GB.

##### Cách 1 — không chép: lối tắt quyền chỉnh sửa, giữ nguyên tên (đang dùng)

B làm việc thẳng trên thư mục của A. Checkpoint mới ghi vào Drive của A; file mới do B tạo thì tính
vào dung lượng của B.

1. **[drive.google.com, tài khoản A]** Mở `trademark_data/runs/trba_ft/<giờ>/checkpoints/`, tạo thư
   mục `trademark_data/backup_trba_ft_8epoch/` và **di chuyển** (không sao chép) 3 file `epoch=…ckpt`
   vào đó. **Để nguyên `last.ckpt`.** Lý do:
   - Khi train ở B, Lightning xoá dần checkpoint cũ khỏi top-3. 3 file này thuộc sở hữu của A, mà
     Drive thường chỉ cho chủ sở hữu xoá file: B xoá không được thì train dừng vì lỗi.
   - Lightning kiểm file còn tồn tại trước khi xoá (`lightning_fabric/plugins/io/torch_io.py`,
     `remove_checkpoint`), nên file đã chuyển đi thì nó bỏ qua.
   - Đã thử trên máy local với trba_ft thu nhỏ: chuyển hết `epoch=…ckpt` đi, hạ điểm top-3 cũ trong
     `last.ckpt` để epoch mới chắc chắn đẩy chúng ra (buộc Lightning xoá các file đã chuyển), rồi
     train tiếp. Không lỗi; 3 checkpoint mới và `last.ckpt` đều được ghi.
   - Trạng thái 8 epoch được giữ nguyên ở thư mục dự phòng.
2. **[drive.google.com, tài khoản A]** Chuột phải thư mục `trademark_data` → Chia sẻ → email của B,
   quyền **Người chỉnh sửa** → Gửi.
3. **[drive.google.com, tài khoản B]** Trong "Drive của tôi" phải **chưa có** thư mục nào tên
   `trademark_data` (có thì: trống → xoá; có dữ liệu → đổi tên). Rồi vào "Được chia sẻ với tôi" →
   chuột phải `trademark_data` → Sắp xếp → Thêm lối tắt → Drive của tôi. **Giữ nguyên tên.**
4. **[Colab, tài khoản B]** Mỗi phiên chạy theo thứ tự:
   - mount Drive bằng tài khoản B, nhưng **không** `os.makedirs(DATA_ROOT)`: lối tắt chưa đồng bộ
     thì lệnh đó âm thầm tạo một thư mục trống che mất lối tắt. Thay bằng kiểm
     `{RUN}/checkpoints/last.ckpt` tồn tại;
   - clone, cài thư viện, kiểm GPU (3 ô đầu `00_setup` + ô kiểm GPU);
   - **kiểm quyền ghi** trước khi train:
     - (a) **phiên đầu ở B:** `checkpoints/` không còn file `epoch=…` nào (cả 3 file của A đã chuyển
       đi). Qua kiểm thì ghi file mốc `{RUN}/B_da_kiem_checkpoint.txt`; các phiên sau thấy mốc thì
       bỏ qua (a). Không thể lọc theo số epoch, vì ngay khi train tiếp B tự ghi lại
       `epoch=7-…ckpt`, trùng tên với file của A;
     - (b) tạo được và xoá được một file của B trong thư mục run;
     - (c) ghi đè được một file nhỏ của A (`hparams.yaml`, ghi lại đúng nội dung cũ) theo đúng cách
       Lightning ghi đè `last.ckpt`: `fsspec` transaction, ghi ra file tạm rồi chuyển đè. Cần kiểm
       (c) vì Lightning 2.6 nuốt `PermissionError` lúc lưu: hỏng ở đây thì train vẫn chạy mà
       `last.ckpt` không được cập nhật;
   - ô 1 + ô 2 của mục 3a. Riêng dòng train dùng đúng kiểu lệnh A đã chạy:
     `DATA_ROOT=/content/lmdb_root bash scripts/05_train.sh trba_ft --max-epochs 20 --num-workers 2 hydra.run.dir={RUN} ckpt_path={RUN}/checkpoints/last.ckpt`.
5. `last.ckpt` hỏng: chép file `epoch=…ckpt` có số epoch lớn nhất (phiên đầu: trong
   `backup_trba_ft_8epoch/`) **vào** `{RUN}/checkpoints/` rồi dùng bản chép làm `ckpt_path`. Bản chép
   phải nằm trong `{RUN}/checkpoints/`, vì `train.py` lấy thư mục run từ `ckpt_path` (lên 2 cấp).
6. Không để A và B cùng train run này một lúc. Xong 20 epoch thì mang về máy local như dưới đây (tải
   từ A hay B đều được); `backup_trba_ft_8epoch/` xoá được sau khi eval xong.

Chưa kiểm chứng được trên Drive thật: việc ghi qua lối tắt trên Colab, và quyền xoá/ghi đè file của
người khác. Vì vậy bước kiểm quyền ghi phải chạy trước khi train. Lỗi ở (b) hoặc (c) thì dùng cách 2.

##### Cách 2 — chép run sang Drive của B

Bản gốc ở A còn nguyên làm dự phòng. Không phụ thuộc quyền ghi qua lối tắt.

1. **[drive.google.com, tài khoản A]** Chia sẻ thư mục `trademark_data` cho B với quyền **Người xem**.
   Quyền xem còn chặn việc lỡ ghi checkpoint mới vào Drive của A: lỡ trỏ nhầm thì báo lỗi quyền truy
   cập, không ghi lẫn.
2. **[drive.google.com, tài khoản B]** "Được chia sẻ với tôi" → chuột phải `trademark_data` → Sắp xếp →
   Thêm lối tắt → Drive của tôi, rồi đổi tên lối tắt thành `trademark_data_A`, để khỏi trùng với
   thư mục `trademark_data` của chính B.
3. **[Colab, tài khoản B]** Chạy 3 ô đầu `00_setup` (`BRANCH = "16-fix-review"`, mount bằng B). Cell 1
   tự tạo `MyDrive/trademark_data/` của B.
4. **[Colab B]** Chép run, chỉ làm một lần, có chốt chặn để không chép đè lên tiến độ đã train ở B:

   ```python
   SRC = "/content/drive/MyDrive/trademark_data_A"                    # lối tắt tới Drive của A (chỉ đọc)
   GIO = "2026-11-05_08-28-47"                                          # tên thư mục run trba_ft bên A
   RUN = f"/content/drive/MyDrive/trademark_data/runs/trba_ft/{GIO}"   # Drive của B, cùng đường dẫn như A
   !ls -la {SRC}/runs/trba_ft/{GIO}/checkpoints
   !test -e {RUN} && echo "ĐÃ CÓ {RUN} — không chép đè" || (mkdir -p /content/drive/MyDrive/trademark_data/runs/trba_ft && cp -r {SRC}/runs/trba_ft/{GIO} /content/drive/MyDrive/trademark_data/runs/trba_ft/ && echo "chép xong")
   !ls -la {RUN}/checkpoints                                           # cùng kích thước như bên A
   ```

   Nếu `/content/drive/MyDrive/` không có `trademark_data_A`: chờ vài phút cho lối tắt đồng bộ rồi
   chạy `drive.mount('/content/drive', force_remount=True)`. Vẫn không thấy thì tải thư mục run từ
   Drive của A về máy, rồi tải lên Drive của B vào đúng `MyDrive/trademark_data/runs/trba_ft/<giờ>/`.
5. **[Colab B]** Ô 1 và ô 2 của mục 3a với `RUN` ở bước 4; dòng `tar` đọc từ lối tắt:
   `!mkdir -p /content/lmdb_root && tar -xf {SRC}/lmdb_trainval.tar -C /content/lmdb_root`.
   Dòng `[05_train] checkpoint -> …` phải là `RUN` trên Drive của **B**.
6. Bị ngắt tiếp thì làm lại bước 3 và 5. Xong 20 epoch thì mang về từ Drive của **B**.

#### 3c. Phiên 2 (và các phiên sau): sang tài khoản C, làm trên máy khác

Theo cách 1, chỉ cần trình duyệt; máy khác không cần cài gì. Notebook dùng chung cho mọi phiên:
[`docs/colab/train_tiep_trba_ft.ipynb`](colab/train_tiep_trba_ft.ipynb). Đây là bản `00_setup` tải
từ A, đã ghi chú từng ô, ô train cũ đã vô hiệu hoá, có thêm ô B1 (kiểm quyền ghi) và B2 (train tiếp).
Ước lượng thời gian: mỗi epoch ~33,5 phút trên T4, lượt SWA cuối ~30 phút, chuẩn bị mỗi phiên
~15 phút. Phiên 1 (B, tối đa ~5 giờ) dự kiến tới khoảng epoch 16/20, phiên 2 còn ~2,5 giờ.

1. **Dừng hẳn phiên trước.** Trên Colab của B: Runtime → Disconnect and delete runtime (nếu chưa tự
   ngắt). Hai tài khoản không được cùng ghi vào một run.
2. **[drive.google.com, tài khoản B]** Mở `trademark_data/runs/trba_ft/2026-11-05_08-28-47/checkpoints/`:
   - nhìn giờ sửa của `last.ckpt` để biết phiên 1 lưu tới đâu;
   - tạo thư mục `trademark_data/backup_trba_ft_phien_B/` rồi **di chuyển** mọi file `epoch=…ckpt`
     đang có vào đó, để nguyên `last.ckpt`. Các file này do B tạo nên thuộc sở hữu của B, C không xoá
     được. Lý do giống bước 1 của cách 1, chỉ khác người làm là B.
3. **[drive.google.com, tài khoản A]** Chia sẻ `trademark_data` cho C, quyền **Người chỉnh sửa**.
4. **[drive.google.com, tài khoản C]** "Drive của tôi" phải chưa có thư mục `trademark_data`. Rồi
   "Được chia sẻ với tôi" → chuột phải `trademark_data` → Sắp xếp → Thêm lối tắt → Drive của tôi.
   **Giữ nguyên tên.**
5. **[Colab, tài khoản C]** Mở notebook:
   - Tệp → Mở sổ tay → tab GitHub, dán link file trên GitHub (repo private: tick "Include private
     repos" và cấp quyền);
   - hoặc tải file `.ipynb` từ GitHub về máy, rồi Tệp → Tải sổ tay lên.

   Chọn Runtime → Change runtime type → T4 GPU. **Ở ô 1 đổi `TK = "B"` thành `TK = "C"`**: mỗi tài
   khoản cần một file mốc riêng ở ô B1, để B1 kiểm lại `checkpoints/` ở phiên đầu của C.
6. **[Colab C]** Chạy 1 → 2 → 3 → 4 → B1 → 6 → B2, không bấm "Run all".
   - B1 phải in đủ (a), (b), (c) OK. Nếu (a) dừng vì còn file `epoch=…` thì làm lại bước 2.
   - Ô 6 in "đã xong N epoch", với `global_step` = N × 801. N là epoch mà phiên 1 đã lưu.
   - B2: thanh tiến trình hiện `Epoch N−1/20` vài phút rồi `Epoch N/20`. Thấy `Epoch 0/20` thì dừng.
   - Thời gian: (20 − N) × ~33,5 phút + ~30 phút cho SWA.
7. **Xong** khi output B2 có `` `Trainer.fit` stopped: `max_epochs=21` reached `` (20 epoch cộng lượt
   SWA của model có BatchNorm; svtrctc ở lần 2 cũng in `max_epochs=11` cho 10 epoch).
   - Phiên 2 lại bị cắt giữa chừng: làm phiên 3 y như trên. Chủ của các file `epoch=…` lúc đó (C)
     di chuyển chúng đi, tài khoản mới đặt `TK` là tên mới.
8. Mang về máy `vunl` như mục dưới đây. Thư mục `backup_trba_ft_*` và các file
   `*_da_kiem_checkpoint.txt` không cần tải. Xoá các thư mục backup sau khi đã eval xong ở `vunl`.

**Mang về máy local:**

1. Từ Drive của tài khoản đang giữ run, tải `trademark_data/runs/trba_ft/<giờ>/`:
   `checkpoints/last.ckpt`, `config/`, `hparams.yaml`, **`events.out.tfevents.*`** (log để vẽ
   đồ thị train; train tiếp sau mỗi lần ngắt sinh thêm một file, lấy **tất cả**) và **tên** các file
   `checkpoints/epoch=…ckpt` (cần để `--continue` biết đã train mấy epoch; chép file nặng, hoặc
   `touch` một file rỗng cùng tên). Đặt vào `data/runs/trba_ft/<giờ>/`.
2. Nếu `val_accuracy` tốt nhất ≥ 1%: `touch data/runs/trba_ft/<giờ>/DONE`. Thiếu `DONE`, lần
   pipeline sau có `trba_ft` trong `--models` sẽ coi là train dở và train tiếp trên CPU.
3. Eval lại mọi model trên bộ test local, vào lúc máy rảnh (không chạy song song với train đêm):
   `bash scripts/09_full_pipeline_cpu.sh --only eval`

### 4. Train thêm `crnn_ft` lên 20 epoch trên một máy Ubuntu khác (yếu hơn)

**Tình trạng 09/11:** máy C không có checkpoint 8 epoch nên **train lại từ đầu 20 epoch** thay vì
`--continue` (diễn biến ở "Lần 4"). Còn thiếu: mang `checkpoints/` về (xem "Việc cần làm tiếp" ở
đầu file). Hướng dẫn dưới đây vẫn đúng cho lần `--continue` sau.

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
       overfit", nêu epoch val_loss thấp nhất.
    3. Checkpoint đem chấm test là epoch nào. Đó là `last.ckpt` của run mới, tức epoch gần nhất
       lọt top-3 val_accuracy, không nhất thiết là epoch tốt nhất hay epoch cuối (xem "Lần 2 →
       Checkpoint được chấm"). Đọc epoch của nó bằng
       `.venv/bin/python -c "import torch,sys;c=torch.load(sys.argv[1],map_location='cpu',weights_only=False);print(c['epoch']+1)" <run>/checkpoints/last.ckpt`
       Số in ra đếm trong run mới, cộng thêm 8 epoch của run cũ. Nếu số in ra lớn hơn
       `trainer.max_epochs` trong `<run>/config/config.yaml` thì đó là model sau SWA, giống
       svtrctc ở lần 2.
    4. Các epoch ngay sau chỗ nối (epoch 9–10) có chững hoặc giảm không. Nếu có, giải thích là do
       learning rate khởi động lại (mục 2.6), không phải overfit.
    5. Test US / VN mới (`reports/crnn_ft_us.summary.json`, `crnn_ft_vn.summary.json`, hoặc
       `final_summary.md`) so với 73,0% / 18,3% của bản 8 epoch.
- **Mục 2.6:** thay đoạn "**Kế hoạch:**" bằng một câu đã làm: máy nào (CPU, số luồng, RAM), batch,
  bao nhiêu phút/epoch (`benchmark.json` của run mới), mấy đêm.
- **Không** đổi các số của lần 1 (8 epoch) ở chỗ khác. Đó là lịch sử, giữ để so sánh.

Ghi thêm một mục "Lần 3 — crnn_ft train tiếp trên máy B" (hoặc số kế tiếp, nếu đã có lần 3) ở đầu
phần nhật ký bên dưới (trên mục mới nhất), cùng khuôn: lệnh, máy, thời gian, kết quả, sự cố. Cập
nhật bảng "Tình trạng 4 mô hình" ở đầu file.

Commit trên `vunl`:

```bash
git add BAO_CAO_TONG_KET_DAT_LIEU_HUAN_LUYEN.md docs/nhat_ky_chay.md reports/figures/
git commit -m "16-fix-review: crnn_ft train tiếp lên 20 epoch: đồ thị + báo cáo"
```

**Không** commit `data/` (đã `.gitignore`, chứa checkpoint vài trăm MB) và `reports/*.csv`.

---

## Lần 4 — `crnn_ft` train lại từ đầu 20 epoch, 07/11 20:04 → 08/11 (máy C, CPU)

### Bối cảnh

- Máy C: CPU, không GPU (torch 2.14.1+cu130, chạy CPU). Repo ở
  `/media/zaibachkhoa/education/GenAI 2025/đồ án chuyên ngành 1`, **có dấu cách**.
- Máy C không có checkpoint 8 epoch của lần 1, nên train lại từ đầu thay vì `--continue`:
  `09_full_pipeline_cpu.sh --from train --skip eval --models crnn_ft --max-epochs 20 --batch-size 16 --num-workers 0`.
- Run `data/runs/crnn_ft/2026-11-07_20-44-52/` đã mang về `vunl` (config, `hparams.yaml`, 5 file log
  TensorBoard, `train.log`), **trừ `checkpoints/`** (trống). Log pipeline của máy C: 8 file
  `data/logs/pipeline_2026-11-0{7,8}_*.log`.

### Diễn biến

| Bắt đầu | Kết quả |
|---|---|
| 07/11 20:04 | Lỗi sau 1 phút: đường dẫn có dấu cách làm Hydra không đọc được `hydra.run.dir=…` (`OverrideParseException`). Máy C sửa bằng cách bọc ngoặc kép; bản sửa đã đưa vào repo, xem "Đã sửa" |
| 07/11 20:11 | `DataLoader worker … killed` (hết RAM) với `--num-workers 1` → chuyển `--num-workers 0` |
| 07/11 20:44 → 08/11 05:18 | Epoch 1–10, khoảng 47 phút/epoch (batch 16), rồi tiến trình bị `Killed`, có lẽ hết RAM, giữa epoch 11 |
| 08/11 14:04 | Lỗi sau 1 phút: `ckpt_path=` chưa bọc ngoặc, cùng lỗi dấu cách → máy C sửa tiếp |
| 08/11 14:09 → 15:03 | Xong epoch 11, bị `Killed` |
| 08/11 15:41 → 16:10 | Bị `Killed`, không xong epoch nào. Lệnh có `--batch-size 8`, nhưng khi train tiếp pipeline giữ batch 16 của run (cố ý: đổi batch làm lệch lịch OneCycle) |
| 08/11 16:15 → 20:03 | Epoch 12–15, bị `Killed` |
| 08/11 21:42 → ? | Chạy **ngoài pipeline** (không có log), **batch 8** (6.404 bước/epoch, thấy từ log TensorBoard): epoch 16–20 |

Từ epoch 16 SWA đã thay lịch OneCycle bằng learning rate cố định, nên đổi sang batch 8 không gây lỗi
lịch learning rate. Tuy vậy, 5 epoch cuối được train với batch khác 15 epoch đầu: báo cáo phải ghi
điều này. `pipeline_train.env` của run vẫn ghi `batch_size=16`.

### Kết quả (val)

Val accuracy theo epoch: 0 → 0,5 → 55,5 → 63,3 → 66,3 → 68,7 → 70,4 → 71,0 → 71,9 → 72,8 → 73,1 →
73,2 → 73,8 → 74,2 → 74,4 → 74,5 → 74,5 → 75,3 → 75,8 → **76,35%** (epoch 1–20). Bảng đầy đủ ở
`reports/figures/train_curves.md`.

- val_loss giảm liên tục tới epoch 20 (0,3896): **chưa overfit**, còn học thêm được.
- Hơn bản 8 epoch của lần 1 (74,07%) **2,3 điểm** val.
- Khởi đầu chậm hơn lần 1 (epoch 2: 0,5% so với 60,9%). Lý do: parseq nhân learning rate với
  batch/256, nên batch 16 cho learning rate đỉnh nhỏ hơn 4 lần so với batch 64 của lần 1.
- Chưa có số test: thiếu checkpoint (xem "Việc cần làm tiếp" ở đầu file).

### Đã sửa

`scripts/09_full_pipeline_cpu.sh` bọc ngoặc kép mọi đường dẫn trong override Hydra (`hydra.run.dir`,
`ckpt_path`, `model.init_from`, `+model.init_from`). `scripts/05_train.sh` bỏ lớp ngoặc trước khi kiểm
thư mục LMDB và in dòng `checkpoint ->`. Bản sửa lấy từ máy C, thêm hai chỗ máy C bỏ sót
(`+model.init_from`, `data.root_dir` trong `05_train.sh`). Kiểm bằng bộ phân tích override của Hydra:
không ngoặc thì `OverrideParseException`, có ngoặc thì nhận đúng đường dẫn. Golden `testcase71..75`;
toàn bộ `pytest tests/` 315 passed; trả `05_train.sh` về bản cũ thì cả 5 testcase mới đỏ.

---

## Lần 3 — `trba_ft` 20 epoch trên Colab T4, 05/11 15:28 → 07/11 (4 phiên)

### Diễn biến

Run `2026-11-05_08-28-47` (tên theo giờ UTC), luôn train tiếp đúng run đó bằng `ckpt_path` (mục 3a–3c).
Giờ bắt đầu lấy từ tên file log TensorBoard, giờ VN:

| Phiên | Bắt đầu | Epoch train được | Kết thúc |
|---|---|---|---|
| 1 | 05/11 15:28 | 1–8 | hết hạn mức GPU giữa epoch 9 (577/801) |
| 2 | 06/11 11:03 | 9–15 | `DataLoader worker … killed` (hết RAM, `--num-workers 2`) giữa epoch 16 |
| 3 | 07/11 10:58 | 16–17 | hỏng giữa epoch 18; chạy lại lúc 12:18 lại hỏng sau ~385 bước |
| 4 | 07/11 17:04 | 18–20 | `DataLoader worker … killed` giữa lượt SWA cuối ("Epoch 20/20", 453/802) |

Tốc độ: phiên 1 khoảng 33,5 phút/epoch (0,45 batch/giây), phiên 4 khoảng 25 phút/epoch (0,58
batch/giây). Notebook của phiên 2 và phiên 4 (có output) lưu ở `data/logs/colab_trba_ft/`; phiên 3
không có notebook, diễn biến suy từ file log TensorBoard.

### Kết quả (val)

- Val accuracy epoch 9–20: 84,8 → 84,7 → 85,1 → 85,8 → 85,6 → 85,9 → 86,1 → 86,2 → 86,3 → 86,3 →
  **86,72** → 86,47%.
- Epoch 6–8 (81,17 / 82,22 / 84,15%) chỉ biết qua tên checkpoint dự phòng. **Thiếu file log
  TensorBoard của phiên 1** (epoch 1–8) trong số file đã tải về, nên đồ thị bắt đầu từ epoch 9. Nếu
  file `events.out.tfevents.*` của phiên 1 còn trong thư mục run trên Drive của tài khoản A thì chép
  thêm vào `data/runs/trba_ft/2026-11-05_08-28-47/` rồi vẽ lại.
- Top-3 cuối cùng: epoch 19 (86,72%), epoch 20 (86,47%), epoch 17 (86,30%). `last.ckpt` = epoch 20
  (`global_step` 16.020 = 20 × 801), là checkpoint sẽ được chấm.
- val_loss dao động 0,46–0,53 nhưng thấp nhất ở epoch 20 (0,4639): **chưa overfit**. Các epoch cuối chỉ
  thêm khoảng 0,2 điểm/epoch.
- Lượt SWA cuối hỏng giữa chừng: trọng số trung bình (epoch 16–20) nằm trong trạng thái callback của
  `last.ckpt`, nhưng chưa bao giờ được đánh giá hay chép vào model. Không ảnh hưởng eval, vì
  `last.ckpt` giữ trọng số epoch 20.
- Đã đánh dấu `DONE` ở `vunl`. Chưa eval trên test.

Đã bỏ (chuyển vào thùng rác của ổ `/mnt/1864…`): 6 checkpoint dự phòng (epoch 6, 7, 8, 12, 14, 15;
4,8 GB), bản sao thiếu `last.ckpt` của run, 2 file zip tải từ Drive, bản sao LMDB train/val (trùng
md5 với `data/lmdb/`), báo cáo cũ và notebook trùng của máy C.

---

## Lần 2 — 05/11/2026 20:59 → 06/11/2026 07:10 (máy `vunl`, CPU)

### Bối cảnh

- Máy như lần 1. Nhánh `16-fix-review`, commit `a299574`. Tập VN vẫn **chưa duyệt tay**.
- Lịch hẹn 21:00 trong phiên Claude chạy chuỗi `data/night_2026-11-05.sh` trong tmux `doan`, dưới
  một khoá `systemd-inhibit` (máy không ngủ suốt đêm):
  1. `09_full_pipeline_cpu.sh --models parseq_ft --retrain --max-hours 9`
  2. còn ≥ 2 giờ trước 08:00 thì `09_full_pipeline_cpu.sh --models svtr --max-hours <phần còn lại> --max-epochs 30`
- Log: `data/logs/night_2026-11-05.log` (quyết định của chuỗi),
  `data/logs/pipeline_2026-11-05_20-59-07.log`, `data/logs/pipeline_2026-11-06_04-21-56.log`.

### Thời gian từng bước

| Giờ | Bước | Kết quả |
|---|---|---|
| 20:59–04:18 | train `parseq_ft` (từ đầu) | đo 4,43 s/batch → 60,5 phút/epoch → ngân sách 9h chọn **7 epoch** (ước 7,1h); xong sau 438 phút |
| 04:18–04:21 | eval + đồ thị | crnn_ft, parseq_ft |
| 04:21 | chuỗi quyết định | còn 3,0h trước 08:00 → chạy svtr với `--max-hours 3.0` |
| 04:21–07:07 | train `svtr` (từ đầu) | đo 1,09 s/batch → 15,0 phút/epoch → chọn **10 epoch** (ước 2,5h); xong sau 164 phút |
| 07:07–07:10 | eval + đồ thị | crnn_ft, parseq_ft, svtrctc → `reports/final_summary.md`, `reports/figures/` |

Dự tính trước đó (8 epoch × 57 phút) lệch vì lần này đo được 60,5 phút/epoch, nên ngân sách chỉ đủ
7 epoch. svtr được phép tới 30 epoch, nhưng 3 giờ chỉ đủ 10. Mỗi lần eval chuyển bộ báo cáo trước đó
vào `reports/archive/<giờ>/` (không đưa vào git).

### Kết quả

Val accuracy theo epoch. Bảng đầy đủ ở `reports/figures/train_curves.md`, đồ thị ở
`BAO_CAO_TONG_KET_DAT_LIEU_HUAN_LUYEN.md` mục 2.5.

- **parseq_ft:** 84,3 → 87,8 → 88,7 → 89,4 → **89,5** → 89,5 → 89,1% (epoch 1–7).
  - val_loss thấp nhất ở epoch 5 (0,479), sau đó lên 0,497 (+4%) trong khi loss train vẫn giảm.
  - Kết luận: **gần bão hoà**. Đúng chiều của overfit nhưng chưa tới ngưỡng +5% của luật.
- **svtrctc:** 0 → 0,7 → 50,5 → 61,4 → 64,9 → 68,1 → 69,2 → 69,8 → 71,1 → **71,8%** (epoch 1–10).
  - Hai epoch đầu gần 0% vì model train từ đầu.
  - val_loss vẫn giảm tới epoch cuối. Kết luận: **chưa bão hoà**.

**Checkpoint được chấm.** `last.ckpt` là epoch gần nhất lọt top-3 val_accuracy (xem đầu file), nên
không phải lúc nào cũng là epoch cuối:

| Model | `last.ckpt` là | val của checkpoint đó | val tốt nhất |
|---|---|---|---|
| crnn_ft | epoch 8 (bước 6.408) | 74,07% | 74,07% (epoch 8) |
| parseq_ft | epoch 6 (bước 4.806); epoch 7 (89,14%) không lọt top-3 nên không được ghi | 89,46% | 89,49% (epoch 5) |
| svtrctc | model sau SWA (bước 8.011) | 70,52% | 71,82% (epoch 10) |

Sau epoch cuối, Lightning chạy thêm một lượt qua tập train để tính lại BatchNorm cho model SWA:
- **svtr:** lượt này làm bộ đếm bước tăng 1 (8.010 → 8.011), nên checkpoint được ghi. Model SWA lọt
  top-3 (70,52% > 69,75% của epoch 8), vì vậy svtrctc được chấm trên model SWA, kém epoch tốt nhất
  1,3 điểm val.
- **crnn_ft:** bộ đếm bước giữ nguyên 6.408, nên Lightning không ghi checkpoint.

**Test** (crnn_ft giữ nguyên số của lần 1):

| Model | Tập | n | accuracy | bỏ hoa/thường | bỏ cả dấu | 1−NED |
|---|---|---|---|---|---|---|
| parseq_ft | US | 2.847 | **88,7%** | 88,7% | 88,9% | 0,957 |
| parseq_ft | VN | 2.000 | **27,6%** | 41,1% | 45,8% | 0,534 |
| svtrctc | US | 2.847 | **69,4%** | 69,4% | 69,5% | 0,848 |
| svtrctc | VN | 2.000 | **11,4%** | 16,4% | 17,2% | 0,275 |
| crnn_ft | US | 2.847 | 73,0% | 73,0% | 73,1% | 0,889 |
| crnn_ft | VN | 2.000 | 18,3% | 27,0% | 28,5% | 0,380 |

Theo loại nhãn hiệu (US, bỏ hoa/thường):

| Model | `standard_character` (2.288) | `typeset_old` (29) | `stylized_words` (112) | `design_with_words` (418) |
|---|---|---|---|---|
| parseq_ft | 98,3% | 96,6% | 67,9% | 40,9% |
| crnn_ft | 84,9% | 96,6% | 42,0% | 14,4% |
| svtrctc | 83,9% | 89,7% | 13,4% | 3,4% |

Tập VN:

| Model | US→VN, bỏ hoa/thường (CI 95%) | US→VN, chỉ 1.442 nhãn VN không dấu | Đọc đúng 558 nhãn VN **có dấu** |
|---|---|---|---|
| parseq_ft | −0,476 [−0,502; −0,453] | −0,319 [−0,347; −0,291] | 2/558 (0,4%) |
| crnn_ft | −0,460 [−0,484; −0,434] | −0,355 [−0,384; −0,325] | 0 |
| svtrctc | −0,530 [−0,554; −0,507] | −0,466 [−0,495; −0,440] | 0 |

Cả ba model gần như không đọc được chữ có dấu, vì dữ liệu train USPTO gần như không có dấu. parseq_ft
tốt nhất ở mọi cột. Khoảng cách lớn nhất nằm ở nhãn có hình vẽ (`design_with_words`): 40,9% so với
14,4% và 3,4%.

### Phát hiện: vì sao parseq_ft lần 1 không học được mà lần 2 học được

Cùng cấu hình, cùng nạp 172 tensor pretrained. Chỉ khác `max_epochs` (2 so với 7), nên lịch OneCycle
dài khác nhau. Lần 1, sau 1.602 bước val vẫn 0%. Lần 2, ngay sau epoch 1 (801 bước) val đã 84,3%.
Như vậy "ít epoch" **tự nó** không giải thích được lần 1.

Loss train theo bước tách nhau từ khoảng bước 100:

| bước | 100 | 200 | 300 | 400 | 600 | 800 |
|---|---|---|---|---|---|---|
| lần 1 (2 epoch) | 5,31 | 5,12 | 4,85 | 4,58 | 4,15 | 3,85 |
| lần 2 (7 epoch) | 5,20 | 3,52 | 3,09 | 1,23 | 0,42 | 0,45 |

Learning rate đỉnh (1,75·10⁻⁴ = 7·10⁻⁴ × 64/256) đạt sau phần warmup, dài 7,5% tổng số bước:

- **Lần 1:** đỉnh ở **bước 120**, lúc loss vẫn ~5,3 = ln 217, tức lớp ra mới khởi tạo còn đoán ngẫu
  nhiên.
- **Lần 2:** đỉnh ở **bước 420**, lúc loss đã xuống 1,23.

**Giả thuyết mạnh nhất:** learning rate lên đỉnh khi lớp ra còn ngẫu nhiên, nên gradient lớn và nhiễu
dội ngược vào phần trọng số pretrained, làm hỏng nó. Model phải học lại gần như từ đầu, cùng với
việc SWA thay lịch LR từ epoch 2 (đã ghi ở lần 1).

Đây **chưa** được kiểm chứng bằng thí nghiệm đối chứng (cùng thứ tự dữ liệu, chỉ đổi độ dài warmup).
Thứ tự dữ liệu hai lần cũng khác nhau, vì `train.py` không đặt seed. Hệ quả thực tế: fine-tune cần
lịch đủ dài. Pipeline đã cảnh báo khi ngân sách chỉ đủ dưới 3 epoch.

### Sự cố và quan sát

- Không lỗi. Cả hai bước train thoát 0, eval chạy hai lần đều xong.
- `last.ckpt` không phải epoch cuối (bảng "Checkpoint được chấm"). Đây là hành vi của Lightning 2.6,
  không phải lỗi pipeline. Chưa sửa. Muốn chấm epoch val tốt nhất thì phải đổi `06_eval.sh` sang
  chọn file `epoch=…ckpt` có `val_accuracy` cao nhất. Việc đó đổi số liệu test của svtrctc (epoch
  tốt nhất hơn model SWA 1,3 điểm val), và cần thống nhất trước khi làm.
- Cuối log có cảnh báo: tập VN chưa duyệt tay.

### Việc tiếp theo

Xem đầu file: trba_ft train tiếp (mục 3), crnn_ft lên 20 epoch (mục 4), duyệt tay VN (mục 1).
svtrctc cũng chưa bão hoà, có thể train thêm bằng
`--from train --models svtr --continue --max-epochs 30 --max-hours <giờ>`.

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
   - **Bổ sung sau lần 2:** cùng cấu hình nhưng 7 epoch thì val đạt 84% ngay sau epoch 1. Vậy nguyên
     nhân chính có lẽ là warmup quá ngắn (LR lên đỉnh ở bước 120), chứ không phải số epoch. Xem
     "Lần 2 → Phát hiện".
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
