# Eval crnn_ft 20 epoch trên máy C (dự phòng: Colab)

Checkpoint của run `crnn_ft/2026-11-07_20-44-52` (20 epoch, val 76,35%) chỉ có ở máy C. Hướng dẫn
này chấm test US + VN cho riêng checkpoint đó, rồi mang kết quả về máy chính để ghép vào báo cáo.

Đi đường nào thì kết quả cũng là 4 file:

- `crnn_ft_us.csv`, `crnn_ft_us.summary.json`
- `crnn_ft_vn.csv`, `crnn_ft_vn.summary.json`

## Máy C có chạy nổi không

Nhiều khả năng là được. Đây là ước lượng, chưa đo trên máy C:

- Eval chỉ chạy suy luận: không có gradient, không có optimizer, batch 8, không có worker phụ.
  Lượng RAM cần ít hơn lúc train rất nhiều, mà lúc train máy C bị kill chính vì thiếu RAM.
- Ở máy chính, CRNN chấm 2.000 ảnh test VN hết khoảng 13 giây; cả hai tập (4.847 ảnh) chưa tới
  1 phút. Máy C có chậm hơn vài lần thì cũng chỉ mất vài phút.
- Bước A3 chạy thử 32 ảnh trước. Hỏng thì giảm batch; vẫn hỏng thì chuyển sang mục B (Colab).

## A. Trên máy C

### A0. Chuẩn bị ở máy chính, trước khi sang C

1. Push nhánh `16-fix-review`, để máy C kéo được file này.
2. Đóng gói tập test. Bản thư mục máy C gửi về hôm 09/11 chỉ có `data/lmdb/train` và
   `data/lmdb/val`, nên nhiều khả năng máy C chưa có tập test:

   ```bash
   cd /mnt/1864514f-3a41-4b8f-9907-2fbcd9d2815b/airi/doanchuyennganh1
   tar -cf ~/Downloads/lmdb_test.tar data/lmdb/test     # khoảng 126 MB
   ```

### A1. Kéo code mới

```bash
cd "/media/zaibachkhoa/education/GenAI 2025/đồ án chuyên ngành 1"
git stash         # cất phần máy C đã sửa (05_train.sh, 09_full_pipeline_cpu.sh, nhat_ky, BAO_CAO)
git pull
git log --oneline -1
```

- Những gì máy C sửa đã được đưa vào commit mới, nên cất đi là an toàn. Bản sửa vẫn nằm trong
  `git stash list` nếu cần xem lại.
- Không có mạng thì bỏ qua A1: code eval hiện có trên máy C đã đủ dùng. File này thì xem trên
  GitHub.

### A2. Kiểm đầu vào

```bash
R=data/runs/crnn_ft/2026-11-07_20-44-52
ls -la $R/checkpoints/last.ckpt $R/config/config.yaml
ls -la data/lmdb/test/test_us/data.mdb data/lmdb/test/test_vn/data.mdb
touch $R/DONE     # run đã đủ 20 epoch; thiếu DONE thì pipeline của máy C sẽ cố train tiếp
```

- Thiếu `config/config.yaml` thì eval không biết lớp model. Khi đó thêm
  `--model-class strhub.models.crnn.system.CRNN` vào lệnh ở A3 và A4.
- Thiếu LMDB test thì chép `lmdb_test.tar` sang, rồi giải nén ngay tại thư mục repo:
  `tar -xf <đường dẫn tới>/lmdb_test.tar`. File sẽ vào đúng `data/lmdb/test/`.

### A3. Chạy thử 32 ảnh

Chạy A3 và A4 trong **cùng một terminal**, vì các biến `export` chỉ sống trong terminal đó.

```bash
export PYTHONPATH="$PWD/src:$PWD/third_party/parseq" DATA_ROOT="$PWD/data"
CK="$PWD/data/runs/crnn_ft/2026-11-07_20-44-52/checkpoints/last.ckpt"
OUT="$PWD/data/eval_crnn_ft"
venv/bin/python -m trademark_ocr.evaluate --ckpt "$CK" --subset test_vn --limit 32 \
  --out "$OUT/thu.csv" --device cpu --batch-size 8 --num-workers 0
```

Chạy đúng thì dòng cuối có dạng `[evaluate] CRNN / test_vn: n=32 acc=…`.

### A4. Chạy thật

```bash
for d in us vn; do
  venv/bin/python -m trademark_ocr.evaluate --ckpt "$CK" --subset test_$d \
    --out "$OUT/crnn_ft_$d.csv" --device cpu --batch-size 8 --num-workers 0 || break
done
ls -la "$OUT"
```

- Phải có 2 dòng `[evaluate] CRNN / test_us: n=2847 …` và `[evaluate] CRNN / test_vn: n=2000 …`.
- Thư mục `data/eval_crnn_ft/` phải có đủ 4 file kết quả. Hai file `thu.*` của A3 thì bỏ qua.

Không dùng `scripts/06_eval.sh` ở máy C. Script đó ghi vào `reports/` và dựng lại
`reports/final_summary.md` từ các CSV có trên máy, mà máy C chỉ có CSV của crnn_ft. Lỡ commit
từ máy C là bảng tổng hợp mất hết các model khác.

### Nếu hỏng

| Dấu hiệu | Cách xử lý |
|---|---|
| In `Killed`, hoặc máy treo | Đóng bớt ứng dụng, chạy lại với `--batch-size 4`. Vẫn `Killed` thì sang mục B. |
| Một tập chạy quá 30 phút | Sang mục B. |
| `ModuleNotFoundError: No module named 'strhub'` | Chạy `bash scripts/00_get_parseq.sh`, rồi chạy lại. |
| Lỗi không mở được LMDB `test_us`/`test_vn` | Quay lại A2. |

### A5. Mang về máy chính

```bash
tar -cf ~/crnn_ft_eval.tar data/eval_crnn_ft/crnn_ft_us.* data/eval_crnn_ft/crnn_ft_vn.* \
  data/runs/crnn_ft/2026-11-07_20-44-52/checkpoints/last.ckpt          # khoảng 135 MB
```

Chuyển file này qua Google Drive hoặc USB như các lần trước. Trên Drive, tải **riêng file**,
đừng tải cả thư mục, vì tải thư mục thì Drive gói lại thành zip.

Nên mang cả `last.ckpt` về. Trên máy chính, `06_eval.sh` chọn run mới nhất **có** `last.ckpt`.
Thiếu file này, một lần eval đầy đủ sau đó sẽ âm thầm chấm lại bản 8 epoch (04/11) và ghi đè
kết quả 20 epoch. Nếu chỉ mang được 4 file kết quả thì từ đó đừng eval crnn_ft trên máy chính.

## B. Dự phòng: Colab

Dùng khi máy C không chạy nổi.

### B1. Đưa checkpoint lên Drive

Từ máy C, tải `data/runs/crnn_ft/2026-11-07_20-44-52/checkpoints/last.ckpt` lên thư mục
`trademark_data/eval_crnn_ft/` trên Drive (tạo thư mục nếu chưa có). Dùng tài khoản nào có quyền
chỉnh sửa `trademark_data` cũng được.

Chỉ cần đúng file này: ô dưới truyền `--model-class`, nên không cần thư mục `config/`.

### B2. Notebook

1. Mở `docs/colab/train_tiep_trba_ft.ipynb` và chạy **ô 1–3**: mount Drive, clone nhánh
   `16-fix-review`, cài thư viện. Không chạy các ô train.
2. Không cần GPU. Hết quota GPU thì chọn CPU ở Runtime → Change runtime type; CRNN chạy CPU vẫn
   nhanh.
3. Thêm một ô mới:

```python
import os, torch
R = os.environ["DATA_ROOT"]                    # /content/drive/MyDrive/trademark_data
E = R + "/eval_crnn_ft"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["PYTHONPATH"] = "/content/major_proj_1/src:/content/major_proj_1/third_party/parseq"
!ls -la "{E}/last.ckpt" "{R}/lmdb/test/test_us/data.mdb" "{R}/lmdb/test/test_vn/data.mdb"
for d in ["us", "vn"]:
    !python -m trademark_ocr.evaluate --ckpt "{E}/last.ckpt" --model-class strhub.models.crnn.system.CRNN --subset test_{d} --out "{E}/crnn_ft_{d}.csv" --data-root "{R}" --device {DEV} --batch-size 64 --num-workers 0
```

- Kết quả đúng giống A4: có 2 dòng `n=2847` và `n=2000`.
- Nếu `ls` báo thiếu LMDB test trên Drive: tải `lmdb_test.tar` (bước A0) lên `trademark_data/`,
  rồi thêm `!tar -xf "{R}/lmdb_test.tar" -C /content` trước vòng `for`, và đổi
  `--data-root "{R}"` thành `--data-root /content/data`.
- Tải 4 file `crnn_ft_*` trong `trademark_data/eval_crnn_ft/` về máy chính. Tải nhiều file một
  lúc thì Drive gói thành zip; giải nén zip vào `data/eval_crnn_ft/`.

## C. Trên máy chính: ghép vào báo cáo

```bash
cd /mnt/1864514f-3a41-4b8f-9907-2fbcd9d2815b/airi/doanchuyennganh1
tar -xf ~/Downloads/crnn_ft_eval.tar                 # đi đường Colab: đã giải nén ở B2
touch data/runs/crnn_ft/2026-11-07_20-44-52/DONE     # chỉ khi đã mang last.ckpt về
```

Sau đó bảo Claude "nhập kết quả eval crnn_ft trong data/eval_crnn_ft", hoặc tự chạy:

```bash
export DATA_ROOT="$PWD/data" PYTHONPATH="$PWD/src:$PWD/third_party/parseq"
arch="reports/archive/$(date +%Y-%m-%d_%H-%M-%S)"; mkdir -p "$arch"
mv reports/crnn_ft_us.* reports/crnn_ft_vn.* "$arch/"          # kết quả bản 8 epoch (04/11)
cp data/eval_crnn_ft/crnn_ft_us.* data/eval_crnn_ft/crnn_ft_vn.* reports/
.venv/bin/python -m trademark_ocr.eval_domain_shift --reports reports/ --out reports/domain_shift.csv
.venv/bin/python -m trademark_ocr.error_analysis --reports reports/ --out-dir data/processed/eda_figures
.venv/bin/python -m trademark_ocr.report --reports reports/ --out reports/final_summary.md
```

Điền số test vào `BAO_CAO_TONG_KET_DAT_LIEU_HUAN_LUYEN.md` mục 2.5 (các ô "chờ eval"), rồi commit
cùng `reports/final_summary.md`.

trba_ft cũng đang chờ eval, nhưng checkpoint của nó đã ở máy chính:
`bash scripts/06_eval.sh --only trba_ft`. Chạy lệnh đó **sau** bước ghép ở trên, vì nó dựng lại
`final_summary.md` từ mọi CSV đang có trong `reports/`.
