# Hướng dẫn bàn giao: eval `crnn_ft` + `trba_ft` (bộ test 2847/2000) trên máy kia

> **Đã xong 10/11** trên máy chính. Kết quả ở `docs/nhat_ky_chay.md`, lần 5. Số epoch dưới đây lấy
> theo tên file checkpoint, đếm từ 0 (`epoch=19` là epoch 20 trong báo cáo).

*Tài liệu cho AI/người thực hiện trên **máy kia** (máy có sẵn test LMDB đúng bộ eval 2847/2000). Cập nhật: 2026-11-09, repo `nguyenlamvu123/major_proj_1`, nhánh `16-fix-review` (HEAD lúc bàn giao: `d376096`).*

## 1. Mục tiêu
- Đưa 2 checkpoint đã train xong 20 epoch (`crnn_ft`, `trba_ft`) sang máy kia.
- Chạy **đánh giá (eval)** trên **đúng bộ test 2847/2000** (khớp `reports/final_summary.md` sinh 2026-11-06 07:10 UTC+07:00: US n=2847, VN n=2000, VN chưa duyệt tay) để so sánh được.
- Cập nhật `reports/final_summary.md`, `reports/figures/*`, `BAO_CAO_TONG_KET_DAT_LIEU_HUAN_LUYEN.md`, `docs/nhat_ky_chay.md`.

> Chưa eval: `crnn_ft` 20 epoch và `trba_ft` 20 epoch **chưa có kết quả test trong `reports/final_summary.md`** tại thời điểm bàn giao.

## 2. Gói đã upload Google Drive (tải về máy kia)
| File | Nội dung | Trích xuất vào |
|---|---|---|
| `crnn_ft_2026-11-07_20-44-52.tar` | Run `data/runs/crnn_ft/2026-11-07_20-44-52` (checkpoint + config + TensorBoard events + `pipeline_train.env`/`hparams`/log) | `<repo>/data/runs/` |
| `trba_ft_2026-11-05_08-28-47.tar` | Run `data/runs/trba_ft/2026-11-05_08-28-47` (tải từ Colab về máy nguồn, 20 epoch) | `<repo>/data/runs/` |

Tạo từ máy nguồn:
```bash
tar -C data/runs -cf /media/zaibachkhoa/AC84-F995/crnn_ft_2026-11-07_20-44-52.tar crnn_ft/2026-11-07_20-44-52
tar -C data/runs -cf /media/zaibachkhoa/AC84-F995/trba_ft_2026-11-05_08-28-47.tar trba_ft/2026-11-05_08-28-47
```

## 3. Yêu cầu có sẵn trên máy kia
- Repo `major_proj_1`, nhánh `16-fix-review` cập nhật:
  ```bash
  cd <repo_máy_kia>
  git fetch origin && git checkout 16-fix-review && git pull origin 16-fix-review
  ```
- Môi trường chạy được `parseq`/eval: `venv` (hoặc python phù hợp) + deps (`requirements.txt`/setup repo) + `third_party/parseq` + `PYTHONPATH`/`--python` như hướng dẫn repo (`docs/eval_crnn_ft_may_C.md`, `scripts/00_setup*.sh`). **Không copy `venv` giữa máy khác nhau** trừ khi cùng OS/Python/path.
- **Test LMDB đúng bộ 2847/2000** (bắt buộc để so sánh): dưới `$DATA_ROOT/lmdb/test/` có `test_us` và `test_vn`.
  - Trên máy nguồn (AC84, `/media/zaibachkhoa/AC84-F995/trademark_data/lmdb/test/`) chỉ có `test_us b'200'`, `test_vn b'50'` — **không dùng** nếu cần so sánh 2847/2000.
  - Chấp nhận build từ `processed/split_test_*` + `raw/images_*` **chỉ khi** ra đúng 2847/2000 như lần 2.
- Không có GPU cũng chạy eval được (chậm hơn) với `--device cpu`/cấu hình mặc định `06_eval.sh`.

## 4. Giải nén run vào đúng cấu trúc
```bash
cd <repo_máy_kia>
mkdir -p data/runs
tar -xf /path/to/crnn_ft_2026-11-07_20-44-52.tar -C data/runs
tar -xf /path/to/trba_ft_2026-11-05_08-28-47.tar -C data/runs   # bỏ nếu máy kia đã có trba_ft run này

ls data/runs/crnn_ft/2026-11-07_20-44-52/checkpoints/last.ckpt
ls data/runs/trba_ft/2026-11-05_08-28-47/checkpoints/last.ckpt
```

## 5. Xác nhận checkpoint (số liệu thực tế tại thời điểm bàn giao)
Chạy:
```bash
PYTHON=${PYTHON:-$PWD/venv/bin/python} "$PYTHON" - << 'PY'
import torch, os, glob
for ck in glob.glob('data/runs/**/checkpoints/last.ckpt', recursive=True):
    c=torch.load(ck,map_location='cpu',weights_only=False)
    print(ck)
    print(' epoch_done=',c.get('epoch',-1)+1,'global_step=',c.get('global_step'))
    for k,v in (c.get('callbacks') or {}).items():
        if 'ModelCheckpoint' in k:
            for p,s in sorted((v.get('best_k_models') or {}).items(), key=lambda x:-x[1]):
                print(' ',os.path.basename(p), round(float(s),4))
PY
```
Kỳ vọng:
- `crnn_ft/2026-11-07_20-44-52`: **20 epoch xong**, `global_step=80050` (6404 step/epoch, batch 8 giai đoạn cuối). Top-3 `val_accuracy`: `epoch=19` **76.3528%** (NED 90.2712%), `epoch=18` 75.8257%, `epoch=17` 75.3338%. (Train CPU, nhiều lần resume do OOM/Killed; đã qua `scripts/05_train.sh`/pipeline; cuối dùng batch 8, `num_workers=0`.)
- `trba_ft/2026-11-05_08-28-47`: **20 epoch xong** (Colab T4, resume nhiều phiên/tài khoản), `global_step=16020` (801 step/epoch). Top-3 `val_accuracy`: `epoch=18` **86.7182%** (NED 93.9936%), `epoch=19` 86.4722%, `epoch=16` 86.2966%.

Đánh dấu hoàn tất (giúp `train_curves`/pipeline nhận diện):
```bash
touch data/runs/crnn_ft/2026-11-07_20-44-52/DONE
# trba_ft: touch DONE nếu run đã đủ 20 epoch và bạn xác nhận
```

## 6. Xác nhận test LMDB = 2847/2000 (bắt buộc trước eval)
```bash
export DATA_ROOT="${DATA_ROOT:-$PWD/data}"   # chỉnh nếu LMDB test ở nơi khác
python3 - << 'PY'
import lmdb, os
base=os.environ.get('DATA_ROOT','data')
for sub in ['lmdb/test/test_us','lmdb/test/test_vn']:
    p=os.path.join(base,sub)
    e=lmdb.open(p,readonly=True,lock=False,readahead=False,meminit=False)
    with e.begin() as t: print(p, t.get(b'num-samples'))
    e.close()
PY
```
**Phải ra:** `test_us b'2847'`, `test_vn b'2000'`. Nếu khác (vd 200/50) → **không so sánh được** với baseline; dừng và tìm đúng bộ test.

## 7. Eval + tổng hợp báo cáo
```bash
cd <repo_máy_kia>
export DATA_ROOT="${DATA_ROOT:-$PWD/data}"
export PYTHONPATH="$PWD/compat:$PWD/src:$PWD/third_party/parseq${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="$PWD/venv/bin/python" bash scripts/06_eval.sh --only crnn_ft,trba_ft
# 06_eval.sh dựng lại final_summary từ các CSV trong reports/ và có bước vẽ curves;
# nếu đã ghép CSV thủ công, có thể eval từng model rồi regen report (xem docs/eval_crnn_ft_may_C.md).

# Đảm bảo regen (nếu cần chạy riêng):
"$PYTHON" -m trademark_ocr.train_curves --runs data/runs --out-dir reports/figures
"$PYTHON" -m trademark_ocr.report --reports reports/ --out reports/final_summary.md
```
Kiểm tra output: `reports/crnn_ft_us.csv`, `reports/crnn_ft_vn.csv`, `reports/trba_ft_us.csv`, `reports/trba_ft_vn.csv` (+ `.summary.json` nếu có), `reports/final_summary.md`, `reports/figures/train_curves*`.

## 8. Cập nhật tài liệu (sau eval)
- `BAO_CAO_TONG_KET_DAT_LIEU_HUAN_LUYEN.md`: mục 2.4.2/2.5, điền số test (ghi rõ n=2847/2000, ngày eval, checkpoint đem chấm), bỏ “chờ eval”, cập nhật trạng thái `crnn_ft`/`trba_ft`. Giữ chú thích `[CŨ]/[HIỆN TẠI]` rõ ràng.
- `docs/nhat_ky_chay.md`: thêm mục “Lần eval crnn_ft 20 epoch + trba_ft 20 epoch (máy kia, bộ 2847/2000)”.
- Commit (chỉ khi được yêu cầu, theo quy ước repo): code/doc + `reports/*_us*.csv,*_vn*.csv,final_summary.md,figures/*`. **Không commit** tar checkpoint lớn, `lmdb_trainval.tar`, dữ liệu/ảnh, secrets, trừ khi có quyết định riêng.

## 9. Gotchas / lưu ý đã gặp
- **Đường dẫn có khoảng trắng** (`GenAI 2025`): Hydra override phải quote giá trị (`ckpt_path="..."`, `hydra.run.dir="..."`). `scripts/09_full_pipeline_cpu.sh`/`scripts/05_train.sh` trên `16-fix-review` đã xử lý; chạy thủ công phải dùng `ckpt_path=\"$CK\"` (xem `docs/eval_crnn_ft_may_C.md`).
- **`09_full_pipeline_cpu.sh` resume**: khi resume có thể lấy `batch_size/max_epochs` từ `pipeline_train.env` của run (đã từng khiến cờ `--batch-size 8` không có hiệu lực và OOM/Killed với batch 16). Nếu máy yếu/OOM, resume trực tiếp `05_train.sh --batch-size 8 --num-workers 0 ...` + `ckpt_path`/`hydra.run.dir` cùng run.
- **Lightning 2.6 + `save_top_k`**: `last.ckpt` cập nhật khi lọt top-k; “epoch_done”/file checkpoint là mốc tin cậy; `DONE` file là marker pipeline/curves.
- **`crnn_ft` 20 epoch** là run **train lại từ đầu 20 epoch** trên CPU (07–08/11), không phải nối trực tiếp epoch-8 cũ bằng `--continue` (không có run epoch-8 trên máy đó) — ghi chú này quan trọng khi đối chiếu với kết quả `crnn_ft` 8 epoch (US 73.0%, VN 18.3%, n 2847/2000, eval 06/11).
- Máy nguồn: `/media/.../education` gần đầy (94%) — đặt dữ liệu/tar trên ổ có chỗ (vd AC84).

## 10. Tham chiếu
- `docs/eval_crnn_ft_may_C.md` — quy trình eval chi tiết (máy C/dự phòng Colab).
- `docs/nhat_ky_chay.md` — nhật ký chạy, mục 3 (trba_ft), mục 4/lần 4 (crnn_ft 20 epoch).
- `BAO_CAO_TONG_KET_DAT_LIEU_HUAN_LUYEN.md` — tổng kết data/train, trạng thái [CŨ]/[HIỆN TẠI].
- `scripts/06_eval.sh`, `scripts/05_train.sh`, `scripts/09_full_pipeline_cpu.sh`.
- `reports/final_summary.md` (baseline eval 06/11: `crnn_ft` 8ep/`parseq_ft`/`svtrctc`, n 2847/2000).
