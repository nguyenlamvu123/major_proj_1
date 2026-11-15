# Nhật ký trao đổi — đánh giá & sửa đồ án OCR nhãn hiệu (2026-10-23)

> Tóm tắt cuộc trò chuyện giữa chủ repo và Claude Code. Ý nào được nhắc lại ở nhiều lượt chỉ
> ghi một lần, ở dạng cuối cùng. Chi tiết kỹ thuật đầy đủ: `REVIEW.md` (nhận định + trạng thái
> xử lý), `DESIGN.md` mục 15–16 (thiết kế), `ACCEPTANCE.md` mục 15–16 (lệnh nghiệm thu).

## 1. Đề tài (`detai.txt`)

Nhận diện chữ trong nhãn hiệu sản phẩm bằng học có giám sát với mô hình học sâu:
1. thu thập nhãn hiệu + nhãn chữ tương ứng (đăng ký tại Mỹ);
2. huấn luyện và đánh giá mô hình nhận diện chữ trong nhãn hiệu;
3. đánh giá trên tập dữ liệu mới (nhãn hiệu đăng ký tại Việt Nam).

## 2. Đánh giá phương án và code (tại nhánh `14-demo-report`)

**Kết luận.** Khung kỹ thuật tốt (pipeline tách module, test offline, tài liệu kỹ, tích hợp
parseq không sửa code gốc, chia tập theo nhóm nhãn, NFC xuyên suốt). Nhưng đồ án **chưa có kết
quả thật**, có một bug chặn việc tăng dữ liệu, và cách đánh giá US→VN **không đo đúng thứ đề tài
muốn đo**.

| Yêu cầu | Trạng thái lúc review |
|---|---|
| Thu thập US | Công cụ tốt (USPTO EOG API, không cần key); mới 4.840 mẫu sạch / mục tiêu 50k–150k |
| Huấn luyện + đánh giá | Config + dry-run cho 4 model; chưa train thật (ckpt 2 batch, accuracy 0); `docs/colab/` trống |
| Đánh giá trên VN | Crawler WIPO Publish; 50 mẫu, chỉ 36 vào được đánh giá / mục tiêu 2k–5k |

Vấn đề, xếp từ nặng đến nhẹ:

| § | Vấn đề | Bằng chứng |
|---|---|---|
| 4.1 | **Bug**: LMDB ≥ 1.000 mẫu đọc lại ra 0 (`txn` gán lại trong khối `with`) | tái hiện: 999 → 999; 1.000 / 1.500 / 2.500 → 0; parseq ném `TypeError` |
| 4.2 | Nhãn US viết HOA, gần như không dấu; nhãn VN lẫn hoa/thường, có dấu → độ sụt US→VN chủ yếu do quy ước nhãn | 200/200 nhãn US viết hoa; **15/36 nhãn VN (42%) model không thể đúng** |
| 4.3 | Cả ảnh nhãn hiệu (logo, nhiều dòng) bị co về 32×128; ~70% US là `standard_character` (chữ đánh máy, dễ) | không có bước phát hiện vùng chữ |
| 4.4 | SVTR: 64 "bước" CTC thực chất là 4 lần quét × **16 cột** < 25 ký tự | kiểm tra `64 ≥ 25` vẫn qua, che lỗi |
| 4.5 | Evaluate lọc ngầm nhãn dài, xoá ký tự lạ khỏi nhãn đúng, `idx` không truy ngược được | VN 50 → 36 mẫu không báo |
| 4.6 | Không nạp được trọng số pretrained (parseq nạp strict, lệch charset); `docs/colab/` trống | |
| 4.7 | Nhãn VN chưa kiểm tra tay, có trùng, lẫn đơn chưa cấp, crawl dừng sớm | `GASU` ×2, `ENGLISH1984`/`ENGLISH 1984` |
| 4.8 | Lặt vặt: README `svtrctc` sai tên, `except` rộng, cỡ ảnh viết cứng, đường dẫn máy cụ thể, parseq chưa ghim, `aiohttp` thừa | |

Đề xuất chính: đánh giá **không phân biệt hoa/thường**, báo cáo thêm mức **bỏ dấu**, tách tập VN
**có dấu / không dấu** — phần chênh lệch còn lại mới là khác biệt hình ảnh, cũng là phần phân
tích giá trị nhất của đề tài.

## 3. Hai nhánh git (yêu cầu: review ra md + test phần chạy tốt; nhánh khác xử lý mọi vấn đề)

### `15-review-tests` (từ `14-demo-report`) — 2 commit

- `REVIEW.md`: toàn bộ nhận định ở mục 2, kèm cách tái hiện.
- 28 test mới khoá các phần đang chạy đúng (177 pass, 1 skip): chuỗi clean → split → LMDB →
  `SceneTextDataModule`; config khớp parseq và hai bẫy tiếng Việt (chứng minh bằng parseq thật);
  charset phủ đủ khối Unicode chữ Việt; tính chất của edit distance; phân trang crawler VN.

### `16-fix-review` (từ nhánh 15) — 8 commit, mỗi lỗi có test tái hiện chạy đỏ trước khi sửa

| Commit | Nội dung |
|---|---|
| 16.1 | LMDB: quản lý transaction tay, commit lô cuối + `num-samples` |
| 16.2 | SVTR: merging stride (2,1), gộp chiều cao, positional embedding → 32 cột |
| 16.3 | Đánh giá trung thực: chấm mọi mẫu với nhãn gốc, `coverage`, `idx` = chỉ số LMDB + serial/`drawing_kind`; metric 3 mức (tuyệt đối / bỏ hoa-thường / bỏ dấu); `img_size` đọc từ checkpoint |
| 16.4 | Tách độ sụt US→VN (hoa-thường / dấu / hình ảnh) + CI 95% bootstrap; báo cáo theo `drawing_kind` |
| 16.5 | Dữ liệu VN: crawl qua trang rỗng, lọc trùng, `--registered-only`, công cụ duyệt tay `vn_review` |
| 16.6 | Fine-tune từ trọng số pretrained: `crnn_ft` / `trba_ft` / `parseq_ft` |
| 16.7 | `DATA_ROOT` mặc định một chỗ (`<repo>/data`), alias `svtrctc`, ghim commit parseq (`scripts/00_get_parseq.sh`), 4 notebook Colab |
| — | `REVIEW.md` §6: trạng thái từng vấn đề |

Kiểm chứng: 236 pass, 0 skip (cả hai nhánh chạy lại trên bản `git archive` sạch); trọng số
pretrained thật nạp đúng (parseq 172/175 tensor, chỉ bỏ phần phụ thuộc charset); train thật 2
batch CPU cho `svtr` và `parseq_ft` qua `scripts/05_train.sh`, rồi evaluate → domain shift →
report chạy hết. Test chạy trong venv tạm (torch 2.14.1+cpu, PL 2.6.6, timm 1.0.30) vì môi
trường máy chưa cài torch/pytest.

Phát hiện thêm khi sửa: CSV thô của VN khai cột `reg_no` nhưng parser lưu `registration_no`
→ số đăng ký luôn rỗng. Đã sửa ở 16.5.

Thay đổi cần biết:
- `DATA_ROOT` mặc định giờ là `<repo>/data` — máy nào không `export DATA_ROOT` phải export.
- Checkpoint SVTR cũ không nạp được (kiến trúc đổi), phải train lại.
- Phải build lại LMDB thì báo cáo mới có cột serial / `drawing_kind`.
- CSV kết quả đánh giá có thêm cột; cột cũ giữ nguyên.

Còn lại (cần dữ liệu và GPU thật, không làm bằng code): thu thập US ≥ 50k và VN ≥ 1–2k mẫu;
duyệt tay ~300 nhãn VN; train cả hai chế độ (từ đầu và `*_ft`); viết phân tích theo bảng phân
tách độ sụt. Đồ án chưa có bước phát hiện vùng chữ → phạm vi là "nhận dạng một dòng chữ trên
cả ảnh nhãn hiệu", cần ghi rõ trong báo cáo.

## 4. Quy trình làm tiếp (hỏi đáp)

### 4.1 Việc đầu tiên là tải dữ liệu rồi mang lên Colab chạy?

Đúng hướng (dữ liệu trước, train sau), với ba lưu ý:
- **Không có bộ dữ liệu sẵn để tải** — phải tự thu thập bằng script, dùng code nhánh
  `16-fix-review` (nhánh cũ còn bug LMDB).
- **Không mang ảnh rời lên Drive/Colab**; thứ cần cho train là thư mục `lmdb/` (vài file
  `data.mdb`, ~3 GB).
- **Phải push code trước**: notebook clone repo từ GitHub (`REPO_URL`, `BRANCH`). Chưa push
  nhánh 16 (hoặc chưa merge vào `main`) thì Colab chạy code cũ. Repo private thì `REPO_URL`
  dạng `https://<token>@github.com/...`.

Tập VN chỉ cần lúc đánh giá, nên có thể crawl + duyệt tay song song trong lúc train.

### 4.2 Chỉ trên Colab mới có GPU

Đúng, và chỉ bước train cần GPU:

| Bước | Cần GPU? | Runtime Colab |
|---|---|---|
| Tải metadata/ảnh, lọc, chia tập, build LMDB; crawl + duyệt VN | Không (tốn mạng, ổ đĩa) | **CPU** |
| Train (`05_train.sh`) | **Có** | **T4** |
| Đánh giá (`06_eval.sh`) | Không bắt buộc, GPU nhanh hơn nhiều | T4, ngay sau train |

Đừng tải dữ liệu khi đang bật runtime GPU: vài giờ không dùng GPU mà vẫn trừ hạn mức miễn phí.

Khi train trên Colab miễn phí:
- phiên có thể bị ngắt (tối đa khoảng 12 giờ, sớm hơn nếu bỏ không);
- checkpoint ghi thẳng vào Drive (`$DATA_ROOT/runs/...`) nên không mất; train tiếp bằng
  override `ckpt_path=.../checkpoints/last.ckpt`;
- mỗi phiên một model; có 7 cấu hình (4 từ đầu, 3 fine-tune) — ưu tiên `parseq_ft`, `crnn_ft`.

### 4.3 Tải dữ liệu trên Colab, lưu vào Drive — làm thế nào

Dùng `docs/colab/01_download_data.ipynb` (đã viết lại; bản cũ ghi thẳng vào Drive giữ ở
`01_download_data_old.ipynb`). Runtime **CPU**, để tab mở suốt lúc tải.

- **Ảnh US không ghi thẳng vào Drive**: ~60k ảnh chung một thư mục, mà thao tác Drive trên Colab
  dễ timeout khi một thư mục quá nhiều file. Phần US chạy với `DATA_ROOT=/content/trademark_data`
  (ổ máy ảo, mất khi hết phiên), rồi chép lên `MyDrive/trademark_data/`: `raw/meta/`,
  `raw/images_us.tar` (một file), `processed/`, `lmdb/`. Chạy lại sau khi bị ngắt: notebook tự
  lấy lại bản sao từ Drive và tải tiếp (`--resume`, ảnh đã có được bỏ qua).
- **Kiểm tra** sau khi chép: `num-samples` của từng LMDB trên Drive phải là một con số (không phải
  `None`), bằng số dòng split tương ứng.
- **Tập VN làm thẳng trên Drive** (~2k ảnh, cần nhiều phiên): crawl → `vn_review make` → tải
  `processed/` và `raw/images_vn/` về máy, giữ nguyên cấu trúc, mở `review_vn.html`, sửa cột
  `visible_text` (để trống = bỏ mẫu), upload đè `review_vn.csv` → `vn_review apply` → build LMDB
  `test_vn`. Ghi `label_noise_rate` vào báo cáo.
- Xong thì đổi runtime sang **T4** → `00_setup` → `02_train` (tự chép `lmdb/` từ Drive ra ổ máy ảo
  trước khi train, vì đọc LMDB ngẫu nhiên trên Drive rất chậm) → `03_eval`.

Cách thay thế: chạy đúng các lệnh đó ở máy local (với `export DATA_ROOT=<ổ ngoài>`), rồi chỉ
upload thư mục `lmdb/` lên `MyDrive/trademark_data/lmdb/`.

### 4.4 Train xong lấy "file siêu tham số" về là chạy được ở local?

Chạy được trên CPU ở local, nhưng phải lấy **checkpoint `.ckpt`**, không phải `hparams.yaml`:
file hparams chỉ có cấu hình, không có trọng số — dùng nó chỉ dựng được model chưa train.
`.ckpt` chứa cả trọng số lẫn siêu tham số.

- Tải **cả thư mục run**, ít nhất `checkpoints/last.ckpt` + `config/config.yaml`; `evaluate` đọc
  `config.yaml` để biết lớp model (thiếu thì SVTR cần `--model-class trademark_ocr.svtr_ctc.SVTRCTC`).
- Dung lượng: có kèm trạng thái optimizer nên nặng — lần chạy thử `parseq_ft` 383 MB,
  `svtrctc` 38 MB.
- Máy local phải dùng code nhánh 16 (kiến trúc SVTR đã đổi).
- Checkpoint train trên GPU nạp được trên máy chỉ có CPU: Lightning tự chuyển tensor CUDA sang
  CPU khi không có CUDA.

```bash
python -m trademark_ocr.demo_app --ckpt runs/parseq_ft/<ts>/checkpoints/last.ckpt --image anh.png
python -m trademark_ocr.evaluate --ckpt runs/parseq_ft/<ts>/checkpoints/last.ckpt --subset test_vn \
    --out reports/parseq_ft_vn.csv --batch-size 16      # cần thư mục lmdb/ ở local
```

## 5. Thay đổi cuối phiên

- Sao lưu `docs/colab/01_download_data.ipynb` → `docs/colab/01_download_data_old.ipynb`, rồi viết
  lại `01_download_data.ipynb` theo mục 4.3 (tự clone repo + cài thư viện vì runtime CPU là
  phiên riêng; tham số `BRANCH`, `US_META_ROWS`, `VN_LIMIT`, `VN_REVIEW_N` ở cell đầu). Các bước
  sao lưu / khôi phục / rsync đã chạy thử bằng bash với thư mục giả.
- `DESIGN.md` 16.7 cập nhật mô tả notebook.
- File này.
