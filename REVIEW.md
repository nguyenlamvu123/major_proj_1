# REVIEW.md — Đánh giá phương án & code (2026-10-23)

> Phạm vi: toàn bộ repo tại nhánh `14-demo-report` (commit `7870540`), đối chiếu với đề tài
> trong `detai.txt`:
>
> *Nhận diện chữ trong nhãn hiệu sản phẩm sử dụng học có giám sát với các mô hình học sâu:
> (1) thu thập nhãn hiệu + nhãn chữ (đăng ký tại Mỹ); (2) huấn luyện và đánh giá mô hình;
> (3) đánh giá trên tập dữ liệu mới (nhãn hiệu đăng ký tại Việt Nam).*
>
> File này ghi lại **nhận định tại thời điểm review**. Trạng thái xử lý từng vấn đề được
> cập nhật ở mục cuối (§6) khi có nhánh sửa.

## 1. Kết luận

Phần khung kỹ thuật làm tốt: pipeline tách module rõ, có test offline, tài liệu kỹ, cách
ghép vào parseq cẩn thận. Nhưng đồ án **chưa có kết quả thật**. Có một bug sẽ chặn việc
tăng dữ liệu. Nghiêm trọng hơn, cách đánh giá US→VN hiện nay **không đo đúng thứ đề tài
muốn đo**.

## 2. Mức hoàn thành theo đề tài

| Yêu cầu | Trạng thái |
|---|---|
| Thu thập nhãn hiệu US kèm nhãn chữ | Có công cụ tốt (USPTO EOG API, không cần key). Mới có **4.840 mẫu sạch**, mục tiêu là 50k–150k |
| Huấn luyện và đánh giá mô hình | Có config và dry-run cho 4 mô hình. **Chưa huấn luyện thật**: checkpoint mới chạy 2 batch, accuracy = 0. Thư mục `docs/colab/` còn trống |
| Đánh giá trên nhãn hiệu VN | Có crawler WIPO Publish. Mới có **50 mẫu, chỉ 36 mẫu vào được đánh giá**, mục tiêu là 2k–5k |

## 3. Điểm mạnh

- Tích hợp parseq mà không sửa code gốc (3 điểm tiếp xúc: hydra `--config-dir`, charset
  group, LMDB đúng layout). Đã phát hiện đúng hai bẫy làm mất dấu tiếng Việt và mất khoảng
  trắng: `normalize_unicode`, `remove_whitespace`; và bẫy `charset_test` 36 ký tự.
- Chia tập theo **nhóm nhãn** nên không rò rỉ nhãn giữa train và test. Nhãn được chuẩn hoá
  **NFC** xuyên suốt (fetch → clean → LMDB → parseq).
- Crawler chạy lại được giữa chừng (resume), tải song song với session thread-local, kiểm
  tra ảnh bằng PIL, ghi file nguyên tử (tmp → replace).
- Test offline đầy đủ (149 pass, 1 skip cần dữ liệu thật), fixture HTML thật cho WIPO Publish.
- Hàm `edit_distance` tự viết đúng chuẩn, không phụ thuộc thư viện ngoài.
- Shim NumPy-2 cho `imgaug` giải quyết đúng chỗ (không hạ numpy, không sửa parseq).

## 4. Vấn đề, xếp từ nặng đến nhẹ

### 4.1 [Bug nghiêm trọng] Từ 1.000 mẫu trở lên, LMDB đọc lại ra 0 mẫu

`src/trademark_ocr/build_lmdb.py:79-91`: code gán lại `txn` **bên trong** khối
`with env.begin(write=True) as txn:`. Khi thoát khối, Python gọi `__exit__` của transaction
**cũ** (đã commit), còn transaction mới — chứa các mẫu sau và khoá `num-samples` — không bao
giờ được commit và bị huỷ khi `env.close()`.

Tái hiện (môi trường tạm chỉ cài `lmdb`):

```
n=999:  create_lmdb báo num_samples=999,  đọc lại được 999 mẫu
n=1000: create_lmdb báo num_samples=1000, đọc lại được 0 mẫu
n=1500: create_lmdb báo num_samples=1500, đọc lại được 0 mẫu
n=2500: create_lmdb báo num_samples=2500, đọc lại được 0 mẫu
```

Với parseq, `LmdbDataset._preprocess_labels` gọi `int(txn.get('num-samples'))` → `int(None)`
→ `TypeError` ngay khi nạp dữ liệu. Test và lệnh nghiệm thu không bắt được lỗi này vì luôn
chạy `--limit 500/200`. **Chưa sửa lỗi này thì không thể train ở quy mô thật.**

### 4.2 [Phương pháp — nghiêm trọng nhất] Nhãn US và VN theo hai quy ước khác nhau

Con số "sụt giảm US→VN" hiện nay phần lớn là do **cách ghi nhãn**, không phải do khác biệt
hình ảnh.

- Nhãn USPTO (`markText`) viết **hoa toàn bộ**: 200/200 mẫu trong `reports/crnn_us.csv`.
  Theo EDA (nhánh 06), cả tập US chỉ có **1 ký tự có dấu tiếng Việt**.
- Nhãn VN (`rs-MK`) viết lẫn hoa thường và có dấu: `Hoff`, `Luân Hồi`,
  `HP Food Đặc sản Hải Phòng`.
- Kết quả: **15/36 nhãn VN (42%) là bất khả thi về mặt cấu trúc** — mô hình chưa từng học
  chữ thường hay chữ có dấu thì chắc chắn sai, dù ảnh rõ đến đâu:

  ```
  Hoff, VT VẠN TÍN, Luân Hồi, Qgreen, UniOVR, THỦY SÂM, APL Aluminium, M Morenza,
  M MINH PHÚ, LJ LEJET CAFÉ BISTRO, LINH MỘC Y ĐƯỜNG, THIÊN VŨ, NÔNG TRẠI CẨM THẠCH,
  1P's Bakery & Cafe To-Go, HP Food Đặc sản Hải Phòng
  ```

- Khoảng 146/217 lớp ký tự gần như không có dữ liệu huấn luyện.

Đề xuất:
- Đánh giá **không phân biệt hoa thường** (đưa cả hai phía về chữ hoa) — cách làm chuẩn
  trong nhận dạng chữ.
- Báo cáo ba mức: chính xác tuyệt đối / không phân biệt hoa thường / bỏ cả dấu.
- Tách tập VN thành hai phần "có dấu" và "không dấu". Như vậy mới tách được ảnh hưởng của
  khác biệt hình ảnh khỏi ảnh hưởng của ký tự chưa gặp — đây chính là phần phân tích giá
  trị nhất của đề tài.

### 4.3 [Phương pháp] Bài toán được đặt như nhận dạng một dòng chữ đã cắt sẵn, nhưng ảnh nhãn hiệu là ảnh nguyên

- Cả ảnh (khoảng 600×250, thường có logo và chữ nhiều dòng) bị co thẳng về 32×128, không
  giữ tỉ lệ (`SceneTextDataModule.get_transform` → `T.Resize(img_size)`). Thiếu bước phát
  hiện vùng chữ.
- Khoảng 70% tập US là loại `standard_character`, tức chữ đánh máy font chuẩn, rất dễ. Điều
  này làm accuracy US cao giả tạo và khuếch đại độ sụt sang VN.
- Đề xuất: báo cáo theo từng `drawing_kind`. Lọc ảnh một dòng, tăng kích thước đầu vào (ví
  dụ 32×256), hoặc thêm bước phát hiện chữ. Cần ghi rõ phạm vi bài toán trong báo cáo.

### 4.4 [Thiết kế SVTR] Trục thời gian của CTC bị đặt sai

- `src/trademark_ocr/svtr_ctc.py:67`: `Merging` giảm **cả chiều rộng** (stride 2).
  `:147`: lưới 4×16 được trải phẳng theo từng hàng (`view(b, h*w, c)`). Vì vậy chuỗi 64
  bước thực chất là 4 lần quét trái→phải, mỗi lần chỉ có **16 cột, ít hơn 25 ký tự tối đa**.
- Phép kiểm tra `L >= max_label_length` ở `:194` vẫn qua (64 ≥ 25) nên che mất lỗi này.
  `tests/test_svtr.py` còn khoá cứng `seq_len == 64`; test overfit chỉ kiểm tra loss giảm
  nên cũng không phát hiện.
- So sánh: CRNN của parseq gộp chiều cao về 1 và giữ ~33 cột ngang.
- Cách sửa giống SVTR gốc: `Merging` dùng stride (2,1), rồi lấy trung bình theo chiều cao,
  được L = 32 cột. Nên thêm positional embedding (attention hiện không có thông tin vị trí).

### 4.5 [Đo lường] Phần đánh giá âm thầm thay đổi tập test

- `src/trademark_ocr/evaluate.py:91`: `LmdbDataset(..., max_label_len=25)` bỏ các nhãn VN
  dài: 50 còn 36, và báo cáo không ghi lại tỉ lệ bị bỏ.
- `CharsetAdapter` của parseq (trong `LmdbDataset`) xoá các ký tự ngoài bảng ký tự ra khỏi
  **chính nhãn đúng** (ground truth).
- Cột `idx` trong CSV là số thứ tự **sau lọc**, không khớp với chỉ số trong LMDB. Vì vậy
  không truy ngược được về ảnh, serial hay `drawing_kind` khi phân tích lỗi.
- Với n=36 thì không đủ ý nghĩa thống kê. Cần ≥1–2k mẫu VN kèm khoảng tin cậy bootstrap.

### 4.6 [Kết quả] Chưa có lộ trình huấn luyện khả thi

- PARSeq và các mô hình dạng ViT train từ đầu trên khoảng 50k ảnh sẽ yếu. Nên khởi tạo từ
  trọng số pretrained của parseq.
- Cờ `pretrained` trong `third_party/parseq/train.py:79-81` nạp trọng số ở chế độ
  **strict**, nên sẽ lỗi vì lớp đầu ra của bảng ký tự VN (218/220 lớp) khác kích thước bản
  pretrained (94 ký tự). Cần nạp một phần, bỏ qua các tensor lệch kích thước.
- `docs/colab/` chỉ có `.gitkeep`, trong khi README/DESIGN trỏ tới `00_setup.ipynb` …
  `03_eval.ipynb`.

### 4.7 [Dữ liệu VN]

- Trường `rs-MK` là phần chữ được đăng ký, **không phải** bản chép lại nội dung ảnh. Cần
  kiểm tra tay vài trăm mẫu và báo cáo tỉ lệ nhãn sai.
- Có nhãn trùng: `GASU` ×2, `ENGLISH1984` / `ENGLISH 1984`.
- `query=*:*` lấy cả đơn chưa được cấp, không chỉ "nhãn hiệu đã đăng ký".
- `src/trademark_ocr/fetch_vn.py:230` dừng crawl ngay khi gặp **một** trang không có mẫu nào
  dùng được (`new == 0`).

### 4.8 [Nhỏ]

- README dòng 65 hướng dẫn `05_train.sh svtrctc`, nhưng script chỉ nhận `svtr` nên sẽ báo
  `unknown model`.
- DESIGN.md (bảng module, model zoo) ghi `models_svtr.py`, file thật là `svtr_ctc.py`.
- `evaluate.py:55-57` dùng `except Exception` quá rộng (mọi lỗi nạp ckpt đều rơi sang
  `SVTRCTC`, che lỗi thật) và đoán lớp mô hình theo **chuỗi con trong đường dẫn**
  (`'parseq' in path`…).
- Kích thước ảnh `(32, 128)` và độ dài nhãn `25` bị viết cứng trong `evaluate.py` và
  `demo_app.py` thay vì đọc từ hparams của checkpoint.
- Đường dẫn mặc định là `/media/zaibachkhoa/...` (lặp ở 10 module, 4 config, 5 script),
  trái với `./data` mà DESIGN §2 mô tả.
- parseq chưa ghim commit (hiện đang ở `1902db0`).
- `aiohttp` có trong requirements nhưng không dùng.

## 5. Thứ tự nên làm tiếp

1. Sửa bug LMDB và thêm một test với hơn 1.000 mẫu.
2. Thống nhất quy ước nhãn (chữ hoa, đánh giá không phân biệt hoa thường), thêm các metric
   bỏ dấu và tách tập VN có dấu / không dấu.
3. Sửa trục thời gian của SVTR.
4. Sửa phần đánh giá để không âm thầm lọc nhãn, ghi tỉ lệ bao phủ và lưu serial vào CSV.
5. Tăng dữ liệu (US ≥50k, VN ≥1–2k đã kiểm tra tay), viết notebook Colab, train từ trọng
   số pretrained.
6. Báo cáo kết quả theo `drawing_kind` và theo nhóm có dấu / không dấu.

### Giới hạn của lần review này

- Kết luận về SVTR rút ra từ đọc code và tính kích thước tensor (32×128 → stem 8×32 →
  merging 4×16 → 64 bước).
- Thống kê nhãn lấy từ hai file `reports/crnn_{us,vn}.csv` có sẵn (không truy cập được
  `DATA_ROOT` gốc).
- Bộ test của dự án được chạy trong một venv tạm (torch 2.14.1+cpu, PL 2.6.6, timm 1.0.30):
  149 pass, 1 skip (`test_evaluate_end_to_end_on_real_lmdb` cần LMDB thật).

## 6. Trạng thái xử lý (nhánh `16-fix-review`)

Mỗi mục có test tái hiện viết **trước** khi sửa (đỏ → xanh); chi tiết thiết kế ở DESIGN.md
mục 16.x, lệnh nghiệm thu ở ACCEPTANCE.md mục 16.x.

| § | Vấn đề | Trạng thái | Ở đâu |
|---|---|---|---|
| 4.1 | LMDB ≥ 1.000 mẫu đọc ra 0 | ✅ Đã sửa | 16.1 — `build_lmdb.create_lmdb` quản lý transaction tay; test 7 mẫu/`commit_every=3` và 1.500 mẫu qua parseq |
| 4.2 | Quy ước nhãn US (HOA) ≠ VN (hoa/thường, có dấu) | ✅ Đã sửa phần đo | 16.3 — metric 3 mức `strict/nocase/base`; 16.4 — `delta_nocase`, `delta_base`, `delta_plain_nocase` (VN không dấu) + CI 95% |
| 4.3 | Ảnh nguyên, nhiều dòng; 70% `standard_character` | 🟡 Đo được, chưa đổi bài toán | 16.4 — bảng theo `drawing_kind`; 16.3 — `img_size` đọc từ ckpt nên thử `model.img_size=[32,256]` được. **Chưa** có bước phát hiện vùng chữ — giữ phạm vi "nhận dạng 1 dòng trên cả ảnh", cần ghi rõ trong báo cáo |
| 4.4 | SVTR: 64 bước = 4 × 16 cột | ✅ Đã sửa | 16.2 — merging stride (2,1), gộp chiều cao, `pos_embed`; L = 32 cột; 32×64 bị từ chối |
| 4.5 | Evaluate lọc ngầm, sửa gt, `idx` không truy ngược được | ✅ Đã sửa | 16.3 — chấm mọi mẫu, gt gốc, `in_scope` + `coverage`, `idx` = chỉ số LMDB + `serial_number`/`drawing_kind`; 16.4 — CI bootstrap |
| 4.6 | Không nạp được pretrained; `docs/colab/` trống | ✅ Đã sửa phần code | 16.6 — `pretrained.finetune` + `tm_{crnn,trba,parseq}_ft` (kiểm với trọng số thật: parseq nạp 172/175); 16.7 — 4 notebook Colab. **Chưa** train thật trên GPU |
| 4.7 | Nhãn VN chưa kiểm tra, trùng, lẫn đơn chưa cấp, crawl dừng sớm | ✅ Đã có công cụ | 16.5 — `max_empty_pages`, `dedupe_key`, `--registered-only`, `vn_review make/apply` (tỉ lệ nhãn sai). **Chưa** duyệt tay dữ liệu thật |
| 4.8 | README/DESIGN lệch, `except` rộng, viết cứng cỡ ảnh, `DATA_ROOT` máy cụ thể, parseq chưa ghim, `aiohttp` | ✅ Đã sửa | 16.3 (`resolve_model_class`, hparams), 16.7 (phần còn lại) |

**Phát hiện thêm khi sửa** (không có trong review ban đầu): `fetch_vn.write_outputs` khai cột
`reg_no` trong khi parser lưu `registration_no` → số đăng ký bị ghi rỗng ở mọi dòng
`raw/vn_marks.csv`. Đã sửa ở 16.5 (cột sinh từ `_FIELDS`).

### Việc còn lại — cần dữ liệu / GPU thật, không làm được bằng code trong repo

1. Thu thập đủ dữ liệu: US ≥ 50k mẫu sạch (hiện 4.840), VN ≥ 1–2k (hiện 50) —
   `docs/colab/01_download_data.ipynb`.
2. Duyệt tay ≥ 300 nhãn VN bằng `vn_review`, ghi `label_noise_rate` vào báo cáo.
3. Train trên GPU cả hai chế độ (từ đầu và `*_ft`) để so sánh công bằng —
   `docs/colab/02_train.ipynb`; SVTR phải train lại (kiến trúc đổi, ckpt smoke cũ không nạp được).
4. Chạy `06_eval.sh`, viết phần phân tích dựa trên bảng phân tách độ sụt (16.4): đọc ở mức
   bỏ hoa/thường và ở nhóm VN không dấu, kèm CI.
5. Build lại LMDB (`build_lmdb`) để có khoá `meta-*` → cột `serial_number`/`drawing_kind`
   trong báo cáo đánh giá.
6. Nếu muốn mở rộng phạm vi: thêm bước phát hiện vùng chữ (4.3) — thay đổi lớn, chưa làm.

Kiểm chứng của nhánh: `pytest -q` → 236 pass, 0 skip (venv torch 2.14.1+cpu / PL 2.6.6 /
timm 1.0.30); smoke train thật 2 batch CPU cho `svtr` và `parseq_ft` qua `scripts/05_train.sh`
trên LMDB tổng hợp, rồi `evaluate` → `eval_domain_shift` → `report` chạy hết.
