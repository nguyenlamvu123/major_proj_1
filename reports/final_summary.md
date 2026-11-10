# Tổng hợp kết quả — nhận dạng chữ nhãn hiệu

_Sinh lúc: 2026-11-10 08:36 (UTC+07:00)_

## Bảng tổng hợp (model × dataset)

| model | dataset | n | accuracy | 1−NED | char-accuracy |
|---|---|---|---|---|---|
| crnn_ft | US | 2847 | 0.7489 | 0.8966 | 0.8599 |
| crnn_ft | VN | 2000 | 0.1940 | 0.3921 | 0.1943 |
| parseq_ft | US | 2847 | 0.8869 | 0.9574 | 0.9418 |
| parseq_ft | VN | 2000 | 0.2760 | 0.5336 | 0.3367 |
| svtrctc | US | 2847 | 0.6937 | 0.8482 | 0.8211 |
| svtrctc | VN | 2000 | 0.1140 | 0.2750 | 0.1289 |
| trba_ft | US | 2847 | 0.8423 | 0.9303 | 0.8958 |
| trba_ft | VN | 2000 | 0.2385 | 0.4847 | 0.2781 |

## Ba mức so khớp + độ bao phủ

| model | dataset | n | coverage | accuracy | acc (bỏ hoa/thường) | acc (bỏ cả dấu) |
|---|---|---|---|---|---|---|
| crnn_ft | US | 2847 | 1.000 | 0.7489 | 0.7489 | 0.7503 |
| crnn_ft | VN | 2000 | 0.778 | 0.1940 | 0.2855 | 0.3035 |
| parseq_ft | US | 2847 | 1.000 | 0.8869 | 0.8869 | 0.8890 |
| parseq_ft | VN | 2000 | 0.778 | 0.2760 | 0.4105 | 0.4580 |
| svtrctc | US | 2847 | 1.000 | 0.6937 | 0.6937 | 0.6951 |
| svtrctc | VN | 2000 | 0.778 | 0.1140 | 0.1640 | 0.1715 |
| trba_ft | US | 2847 | 1.000 | 0.8423 | 0.8423 | 0.8455 |
| trba_ft | VN | 2000 | 0.778 | 0.2385 | 0.3605 | 0.3980 |

_coverage = tỉ lệ mẫu model CÓ THỂ đúng (nhãn ≤ max_label_length, mọi ký tự thuộc charset); mẫu ngoài phạm vi vẫn tính là sai. Nhãn USPTO luôn VIẾT HOA và gần như không dấu, nên so sánh US↔VN nên đọc ở cột bỏ hoa/thường._

## Sụt giảm US → VN (domain shift)

| model | acc_us | acc_vn | delta | n_us | n_vn |
|---|---|---|---|---|---|
| parseq_ft | 0.8868984896382157 | 0.276 | -0.6108984896382157 | 2847 | 2000 |
| trba_ft | 0.8422901299613629 | 0.2385 | -0.6037901299613628 | 2847 | 2000 |
| svtrctc | 0.6937126800140498 | 0.114 | -0.5797126800140499 | 2847 | 2000 |
| crnn_ft | 0.7488584474885844 | 0.194 | -0.5548584474885845 | 2847 | 2000 |

_delta âm = độ chính xác giảm khi chuyển sang nhãn hiệu Việt Nam._

### Phân tách nguồn gốc độ sụt (CI 95% bootstrap)

| model | Δ tuyệt đối | Δ bỏ hoa/thường | Δ bỏ cả dấu | Δ VN không dấu, bỏ hoa/thường | n VN không dấu / có dấu |
|---|---|---|---|---|---|
| parseq_ft | -0.6109 | -0.4764 [-0.502, -0.453] | -0.4310 | -0.3189 [-0.347, -0.291] | 1442 / 558 |
| trba_ft | -0.6038 | -0.4818 [-0.506, -0.458] | -0.4475 | -0.3423 [-0.371, -0.313] | 1442 / 558 |
| svtrctc | -0.5797 | -0.5297 [-0.554, -0.507] | -0.5236 | -0.4663 [-0.495, -0.440] | 1442 / 558 |
| crnn_ft | -0.5549 | -0.4634 [-0.490, -0.439] | -0.4468 | -0.3529 [-0.384, -0.323] | 1442 / 558 |

_Δ bỏ hoa/thường loại phần sai do quy ước ghi nhãn; Δ bỏ cả dấu loại thêm phần sai do chưa học chữ có dấu; Δ trên nhóm VN không dấu là ước lượng sạch nhất của khác biệt **hình ảnh**. CI rộng = n quá nhỏ để kết luận._

## Theo loại nhãn hiệu (drawing_kind)

| model | dataset | drawing_kind | n | acc (bỏ hoa/thường) |
|---|---|---|---|---|
| crnn_ft | US | design_with_words | 418 | 0.1555 |
| crnn_ft | US | standard_character | 2288 | 0.8702 |
| crnn_ft | US | stylized_words | 112 | 0.4286 |
| crnn_ft | US | typeset_old | 29 | 0.9655 |
| parseq_ft | US | design_with_words | 418 | 0.4091 |
| parseq_ft | US | standard_character | 2288 | 0.9834 |
| parseq_ft | US | stylized_words | 112 | 0.6786 |
| parseq_ft | US | typeset_old | 29 | 0.9655 |
| svtrctc | US | design_with_words | 418 | 0.0335 |
| svtrctc | US | standard_character | 2288 | 0.8392 |
| svtrctc | US | stylized_words | 112 | 0.1339 |
| svtrctc | US | typeset_old | 29 | 0.8966 |
| trba_ft | US | design_with_words | 418 | 0.2368 |
| trba_ft | US | standard_character | 2288 | 0.9642 |
| trba_ft | US | stylized_words | 112 | 0.5804 |
| trba_ft | US | typeset_old | 29 | 0.9655 |

_`standard_character` là chữ đánh máy font chuẩn do USPTO dựng — dễ hơn nhiều so với `stylized_words`/`design_with_words`; accuracy US gộp chung phụ thuộc mạnh vào tỉ lệ loại này._

## Hình minh hoạ

- `/mnt/1864514f-3a41-4b8f-9907-2fbcd9d2815b/airi/doanchuyennganh1/data/processed/eda_figures/err_by_length.png`
- `/mnt/1864514f-3a41-4b8f-9907-2fbcd9d2815b/airi/doanchuyennganh1/data/processed/eda_figures/err_char_confusions.png`
- `/mnt/1864514f-3a41-4b8f-9907-2fbcd9d2815b/airi/doanchuyennganh1/data/processed/eda_figures/err_top_samples.png`
