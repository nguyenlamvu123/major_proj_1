# Tổng hợp kết quả — nhận dạng chữ nhãn hiệu

_Sinh lúc: 2026-10-15 18:41 UTC_

## Bảng tổng hợp (model × dataset)

| model | dataset | n | accuracy | 1−NED | char-accuracy |
|---|---|---|---|---|---|
| crnn | US | 200 | 0.0000 | 0.0233 | -0.1266 |
| crnn | VN | 36 | 0.0000 | 0.0295 | -0.0862 |

## Sụt giảm US → VN (domain shift)

| model | acc_us | acc_vn | delta | n_us | n_vn |
|---|---|---|---|---|---|
| crnn | 0.0 | 0.0 | 0.0 | 200 | 36 |

_delta âm = độ chính xác giảm khi chuyển sang nhãn hiệu Việt Nam._

## Hình minh hoạ

- `/media/zaibachkhoa/AC84-F995/trademark_data/processed/eda_figures/fig_char_freq.png`
- `/media/zaibachkhoa/AC84-F995/trademark_data/processed/eda_figures/fig_drawing_kind.png`
- `/media/zaibachkhoa/AC84-F995/trademark_data/processed/eda_figures/fig_img_size.png`
- `/media/zaibachkhoa/AC84-F995/trademark_data/processed/eda_figures/fig_len_hist.png`
- `/media/zaibachkhoa/AC84-F995/trademark_data/processed/eda_figures/fig_nwords.png`
- `/media/zaibachkhoa/AC84-F995/trademark_data/processed/eda_figures/err_by_length.png`
- `/media/zaibachkhoa/AC84-F995/trademark_data/processed/eda_figures/err_char_confusions.png`
- `/media/zaibachkhoa/AC84-F995/trademark_data/processed/eda_figures/err_top_samples.png`
