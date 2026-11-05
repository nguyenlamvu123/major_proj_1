# Đường cong huấn luyện

Sinh tự động bởi `python -m trademark_ocr.train_curves` từ log TensorBoard của từng run. `epoch` đánh số từ 1, tính dồn qua các lần train tiếp. Loss train tính trên ảnh đã augment nên thường cao hơn val_loss — so **xu hướng** hai đường, không so độ lớn.

## crnn_ft

Run `crnn_ft/2026-11-04_21-58-41` — **train xong** (`DONE`).

![crnn_ft](train_curves_crnn_ft.png)

| epoch | loss train | val_loss | val_accuracy (%) | val 1−NED (%) | ghi chú |
|---|---|---|---|---|---|
| 1 | 6,8885 | 2,5449 | 0,00 | 5,30 |  |
| 2 | 1,3157 | 0,6365 | 60,93 | 85,74 |  |
| 3 | 0,7338 | 0,5189 | 67,64 | 87,49 |  |
| 4 | 0,6431 | 0,4765 | 70,48 | 88,59 |  |
| 5 | 0,6020 | 0,4575 | 71,61 | 88,86 |  |
| 6 | 0,5769 | 0,4456 | 72,63 | 89,08 |  |
| 7 | 0,5632 | 0,4354 | 73,05 | 89,35 |  |
| 8 | 0,5503 | 0,4254 | 74,07 | 89,52 | tốt nhất |
| 8 | — | 0,4445 | 72,59 | 89,10 | model sau SWA |

**Kết luận:** chưa thấy overfit: val_loss vẫn giảm tới epoch 8 (0,4254); model còn học thêm được.

## parseq_ft

Run `parseq_ft/2026-11-05_00-27-27` — **train xong nhưng chưa học được** (`UNDERTRAINED`).

![parseq_ft](train_curves_parseq_ft.png)

| epoch | loss train | val_loss | val_accuracy (%) | val 1−NED (%) | ghi chú |
|---|---|---|---|---|---|
| 1 | 4,6220 | 3,8378 | 0,00 | 4,15 |  |
| 2 | 3,4865 | 3,2631 | 0,00 | 7,03 |  |

**Kết luận:** mới có 2 epoch có val_loss — chưa đủ để kết luận.
