"""Nhánh 10 (phần 1) — metric nhận dạng chữ (không phụ thuộc thư viện ngoài).

Định nghĩa (lấy chuẩn ICDAR 2019 / parseq để so sánh được):

* **accuracy** — tỉ lệ *khớp tuyệt đối* (`pred == gt`), phân biệt hoa/thường & dấu.
* **NED** — Normalized Edit Distance: trung bình `levenshtein(pred, gt) / max(|pred|, |gt|)`.
* **1 − NED** — độ "gần đúng" trung bình (càng gần 1 càng tốt).
* **CER** — Character Error Rate ở mức *corpus*: `Σ edit / Σ |gt|`.
* **char_accuracy** — `1 − CER`.

Tất cả hàm nhận `Sequence[str]` và trả `dict` để tiện ghi CSV/JSON.
`edit_distance` viết tay (DP 2 hàng) → **không cần** `python-Levenshtein`, chạy offline.
"""

from __future__ import annotations

from typing import Sequence

METRIC_KEYS: tuple[str, ...] = (
    "n", "accuracy", "ned", "one_minus_ned", "cer", "char_accuracy", "mean_label_len",
)

#: cột CSV báo cáo từng mẫu (nhánh 10) — giữ ổn định cho nhánh 12/14
REPORT_COLUMNS: tuple[str, ...] = (
    "idx", "pred", "gt", "correct", "ned", "char_err", "pred_len", "gt_len",
)


def edit_distance(a: str, b: str) -> int:
    """Khoảng cách Levenshtein (chèn/xóa/thay) bằng DP 2 hàng."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,                                  # xóa
                cur[j - 1] + 1,                               # chèn
                prev[j - 1] + (0 if ca == cb else 1),         # thay
            ))
        prev = cur
    return prev[-1]


def normalized_edit_distance(pred: str, gt: str) -> float:
    """`edit / max(|pred|, |gt|)`; 0.0 nếu cả hai rỗng."""
    denom = max(len(pred), len(gt))
    if denom == 0:
        return 0.0
    return edit_distance(pred, gt) / denom


def char_error_rate(pred: str, gt: str) -> float:
    """`edit / |gt|` (per-sample); 0.0 nếu `gt` rỗng."""
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return edit_distance(pred, gt) / len(gt)


def evaluate_metrics(preds: Sequence[str], gts: Sequence[str]) -> dict[str, float]:
    """Tính toàn bộ metric trên một danh sách cặp (pred, gt)."""
    if len(preds) != len(gts):
        raise ValueError(f"len(preds)={len(preds)} != len(gts)={len(gts)}")
    n = len(preds)
    if n == 0:
        return {k: 0.0 for k in METRIC_KEYS}

    n_correct = 0
    sum_ned = 0.0
    sum_edit = 0
    sum_gt_len = 0
    sum_pred_len = 0
    for pred, gt in zip(preds, gts):
        n_correct += int(pred == gt)
        sum_ned += normalized_edit_distance(pred, gt)
        sum_edit += edit_distance(pred, gt)
        sum_gt_len += len(gt)
        sum_pred_len += len(pred)

    accuracy = n_correct / n
    ned = sum_ned / n
    cer = (sum_edit / sum_gt_len) if sum_gt_len else 0.0
    return {
        "n": float(n),
        "accuracy": accuracy,
        "ned": ned,
        "one_minus_ned": 1.0 - ned,
        "cer": cer,
        "char_accuracy": 1.0 - cer,
        "mean_label_len": sum_gt_len / n,
        "mean_pred_len": sum_pred_len / n,
    }


def per_sample(pred: str, gt: str) -> dict[str, object]:
    """Hàng CSV cho một mẫu (khớp `REPORT_COLUMNS`)."""
    return {
        "pred": pred,
        "gt": gt,
        "correct": int(pred == gt),
        "ned": normalized_edit_distance(pred, gt),
        "char_err": char_error_rate(pred, gt),
        "pred_len": len(pred),
        "gt_len": len(gt),
    }


def format_summary(metrics: dict[str, float]) -> str:
    return (f"n={int(metrics.get('n', 0))} acc={metrics.get('accuracy', 0):.4f} "
            f"1-NED={metrics.get('one_minus_ned', 0):.4f} "
            f"char_acc={metrics.get('char_accuracy', 0):.4f} "
            f"CER={metrics.get('cer', 0):.4f}")
