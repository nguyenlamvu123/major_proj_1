"""Nhánh 09 — SVTR-CTC tự viết (bản rút gọn "SVTR-Lite").

Cài đặt như một `strhub.models.base.CTCSystem` (KHÔNG sửa parseq) để dùng lại nguyên
vẹn vòng train/eval/metric của parseq:

* `forward(images)` -> logits `(N, L, C)`; CTC loss + decode do lớp cha lo.
* Kiến trúc SVTR-style: **stem conv** (giảm 4×) + **positional embedding** → các
  **MixingBlock** (Local Mixing = depthwise conv 3×3, Global Mixing = self-attention) →
  **merging** (chỉ giảm chiều cao, stride (2,1)) → **gộp chiều cao về 1** → head tuyến tính
  theo từng cột (CTC).

Trục thời gian CTC là **chiều ngang**: mỗi bước thời gian là một cột của feature map, nên
chuỗi đi trái→phải đúng thứ tự đọc. Với `img_size=[32,128]`: stem → 8×32, merge → 4×32,
gộp cao → `L = 32` cột. Ràng buộc: `L` (số cột) phải **≥ `max_label_length`** (25), nếu
không CTC không thể phát nhãn dài nhất — class kiểm tra lúc dựng.

(Bản đầu tiên giảm cả chiều rộng rồi trải phẳng lưới 4×16 theo hàng thành 64 bước: kiểm tra
`64 ≥ 25` vẫn qua nhưng thực chất chỉ có 16 cột — xem REVIEW.md §4.4.)

Tham chiếu: SVTR (Yu et al., 2022) + PP-OCRv3 "SVTR-Lite" (bỏ bớt global mixing ở
stage đầu, dùng depthwise conv). Đây là bản **tự viết** phục vụ so sánh, không phải
bản chính thức của OpenOCR.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from strhub.models.base import CTCSystem

try:  # timm >= 0.9 có DropPath ở timm.layers
    from timm.layers import DropPath
except Exception:  # pragma: no cover - fallback
    class DropPath(nn.Module):  # type: ignore[no-redef]
        def __init__(self, drop_prob: float = 0.0) -> None:
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x: Tensor) -> Tensor:
            return x


def _ln(dim: int) -> nn.LayerNorm:
    return nn.LayerNorm(dim, eps=1e-6)


class ConvBN(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU(),
        )


class PatchEmbed(nn.Module):
    """Stem: 2 conv 3×3 stride 2 → giảm 4× (H/4, W/4)."""

    def __init__(self, in_ch: int = 3, embed_dim: int = 192) -> None:
        super().__init__()
        c1 = embed_dim // 2
        self.proj = nn.Sequential(ConvBN(in_ch, c1, 3, 2, 1), ConvBN(c1, embed_dim, 3, 2, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class Merging(nn.Sequential):
    """Giảm 2× **chiều cao**, giữ nguyên chiều rộng (conv 3×3 stride (2, 1)).

    Giữ chiều rộng để số cột — tức số bước thời gian CTC — không bị chia đôi (như SVTR gốc).
    """

    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__(nn.Conv2d(dim, out_dim, 3, (2, 1), 1, bias=False), nn.BatchNorm2d(out_dim))


class MixingBlock(nn.Module):
    """SVTR mixing block: Local Mixing + Global Mixing (attn) + MLP, residual."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop: float = 0.0, drop_path: float = 0.0, global_mix: bool = True) -> None:
        super().__init__()
        self.global_mix = global_mix
        self.norm1 = _ln(dim)
        self.local_mix = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True) \
            if global_mix else None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = _ln(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:            # x: (B, C, H, W)
        b, c, h, w = x.shape
        xn = self.norm1(x.permute(0, 2, 3, 1))          # LN trên kênh
        xt = xn.permute(0, 3, 1, 2)
        local = self.local_mix(xt)
        mix = local
        if self.attn is not None:
            seq = xt.flatten(2).transpose(1, 2)          # (B, N, C)
            attn, _ = self.attn(seq, seq, seq, need_weights=False)
            mix = mix + attn.transpose(1, 2).view(b, c, h, w)
        x = x + self.drop_path(mix)
        y = self.mlp(self.norm2(x.permute(0, 2, 3, 1)))
        return x + self.drop_path(y.permute(0, 3, 1, 2))


class SVTRNet(nn.Module):
    def __init__(self, img_size: Sequence[int], num_classes: int, embed_dim: int = 192,
                 depths: Sequence[int] = (2, 2), num_heads: int = 6, mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0, drop_path_rate: float = 0.1,
                 global_mix: Sequence[bool] = (True, True)) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(3, embed_dim)
        # Mỗi conv 3×3 stride 2 pad 1 cho ra ceil(n/2); hai lần liên tiếp = ceil(n/4).
        img_h, img_w = img_size
        grid = (math.ceil(img_h / 4), math.ceil(img_w / 4))
        # Self-attention bất biến với hoán vị token; positional embedding học được (như SVTR)
        # cho attention biết thứ tự cột, nhánh depthwise conv chỉ thấy lân cận 3×3.
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *grid))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist() if drop_path_rate > 0 \
            else [0.0] * sum(depths)
        stages = []
        idx = 0
        dim = embed_dim
        for si, depth in enumerate(depths):
            blocks = []
            for _ in range(depth):
                blocks.append(MixingBlock(dim, num_heads, mlp_ratio, drop_rate, dpr[idx],
                                          global_mix=bool(global_mix[si])))
                idx += 1
            stages.append(nn.Sequential(*blocks))
            if si < len(depths) - 1:
                stages.append(Merging(dim, dim))
        self.stages = nn.Sequential(*stages)
        self.norm = _ln(dim)
        self.head = nn.Linear(dim, num_classes)
        self.apply(self._init_weights)
        self.num_features = dim

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x) + self.pos_embed
        x = self.stages(x)                              # (B, C, H', W/4)
        # Gộp chiều cao về 1 ("combing" của SVTR): mỗi cột thành một bước thời gian, nên
        # chuỗi CTC đi trái→phải theo đúng thứ tự đọc. KHÔNG trải phẳng H'×W theo hàng —
        # làm vậy biến chuỗi thành H' lần quét lặp lại (REVIEW.md §4.4).
        x = x.mean(dim=2).transpose(1, 2)               # (B, W/4, C)
        return self.head(self.norm(x))                  # (N, L=W/4, num_classes) — CTC head


class SVTRCTC(CTCSystem):
    """`CTCSystem` bọc `SVTRNet` (dùng làm `_target_` trong `configs/hydra/tm_svtrctc.yaml`)."""

    def __init__(
        self,
        charset_train: str,
        charset_test: str,
        max_label_length: int,
        batch_size: int,
        lr: float,
        warmup_pct: float,
        weight_decay: float,
        img_size: Sequence[int],
        embed_dim: int = 192,
        depths: Sequence[int] = (2, 2),
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        global_mix: Sequence[bool] = (True, True),
        **kwargs: Any,
    ) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()
        self.max_label_length = max_label_length
        img_h, img_w = img_size
        self.model = SVTRNet(
            img_size=(img_h, img_w),
            num_classes=len(self.tokenizer),
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            global_mix=global_mix,
        )
        # Invariant CTC: số cột L (= W/4) phải đủ cho nhãn dài nhất.
        with torch.no_grad():
            self.eval()
            dummy = torch.zeros(1, 3, img_h, img_w)
            L = self.model(dummy).shape[1]
            self.train()
        assert L >= max_label_length, (
            f"SVTRCTC: số cột/timestep L={L} < max_label_length={max_label_length}; "
            "giảm max_label_length hoặc tăng chiều rộng img_size."
        )
        self.seq_len = L

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        return self.model(images)

    def training_step(self, batch, batch_idx) -> Tensor:
        # CTCSystem không định nghĩa training_step -> phải tự khai báo (như CRNN).
        images, labels = batch
        loss = self.forward_logits_loss(images, labels)[1]
        self.log('loss', loss)
        return loss
