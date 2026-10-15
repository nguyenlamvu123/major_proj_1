"""Test cho nhánh 08 — forward + loss của CRNN / TRBA / PARSeq trên charset tiếng Việt.

Mục tiêu: bắt sớm mọi lỗi wiring (shape, charset, decode) **trước** khi train thật.

Tất cả chạy ở `model.eval()` vì `TRBA.forward` chỉ hợp lệ khi eval (xem nhánh 07).
"""

from __future__ import annotations

import hydra
import pytest
import torch

from trademark_ocr.charset import CHARSET_VI
from trademark_ocr.train_utils import available_configs, compose_config

NAMES = ("tm_crnn", "tm_trba", "tm_parseq")
LABELS = ["ĐÔNG Á", "HANOI VIET"]      # có dấu + không dấu


@pytest.fixture(scope="module")
def images() -> torch.Tensor:
    torch.manual_seed(1111)
    return torch.rand(2, 3, 32, 128)


@pytest.fixture(scope="module")
def models() -> dict[str, torch.nn.Module]:
    out = {}
    for name in NAMES:
        cfg = compose_config(name, ["model.batch_size=2"])
        m = hydra.utils.instantiate(cfg.model)
        m.eval()
        out[name] = m
    return out


def test_configs_cover_expected_models():
    assert set(NAMES) <= set(available_configs())


@pytest.mark.parametrize("name", NAMES)
def test_forward_logits_shape(name, models, images):
    model = models[name]
    with torch.no_grad():
        logits = model(images)
    assert logits.ndim == 3, f"{name}: mong đợi (N, L, C), got {tuple(logits.shape)}"
    assert logits.shape[0] == 2
    # C: độ lớn = charset + special tokens (CTC: 4, CE: 2-3) -> luôn > |charset|
    assert logits.shape[2] > len(CHARSET_VI)
    assert logits.shape[1] >= 1


@pytest.mark.parametrize("name", NAMES)
def test_loss_non_negative(name, models, images):
    model = models[name]
    with torch.no_grad():
        logits, loss, numel = model.forward_logits_loss(images, LABELS)
    assert torch.isfinite(loss), f"{name}: loss không hữu hạn"
    assert loss.item() >= 0
    assert int(numel) > 0
    assert logits.shape[0] == 2


@pytest.mark.parametrize("name", NAMES)
def test_decode_returns_labels_in_charset(name, models, images):
    model = models[name]
    with torch.no_grad():
        logits = model(images)
        preds, probs = model.tokenizer.decode(logits.softmax(-1))
    assert isinstance(preds, list) and len(preds) == 2
    for pred in preds:
        assert isinstance(pred, str)
        assert all(ch in CHARSET_VI for ch in pred), \
            f"{name}: ký tự ngoài charset: {sorted(set(pred) - set(CHARSET_VI))}"
    # decode trả (preds, probs): probs là list tensor xác suất từng ký tự
    assert all(0.0 <= float(p.prod()) <= 1.0 for p in probs)


def test_decode_length_bounded_by_timesteps(models, images):
    """Decode không vượt số timestep (N,L) của logits — tránh loop/không tương ứng."""
    for name, model in models.items():
        with torch.no_grad():
            logits = model(images)
            preds, _ = model.tokenizer.decode(logits.softmax(-1))
        assert max(len(p) for p in preds) <= int(logits.shape[1]), \
            f"{name}: decode ({max(len(p) for p in preds)}) > L ({logits.shape[1]})"
