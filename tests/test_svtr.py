"""Test cho nhánh 09 — SVTR-CTC tự viết (`trademark_ocr.svtr_ctc`)."""

from __future__ import annotations

import hydra
import pytest
import torch

from trademark_ocr.charset import CHARSET_VI
from trademark_ocr.svtr_ctc import SVTRCTC, SVTRNet
from trademark_ocr.train_utils import compose_config

LABELS = ["ĐÔNG Á", "HANOI VIET", "A", "TRADEMARK SAMPLE"]


@pytest.fixture(scope="module")
def cfg():
    return compose_config("tm_svtrctc", ["model.batch_size=2"])


@pytest.fixture(scope="module")
def model(cfg):
    m = hydra.utils.instantiate(cfg.model)
    m.eval()
    return m


@pytest.fixture(scope="module")
def images():
    torch.manual_seed(1111)
    return torch.rand(2, 3, 32, 128)


def test_is_ctc_system(model):
    from strhub.models.base import CTCSystem
    assert isinstance(model, CTCSystem)
    assert model.tokenizer.blank_id == model.blank_id
    assert model.max_label_length == 25


def test_forward_logits_shape(model, images):
    with torch.no_grad():
        logits = model(images)
    assert logits.ndim == 3
    assert logits.shape[0] == 2
    assert logits.shape[2] > len(CHARSET_VI)
    assert logits.shape[1] == model.seq_len == 64     # 4x16 timestep


def test_seq_len_covers_max_label_length(model):
    """Điều kiện sống còn của CTC: L >= max_label_length."""
    assert model.seq_len >= model.max_label_length == 25


def test_loss_non_negative_and_finite(model, images):
    model.train()
    with torch.no_grad():
        logits, loss, numel = model.forward_logits_loss(images, LABELS[:2])
    assert torch.isfinite(loss) and loss.item() >= 0
    assert int(numel) == 2                            # CTCSystem trả N
    assert logits.shape[0] == 2
    model.eval()


def test_decode_chars_in_charset(model, images):
    with torch.no_grad():
        preds, _ = model.tokenizer.decode(model(images).softmax(-1))
    assert len(preds) == 2
    for p in preds:
        assert all(ch in CHARSET_VI for ch in p)


def test_param_count_reasonable(model):
    n = sum(p.numel() for p in model.parameters())
    assert 0.5e6 < n < 20e6, f"SVTR-Lite nên ở khoảng ~1-5M tham số, got {n}"


def test_can_overfit_tiny_batch():
    """SVTR-CTC phải HỌC được: loss giảm rõ rệt khi overfit vài mẫu (kiểm tra wiring/optim)."""
    from strhub.data.utils import CTCTokenizer

    torch.manual_seed(0)
    charset = "abcdefghijklmnopqrstuvwxyz"
    tok = CTCTokenizer(charset)                # 0 = blank, 1.. = ký tự
    m = SVTRNet(img_size=(32, 128), num_classes=len(charset) + 1,
                drop_rate=0.0, drop_path_rate=0.0)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    images = torch.rand(2, 3, 32, 128)
    labels = ["abc", "xy"]
    targets = tok.encode(labels, torch.device("cpu"))
    losses = []
    for _ in range(40):
        opt.zero_grad()
        logits = m(images)
        lp = logits.log_softmax(-1).transpose(0, 1)
        T, N, _ = lp.shape
        loss = torch.nn.functional.ctc_loss(
            lp, targets, torch.full((N,), T, dtype=torch.long),
            torch.tensor([len(x) for x in labels]), blank=tok.blank_id, zero_infinity=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.7, f"không giảm loss: {losses[0]:.2f} -> {losses[-1]:.2f}"


def test_training_step_defined_on_class():
    """CTCSystem KHÔNG có training_step -> SVTRCTC phải tự khai báo (nếu thiếu, PL báo
    'No `training_step()` method defined' và train đổ)."""
    assert "training_step" in SVTRCTC.__dict__


def test_seq_len_satisfies_ctc_capacity(model):
    """CTC cần L >= max_target_length (len nhãn + có thể khoảng trắng bị CTC gộp)."""
    assert model.seq_len == 64 >= model.max_label_length == 25
