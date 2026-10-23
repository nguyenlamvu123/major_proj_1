"""Test offline cho ``trademark_ocr.pretrained`` — fine-tune từ trọng số pretrained (REVIEW.md §4.6).

Không tải mạng: "trọng số pretrained" được giả lập bằng một PARSeq nhỏ dựng với charset
36 ký tự (như bản phát hành của parseq) rồi lưu ra file.
"""

from __future__ import annotations

import hydra
import pytest
import torch

from trademark_ocr import pretrained as pt
from trademark_ocr.charset import CHARSET_VI
from trademark_ocr.train_utils import compose_config

PARSEQ = "strhub.models.parseq.system.PARSeq"
#: PARSeq thu nhỏ cho test nhanh (kiến trúc giữ nguyên, chỉ bớt chiều/độ sâu)
SMALL = ["model.embed_dim=48", "model.enc_num_heads=2", "model.dec_num_heads=2",
         "model.enc_depth=1", "model.batch_size=2"]


def _kwargs(charset: str) -> dict:
    cfg = compose_config("tm_parseq", SMALL)
    kw = {k: v for k, v in hydra.utils.instantiate(cfg.model, _partial_=True).keywords.items()}
    kw.update(charset_train=charset, charset_test=charset)
    return kw


@pytest.fixture(scope="module")
def english_weights(tmp_path_factory):
    """State dict của ``PARSeq.model`` (đúng dạng file phát hành của parseq), charset 36."""
    torch.manual_seed(0)
    src = pt._import_class(PARSEQ)(**_kwargs("0123456789abcdefghijklmnopqrstuvwxyz"))
    path = tmp_path_factory.mktemp("w") / "parseq_en.pt"
    torch.save(src.model.state_dict(), path)
    return path, src.model.state_dict()


def test_strict_load_fails_for_vietnamese_charset(english_weights):
    # ROOT CAUSE:
    #
    # `pretrained=` của parseq train.py gọi load_state_dict STRICT. Charset VN (217 ký tự)
    # làm head/text_embed khác kích thước bản pretrained (36/94 ký tự) → không khởi tạo được,
    # nên mọi model chỉ train từ đầu trên vài chục nghìn ảnh.
    #
    # Đã sửa bằng pretrained.finetune: nạp các tensor cùng tên + cùng kích thước, bỏ phần lệch.
    _path, state = english_weights
    model = pt._import_class(PARSEQ)(**_kwargs(CHARSET_VI))
    with pytest.raises(RuntimeError, match="size mismatch"):
        model.model.load_state_dict(state)


def test_load_matching_copies_shared_tensors_and_skips_charset_dependent(english_weights):
    _path, state = english_weights
    model = pt._import_class(PARSEQ)(**_kwargs(CHARSET_VI))
    report = pt.load_matching(model.model, state)
    assert set(report["skipped_shape"]) == {"head.weight", "head.bias", "text_embed.embedding.weight"}
    assert not report["missing"] and not report["unexpected"]
    assert len(report["loaded"]) == len(state) - 3
    enc = "encoder.blocks.0.attn.qkv.weight"
    assert torch.equal(model.model.state_dict()[enc], state[enc])


def test_finetune_factory_from_file_targets_inner_model(english_weights):
    path, state = english_weights
    model = pt.finetune(target=PARSEQ, init_from=str(path), **_kwargs(CHARSET_VI))
    assert type(model).__name__ == "PARSeq"
    assert model.hparams.charset_train == CHARSET_VI
    assert "init_from" not in model.hparams and "target" not in model.hparams
    enc = "encoder.blocks.0.mlp.fc1.weight"
    assert torch.equal(model.model.state_dict()[enc], state[enc])


def test_finetune_accepts_lightning_checkpoint_state_dict(english_weights, tmp_path):
    """Ckpt Lightning (khoá ``model.*`` cấp system) → tự chọn đích là cả system."""
    _path, state = english_weights
    ckpt = tmp_path / "last.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in state.items()}}, ckpt)
    model = pt.finetune(target=PARSEQ, init_from=str(ckpt), **_kwargs(CHARSET_VI))
    enc = "encoder.blocks.0.attn.proj.weight"
    assert torch.equal(model.model.state_dict()[enc], state[enc])


def test_finetune_rejects_weights_that_match_nothing(tmp_path):
    bogus = tmp_path / "bogus.pt"
    torch.save({"unrelated.weight": torch.zeros(3)}, bogus)
    with pytest.raises(ValueError, match="không khớp"):
        pt.finetune(target=PARSEQ, init_from=str(bogus), **_kwargs(CHARSET_VI))


def test_tm_parseq_ft_config_instantiates_without_download():
    cfg = compose_config("tm_parseq_ft", SMALL + ["model.init_from=null"])
    assert cfg.model._target_ == "trademark_ocr.pretrained.finetune"
    assert cfg.model.target == PARSEQ
    assert cfg.pretrained is None                  # nhánh strict của parseq train.py không chạy
    model = hydra.utils.instantiate(cfg.model)
    assert type(model).__name__ == "PARSeq"


def test_evaluate_resolves_finetune_runs_to_the_real_class(tmp_path):
    import yaml
    from trademark_ocr.evaluate import resolve_model_class

    run = tmp_path / "runs" / "parseq_ft" / "ts"
    (run / "config").mkdir(parents=True)
    (run / "config" / "config.yaml").write_text(yaml.safe_dump(
        {"model": {"_target_": pt.FINETUNE_TARGET, "target": PARSEQ}}), encoding="utf-8")
    assert resolve_model_class(run / "checkpoints" / "last.ckpt").__name__ == "PARSeq"
