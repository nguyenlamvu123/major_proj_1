"""Test offline cho nhánh 07 — `trademark_ocr.train_utils` (compose config + shim)."""

from __future__ import annotations

import numpy as np

from trademark_ocr import _numpy2_compat
from trademark_ocr.charset import CHARSET_VI
from trademark_ocr.train_utils import (
    available_configs,
    compose_config,
    train_command,
)


def test_numpy2_shim_idempotent():
    assert _numpy2_compat.apply_numpy2_compat() in (True, False)   # lần 1
    assert _numpy2_compat.apply_numpy2_compat() is False           # đã patch
    assert hasattr(np, "sctypes")


def test_configs_available():
    cfgs = available_configs()
    assert {"tm_crnn", "tm_trba", "tm_parseq"} <= set(cfgs)


def test_compose_tm_crnn_uses_vietnamese_charset_and_no_nfkD():
    cfg = compose_config("tm_crnn")
    assert cfg.model.name == "crnn"
    assert cfg.model._target_.startswith("strhub.models.")
    # charset tiếng Việt (217 ký tự) — đúng như nhánh 01
    assert cfg.model.charset_train == CHARSET_VI
    assert cfg.model.charset_test == CHARSET_VI
    # BẮT BUỘC: nếu sai ở đây thì mọi nhãn có dấu sẽ hỏng khi train
    assert cfg.data.normalize_unicode is False
    assert cfg.data.remove_whitespace is False
    assert len(cfg.model.charset_train) == 217


def test_compose_lmdb_layout():
    cfg = compose_config("tm_crnn")
    assert str(cfg.data.root_dir).endswith("/lmdb")
    assert cfg.data.train_dir == "trademark"       # == lmdb/train/trademark
    assert cfg.model.img_size == [32, 128]
    assert cfg.model.max_label_length == 25


def test_trainer_smoke_knobs_present():
    """Các key mà scripts/05_train.sh ghi đè phải KHÔNG cần '+'."""
    cfg = compose_config("tm_crnn")
    for key in ("limit_train_batches", "limit_val_batches", "max_steps",
                "max_epochs", "accelerator", "log_every_n_steps"):
        assert key in cfg.trainer, key
    assert cfg.trainer.log_every_n_steps == 1      # loss hiện trong smoke test 2 batch
    assert cfg.trainer.val_check_interval == 1.0   # dữ liệu nhỏ -> validate mỗi epoch
    assert cfg.trainer.accelerator == "gpu"        # Colab; local đổi qua --accelerator cpu


def test_compose_all_configs():
    for name in available_configs():
        cfg = compose_config(name)
        assert cfg.model.charset_train == CHARSET_VI
        assert cfg.data.normalize_unicode is False
        assert cfg.trainer.get("limit_train_batches") is not None


def test_train_command_points_to_parseq_entry():
    cmd = train_command("tm_crnn", ["trainer.accelerator=cpu"])
    assert "--config-dir configs/hydra" in cmd
    assert "--config-name tm_crnn" in cmd
    assert "trainer.accelerator=cpu" in cmd
    # không sửa parseq -> chỉ dùng entrypoint của nó
    assert "third_party/parseq/train.py" in cmd
