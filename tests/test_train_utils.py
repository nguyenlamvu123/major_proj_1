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


# ------------- train theo ngân sách giờ (nhánh 16.8, scripts/09_full_pipeline_cpu.sh) -------------

def _tiny_lmdb(root, n_train=12, n_val=4):
    from PIL import Image
    from trademark_ocr.build_lmdb import create_lmdb

    def rows(n, tag):
        out = []
        for i in range(n):
            p = root / "img" / f"{tag}{i}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (120, 40), (255, 255, 255)).save(p)
            out.append({"image_path": str(p), "word_mark": f"AB{i}"})
        return out
    create_lmdb(rows(n_train, "tr"), root / "lmdb" / "train" / "trademark", data_root=root)
    create_lmdb(rows(n_val, "va"), root / "lmdb" / "val", data_root=root)


SMALL_SVTR = ["model.embed_dim=32", "model.num_heads=2", "model.batch_size=4",
              "data.num_workers=0", "data.augment=false", "trainer.accelerator=cpu"]


def test_benchmark_measures_real_steps_and_skips_pretrained_download(tmp_path, monkeypatch):
    import strhub.models.utils as su
    from trademark_ocr.train_utils import benchmark

    _tiny_lmdb(tmp_path)
    # đo tốc độ không được tải trọng số pretrained (đêm chạy tự động, mạng có thể chập chờn)
    monkeypatch.setattr(su, "get_pretrained_weights",
                        lambda *_: (_ for _ in ()).throw(AssertionError("không được tải mạng")))
    cfg = compose_config("tm_crnn_ft", [f"data.root_dir={tmp_path / 'lmdb'}", "model.batch_size=4",
                                        "data.num_workers=0", "data.augment=false",
                                        "trainer.accelerator=cpu"])
    assert cfg.model.init_from == "crnn"
    r = benchmark(cfg, warmup=1, num_batches=2, val_batches=1)
    assert (r["train_samples"], r["val_samples"], r["batch_size"]) == (12, 4, 4)
    assert r["sec_per_train_batch"] > 0 and r["sec_per_val_batch"] > 0


def _fit_tiny(tmp_path, *, save_top_k):
    """Fit THẬT 2 epoch với ModelCheckpoint như parseq train.py (monitor val_accuracy, save_last)."""
    import hydra
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

    _tiny_lmdb(tmp_path)
    cfg = compose_config("tm_svtrctc", [f"data.root_dir={tmp_path / 'lmdb'}", *SMALL_SVTR])
    model = hydra.utils.instantiate(cfg.model)
    dm = hydra.utils.instantiate(cfg.data)
    ckpt_dir = tmp_path / "run" / "checkpoints"
    pl.Trainer(max_epochs=2, limit_train_batches=2, limit_val_batches=1, max_time="00:00:10:00",
               accelerator="cpu", logger=False, enable_progress_bar=False, enable_model_summary=False,
               callbacks=[ModelCheckpoint(dirpath=ckpt_dir, monitor="val_accuracy", mode="max",
                                          save_top_k=save_top_k, save_last=True)]).fit(model, datamodule=dm)
    return ckpt_dir / "last.ckpt"


def test_ckpt_progress_reads_epoch_and_timer_from_real_fit(tmp_path):
    from trademark_ocr.train_utils import ckpt_progress

    # save_top_k=3 như parseq: 2 epoch đầu luôn vào top-3 -> last.ckpt đi theo từng epoch
    info = ckpt_progress(_fit_tiny(tmp_path, save_top_k=3))
    assert info["epoch"] == 1
    assert info["global_step"] == 4
    assert info["train_seconds"] > 0          # Timer có trạng thái -> train tiếp cộng dồn được


def test_last_ckpt_does_not_advance_without_a_topk_save(tmp_path):
    # ROOT CAUSE:
    #
    # Pipeline định đánh DONE khi ckpt["epoch"] + 1 >= max_epochs. Với pytorch-lightning 2.6,
    # ModelCheckpoint.on_validation_end chỉ ghi last.ckpt khi lượt đó CÓ lưu checkpoint top-k
    # ("Only save last checkpoint if a checkpoint was actually saved in this step"). parseq
    # train.py dùng save_top_k=3 theo val_accuracy: epoch không lọt top-3 -> last.ckpt đứng yên ở
    # epoch cũ -> model đã train đủ bị coi là dở và bị train lại mãi.
    #
    # Đã sửa: scripts/09_full_pipeline_cpu.sh quyết định DONE theo thời gian chạy so với trần
    # max_time (chạm trần = bị cắt), không theo epoch trong last.ckpt.
    from trademark_ocr.train_utils import ckpt_progress

    # save_top_k=1, val_accuracy của model chưa học = 0 ở cả 2 epoch -> epoch 2 không "tốt hơn"
    info = ckpt_progress(_fit_tiny(tmp_path, save_top_k=1))
    assert info["epoch"] == 0                 # dù fit đã chạy đủ 2 epoch
    assert info["global_step"] == 2


def test_max_time_override_parses_as_string():
    cfg = compose_config("tm_crnn", ["+trainer.max_time='00:03:27:00'"])
    assert cfg.trainer.max_time == "00:03:27:00"
