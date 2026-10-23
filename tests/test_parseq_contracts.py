"""Hợp đồng giữa code của đồ án và parseq (``third_party/parseq``) — các phần đã chạy đúng.

DESIGN.md §4 liệt kê các điểm tiếp xúc với parseq và các bẫy tiếng Việt. Test ở đây khoá
lại để một lần sửa config / nâng parseq không âm thầm phá chúng:

- mọi ``configs/hydra/tm_*.yaml`` nhất quán giữa khối ``model`` và ``data``;
- vì sao ``normalize_unicode`` / ``remove_whitespace`` phải tắt (chứng minh bằng parseq thật);
- CRNN có đủ số bước thời gian CTC theo chiều ngang;
- shim NumPy-2 làm RandAugment của parseq chạy được.
"""

from __future__ import annotations

import hydra
import pytest
import torch
from PIL import Image

from trademark_ocr import build_lmdb
from trademark_ocr.charset import CHARSET_VI
from trademark_ocr.train_utils import available_configs, compose_config

VN_LABEL = "ĐÔNG Á"


@pytest.mark.parametrize("name", available_configs())
def test_config_model_and_data_blocks_agree(name):
    cfg = compose_config(name)
    assert cfg.model.charset_train == cfg.model.charset_test == CHARSET_VI
    assert cfg.data.charset_train == cfg.model.charset_train
    assert cfg.data.charset_test == cfg.model.charset_test
    assert cfg.data.max_label_length == cfg.model.max_label_length
    assert list(cfg.data.img_size) == list(cfg.model.img_size)
    assert cfg.data.batch_size == cfg.model.batch_size
    # hai bẫy tiếng Việt (DESIGN §4.2) phải tắt ở MỌI config, không chỉ tm_crnn
    assert cfg.data.normalize_unicode is False
    assert cfg.data.remove_whitespace is False
    # layout LMDB do build_lmdb sinh ra
    assert str(cfg.data.root_dir).endswith("/lmdb")
    assert build_lmdb.lmdb_path("R", "train", cfg.data.train_dir).as_posix() == "R/lmdb/train/trademark"


def _one_label_lmdb(tmp_path, label):
    img = tmp_path / "img.png"
    Image.new("RGB", (64, 32), "white").save(img)
    out = tmp_path / "lmdb"
    build_lmdb.create_lmdb([{"image_path": str(img), "word_mark": label}], out,
                           data_root=tmp_path)
    return out


@pytest.mark.parametrize("remove_whitespace,normalize_unicode,want", [
    (False, False, "ĐÔNG Á"),   # config của đồ án: giữ nguyên
    (True, False, "ĐÔNGÁ"),     # mặc định parseq: mất khoảng trắng
    (False, True, "ONG A"),     # NFKD → ASCII: mất dấu, 'Đ' (không phân rã) bị xoá hẳn
])
def test_parseq_preprocessing_flags_explain_config_choice(tmp_path, remove_whitespace,
                                                          normalize_unicode, want):
    from strhub.data.dataset import LmdbDataset

    ds = LmdbDataset(str(_one_label_lmdb(tmp_path, VN_LABEL)), CHARSET_VI, max_label_len=25,
                     remove_whitespace=remove_whitespace, normalize_unicode=normalize_unicode)
    assert ds.labels == [want]


def test_charset_adapter_keeps_case_for_mixed_case_charset():
    """CHARSET_VI có cả hoa lẫn thường → parseq không tự lower/upper nhãn."""
    from strhub.data.utils import CharsetAdapter

    adapter = CharsetAdapter(CHARSET_VI)
    assert adapter("Luân Hồi") == "Luân Hồi"
    assert adapter("THỦY SÂM") == "THỦY SÂM"


def test_crnn_has_enough_horizontal_ctc_steps():
    """CTC chỉ phát được nhãn dài L ký tự khi có ≥ L bước thời gian trái→phải."""
    cfg = compose_config("tm_crnn", ["model.batch_size=1"])
    model = hydra.utils.instantiate(cfg.model).eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 3, *cfg.model.img_size))
    assert logits.shape[1] >= cfg.model.max_label_length


def test_numpy2_shim_lets_parseq_randaugment_run():
    pytest.importorskip("imgaug")
    from trademark_ocr._numpy2_compat import apply_numpy2_compat
    from strhub.data.module import SceneTextDataModule

    apply_numpy2_compat()
    transform = SceneTextDataModule.get_transform((32, 128), augment=True)
    torch.manual_seed(0)
    out = transform(Image.new("RGB", (300, 120), (200, 30, 30)))
    assert out.shape == (3, 32, 128)
