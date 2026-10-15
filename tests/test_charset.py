"""Tests cho bảng ký tự tiếng Việt (nhánh 01)."""
import unicodedata

import pytest

from trademark_ocr.charset import (
    CHARSET_VI,
    Charset,
    build_charset,
    contains_only_supported,
    emit_hydra_yaml,
    filter_supported,
    stats,
    to_nfc,
)


def test_charset_unique_and_deterministic():
    assert len(set(CHARSET_VI)) == len(CHARSET_VI)
    assert build_charset() == CHARSET_VI  # deterministic


def test_charset_is_nfc():
    for ch in CHARSET_VI:
        assert unicodedata.normalize("NFC", ch) == ch


def test_vietnamese_letters_present():
    for ch in "ăâđêôơưĂÂĐÊÔƠƯáàảãạắằẳẵặấầẩẫậếềểễệốồổỗộớờởỡợứừửữựýỳỷỹỵ":
        assert ch in CHARSET_VI, f"thiếu {ch!r}"


def test_digits_ascii_punct_space():
    for ch in "0123456789abcXYZ.-&'":
        assert ch in CHARSET_VI
    assert " " in CHARSET_VI


def test_to_nfc_composes_decomposed_input():
    decomposed = "a\u0301" * 1 + "e\u0302" + "o\u031b\u0301"  # á ê ớ (dạng rời)
    composed = to_nfc(decomposed)
    assert composed == "áêớ"
    assert contains_only_supported(composed)


def test_filter_supported_drops_symbols():
    assert filter_supported("ABC®™123") == "ABC123"


@pytest.mark.parametrize(
    "text",
    [
        "CÀ PHÊ SỮA ĐÁ 123",
        "NƯỚC MẮM Ớt",
        "Bánh mì Hội An",
        "VIETCOMBANK & TECHCOMB",
    ],
)
def test_encode_decode_roundtrip(text):
    cs = Charset()
    ids = cs.encode(to_nfc(text))
    assert cs.decode(ids) == to_nfc(text)


def test_ctc_decode_collapses_repeats_and_blanks():
    cs = Charset()
    a = cs.stoi["a"]
    b = cs.stoi["b"]
    # a a <blank> a b b  ->  "aab" (CTC best-path)
    ids = [a, a, cs.blank_id, a, b, b]
    assert cs.decode(ids) == "aab"


def test_emit_hydra_yaml(tmp_path):
    import yaml

    out = emit_hydra_yaml(tmp_path / "charset_vi.yaml")
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["model"]["charset_train"] == CHARSET_VI
    assert data["model"]["charset_test"] == CHARSET_VI
    assert "@package _global_" in out.read_text(encoding="utf-8")


def test_stats_sane():
    s = stats()
    assert s["n_chars"] == len(CHARSET_VI)
    assert s["n_classes_ctc"] == len(CHARSET_VI) + 1
    assert s["n_vietnamese_letters"] >= 80  # ~89 chữ có dấu + đ/Đ
    assert s["has_space"] and s["nfc_stable"]
