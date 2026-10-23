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


# ------------------------------------------------------------------------
# Độ phủ bảng chữ tiếng Việt (kiểm theo Unicode, không theo danh sách tự gõ)
# ------------------------------------------------------------------------

#: Khối "Latin Extended Additional" U+1EA0..U+1EF9 = toàn bộ chữ Việt có thanh dựng sẵn.
_VN_BLOCK = [chr(cp) for cp in range(0x1EA0, 0x1EFA)]
#: chữ Việt dựng sẵn nằm ở Latin-1 / Latin Extended-A / B (ngoài khối trên)
_VN_LATIN = "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯư"


def test_charset_covers_whole_vietnamese_unicode_block():
    missing = [c for c in _VN_BLOCK + list(_VN_LATIN) if c not in CHARSET_VI]
    assert not missing, f"thiếu {missing}"


def test_charset_has_exactly_134_vietnamese_letters_and_no_combining_marks():
    vn = set(_VN_BLOCK) | set(_VN_LATIN)
    assert len(vn) == 134                                     # 67 thường + 67 hoa
    assert sum(1 for c in CHARSET_VI if c in vn) == 134
    assert not any(unicodedata.combining(c) for c in CHARSET_VI)


@pytest.mark.parametrize("label", [
    # nhãn thật lấy từ tập VN (reports/crnn_vn.csv)
    "Luân Hồi", "LINH MỘC Y ĐƯỜNG", "NÔNG TRẠI CẨM THẠCH",
    "HP Food Đặc sản Hải Phòng", "1P's Bakery & Cafe To-Go", "LJ LEJET CAFÉ BISTRO",
])
def test_real_vn_labels_fully_supported_after_nfc(label):
    nfd = unicodedata.normalize("NFD", label)
    if nfd != label:                             # nhãn có dấu
        assert not contains_only_supported(nfd)  # dấu rời (NFD) không phải token
    assert contains_only_supported(to_nfc(nfd))  # → mọi khâu phải NFC trước
    assert filter_supported(to_nfc(nfd)) == label
