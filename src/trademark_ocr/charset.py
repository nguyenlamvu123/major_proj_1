"""Bảng ký tự (charset) cho nhận diện chữ trong nhãn hiệu — Unicode tiếng Việt (NFC).

Vì sao cần module này
---------------------
`baudm/parseq` mặc định dùng charset Latin 36/62/94 ký tự. Nhãn hiệu Việt Nam
có dấu (ă â đ ê ô ơ ư + 5 thanh) nên phải định nghĩa charset riêng:

- Dạng chuẩn hoá: **NFC** (ký tự tổ hợp sẵn, ví dụ 'ắ' = U+1EAF) để 1 glyph = 1 token.
  ⚠️ parseq `LmdbDataset` mặc định NFKD→ASCII (làm mất dấu) nên config huấn luyện
  **bắt buộc** đặt `data.normalize_unicode: false` (xem DESIGN.md §4.2).
- Thứ tự ký tự quyết định token id của mô hình → builder phải **deterministic**.
- Không trùng ký tự (nếu trùng, `len(charset)` ≠ số token thật → sai số class đầu ra).

Sử dụng
-------
    from trademark_ocr.charset import CHARSET_VI, to_nfc, filter_supported

    CHARSET_VI                      # chuỗi charset đầy đủ
    to_nfc("a\u0301")               # -> 'á'
    filter_supported("ABC®", CHARSET_VI)   # -> 'ABC'

CLI:
    python -m trademark_ocr.charset --check
    python -m trademark_ocr.charset --emit-yaml        # -> configs/hydra/charset/charset_vi.yaml
    python -m trademark_ocr.charset --emit-txt         # -> configs/charset_vi.txt
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# 1. Thành phần charset
# --------------------------------------------------------------------------

# Nguyên âm tiếng Việt có thể mang thanh (dạng NFC sau khi tổ hợp)
_VOWELS_LOWER = ["a", "ă", "â", "e", "ê", "i", "o", "ô", "ơ", "u", "ư", "y"]

# 5 thanh điệu (dấu tổ hợp trong Unicode combining marks)
_TONES = [
    "",              # không dấu
    "\u0301",        # sắc  ́
    "\u0300",        # huyền ̀
    "\u0309",        # hỏi  ̉
    "\u0303",        # ngã  ̃
    "\u0323",        # nặng  ̣
]

# Ký tự đặc thù tiếng Việt không sinh từ (nguyên âm + thanh)
_SPECIAL_LOWER = ["đ"]

_DIGITS = "0123456789"
_ASCII_LOWER = "abcdefghijklmnopqrstuvwxyz"
_ASCII_UPPER = "abcdefghijklmnopqrstuvwxyz".upper()

# Dấu câu/space hay gặp trong nhãn hiệu (curated — cố ý KHÔNG gồm ™ © ® vì
# glyph nhỏ, khó đọc, dễ gây nhiễu; xem DESIGN.md).
_PUNCT = " .,-'’&/()!+*@#%:;=?_"


def _diacritic_variants(base: str) -> list[str]:
    """Sinh các biến thể có thanh của 1 nguyên âm (dạng NFC, đã tổ hợp)."""
    out = []
    for tone in _TONES:
        ch = unicodedata.normalize("NFC", base + tone)
        # Chỉ nhận biến thể tổ hợp được thành 1 codepoint (tránh 'a' + dấu rời)
        if len(ch) == 1 or tone == "":
            out.append(ch)
    return out


def build_charset() -> str:
    """Trả về charset đầy đủ (deterministic, không trùng ký tự).

    Thứ tự: chữ số → Latin thường → Latin hoa → ký tự đặc thù Việt →
    dấu câu/space. Ký tự trùng được loại bỏ, giữ vị trí xuất hiện đầu.
    """
    chars: list[str] = []
    for v in _VOWELS_LOWER:
        chars.extend(_diacritic_variants(v))
    chars.extend(_SPECIAL_LOWER)

    lower = "".join(chars)
    upper = lower.upper()

    ordered = (
        list(_DIGITS)
        + list(_ASCII_LOWER)
        + list(_ASCII_UPPER)
        + list(_SPECIAL_LOWER)
        + list(lower)
        + list(upper)
        + list(_PUNCT)
    )

    seen: set[str] = set()
    dedup: list[str] = []
    for ch in ordered:
        if ch not in seen:
            seen.add(ch)
            dedup.append(ch)
    return "".join(dedup)


#: Charset chuẩn của đồ án (được nhúng vào model + config hydra).
CHARSET_VI: str = build_charset()

#: Số token của CTC tokenizer parseq: len(charset) + 1 (BLANK ở index 0).
NUM_CLASSES_CTC: int = len(CHARSET_VI) + 1


# --------------------------------------------------------------------------
# 2. Chuẩn hoá & lọc
# --------------------------------------------------------------------------

def to_nfc(text: str) -> str:
    """Chuẩn hoá NFC (tổ hợp dấu sẵn) — dùng cho MỌI nhãn trước khi lưu/train."""
    return unicodedata.normalize("NFC", text)


def collapse_ws(text: str) -> str:
    """Gộp khoảng trắng liên tiếp thành 1 space và strip 2 đầu."""
    return " ".join(text.split())


def filter_supported(text: str, charset: str = CHARSET_VI) -> str:
    """Giữ lại các ký tự thuộc charset (chạy sau to_nfc)."""
    allowed = set(charset)
    return "".join(ch for ch in text if ch in allowed)


def contains_only_supported(text: str, charset: str = CHARSET_VI) -> bool:
    return all(ch in set(charset) for ch in text)


# --------------------------------------------------------------------------
# 3. Encoder/decoder đơn giản (CTC style, blank id = 0) — dùng cho test/EDA
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Charset:
    """Encoder/decoder tham chiếu (không thay thế tokenizer của parseq)."""

    chars: str = CHARSET_VI

    def __post_init__(self) -> None:
        assert len(set(self.chars)) == len(self.chars), "charset có ký tự trùng"

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def stoi(self) -> dict[str, int]:
        return {c: i + 1 for i, c in enumerate(self.chars)}  # +1 vì blank = 0

    def encode(self, text: str) -> list[int]:
        table = self.stoi
        return [table[c] for c in text if c in table]

    def decode(self, ids: list[int], collapse: bool = True) -> str:
        """Giải mã best-path CTC: bỏ blank, gộp ký tự lặp liền kề (nếu collapse)."""
        itos = {i + 1: c for i, c in enumerate(self.chars)}
        out: list[str] = []
        prev = None
        for i in ids:
            if i == self.blank_id:
                prev = i
                continue
            if collapse and i == prev:
                prev = i
                continue
            out.append(itos.get(i, "?"))
            prev = i
        return "".join(out)


# --------------------------------------------------------------------------
# 4. Sinh file cấu hình
# --------------------------------------------------------------------------

def emit_hydra_yaml(path: str | Path | None = None, charset: str = CHARSET_VI) -> Path:
    """Sinh config group `charset` cho hydra (charset_train == charset_test).

    Mặc định: ``configs/hydra/charset/charset_vi.yaml``
    (đặt đúng dạng thư mục group để `charset: charset_vi` resolve được,
     xem DESIGN.md §8).
    """
    path = Path(path) if path else REPO_ROOT / "configs" / "hydra" / "charset" / "charset_vi.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": {"charset_train": charset, "charset_test": charset}}
    body = yaml.safe_dump(
        payload,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=10**9,
    )
    header = (
        "# @package _global_\n"
        "# AUTO-GENERATED by `python -m trademark_ocr.charset --emit-yaml`. KHÔNG sửa tay.\n"
        "# charset_train == charset_test: bảng ký tự Unicode tiếng Việt (NFC).\n"
    )
    path.write_text(header + body, encoding="utf-8")
    return path


def emit_txt(path: str | Path | None = None, charset: str = CHARSET_VI) -> Path:
    """Ghi charset dạng text thuần (1 dòng) — cho công cụ khác/AI đọc."""
    path = Path(path) if path else REPO_ROOT / "configs" / "charset_vi.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(charset + "\n", encoding="utf-8")
    return path


def stats(charset: str = CHARSET_VI) -> dict:
    """Thống kê phục vụ báo cáo."""
    vn_letters = set(
        "".join(ch for v in _VOWELS_LOWER for ch in _diacritic_variants(v))
        + "".join(_SPECIAL_LOWER)
    )
    vn_letters |= {c.upper() for c in vn_letters}
    return {
        "n_classes_ctc": len(charset) + 1,
        "n_chars": len(charset),
        "n_vietnamese_letters": sum(1 for c in charset if c in vn_letters),
        "n_ascii_lower": sum(1 for c in charset if c in _ASCII_LOWER),
        "n_ascii_upper": sum(1 for c in charset if c in _ASCII_UPPER),
        "n_digits": sum(1 for c in charset if c in _DIGITS),
        "n_punct": sum(1 for c in charset if c in _PUNCT),
        "has_space": " " in charset,
        "nfc_stable": all(to_nfc(c) == c for c in charset),
    }


# --------------------------------------------------------------------------
# 5. CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bảng ký tự tiếng Việt (NFC) cho trademark OCR")
    ap.add_argument("--print", action="store_true", help="in charset ra stdout")
    ap.add_argument("--check", action="store_true", help="kiểm tra & in thống kê")
    ap.add_argument("--emit-yaml", nargs="?", const="", default=None,
                    help="sinh configs/hydra/charset/charset_vi.yaml (hoặc path chỉ định)")
    ap.add_argument("--emit-txt", nargs="?", const="", default=None,
                    help="sinh configs/charset_vi.txt (hoặc path chỉ định)")
    args = ap.parse_args(argv)

    if args.print:
        print(CHARSET_VI)
    if args.emit_yaml is not None:
        p = emit_hydra_yaml(args.emit_yaml or None)
        print(f"[charset] wrote {p}")
    if args.emit_txt is not None:
        p = emit_txt(args.emit_txt or None)
        print(f"[charset] wrote {p}")
    if args.check or not any([args.print, args.emit_yaml is not None, args.emit_txt is not None]):
        s = stats()
        for k, v in s.items():
            print(f"{k:>22}: {v}")
        assert len(set(CHARSET_VI)) == len(CHARSET_VI), "charset trùng ký tự!"
        assert s["nfc_stable"], "charset chưa NFC!"
        print("charset OK (unique + NFC)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
