"""Smoke tests cho nhánh 00-init: package import được, parseq tìm thấy."""


def test_package_importable():
    import trademark_ocr

    assert trademark_ocr.__version__


def test_parseq_vendored_importable():
    # pythonpath của pytest trỏ third_party/parseq (xem pyproject.toml)
    import strhub  # noqa: F401
    from strhub.data.dataset import LmdbDataset  # noqa: F401
    from strhub.models.crnn.system import CRNN  # noqa: F401
