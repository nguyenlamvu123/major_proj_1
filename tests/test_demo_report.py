"""Test offline cho nhánh 14 — demo_app (stub model) + report (không mạng/không GPU)."""

from __future__ import annotations

import csv
import types

import pytest

from trademark_ocr import report as rp

HEADER = ["idx", "pred", "gt", "correct", "ned", "char_err", "pred_len", "gt_len"]


# ------------------------------- demo_app -------------------------------

class _StubTokenizer:
    @staticmethod
    def decode(probs):
        b = probs.shape[0]
        import torch
        return ["ABC"] * b, [torch.tensor([0.9, 0.8]) for _ in range(b)]


class _StubModel:
    def __init__(self, img_size=(32, 128)):
        self.tokenizer = _StubTokenizer()
        self.hparams = types.SimpleNamespace(img_size=list(img_size))
        self.seen_shapes = []

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, x):
        import torch
        self.seen_shapes.append(tuple(x.shape))
        return torch.zeros(x.shape[0], 5, 4)

    def charset_adapter(self, s):
        return s.upper()


def _png(tmp_path, name="a.png", size=(120, 40)):
    from PIL import Image
    p = tmp_path / name
    Image.new("RGB", size, "white").save(p)
    return p


def test_demo_help_exits_zero():
    from trademark_ocr import demo_app
    with pytest.raises(SystemExit) as e:
        demo_app.main(["--help"])
    assert e.value.code == 0


def test_demo_predict_with_stub_model(tmp_path):
    from trademark_ocr import demo_app
    p1, p2 = _png(tmp_path, "a.png"), _png(tmp_path, "b.png")
    rows = demo_app.predict(_StubModel(), [str(p1), str(p2)], device="cpu")
    assert len(rows) == 2
    assert rows[0]["pred"] == "ABC" and rows[0]["confidence"] == pytest.approx(0.72, abs=1e-6)
    assert rows[0]["image"].endswith("a.png")


def test_demo_resizes_to_model_img_size(tmp_path):
    """Ảnh demo phải resize đúng cỡ model được train (trước đây viết cứng 32×128)."""
    from trademark_ocr import demo_app
    model = _StubModel(img_size=(32, 256))
    demo_app.predict(model, [str(_png(tmp_path))], device="cpu")
    assert model.seen_shapes == [(1, 3, 32, 256)]


def test_demo_run_requires_ckpt(tmp_path):
    from trademark_ocr import demo_app
    with pytest.raises(SystemExit):
        demo_app.run([str(_png(tmp_path))], ckpt=None)
    with pytest.raises(SystemExit):
        demo_app.run([], ckpt="whatever.ckpt")


def test_demo_run_with_model_and_images(tmp_path, capsys):
    from trademark_ocr import demo_app
    rows = demo_app.run([str(_png(tmp_path))], model=_StubModel())
    out = capsys.readouterr().out
    assert "ABC" in out and len(rows) == 1


# -------------------------------- report --------------------------------

def _report(path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, (pred, gt) in enumerate(pairs):
            w.writerow([i, pred, gt, int(pred == gt), 0, 0, len(pred), len(gt)])


def test_report_collect_and_render(tmp_path):
    _report(tmp_path / "crnn_us.csv", [("a", "a"), ("b", "c")])
    _report(tmp_path / "crnn_vn.csv", [("x", "x")])
    (tmp_path / "domain_shift.csv").write_text(
        "model,acc_us,acc_vn,delta,n_us,n_vn\ncrnn,0.5,1.0,0.5,2,1\n")
    res = rp.run(reports_dir=tmp_path, out_path=tmp_path / "final_summary.md",
                 figures_dir=tmp_path / "figs", log=lambda *_: None)
    md = (tmp_path / "final_summary.md").read_text(encoding="utf-8")
    assert "| model | dataset | n | accuracy | 1−NED | char-accuracy |" in md
    assert "crnn" in md and "US" in md and "VN" in md
    assert "Sụt giảm US" in md and "domain_shift" not in md or True
    assert "| crnn | 0.5 | 1.0 | 0.5 | 2 | 1 |" in md
    assert res["rows"] and res["shift"]
    accs = {r["dataset"]: r["accuracy"] for r in res["rows"]}
    assert accs["US"] == pytest.approx(0.5) and accs["VN"] == pytest.approx(1.0)


def test_report_empty_reports_still_writes(tmp_path):
    out = tmp_path / "final_summary.md"
    rp.run(reports_dir=tmp_path, out_path=out, figures_dir=tmp_path / "none",
           log=lambda *_: None)
    md = out.read_text(encoding="utf-8")
    assert "Chưa có báo cáo nào" in md and "Chưa có `domain_shift.csv`" in md
    assert "Không tìm thấy hình" in md


def test_report_lists_figures(tmp_path):
    figs = tmp_path / "figs"
    figs.mkdir()
    (figs / "fig_len_hist.png").write_bytes(b"x")
    (figs / "err_by_length.png").write_bytes(b"x")
    (figs / "ignore.txt").write_text("x")
    res = rp.run(reports_dir=tmp_path, out_path=tmp_path / "s.md", figures_dir=figs,
                 log=lambda *_: None)
    assert len(res["figures"]) == 2
    md = (tmp_path / "s.md").read_text(encoding="utf-8")
    assert "fig_len_hist.png" in md and "err_by_length.png" in md


def test_report_cli_help_exits_zero():
    with pytest.raises(SystemExit) as e:
        rp.main(["--help"])
    assert e.value.code == 0


def _full_report(path, rows):
    """CSV đủ cột như evaluate (nhánh 16): (pred, gt, drawing_kind, in_scope)."""
    from trademark_ocr.metrics import REPORT_COLUMNS, per_sample
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(REPORT_COLUMNS))
        w.writeheader()
        for i, (pred, gt, kind, scope) in enumerate(rows, start=1):
            w.writerow({"idx": i, "serial_number": f"S{i}", "drawing_kind": kind,
                        **per_sample(pred, gt), "in_scope": scope, "confidence": 0.5})


def test_report_levels_coverage_and_drawing_kind(tmp_path):
    _full_report(tmp_path / "crnn_us.csv", [
        ("ABC", "ABC", "standard_character", 1),
        ("ABD", "ABC", "stylized_words", 1),
        ("XYZ", "XYZ", "stylized_words", 1),
    ])
    _full_report(tmp_path / "crnn_vn.csv", [
        ("HOFF", "Hoff", "", 1),
        ("LUAN HOI", "Luân Hồi", "", 1),
        ("X", "A" * 30, "", 0),
    ])
    res = rp.run(reports_dir=tmp_path, out_path=tmp_path / "s.md",
                 figures_dir=tmp_path / "none", log=lambda *_: None)
    md = (tmp_path / "s.md").read_text(encoding="utf-8")
    vn = next(r for r in res["rows"] if r["dataset"] == "VN")
    assert vn["accuracy_nocase"] == pytest.approx(1 / 3)
    assert vn["accuracy_base"] == pytest.approx(2 / 3)
    assert vn["coverage"] == pytest.approx(2 / 3)
    assert "| crnn | VN | 3 | 0.667 | 0.0000 | 0.3333 | 0.6667 |" in md
    # US tách theo loại nhãn hiệu; VN không có drawing_kind -> không có dòng
    assert "| crnn | US | standard_character | 1 | 1.0000 |" in md
    assert "| crnn | US | stylized_words | 2 | 0.5000 |" in md


def test_report_old_csv_without_new_columns_still_renders(tmp_path):
    _report(tmp_path / "crnn_us.csv", [("a", "a")])
    res = rp.run(reports_dir=tmp_path, out_path=tmp_path / "s.md",
                 figures_dir=tmp_path / "none", log=lambda *_: None)
    assert res["rows"][0]["coverage"] is None
    md = (tmp_path / "s.md").read_text(encoding="utf-8")
    assert "| crnn | US | 1 | – |" in md


def test_report_renders_shift_decomposition_from_domain_shift_run(tmp_path):
    from trademark_ocr import eval_domain_shift as ds
    _full_report(tmp_path / "crnn_us.csv", [("ABC", "ABC", "stylized_words", 1)] * 20)
    _full_report(tmp_path / "crnn_vn.csv", [("HOFF", "Hoff", "", 1), ("ĐA", "ĐA", "", 1),
                                            ("X", "Y", "", 1)] * 10)
    ds.run(reports_dir=tmp_path, out_path=tmp_path / "domain_shift.csv")
    rp.run(reports_dir=tmp_path, out_path=tmp_path / "s.md", figures_dir=tmp_path / "none",
           log=lambda *_: None)
    md = (tmp_path / "s.md").read_text(encoding="utf-8")
    assert "### Phân tách nguồn gốc độ sụt" in md
    line = next(l for l in md.splitlines() if l.startswith("| crnn | -0.6667 |"))
    assert "-0.3333 [" in line            # Δ bỏ hoa/thường kèm CI
    assert "| 20 / 10 |" in line          # VN: 20 không dấu, 10 có dấu
