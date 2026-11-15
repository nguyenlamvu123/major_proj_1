"""Nhánh 14 (phần 1) — demo suy luận ảnh → chữ cho nhãn hiệu.

Nạp checkpoint (parseq hoặc SVTRCTC tự viết — dùng lại ``evaluate.load_model``),
tiền xử lý ảnh giống lúc train (``SceneTextDataModule.get_transform((32,128))``),
rồi in dự đoán. Có thêm chế độ web (Gradio) tuỳ chọn.

CLI
---
    python3 -m trademark_ocr.demo_app --help
    python3 -m trademark_ocr.demo_app --ckpt runs/crnn/.../last.ckpt --image a.png b.jpg
    python3 -m trademark_ocr.demo_app --ckpt <ckpt> --serve            # mở web UI
    python3 -m trademark_ocr.demo_app --ckpt <ckpt> --serve --share    # link công khai

⚠️ Không có ``--ckpt`` thì không thể suy luận (in hướng dẫn). ``--help`` luôn chạy được.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

DEFAULT_IMG_SIZE = (32, 128)


def load_transform(img_size: tuple[int, int] = DEFAULT_IMG_SIZE):
    """Transform ảnh giống lúc train (không augment)."""
    from strhub.data.module import SceneTextDataModule
    return SceneTextDataModule.get_transform(img_size, augment=False)


def preprocess(paths: Sequence[str], img_size: tuple[int, int] = DEFAULT_IMG_SIZE):
    """Danh sách đường dẫn ảnh -> tensor ``(B, 3, H, W)``."""
    import torch
    from PIL import Image
    tf = load_transform(img_size)
    tensors = [tf(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(tensors, dim=0)


def decode_predictions(model: Any, logits: Any) -> list[tuple[str, float]]:
    """logits ``(B, T, C)`` -> list ``(pred, confidence)`` (giống parseq ``_eval_step``)."""
    probs = logits.softmax(-1)
    preds, conf = model.tokenizer.decode(probs)
    out: list[tuple[str, float]] = []
    for i, pred in enumerate(preds):
        pred = model.charset_adapter(pred)
        c = conf[i] if len(conf) else probs[i].max()
        try:
            score = float(c.prod())
        except AttributeError:
            score = float(c)
        out.append((pred, score))
    return out


def predict(model: Any, paths: Sequence[str], device: str = "cpu",
            img_size: tuple[int, int] = DEFAULT_IMG_SIZE) -> list[dict[str, Any]]:
    """Suy luận trên nhiều ảnh; trả list ``{image, pred, confidence}``."""
    import torch
    model = model.eval().to(device)
    imgs = preprocess(paths, img_size).to(device)
    with torch.inference_mode():
        logits = model(imgs)
    decoded = decode_predictions(model, logits)
    return [{"image": str(p), "pred": pred, "confidence": round(conf, 4)}
            for p, (pred, conf) in zip(paths, decoded)]


def run(images: Sequence[str], ckpt: str | Path | None = None, device: str = "cpu",
        model: Any | None = None, img_size: tuple[int, int] = DEFAULT_IMG_SIZE,
        log=print) -> list[dict[str, Any]]:
    if not images:
        raise SystemExit("cần --image <đường dẫn ảnh> ...")
    if model is None:
        if not ckpt:
            raise SystemExit("cần --ckpt <checkpoint> để nạp model")
        from trademark_ocr.evaluate import load_model
        model = load_model(ckpt, device=device)
    rows = predict(model, images, device=device, img_size=img_size)
    for r in rows:
        log(f"{Path(r['image']).name}: {r['pred']!r} (conf={r['confidence']:.3f})")
    return rows


def build_interface(model: Any, device: str = "cpu",
                    img_size: tuple[int, int] = DEFAULT_IMG_SIZE):
    """Giao diện Gradio (nhãn hiệu -> chữ)."""
    import gradio as gr
    from PIL import Image

    def _predict(pil):
        import torch
        tf = load_transform(img_size)
        x = tf(pil.convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model.eval().to(device)(x)
        pred, conf = decode_predictions(model, logits)[0]
        return pred, round(conf, 4)

    return gr.Interface(fn=_predict, inputs=gr.Image(type="pil"),
                        outputs=[gr.Textbox(label="Dự đoán"),
                                 gr.Number(label="Độ tin cậy")],
                        title="OCR nhãn hiệu",
                        description="Tải ảnh nhãn hiệu (wordmark) để nhận dạng chữ.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Demo OCR nhãn hiệu (nhánh 14).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="checkpoint (.ckpt)")
    ap.add_argument("--image", nargs="+", default=None, help="1 hoặc nhiều ảnh")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--serve", action="store_true", help="mở Gradio web UI")
    ap.add_argument("--share", action="store_true", help="tạo link công khai (cần --serve)")
    return ap


def main(argv: "list[str] | None" = None) -> int:
    a = build_parser().parse_args(argv)
    if a.serve:
        if not a.ckpt:
            raise SystemExit("--serve cần --ckpt")
        from trademark_ocr.evaluate import load_model
        model = load_model(a.ckpt, device=a.device)
        build_interface(model, device=a.device).launch(share=a.share)
        return 0
    if a.image:
        run(a.image, ckpt=a.ckpt, device=a.device)
        return 0
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
