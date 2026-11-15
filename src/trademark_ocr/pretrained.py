"""Nhánh 16 — fine-tune từ trọng số pretrained của parseq với charset tiếng Việt (REVIEW.md §4.6).

Vấn đề: ``pretrained=<tên>`` trong ``third_party/parseq/train.py`` gọi
``load_state_dict`` **strict**. Charset VN (217 ký tự) làm các tensor phụ thuộc charset
(``head``, ``text_embed`` của PARSeq; lớp ra của CRNN/TRBA) khác kích thước bản phát hành
(charset 94/36 ký tự) → không khởi tạo được, mọi model phải train từ đầu trên vài chục nghìn
ảnh — quá ít cho ViT (PARSeq).

Cách làm (không sửa parseq): :func:`finetune` là một **factory** dùng làm ``model._target_``
trong config hydra (``configs/hydra/tm_parseq_ft.yaml``). ``train.py`` của parseq vẫn gọi
``hydra.utils.instantiate(config.model)`` như thường; factory dựng đúng lớp ``target`` rồi
nạp các tensor **cùng tên và cùng kích thước** từ ``init_from``, bỏ qua phần lệch (được khởi
tạo ngẫu nhiên như cũ). ``target``/``init_from`` không truyền vào model, nên checkpoint sinh
ra là checkpoint ``PARSeq`` bình thường.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

#: giá trị ``model._target_`` của các config fine-tune; ``evaluate`` dùng để tìm lớp thật
FINETUNE_TARGET = "trademark_ocr.pretrained.finetune"


def _import_class(dotted: str) -> type:
    module, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(module), name)


def load_matching(module: Any, state_dict: dict[str, Any]) -> dict[str, list[str]]:
    """Chép vào ``module`` mọi tensor của ``state_dict`` trùng tên **và** trùng kích thước.

    Trả báo cáo ``{loaded, skipped_shape, missing, unexpected}`` (danh sách tên tensor):
    ``skipped_shape`` = trùng tên nhưng lệch kích thước (phần phụ thuộc charset),
    ``missing`` = có trong module mà không có trong file, ``unexpected`` = ngược lại.
    """
    own = module.state_dict()
    report: dict[str, list[str]] = {"loaded": [], "skipped_shape": [], "missing": [],
                                    "unexpected": sorted(set(state_dict) - set(own))}
    to_load = {}
    for name, tensor in own.items():
        if name not in state_dict:
            report["missing"].append(name)
        elif tuple(state_dict[name].shape) != tuple(tensor.shape):
            report["skipped_shape"].append(name)
        else:
            to_load[name] = state_dict[name]
            report["loaded"].append(name)
    # strict=False chỉ vì to_load cố ý thiếu các tensor bị bỏ; mọi khoá trong to_load đều
    # đã được kiểm trùng tên + kích thước ở trên.
    module.load_state_dict(to_load, strict=False)
    return report


def _read_state(init_from: str) -> dict[str, Any]:
    """``init_from`` là đường dẫn file (``.pt`` state dict hoặc ``.ckpt`` Lightning), hoặc
    tên bản phát hành của parseq (``parseq``, ``parseq-tiny``, ``crnn``, ``trba``…, tải qua
    ``torch.hub`` — cần mạng lần đầu, sau đó dùng cache)."""
    import torch

    if Path(init_from).exists():
        state = torch.load(init_from, map_location="cpu", weights_only=False)
        return state["state_dict"] if "state_dict" in state else state
    from strhub.models.utils import get_pretrained_weights
    return get_pretrained_weights(init_from)


def finetune(target: str, init_from: str | None = None, **kwargs: Any) -> Any:
    """Dựng ``target`` (dotted path lớp system, vd ``strhub.models.parseq.system.PARSeq``) với
    ``kwargs`` rồi nạp một phần trọng số ``init_from`` (xem :func:`_read_state`).

    Đích nạp được chọn theo dữ liệu, không theo tên lớp: bản phát hành ``parseq`` chứa state
    dict của ``system.model`` (khoá ``encoder.*``), bản ``crnn``/``trba`` và ckpt Lightning
    chứa state dict cấp system (khoá ``model.*``) — chọn module có nhiều tên khoá trùng hơn.

    ``init_from=None`` → model khởi tạo ngẫu nhiên (giống ``_target_`` trực tiếp).
    """
    model = _import_class(target)(**kwargs)
    if not init_from:
        return model
    state = _read_state(init_from)
    candidates = [model] + ([model.model] if hasattr(model, "model") else [])
    dest = max(candidates, key=lambda m: len(set(m.state_dict()) & set(state)))
    report = load_matching(dest, state)
    if not report["loaded"]:
        raise ValueError(f"init_from={init_from!r}: không khớp tensor nào với {target} "
                         "(sai file hoặc sai kiến trúc?)")
    print(f"[pretrained] {init_from}: nạp {len(report['loaded'])} tensor, bỏ "
          f"{len(report['skipped_shape'])} lệch kích thước {report['skipped_shape']}, "
          f"{len(report['missing'])} thiếu, {len(report['unexpected'])} thừa")
    return model
