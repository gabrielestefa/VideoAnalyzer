"""
Install a no-op mmcv._ext stub into the venv's site-packages.

Background: mmcv-lite is a pure-Python build of mmcv (no compiled CUDA ops).
mmcv.ops still tries to load `mmcv._ext` at import-time, so when mmpose's
transformer heads transitively import mmcv.ops the chain fails with
ModuleNotFoundError.  This stub uses Python 3.7+ module-level __getattr__
to satisfy ext_loader.load_ext()'s hasattr() check without crashing.  RTMW
inference never *calls* these ops, so a stub is enough.

Run from the project root, with the venv interpreter:
    .venv\\Scripts\\python.exe install_mmcv_ext_stub.py
"""
from __future__ import annotations

import os
import sys


STUB_SOURCE = '''\
"""
mmcv._ext stub installed by install_mmcv_ext_stub.py.

mmcv-lite omits the compiled CUDA extensions.  mmcv.ops still tries to load
them eagerly when imported (which mmpose triggers via transformer heads).
This module returns a NotImplementedError-raising callable for any attribute
the loader asks for — enough to make the import chain complete.

For ops we know are actually called at inference time (NMS), we provide a
real implementation backed by torchvision (which uses PyTorch CUDA kernels
directly — no mmcv compilation needed).  Add more shims here as needed.

If you ever install the full mmcv (which needs the CUDA Toolkit), it will
overwrite this stub with the real binary module.
"""
import warnings


def _make_stub(name):
    def _stub(*args, **kwargs):
        raise NotImplementedError(
            "mmcv-lite does not provide the CUDA op '" + name + "'.  "
            "Add a shim in mmcv/_ext.py or install full mmcv (needs CUDA Toolkit)."
        )
    _stub.__name__ = name
    return _stub


def nms(boxes, scores, iou_threshold, offset=0, score_threshold=0.0, max_num=-1):
    """C++ mmcv._ext.nms returns just kept indices; outer wrapper builds dets."""
    import torch
    from torchvision.ops import nms as _tv_nms
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)
    inds = _tv_nms(boxes, scores, float(iou_threshold)).long()
    if max_num > 0:
        inds = inds[:max_num]
    return inds


def softnms(*args, **kwargs):
    return nms(*args, **kwargs)


def __getattr__(name):  # PEP 562
    return _make_stub(name)


warnings.warn(
    "mmcv._ext is being served by a hand-rolled stub (mmcv-lite is installed). "
    "NMS works via torchvision; other CUDA ops will raise NotImplementedError.",
    stacklevel=2,
)
'''


def main() -> int:
    try:
        import mmcv
    except ImportError:
        print("[stub] mmcv not installed; nothing to do.")
        return 0

    target = os.path.join(os.path.dirname(mmcv.__file__), "_ext.py")
    if os.path.exists(target):
        # Likely the real compiled extension already exists (e.g. user installed
        # full mmcv at some point).  Don't clobber it.
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            existing = f.read()
        # Recognise both the old and new stub header strings; refresh either.
        is_our_stub = (
            "No-op stub for mmcv._ext" in existing
            or "mmcv._ext stub installed by" in existing
        )
        if not is_our_stub:
            print(f"[stub] {target} already exists and is not our stub; leaving it alone.")
            return 0
        print(f"[stub] Refreshing existing stub at {target}.")

    with open(target, "w", encoding="utf-8") as f:
        f.write(STUB_SOURCE)
    print(f"[stub] Wrote {target}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
