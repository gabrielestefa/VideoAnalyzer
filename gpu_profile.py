"""
GPU profile selection.

Picks one of {cpu, lite, standard, full, max} from total VRAM, with an env-var
override (VIDEOANALYZER_PROFILE).  Each profile names concrete model variants
so downstream code (pose backends, future segmentation / SAM2) can ask for
the right size without re-reading hardware specs.

Profiles documented in detail in VISION_UPGRADE_PLAN.md §5.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

ProfileName = Literal["cpu", "lite-noseg", "lite", "standard", "full", "max"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Profile:
    """Concrete model selections for a given hardware tier."""
    name: ProfileName
    # Pose
    pose_model: str          # e.g. "rtmw-s", "rtmw-m", "rtmw-l", "rtmw-x"
    pose_device: str         # "cuda" or "cpu"
    pose_input_size: int     # 256 / 384 / 448 …
    # Detection (used as person detector for top-down pose)
    det_model: str           # "yolo11n" / "yolo11s" / "yolo11m" / "yolo11l"
    # Future stages
    seg_model: str | None    # "sapiens-0.3b" / "sapiens-0.6b" / "sapiens-1b" / "schp" / None
    seg_every_n_frames: int  # 0 means skip segmentation entirely
    sam2_model: str | None   # "tiny" / "small" / "base" / "large" / None
    # 3D hand mesh
    wilor: bool              # WiLoR every frame
    hamer_on_contact: bool   # HaMeR only when contact event fires
    # Whisper coexistence policy
    whisper_concurrent: bool  # True = ok to run vision + whisper at once


_PROFILES: dict[str, Profile] = {
    "cpu": Profile(
        name="cpu",
        pose_model="mediapipe", pose_device="cpu", pose_input_size=0,
        det_model="none",
        seg_model=None, seg_every_n_frames=0,
        sam2_model=None,
        wilor=False, hamer_on_contact=False,
        whisper_concurrent=False,
    ),
    "lite-noseg": Profile(
        name="lite-noseg",
        pose_model="rtmw-s", pose_device="cuda", pose_input_size=256,
        det_model="yolo11n",
        seg_model=None, seg_every_n_frames=0,
        sam2_model="tiny",
        wilor=False, hamer_on_contact=False,
        whisper_concurrent=False,
    ),
    "lite": Profile(
        name="lite",
        pose_model="rtmw-s", pose_device="cuda", pose_input_size=256,
        det_model="yolo11n",
        seg_model="sapiens-0.3b", seg_every_n_frames=10,
        sam2_model="tiny",
        wilor=False, hamer_on_contact=False,
        whisper_concurrent=False,
    ),
    "standard": Profile(
        name="standard",
        pose_model="rtmw-m", pose_device="cuda", pose_input_size=384,
        det_model="yolo11s",
        seg_model="sapiens-0.3b", seg_every_n_frames=5,
        sam2_model="small",
        wilor=False, hamer_on_contact=False,
        whisper_concurrent=False,
    ),
    "full": Profile(
        name="full",
        pose_model="rtmw-l", pose_device="cuda", pose_input_size=384,
        det_model="yolo11m",
        seg_model="sapiens-0.6b", seg_every_n_frames=5,
        sam2_model="base",
        wilor=True, hamer_on_contact=False,
        whisper_concurrent=False,
    ),
    "max": Profile(
        name="max",
        pose_model="rtmw-x", pose_device="cuda", pose_input_size=448,
        det_model="yolo11l",
        seg_model="sapiens-1b", seg_every_n_frames=3,
        sam2_model="large",
        wilor=True, hamer_on_contact=True,
        whisper_concurrent=True,
    ),
}


def _detect_vram_gb() -> float | None:
    """Total GPU memory in GB, or None if no CUDA device is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception as e:
        logger.warning("VRAM detection failed: %s", e)
        return None


def _auto_select() -> ProfileName:
    vram = _detect_vram_gb()
    if vram is None or vram < 3.5:
        # No GPU, or so little VRAM that even the smallest RTMW + SAM2 tiny
        # stack (~3 GB) can't fit alongside driver overhead.  Fall back to the
        # MediaPipe-only pipeline.  Skip the heavy mmpose install too.
        return "cpu"
    if vram >= 22:   return "max"        # 4090 / 5090
    if vram >= 11:   return "full"       # 4070 / 5070 / 5080
    if vram >= 7.5:  return "standard"   # 3070 / 4060
    if vram >= 5.5:  return "lite"       # 3060 laptop / 2060
    return "lite-noseg"                   # 4 GB cards


def select_profile() -> Profile:
    """
    Pick a Profile based on VRAM, honouring VIDEOANALYZER_PROFILE override.
    Result is cached for the process lifetime — call at startup, reuse forever.
    """
    override = os.environ.get("VIDEOANALYZER_PROFILE", "").strip().lower()
    if override and override in _PROFILES:
        logger.info("Profile override via env: %s", override)
        return _PROFILES[override]
    if override:
        logger.warning("Unknown VIDEOANALYZER_PROFILE=%r; ignoring", override)

    name = _auto_select()
    vram = _detect_vram_gb()
    if vram is not None:
        logger.info("Auto-selected profile %r (VRAM: %.1f GB)", name, vram)
    else:
        logger.info("Auto-selected profile %r (no CUDA)", name)
    return _PROFILES[name]


def required_components(profile: Profile | None = None) -> set[str]:
    """
    Heavy install-time components a profile actually uses.

    Both setup.bat (via this module run as a script) and dependency_check.py
    read this so we never install 2 GB of mmpose on a machine that can't
    actually run it.

    Returns identifiers; the install layer maps each to its pip / mim
    commands.  Identifiers used today:
      - "rtmw"     → mmpose + mmcv + mmdet + mmengine + openmim
      - "sapiens"  → Sapiens body-part segmentation (Phase 2, not yet wired)
      - "sam2"     → SAM 2 object tracking (Phase 3)
      - "wilor"    → WiLoR 3D hand mesh (Phase 5)
      - "hamer"    → HaMeR refinement on contact frames (Phase 5)
    """
    if profile is None:
        profile = select_profile()
    comps: set[str] = set()
    if profile.pose_model != "mediapipe":
        comps.add("rtmw")
    if profile.seg_model and profile.seg_model.startswith("sapiens"):
        comps.add("sapiens")
    if profile.sam2_model:
        comps.add("sam2")
    if profile.wilor:
        comps.add("wilor")
    if profile.hamer_on_contact:
        comps.add("hamer")
    return comps


if __name__ == "__main__":
    # Script mode used by setup.bat:
    #   python gpu_profile.py             → print profile name (e.g. "lite")
    #   python gpu_profile.py components  → print space-separated component ids
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import sys as _sys
    p = select_profile()
    if len(_sys.argv) > 1 and _sys.argv[1] == "components":
        print(" ".join(sorted(required_components(p))))
    elif len(_sys.argv) > 1 and _sys.argv[1] == "name":
        print(p.name)
    else:
        print(p)
