"""
Pose backends — pluggable per-frame human-pose estimators.

Phase 0 / Phase 1 of the vision pipeline upgrade (see VISION_UPGRADE_PLAN.md).

Two backends ship today:

  * MediapipeBackend  — the original, default.  21 hand keypoints, 7-class
    built-in gesture classifier.  Hand only.  Lightweight, CPU-friendly.

  * RtmwBackend       — RTMW from OpenMMLab.  17 body + 21+21 hand + 68 face
    keypoints (133 whole-body).  Multi-person.  GPU recommended.  Gestures
    are produced by a small rule-based classifier (PoseGestureClassifier)
    so the existing `gesture_counts` UI keeps working unchanged.

Both backends return identical-shape `FrameResult`s so callers (engine, UI,
serialization) need not know which one is active.  Body/face keypoints are
optional — None when running MediaPipe, populated when running RTMW.

Selection happens in VideoAnalysisEngine via VIDEOANALYZER_BACKEND env var
(default "auto" → MediaPipe unless RTMW is installed and a GPU is present).
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------

class Landmark:
    """
    Minimal MediaPipe-compatible landmark.  Same attribute layout as
    LandmarkCompat in video_analysis_engine.py — kept distinct here so this
    module has no upward import.  x/y are in [0,1] (image-normalized).
    """
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)

    def __repr__(self) -> str:  # pragma: no cover
        return f"L({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


@dataclass
class FrameResult:
    """
    Backend-agnostic per-frame output.

    `hands` matches the existing engine shape: list of (landmarks, label)
    where landmarks is a 21-element list of Landmark objects and label is
    "Left" or "Right".  This is what `landmarks_list` and downstream
    visualization code already expect.

    `body` / `face` are populated by whole-body backends (RTMW, Sapiens)
    and left None by MediaPipe.  Downstream code can simply check for None.

    `body_per_person` / `face_per_person` carry keypoints for *all* detected
    people in the frame (RTMW only).  `body` / `face` are the top-confidence
    person's keypoints — kept for backward compatibility with single-subject
    consumers (heatmap, traces, episode analyzer).  Multi-person UI rendering
    iterates the `*_per_person` lists.
    """
    hands: list[tuple[list[Landmark], str]] = field(default_factory=list)
    gesture_left: Optional[tuple[str, float]] = None
    gesture_right: Optional[tuple[str, float]] = None
    body: Optional[list[Landmark]] = None       # 17 COCO body keypoints (top person)
    face: Optional[list[Landmark]] = None       # 68 face keypoints (top person)
    body_per_person: list[list[Landmark]] = field(default_factory=list)
    face_per_person: list[list[Landmark]] = field(default_factory=list)


class BackendBase:
    """Interface every pose backend implements."""
    name: str = "base"
    supports_body: bool = False
    supports_face: bool = False
    supports_gestures: bool = False

    def initialize(self, video_width: int, video_height: int, fps: float) -> None:
        """Allocate models / state for a new video."""
        raise NotImplementedError

    def process_frame(self, rgb_frame: np.ndarray, timestamp_ms: int) -> FrameResult:
        """Run inference on one RGB HxWx3 uint8 frame; return a FrameResult."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources.  Safe to call multiple times."""
        pass


# ---------------------------------------------------------------------------
# Rule-based gesture classifier (used by RTMW backend for parity with the
# MediaPipe built-in gestures).  Produces the same label set:
#   None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Up, Thumb_Down,
#   Victory, ILoveYou
# ---------------------------------------------------------------------------

class PoseGestureClassifier:
    """
    Lightweight rule-based gesture classifier on 21 hand keypoints
    (MediaPipe / MANO index order).  Output matches the labels emitted by
    MediaPipe's GestureRecognizer so existing UI code is unchanged.

    Not as nuanced as MediaPipe's TFLite classifier but adequate for the
    coarse hand-shape stats we currently aggregate.
    """

    # Fingertip / IP indices used for finger-extended tests
    _FINGERS = {
        # finger_name -> (tip, pip, mcp)
        "index":  (8, 6, 5),
        "middle": (12, 10, 9),
        "ring":   (16, 14, 13),
        "pinky":  (20, 18, 17),
    }
    _THUMB = (4, 3, 2)
    _WRIST = 0

    def __call__(self, hand: list[Landmark]) -> Optional[tuple[str, float]]:
        if len(hand) < 21:
            return None
        extended = {
            f: self._is_finger_extended(hand, tip, pip, mcp)
            for f, (tip, pip, mcp) in self._FINGERS.items()
        }
        thumb_up = self._is_thumb_up(hand)
        thumb_down = self._is_thumb_down(hand)
        ext_count = sum(extended.values())

        # Closed fist: no fingers extended and thumb not up/down
        if ext_count == 0 and not thumb_up and not thumb_down:
            return ("Closed_Fist", 0.85)

        # Open palm: all 4 fingers extended (thumb state ignored)
        if ext_count == 4:
            return ("Open_Palm", 0.9)

        # Pointing up: only index extended
        if extended["index"] and ext_count == 1:
            return ("Pointing_Up", 0.85)

        # Victory: index + middle extended only
        if extended["index"] and extended["middle"] and not extended["ring"] and not extended["pinky"]:
            return ("Victory", 0.85)

        # ILoveYou: thumb + index + pinky (no middle, no ring)
        if thumb_up and extended["index"] and extended["pinky"] and not extended["middle"] and not extended["ring"]:
            return ("ILoveYou", 0.8)

        if thumb_up and ext_count == 0:
            return ("Thumb_Up", 0.85)
        if thumb_down and ext_count == 0:
            return ("Thumb_Down", 0.85)

        return None

    @staticmethod
    def _is_finger_extended(h: list[Landmark], tip: int, pip: int, mcp: int) -> bool:
        # Finger is extended when tip is farther from wrist than pip
        # AND tip is "above" pip in image-y (smaller y = higher on screen).
        wrist = h[0]
        d_tip = math.hypot(h[tip].x - wrist.x, h[tip].y - wrist.y)
        d_pip = math.hypot(h[pip].x - wrist.x, h[pip].y - wrist.y)
        return d_tip > d_pip * 1.1 and h[tip].y < h[mcp].y

    @staticmethod
    def _is_thumb_up(h: list[Landmark]) -> bool:
        # Thumb tip clearly above wrist (small y) and roughly vertical.
        return h[4].y < h[2].y - 0.05

    @staticmethod
    def _is_thumb_down(h: list[Landmark]) -> bool:
        return h[4].y > h[2].y + 0.05


# ---------------------------------------------------------------------------
# MediaPipe backend (default — original behaviour preserved verbatim)
# ---------------------------------------------------------------------------

class MediapipeBackend(BackendBase):
    """Wraps the existing MediaPipe GestureRecognizer."""
    name = "mediapipe"
    supports_body = False
    supports_face = False
    supports_gestures = True

    def __init__(self, model_path: str, sensitivity: float = 0.1, num_hands: int = 2):
        self.model_path = model_path
        self.sensitivity = sensitivity
        self.num_hands = num_hands
        self._recognizer = None  # type: ignore[assignment]
        self._fps = 30.0

    def initialize(self, video_width: int, video_height: int, fps: float) -> None:
        # Imports deferred so module loads even without mediapipe present.
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python.vision import (
            GestureRecognizer, GestureRecognizerOptions, RunningMode,
        )
        self._mp = mp
        self._fps = fps or 30.0
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        self._recognizer = GestureRecognizer.create_from_options(
            GestureRecognizerOptions(
                base_options=base_options,
                running_mode=RunningMode.VIDEO,
                num_hands=self.num_hands,
                min_hand_detection_confidence=self.sensitivity,
                min_hand_presence_confidence=self.sensitivity,
                min_tracking_confidence=self.sensitivity,
            )
        )
        logger.info("MediapipeBackend initialised (sensitivity=%.2f, num_hands=%d)",
                    self.sensitivity, self.num_hands)

    def process_frame(self, rgb_frame: np.ndarray, timestamp_ms: int) -> FrameResult:
        if self._recognizer is None:
            raise RuntimeError("MediapipeBackend.initialize() not called")
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._recognizer.recognize_for_video(mp_image, timestamp_ms)

        out = FrameResult()
        if not result.hand_landmarks:
            return out
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].category_name  # "Left" / "Right"
            lms = [Landmark(lm.x, lm.y, getattr(lm, "z", 0.0)) for lm in hand_landmarks]
            out.hands.append((lms, label))

            if result.gestures and len(result.gestures) > i:
                g = result.gestures[i][0]
                if label == "Left":
                    out.gesture_left = (g.category_name, g.score)
                else:
                    out.gesture_right = (g.category_name, g.score)
        return out

    def close(self) -> None:
        if self._recognizer is not None:
            try:
                self._recognizer.close()
            except Exception:
                pass
            self._recognizer = None


# ---------------------------------------------------------------------------
# RTMW backend (Phase 1 — opt-in, requires mmpose stack)
# ---------------------------------------------------------------------------

# COCO-WholeBody keypoint slicing (133 keypoints):
#   body  : 0..16   (17)
#   feet  : 17..22  (6)
#   face  : 23..90  (68)
#   l_hand: 91..111 (21)
#   r_hand: 112..132 (21)
#
# RTMW's left/right convention: indices 91-111 = left hand of the person
# (mirror of viewer).  We pass these straight through; the engine treats
# "left" as the *person's* left, same convention MediaPipe uses.

RTMW_BODY_RANGE = (0, 17)
RTMW_FACE_RANGE = (23, 91)
RTMW_LEFT_HAND_RANGE = (91, 112)
RTMW_RIGHT_HAND_RANGE = (112, 133)


class RtmwBackend(BackendBase):
    """
    RTMW whole-body backend via the mmpose / mmdetection inferencer API.

    Heavy import surface — only loaded on demand.  If mmpose is not
    installed, construction raises ImportError which the engine catches
    and falls back to MediaPipe.
    """
    name = "rtmw"
    supports_body = True
    supports_face = True
    supports_gestures = True  # via rule-based PoseGestureClassifier

    # mmpose 1.3 publishes RTMW under these full config IDs (verified via
    # `mim search mmpose --remote`).  No "small" variant exists upstream;
    # the smallest is "m" at 256×192.  Aliases below pick the best-matching
    # public checkpoint for each (profile, input_size) pair.
    _CONFIGS = {
        # 256×192 input (smaller, faster)
        ("rtmw-m", 256): "rtmw-m_8xb1024-270e_cocktail14-256x192",
        ("rtmw-l", 256): "rtmw-l_8xb1024-270e_cocktail14-256x192",
        ("rtmw-x", 256): "rtmw-x_8xb704-270e_cocktail14-256x192",
        # 384×288 input (larger, more accurate; no "m" variant exists)
        ("rtmw-l", 384): "rtmw-l_8xb320-270e_cocktail14-384x288",
        ("rtmw-x", 384): "rtmw-x_8xb320-270e_cocktail14-384x288",
    }

    def _resolve_config(self) -> str:
        """Pick the closest available RTMW config for our (size, input)."""
        # Map "s" → "m" since mmpose doesn't publish a Small variant.
        size = "rtmw-m" if self.model_size == "rtmw-s" else self.model_size
        input_size = self.input_size if self.input_size in (256, 384) else 256
        cfg = self._CONFIGS.get((size, input_size))
        if cfg is not None:
            return cfg
        # Fallback chain: drop to 256 if 384 unavailable; then drop the model
        # size by one tier rather than bailing.
        cfg = self._CONFIGS.get((size, 256))
        if cfg is not None:
            return cfg
        # Last resort: any rtmw-l 256 config.
        return self._CONFIGS[("rtmw-l", 256)]

    def __init__(self, model_size: str = "rtmw-s", device: str = "cuda",
                 input_size: int = 256):
        self.model_size = model_size
        self.device = device
        self.input_size = input_size
        self._inferencer = None
        self._gesture_clf = PoseGestureClassifier()

    def initialize(self, video_width: int, video_height: int, fps: float) -> None:
        try:
            from mmpose.apis import MMPoseInferencer
        except ImportError as e:
            raise ImportError(
                "mmpose is not installed. Install with:\n"
                "  pip install -r requirements-rtmw.txt\n"
                f"Original error: {e}"
            ) from e

        cfg = self._resolve_config()
        logger.info("RtmwBackend initialising (config=%s, device=%s)", cfg, self.device)
        self._inferencer = MMPoseInferencer(pose2d=cfg, device=self.device)
        self._video_w = video_width
        self._video_h = video_height

    def process_frame(self, rgb_frame: np.ndarray, timestamp_ms: int) -> FrameResult:
        if self._inferencer is None:
            raise RuntimeError("RtmwBackend.initialize() not called")

        # MMPoseInferencer takes BGR ndarray or path; we already have RGB so
        # flip to BGR for consistency with mmcv conventions.
        bgr = rgb_frame[..., ::-1].copy()
        result_gen = self._inferencer(bgr, show=False, return_vis=False)
        result = next(result_gen)

        predictions = result.get("predictions", [])
        if not predictions or not predictions[0]:
            return FrameResult()

        # predictions[0] is a list of per-person dicts (one frame, n people)
        people = predictions[0]
        # Sort by detection score descending so the most-confident person is first
        people = sorted(people, key=lambda p: -float(p.get("bbox_score", 0.0)))

        out = FrameResult()

        # Collect body + face for ALL detected people (multi-subject overlay).
        for person in people:
            kpts = np.asarray(person["keypoints"], dtype=np.float32)
            scores = np.asarray(person.get("keypoint_scores", [1.0] * len(kpts)),
                                dtype=np.float32)
            out.body_per_person.append(self._slice(kpts, scores, *RTMW_BODY_RANGE))
            out.face_per_person.append(self._slice(kpts, scores, *RTMW_FACE_RANGE))

        # Top person also fills the legacy single-subject fields (`body`,
        # `face`, `hands`, gestures) so heatmaps / traces / episode analyzer
        # continue to work unchanged.
        top = people[0]
        kpts = np.asarray(top["keypoints"], dtype=np.float32)
        scores = np.asarray(top.get("keypoint_scores", [1.0] * len(kpts)),
                            dtype=np.float32)
        out.body = out.body_per_person[0]
        out.face = out.face_per_person[0]

        left_hand = self._slice(kpts, scores, *RTMW_LEFT_HAND_RANGE)
        right_hand = self._slice(kpts, scores, *RTMW_RIGHT_HAND_RANGE)

        if left_hand and self._hand_visible(left_hand):
            out.hands.append((left_hand, "Left"))
            g = self._gesture_clf(left_hand)
            if g:
                out.gesture_left = g
        if right_hand and self._hand_visible(right_hand):
            out.hands.append((right_hand, "Right"))
            g = self._gesture_clf(right_hand)
            if g:
                out.gesture_right = g

        return out

    def _slice(self, kpts: np.ndarray, scores: np.ndarray,
               start: int, end: int) -> list[Landmark]:
        sub_k = kpts[start:end]
        sub_s = scores[start:end]
        return [
            Landmark(x / self._video_w, y / self._video_h, 0.0, float(s))
            for (x, y), s in zip(sub_k, sub_s)
        ]

    @staticmethod
    def _hand_visible(hand: list[Landmark], min_visibility: float = 0.3) -> bool:
        # Average keypoint score — RTMW reports 0 for occluded / off-frame.
        if not hand:
            return False
        return sum(lm.visibility for lm in hand) / len(hand) >= min_visibility

    def close(self) -> None:
        # MMPoseInferencer doesn't expose an explicit close; release ref and
        # nudge the GPU allocator.
        self._inferencer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_backend(name: str, mediapipe_model_path: str,
                 sensitivity: float = 0.1) -> BackendBase:
    """
    Build a backend by name.  "auto" picks RTMW if available + GPU present,
    otherwise MediaPipe.  Unknown names fall back to MediaPipe with a warning.
    """
    name = (name or "auto").strip().lower()

    if name == "auto":
        name = _auto_backend()

    if name == "rtmw":
        # Fail fast if mmpose isn't installed — RtmwBackend would otherwise
        # raise during initialize() and the engine would be half-set-up.
        import importlib.util
        if importlib.util.find_spec("mmpose") is None:
            logger.warning(
                "VIDEOANALYZER_BACKEND=rtmw but mmpose is not installed. "
                "Install with: pip install -r requirements-rtmw.txt — falling "
                "back to MediaPipe for this run."
            )
            return MediapipeBackend(mediapipe_model_path, sensitivity=sensitivity)
        try:
            from gpu_profile import select_profile
            prof = select_profile()
            if prof.pose_model == "mediapipe":
                logger.info("Profile %s does not support RTMW; using MediaPipe", prof.name)
                return MediapipeBackend(mediapipe_model_path, sensitivity=sensitivity)
            return RtmwBackend(model_size=prof.pose_model, device=prof.pose_device,
                               input_size=prof.pose_input_size)
        except ImportError as e:
            logger.warning("RTMW backend unavailable (%s); falling back to MediaPipe", e)
            return MediapipeBackend(mediapipe_model_path, sensitivity=sensitivity)

    if name != "mediapipe":
        logger.warning("Unknown backend %r; using MediaPipe", name)
    return MediapipeBackend(mediapipe_model_path, sensitivity=sensitivity)


def _auto_backend() -> str:
    """
    Decide between mediapipe and rtmw at startup.

    Conservative default — mediapipe unless ALL of the following hold:
      1. mmpose is importable
      2. a CUDA GPU is present
      3. the chosen profile is not "cpu"
    """
    try:
        import importlib.util
        if importlib.util.find_spec("mmpose") is None:
            return "mediapipe"
        from gpu_profile import select_profile
        prof = select_profile()
        if prof.pose_model == "mediapipe":
            return "mediapipe"
        return "rtmw"
    except Exception as e:
        logger.warning("Auto-backend selection failed (%s); defaulting to MediaPipe", e)
        return "mediapipe"
