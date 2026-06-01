# VideoAnalyzer — Vision Pipeline Upgrade Plan

**Goal:** replace MediaPipe with a modern, modular vision stack that produces:

1. **Whole-body pose** (body + hands + face keypoints, 2D and optionally 3D),
2. **Body-part segmentation** spatially aligned to the keypoints,
3. **Object detection + segmentation** of things in the scene, and
4. **Interaction events** — when a hand/arm/leg makes contact with an object,
   we can name *what part* of *what object* it interacted with.

Today the pipeline runs MediaPipe `GestureRecognizer` per frame and stores 21
hand landmarks per detected hand ([video_analysis_engine.py:147-234](video_analysis_engine.py:147)).
Body pose, face, and any object grounding are absent. Segmentation does not
exist at all.

---

## 1. Why move off MediaPipe

| Limitation | Impact today |
|---|---|
| Hand-only; no body / arm / leg / face | Can't analyse leg or arm gestures, posture, full-body interaction |
| Single subject in many configurations | Can't analyse multi-person videos |
| Accuracy lags modern transformers in low light, fast motion, occlusion | Misses landmarks during the events you most want to analyse |
| 21 hand keypoints; no contact / object grounding | "Hand moved here" but never "hand grabbed cup" |
| Maintenance has slowed; recent releases are mostly bug-fix | Long-term bet looks weak |
| 3D keypoints from RGB are weakly supervised | 3D position estimates drift |

MediaPipe is still **the fastest CPU pose runner**, so for low-end hardware
without a GPU it remains useful as a fallback ([learnopencv comparison](https://learnopencv.com/yolov7-pose-vs-mediapipe-in-human-pose-estimation/)).
On the GPU we have today, modern models are both faster *and* more accurate.

---

## 2. Survey of candidate components

The pipeline naturally splits into four stages. For each I list two or three
viable choices with notes drawn from current literature and project pages.

### 2.1 Whole-body pose (body + hands + face)

| Model | Notes | License | Refs |
|---|---|---|---|
| **RTMW / RTMPose** (OpenMMLab) | Real-time multi-person 2D + 3D *whole-body*; RTMW-L is the first open-source model > 70 mAP on COCO-WholeBody. 17 body + 21+21 hand + 68 face = 133 keypoints. 70+ FPS on a Snapdragon, much faster on desktop GPUs. | Apache 2.0 | [RTMW paper](https://arxiv.org/abs/2407.08634), [MMPose repo](https://github.com/open-mmlab/mmpose) |
| **Sapiens (Meta)** | Transformer pre-trained on 300 M human images. Four tasks share a backbone: pose, segmentation, depth, surface normal. Highest accuracy of any open model (+7.6 mAP over prior SOTA on Humans-5K). Heavy — 8 GB VRAM minimum, slower per frame than RTMW. | Apache 2.0 (code), CC-BY-NC for weights | [Sapiens repo](https://github.com/facebookresearch/sapiens), [ECCV'24 paper](https://rawalkhirodkar.github.io/sapiens/), [PyTorch inference port](https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference) |
| **DWPose** | RTMPose distilled for whole-body; 18 body / 21×2 hand / 68 face. Slightly less accurate than RTMW but lower latency. Heavily used in ControlNet ecosystems. | Apache 2.0 | [DWPose paper](https://arxiv.org/abs/2307.15880) |
| **YOLO11-Pose** (Ultralytics) | Body only (17 COCO keypoints), but trivial Python API and multi-object tracking baked in. Useful as a *detector* even if pose comes from elsewhere. | AGPL-3.0 / commercial | [YOLO11 docs](https://docs.ultralytics.com/tasks/pose) |

**Pick:** **RTMW-L** as the primary; YOLO11 as the per-frame person detector
that feeds RTMW; Sapiens-Pose available as an opt-in "best quality" mode.

### 2.2 Hand reconstruction (3D MANO mesh, if needed)

If you eventually want **3D hand pose** rather than 2D landmarks (better for
contact geometry, AR, ergonomics):

| Model | Notes | Refs |
|---|---|---|
| **HaMeR** | Transformer-based 3D hand reconstruction, MANO output, SOTA on benchmarks but heavy. | [HaMeR site](https://geopavlakos.github.io/hamer/), [repo](https://github.com/geopavlakos/hamer) |
| **WiLoR** | End-to-end localization + reconstruction. **Medium: >130 FPS, small: 175 FPS — ~45× faster than HaMeR** at near-equivalent accuracy. | [WiLoR paper](https://arxiv.org/html/2409.12259v2), [repo](https://github.com/rolpotamias/WiLoR) |

**Pick:** **WiLoR** if/when 3D hand mesh becomes a feature requirement.

### 2.3 Segmentation

| Model | What it gives you | Why we want it |
|---|---|---|
| **Sapiens-Seg** | 28-class body-part segmentation (face, hair, torso, arms, hands, legs, feet — fine-grained) at 1K resolution. Smallest variant (`0.3B`) is ~2 GB FP16 — fits comfortably on a 6 GB card alongside pose. | Tells us *which body part* a pixel belongs to. Aligns natively with Sapiens-Pose. |
| **SCHP** (Self-Correction Human Parsing) | 20-class body parsing, ~0.8 GB VRAM, ~40 ms / frame. Older (2020) but lightweight and battle-tested. | Fallback when even Sapiens-Seg-0.3B is too heavy. |
| **SAM 2** (Meta) | Promptable instance segmentation **with video tracking** through a streaming memory module — 6× faster than SAM 1 on images, real-time on video; tracks masks across frames automatically. Tiny variant (`Hiera-T`) is ~1 GB FP16. | Given a keypoint or bbox from the pose stage, return a temporally consistent mask. Critical for ID-preserving object tracking. |

**Pick:** **Sapiens-Seg** for body-part labels + **SAM 2** for arbitrary
object masks prompted by detected objects' bboxes/keypoints.

[SAM 2 paper](https://arxiv.org/pdf/2408.00714) · [Ultralytics SAM 2 docs](https://docs.ultralytics.com/models/sam-2) · [Pose-prompted SAM example (SAM-pose2seg)](https://arxiv.org/abs/2601.08982)

### 2.4 Object detection & open-vocabulary grounding

For "what is the hand touching?" we need to know what objects exist in the
scene. Closed-set COCO detection isn't enough — videos contain arbitrary
props.

| Model | Notes | Refs |
|---|---|---|
| **YOLO11-Seg** | Closed-set, very fast, good for the 80 COCO classes. | [YOLO11 docs](https://docs.ultralytics.com/models/yolo11/) |
| **Grounding DINO 1.5/Grounding DINO** | **Open-vocabulary** — query with text like "guitar", "coffee cup", "phone". | [Grounding DINO video tutorial](https://pyimagesearch.com/2025/12/08/grounding-dino-open-vocabulary-object-detection-on-videos/) |
| **DINOv3** (used in SL-HOI) | Newer; powers the current SOTA in open-vocabulary HOI detection. | [SL-HOI paper](https://arxiv.org/abs/2603.27500) |

**Pick:** **YOLO11-Seg** as the fast default; **Grounding DINO** as the
opt-in "describe what to look for" mode for novel objects.

### 2.5 Human-object interaction (HOI)

| Approach | Notes |
|---|---|
| **Geometric contact** | Project hand/limb keypoints onto object masks; declare contact when keypoint is inside (or within N pixels of) a mask edge for K consecutive frames. Simple, fast, ~85% accuracy for clear contact events. |
| **Grasp-quality metrics** | Feature-engineered grasp quality from hand pose + object pose. ~90% accuracy on DexYCB. | [Hand-object contact via grasp metrics](https://arxiv.org/abs/2501.06987) |
| **End-to-end HOI transformer** (SL-HOI, generative HOI) | One model returns `<subject, verb, object>` triplets. Heaviest, most accurate. | [SL-HOI](https://arxiv.org/abs/2603.27500) |

**Pick:** start with **geometric contact** on top of pose+segmentation
(cheap, debuggable, no new model). Upgrade to an HOI transformer only if
the geometric heuristic proves insufficient.

---

## 3. Recommended target architecture

```
                  ┌─────────────────────────────────────────────┐
                  │             Per-frame pipeline              │
                  ├─────────────────────────────────────────────┤
       Frame ──►  │ 1. YOLO11 detector  → person & object bboxes │
                  │ 2. RTMW (per person) → 133 whole-body keypts │
                  │ 3. Sapiens-Seg (1 / N frames) → body-part map│
                  │ 4. SAM 2 (track mode) → object instance masks│
                  └─────────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────────┐
                  │             Per-frame fusion                │
                  │  • assign person ID via tracker             │
                  │  • for each limb/hand keypoint, look up:    │
                  │      - body-part label at keypoint pixel    │
                  │      - nearest object mask + distance       │
                  │  • emit contact event if dist < threshold   │
                  │    for K frames in a row                    │
                  └─────────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────────┐
                  │             Event log (engine state)        │
                  │  (t_start, t_end, person_id, body_part,     │
                  │   object_label, mean_distance, peak_force?) │
                  └─────────────────────────────────────────────┘
```

Three things make this work as a unit and not a pile of models:

1. **One tracker ID** (from YOLO11's BoT-SORT or ByteTrack) ties pose,
   segmentation, and object masks together over time.
2. **The keypoint is the join key.** A landmark at `(x, y)` looks up its
   body-part label in Sapiens-Seg's map and its object-mask membership in
   SAM 2's masks — both are dense per-pixel images. The fusion step is
   `O(num_keypoints)`, not `O(pixels)`.
3. **Segmentation cadence is decoupled from pose.** Sapiens-Seg is heavy
   (~250 ms at 1K res); run it every 5–10 frames and interpolate / hold the
   last map. Pose runs every frame.

---

## 4. Phased migration

### Phase 0 — Wrap the current engine behind an interface (no behaviour change)

The current code calls MediaPipe directly inside `VideoAnalysisEngine`
([video_analysis_engine.py:147-234](video_analysis_engine.py:147)).
Extract a small abstract base class so the engine doesn't care which model is
running:

```python
class PoseBackend(Protocol):
    name: str
    def process_frame(self, rgb: np.ndarray, t_seconds: float) -> FrameResult: ...

class MediapipeBackend(PoseBackend):  # current behaviour, untouched
    ...
```

`FrameResult` carries body keypoints, hand keypoints, face keypoints (any
may be empty), an optional body-part mask, and detected object masks. The UI
and serialization layer ([hand_heatmap_modern.py](hand_heatmap_modern.py))
work on `FrameResult`, not raw MediaPipe types.

This phase touches:
- `video_analysis_engine.py` — extract backend interface; the existing
  MediaPipe call becomes `MediapipeBackend`.
- `hand_heatmap_modern.py` — replace `hand_landmarks[i].x` access patterns
  with backend-agnostic accessors.
- Save format — version-tag the pickle so old sessions still load.

**Outcome:** zero functional change; foundation for Phase 1+.

### Phase 1 — RTMW backend (whole-body, drop-in)

Add `RtmwBackend` using the [`mmpose`](https://github.com/open-mmlab/mmpose)
package and the RTMW-L checkpoint. The engine gains:

- 17 body keypoints (shoulders, elbows, hips, knees, ankles…)
- 21+21 hand keypoints (same indices as MediaPipe — easy parity)
- 68 face keypoints
- Native multi-person support (uses YOLO11 detector internally)

New dependency cost: `mmpose`, `mmcv`, `mmengine`, `mmdet` — heavy install
but stable wheels exist. Add to `requirements.txt` behind an `[advanced]`
extra so the lightweight install still works.

This phase touches:
- `requirements.txt` — `mmpose>=1.3.0`, `mmcv>=2.0.1`, `mmdet>=3.0`
- `video_analysis_engine.py` — new `RtmwBackend`, default backend selection
  logic (env var / CLI flag for now)
- `hand_heatmap_modern.py` — connection map for new keypoints (the
  `HAND_CONNECTIONS` frozenset there only covers hand; add `BODY_CONNECTIONS`)
- Heatmap visualiser — extend to render body skeleton + hands; nothing to
  remove

**Outcome:** existing hand analyses keep working, plus body & face data.

### Phase 2 — Sapiens-Seg body-part map

Add `SapiensSegmenter` running on Sapiens-Seg-0.3B (smallest variant, fits
in 4 GB VRAM). Don't run it every frame — sample every 5th, store the most
recent map. Each pixel is one of 28 classes; we save a `uint8` map per
sampled frame.

For each detected keypoint, look up the pixel value → human-readable label
("left_arm", "right_hand", "torso", "right_leg" …). Store the label on the
keypoint and surface it in the UI tooltip ("left wrist · right_arm region").

This phase touches:
- `requirements.txt` — Sapiens dependencies via the [ibaiGorordo PyTorch
  port](https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference) (lighter
  than the full Meta repo)
- `video_analysis_engine.py` — new `SapiensSegmenter`, store
  `body_part_map: np.ndarray | None` on each `FrameResult`
- Save format — bump version, store sampled maps as PNG (zlib) inside the
  session bundle to keep file size sane

**Outcome:** every keypoint event carries a body-part label without manual
annotation.

### Phase 3 — Object tracking with SAM 2 + YOLO11

`ObjectTracker` runs YOLO11 every frame to find objects (or Grounding DINO
when the user wants to track something specific by name) and feeds the
bounding boxes into SAM 2 for temporally consistent masks. SAM 2's built-in
streaming-memory tracker maintains object IDs across frames.

For each frame we end with:
- `objects: list[ObjectMask]` — `(object_id, class_label, mask, bbox, score)`

This phase touches:
- `requirements.txt` — `ultralytics` (already YOLO11), `sam2`
  ([Ultralytics SAM 2 wrapper](https://docs.ultralytics.com/models/sam-2)
  is easier than Meta's repo)
- `video_analysis_engine.py` — `ObjectTracker`, `FrameResult.objects`

**Outcome:** scene now has named objects with IDs that persist.

### Phase 4 — Interaction fusion & event log

A small pure-Python module sits at the end of the pipeline:

```python
def detect_contacts(frame: FrameResult, prev: ContactState) -> ContactState:
    # For each (hand|foot|elbow|knee) keypoint:
    #   - find nearest object mask (distance transform)
    #   - if distance < TOUCH_PX for K consecutive frames → open event
    #   - if distance > RELEASE_PX for K consecutive frames → close event
    ...
```

State machine is per `(person_id, body_part, object_id)` triple. Output is
appended to a new `interaction_log` list on the engine, with the schema:

```python
{
    "t_start": 12.4, "t_end": 14.1,
    "person_id": 0,
    "body_part": "right_hand",
    "object_label": "cup",
    "object_id": 17,
    "mean_distance_px": 3.2,
    "frames": 51,
}
```

This phase touches:
- `video_analysis_engine.py` — new `interaction_log`, fusion step
- `hand_heatmap_modern.py` — interaction-log panel in the UI; timeline
  marker per event; click-to-seek
- Save format — append `interaction_log` (forwards-compatible: old viewers
  ignore unknown keys)

**Outcome:** the feature the user actually asked for.

### Phase 5 — Optional refinements

- **WiLoR for 3D hands** if/when contact geometry matters more than 2D
  proximity ([WiLoR repo](https://github.com/rolpotamias/WiLoR))
- **HOI transformer** (SL-HOI / Open-Vocabulary HOI) to validate the
  geometric heuristic; useful as a high-confidence override when the
  heuristic is uncertain
- **Grounding DINO text-prompted detection** to let the user say "track the
  guitar" and have it segmented even though it isn't a COCO class
- **Fine-grained gesture vocabulary for RTMW.** Today, when RTMW is the
  active pose backend, gestures are produced by `PoseGestureClassifier`
  ([pose_backends.py](pose_backends.py)) — a small rule-based classifier
  that reproduces the 7 labels MediaPipe's built-in model emits
  (`Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Thumb_Up`, `Thumb_Down`,
  `Victory`, `ILoveYou`). This is parity-preserving but coarse; many real
  hand configurations fall through into the `None` bucket. Two upgrade
  paths once segmentation + HOI are live:

  1. **Train a lightweight MLP on RTMW hand keypoints** (21 × 2 or 21 × 3
     normalized coords → softmax over an expanded label set). Datasets:
     [HaGRID](https://github.com/hukenovs/hagrid) (~552 K samples,
     18 gesture classes) or [Jester](https://www.qualcomm.com/developer/software/jester-dataset)
     for dynamic gestures. A 3-layer MLP trains in minutes on a single
     GPU and runs in < 1 ms per hand. Drop it in beside
     `PoseGestureClassifier` as a second strategy with a confidence
     threshold; fall back to the rule-based classifier when the MLP isn't
     confident.
  2. **Add a dynamic-gesture head** (LSTM / small Transformer) that
     consumes a sliding window of N RTMW frames so the system can label
     swipes, waves, finger-snaps, and other motions the per-frame
     classifier can never see. Output joins `gesture_history_*` with a
     time-span instead of a single frame.

  Either path lives behind the existing backend interface — no engine
  changes required beyond swapping the classifier instance.

---

## 5. Hardware budget

The pipeline targets a wide spread of NVIDIA hardware: from a 6 GB
**RTX 3060 laptop** at the bottom to a 32 GB **RTX 5090** at the top.
Four profiles, auto-selected from `torch.cuda.get_device_properties(0).total_memory`,
let every machine run something useful. All numbers are FP16.

### Profile `lite` — RTX 3060 laptop / RTX 2060 (6 GB)

| Stage | Model | VRAM | Per-frame on RTX 3060M |
|---|---|---|---|
| Detection + tracking | **YOLO11-N** | 0.7 GB | 12 ms |
| Pose | **RTMPose-S whole-body** (or RTMW-S) | 1.2 GB | 35 ms |
| Body-part seg | **Sapiens-Seg-0.3B** every 10th frame | 2.0 GB | ~12 ms amortised |
| Object seg + track | **SAM 2 Tiny (Hiera-T)** | 1.0 GB | 45 ms |
| Fusion | CPU | — | <1 ms |
| **Peak** | | **~5.0 GB** | **~105 ms / frame ≈ 9–10 FPS** |

If 5 GB is still tight (driver overhead, browser open):
- Drop Sapiens-Seg → save 2 GB; use keypoint-only geometric body-part
  labelling.
- Or swap to **SCHP** (Self-Correction Human Parsing) — ~0.8 GB, 20 body
  parts, ~40 ms / frame.
- Or move segmentation to CPU (Sapiens-Seg-0.3B ≈ 1.5 s / sample — fine
  at every 30th frame).

### Profile `standard` — RTX 3060 desktop / 3070 / 4060 (8 GB)

| Stage | Model | VRAM | Per-frame on RTX 4060 |
|---|---|---|---|
| Detection | YOLO11-S | 1.0 GB | 9 ms |
| Pose | **RTMW-M** | 1.6 GB | 28 ms |
| Body-part seg | Sapiens-Seg-0.3B every 5th frame | 2.0 GB | ~17 ms amortised |
| Object seg + track | SAM 2 Hiera-S | 1.5 GB | 30 ms |
| **Peak** | | **~6.5 GB** | **~85 ms / frame ≈ 12 FPS** |

### Profile `full` — RTX 4070+ / 5070 / 5080 (12–16 GB)

| Stage | Model | VRAM | Per-frame on RTX 4070 / 5080 |
|---|---|---|---|
| Detection | YOLO11-M | 1.5 GB | 8 ms |
| Pose | **RTMW-L** (whole-body, 133 keypoints) | 2.0 GB | 25 ms / 18 ms |
| Body-part seg | **Sapiens-Seg-0.6B** every 5th frame | 3.5 GB | ~14 ms amortised |
| Object seg + track | **SAM 2 Hiera-B** | 2.0 GB | 28 ms / 20 ms |
| 3D hand mesh (opt-in) | **WiLoR small** (175 FPS class) | 1.5 GB | 6 ms |
| **Peak** | | **~10.5 GB** | **~80 ms / frame ≈ 12–14 FPS** |

WiLoR fits in this profile because it's tiny and very fast — gives you
MANO 3D hand mesh on top of the 2D keypoints, useful for precise contact
geometry. Run on demand (e.g. only on frames where a 2D hand keypoint is
within `TOUCH_PX` of an object mask) to keep frame time low.

### Profile `max` — RTX 4090 / 5090 (24–32 GB)

Everything on at once, larger backbones throughout, and **HaMeR** added
alongside WiLoR — WiLoR drives real-time speed, HaMeR runs as a
"high-quality refiner" on contact frames.

| Stage | Model | VRAM | Per-frame on RTX 5090 |
|---|---|---|---|
| Detection | YOLO11-L | 2.5 GB | 6 ms |
| Pose | **RTMW-X / Sapiens-Pose-1B** | 4.0 GB | 18 ms |
| Body-part seg | **Sapiens-Seg-1B** every 3rd frame | 5.0 GB | ~12 ms amortised |
| Object seg + track | **SAM 2 Hiera-L** | 3.5 GB | 14 ms |
| 3D hand mesh (real-time) | **WiLoR small** every frame | 1.5 GB | 5 ms |
| 3D hand refinement (on contact) | **HaMeR** on contact frames only | 3.5 GB | ~80 ms / refined frame |
| Surface normals / depth (opt-in) | **Sapiens-Depth-0.6B** every 10th frame | 3.5 GB | ~10 ms amortised |
| **Peak (without HaMeR refine)** | | **~20 GB** | **~55 ms / frame ≈ 18 FPS** |
| **Peak (HaMeR active)** | | **~24 GB** | **~135 ms on contact frames** |

The 5090's 32 GB also lets you keep **Whisper `large-v3` resident in GPU
memory** alongside vision (≈ 4 GB) — useful if you ever want live
audio-aligned analysis instead of the current "transcribe first, then
vision" ordering.

### Auto-profile selection

```python
def select_profile() -> str:
    if not torch.cuda.is_available():
        return "cpu"          # MediaPipe-style fallback, single-person hand only
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 22:    return "max"        # RTX 4090 / 5090
    if vram_gb >= 11:    return "full"       # 4070 / 5070 / 5080
    if vram_gb >= 7.5:   return "standard"   # 3070 / 4060 / 8 GB cards
    if vram_gb >= 5.5:   return "lite"       # RTX 3060 laptop / 2060
    return "lite-noseg"                       # 4 GB cards: drop Sapiens-Seg
```

Manual override via env var: `VIDEOANALYZER_PROFILE=lite` (or `standard` /
`full` / `max`). Useful for testing, benchmarking, and for users who want
to leave headroom for other GPU apps.

### Where HaMeR vs WiLoR earns its keep

- **WiLoR** is the default 3D hand backend from `full` upward — small,
  fast (130–175 FPS class), MANO output, runs every frame without breaking
  the budget.
- **HaMeR** is heavier and slower but still leads several benchmarks on
  reconstruction quality. On `max`, run it **only on contact frames**
  (frames where the geometric contact heuristic just fired) to get the
  most accurate possible grasp pose for the event log — e.g. for studies
  where the exact finger configuration matters (sign language, surgical
  technique, instrument fingering).
- On `full` and below, WiLoR alone is enough.

### Whisper coexistence

Whisper's `large-v3` worker holds ~4 GB on the GPU while transcribing.

| Profile | Coexistence policy |
|---|---|
| `lite`, `standard` | Transcribe **first**, then run vision (current behaviour — enforce in code) |
| `full` | Same default; concurrent possible if you downgrade Whisper to `medium` (~2 GB) |
| `max` | Concurrent is fine — there's enough VRAM for both even at `large-v3` |

---

## 6. Risks & trade-offs

1. **Heavy install surface.** mmpose + sapiens + sam2 push the venv past 8 GB.
   Mitigate with extras (`pip install videoanalyzer[advanced]`) so the
   default install still works as a hand-only tool.
2. **License watch-out.** Ultralytics is AGPL — fine for end-user research,
   commercial deployment needs a license. Sapiens code is Apache, weights
   are CC-BY-NC. RTMPose / RTMW / SAM 2 are Apache. Document this clearly.
3. **Model downloads are large.** RTMW-L ≈ 200 MB, Sapiens-Seg-0.3B ≈ 600 MB,
   SAM 2 Hiera-S ≈ 180 MB, YOLO11-M ≈ 40 MB. Reuse the existing HF cache +
   resume logic from the Whisper work.
4. **Backwards compatibility for saved sessions.** Add a `schema_version`
   field; old MediaPipe pickles must still load. The Phase 0 interface
   refactor makes this clean.
5. **Sapiens speed.** Even the smallest variant is slower than RTMW. The
   "every Nth frame + carry-forward" trick is essential — never run it per
   frame.
6. **Multi-person ID stability.** The fusion logic relies on stable object
   and person IDs. YOLO11's BoT-SORT is good but not perfect; expect 1–5%
   ID switches on busy scenes. Build the interaction-log UI so an analyst
   can merge IDs after the fact.

---

## 7. What to read next

Core papers / pages, in roughly the order I'd read them:

- [Sapiens — Foundation for Human Vision Models (Meta, ECCV'24)](https://rawalkhirodkar.github.io/sapiens/) — the most important single read
- [RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation](https://arxiv.org/abs/2407.08634)
- [DWPose: Effective Whole-body Pose Estimation with Two-stages Distillation](https://arxiv.org/abs/2307.15880)
- [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/pdf/2408.00714)
- [WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild](https://arxiv.org/html/2409.12259v2)
- [Hand-Object Contact Detection using Grasp Quality Metrics](https://arxiv.org/abs/2501.06987)
- [SL-HOI — Streamlined Open-Vocabulary HOI Detection (DINOv3-based)](https://arxiv.org/abs/2603.27500)
- [Sapiens PyTorch Inference (ibaiGorordo)](https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference) — easiest entry point to Sapiens
- [Ultralytics SAM 2 docs](https://docs.ultralytics.com/models/sam-2) — easiest entry point to SAM 2
- [MMPose repo](https://github.com/open-mmlab/mmpose) — RTMW lives here
- [Grounding DINO on videos (PyImageSearch)](https://pyimagesearch.com/2025/12/08/grounding-dino-open-vocabulary-object-detection-on-videos/) — gentle intro to open-vocab detection

---

## 8. Recommended starting point

If you only want to do one phase: **Phase 1 (RTMW)**. It immediately
unblocks the rest of the body, keeps the existing hand analyses intact, and
the interface refactor in Phase 0 is the door that everything else walks
through. Segmentation and HOI are exciting but only useful once we have
full-body keypoints to anchor them to.
