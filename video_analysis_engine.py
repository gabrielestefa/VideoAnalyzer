import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions, RunningMode
import collections
import logging
import os
import re
import pickle
import math
import multiprocessing
import time
from datetime import datetime
from episode_analyzer import EpisodeAnalyzer, Episode

# Top-level imports for stability
try:
    from moviepy import VideoFileClip
except ImportError:
    VideoFileClip = None

try:
    import torch
except ImportError:
    torch = None

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = None
    FASTER_WHISPER_AVAILABLE = False

try:
    import speech_recognition as sr
except ImportError:
    sr = None

# Suppress TensorFlow/MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Module-level constants ---
FRAME_SAMPLE_RATE = 10        # Store every Nth frame to cap memory usage
HEATMAP_BLUR_KERNEL = 51      # Gaussian blur kernel size (must be odd)
HEATMAP_BLOCK_RADIUS = 5      # Pixel radius of speed contribution per point
AUDIO_CHUNK_SECONDS = 10      # Length of each Google Speech API chunk (seconds)

STOP_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there",
    "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no",
    "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well",
    "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us", "is", "are", "was", "were", "has", "had"
}

class LandmarkCompat:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class VideoAnalysisEngine:
    def __init__(self, model_path):
        self.model_path = model_path
        self.video_path = None
        self.frames = []
        self.heatmap = None
        self.speed_heatmap = None
        self.landmarks_list = []
        self.trace_path_left = []
        self.trace_path_right = []
        self.gesture_history_left = []
        self.gesture_history_right = []
        self.gesture_counts = collections.defaultdict(int)
        
        self.video_width = 0
        self.video_height = 0
        self.fps = 30.0
        self.total_frames = 0
        
        # Episode Analysis
        self.episodes_left = []
        self.episodes_right = []
        self.episode_analyzer = EpisodeAnalyzer()
        
        # Audio/Transcript Data
        self.transcript = []
        self.audio_summary = ""
        
        # Flags
        self.has_hands = False
        
        # Default Filter Settings
        self.min_trace_len = 5
        self.max_jump_pct = 0.10 # 10%

    def load_video(self, path):
        """Open the video file and read basic metadata (dimensions, fps, frame count)."""
        self.video_path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
            
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.fps <= 0: self.fps = 30.0
        
        cap.release()
        return True

    def process_video(self, sensitivity=0.1, progress_callback=None, enable_audio=False):
        """
        Main analysis loop with chunk-based processing for memory efficiency.
        Processes video in 1-minute chunks to handle long videos.
        progress_callback(percent, message) is called periodically.
        """
        if not self.video_path:
            raise ValueError("No video loaded")

        cap = cv2.VideoCapture(self.video_path)
        
        # Calculate chunks
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_secs = total_frames / self.fps if self.fps > 0 else 0
        chunk_duration = 60  # 60 seconds = 1 minute per chunk
        frames_per_chunk = int(chunk_duration * self.fps)
        num_chunks = int(duration_secs / chunk_duration) + 1
        
        if progress_callback:
            progress_callback(0, f"Initializing analysis \n ({num_chunks} chunks, ~1 min each)")
        
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        recognizer = GestureRecognizer.create_from_options(
            GestureRecognizerOptions(
                base_options=base_options,
                running_mode=RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=sensitivity,
                min_hand_presence_confidence=sensitivity,
                min_tracking_confidence=sensitivity
            )
        )

        # Initialize storage
        self.frames = []
        self.landmarks_list = []
        self.gesture_history_left = []
        self.gesture_history_right = []
        self.gesture_counts.clear()
        
        # Audio Data
        self.transcript = []
        self.audio_summary = ""

        frame_index = 0
        current_chunk = 0
        sampled_frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Keep only sampled frames in memory; still run detection on every frame
            if frame_index % FRAME_SAMPLE_RATE == 0:
                self.frames.append(frame.copy())
                sampled_frame_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(frame_index * 1000 / self.fps)
            try:
                result = recognizer.recognize_for_video(mp_image, timestamp_ms)
            except Exception as e:
                logging.warning("Frame %d detection error: %s", frame_index, e)
                frame_index += 1
                continue

            frame_index += 1

            # Progress updates
            if progress_callback and frame_index % 30 == 0:
                current_chunk = frame_index // frames_per_chunk
                chunk_prog = (frame_index % frames_per_chunk) / frames_per_chunk
                overall_percent = int((current_chunk / num_chunks) * 85 + (chunk_prog / num_chunks) * 85)
                progress_callback(overall_percent, f"Processing chunk {current_chunk + 1}/{num_chunks} \n ({sampled_frame_count} frames stored)")

            # Chunk boundary progress update
            if frame_index % frames_per_chunk == 0:
                current_chunk = frame_index // frames_per_chunk
                if progress_callback:
                    progress_callback(int((current_chunk / num_chunks) * 85),
                                    f"Completed chunk {current_chunk}/{num_chunks} \n {sampled_frame_count} frames stored")


            # Extract Data
            gesture_left, gesture_right = None, None
            frame_data = [] # List of (landmarks, label)

            if result.hand_landmarks:
                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    label = result.handedness[i][0].category_name # "Left" or "Right"
                    
                    if result.gestures and len(result.gestures) > i:
                        gesture = result.gestures[i][0]
                        if label == "Left":
                            gesture_left = (gesture.category_name, gesture.score)
                        else:
                            gesture_right = (gesture.category_name, gesture.score)
                    
                    frame_data.append((hand_landmarks, label))
            
            self.gesture_history_left.append(gesture_left)
            self.gesture_history_right.append(gesture_right)

            if gesture_left: self.gesture_counts[f"Left_{gesture_left[0]}"] += 1
            if gesture_right: self.gesture_counts[f"Right_{gesture_right[0]}"] += 1

            self.landmarks_list.append(frame_data)

        cap.release()
        
        # Audio Step
        if enable_audio:
            if progress_callback: progress_callback(90, "Transcribing Audio...")
            self.transcribe_audio()
            
        if progress_callback:
            progress_callback(95, "Analyzing episodes...")
        logging.debug("About to call analyze_episodes")
        self.analyze_episodes()
        logging.debug("analyze_episodes returned")
        
        if progress_callback:
            progress_callback(100, "Analysis complete")

    def transcribe_audio(self, progress_callback=None):
        """
        Extract audio from video using moviepy and transcribe using faster-whisper.
        Falls back to Google if unavailable, but faster-whisper is preferred.
        Uses a separate process for faster-whisper to prevent crashes.
        """
        import tempfile
        import traceback
        temp_audio = None
        try:
            if VideoFileClip is None:
                logging.error("moviepy not installed — cannot extract audio")
                if progress_callback:
                    progress_callback(100, "Error: moviepy not installed")
                return

            if progress_callback:
                progress_callback(5, "Loading video file...")

            clip = VideoFileClip(self.video_path)
            if clip.audio is None:
                logging.warning("No audio track found in %s", self.video_path)
                if progress_callback:
                    progress_callback(100, "No audio track found")
                return

            if progress_callback:
                progress_callback(10, "Extracting audio...")

            # Extract full audio to temp file; capture duration before closing clip.
            duration = clip.duration
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            try:
                clip.audio.write_audiofile(temp_audio, logger=None)
            except OSError as e:
                logging.error("Audio extraction failed: %s", e)
                return
            finally:
                clip.close()

            self.transcript = []

            # Try faster-whisper (Multiprocessing)
            if FASTER_WHISPER_AVAILABLE:
                if progress_callback:
                    progress_callback(15, "Starting transcription process...")

                msg_queue = multiprocessing.Queue()
                p = multiprocessing.Process(target=_worker_whisper_transcribe, args=(temp_audio, "large-v3", msg_queue))
                p.start()

                # Drain the queue WHILE the process runs — calling p.join() first
                # would deadlock if the worker fills the OS pipe buffer before we read.
                import queue as _queue_mod
                while True:
                    try:
                        msg = msg_queue.get(timeout=0.1)
                        if msg[0] == "progress":
                            if progress_callback:
                                progress_callback(msg[1], msg[2])
                        elif msg[0] == "result":
                            self.transcript = msg[1]
                        elif msg[0] == "error":
                            logging.warning("Transcription worker error: %s", msg[1])
                            self.transcript = []
                    except _queue_mod.Empty:
                        if not p.is_alive():
                            break

                p.join()

                # Final drain for any messages that arrived in the last window
                while not msg_queue.empty():
                    try:
                        msg = msg_queue.get_nowait()
                        if msg[0] == "result":
                            self.transcript = msg[1]
                        elif msg[0] == "error":
                            logging.warning("Transcription worker error: %s", msg[1])
                    except Exception:
                        break

                if self.transcript:
                    logging.info("[faster-whisper] Transcribed %d sentences", len(self.transcript))
                    if progress_callback: progress_callback(95, "Generating summary...")
                    self.generate_summary()
                    if progress_callback: progress_callback(100, "Transcription complete")
                    return

            # Fallback to Google Speech API
            if not sr:
                logging.error("speech_recognition not installed — no fallback available")
                return

            if progress_callback: progress_callback(15, "Using Google Speech API (fallback)...")

            chunk_len = AUDIO_CHUNK_SECONDS
            r = sr.Recognizer()
            total_chunks = int(duration / chunk_len) + 1

            for chunk_idx, t_start in enumerate(range(0, int(duration), chunk_len)):
                t_end = min(t_start + chunk_len, duration)
                chunk_progress = 15 + int((chunk_idx / total_chunks) * 70)
                if progress_callback: progress_callback(chunk_progress, f"Transcribing chunk {chunk_idx + 1}/{total_chunks}")

                clip_temp = VideoFileClip(self.video_path)
                sub = clip_temp.subclipped(t_start, t_end)
                temp_chunk = os.path.join(tempfile.gettempdir(), f"temp_chunk_{t_start}.wav")
                try:
                    sub.audio.write_audiofile(temp_chunk, logger=None)
                    clip_temp.close()
                    with sr.AudioFile(temp_chunk) as source:
                        audio_data = r.record(source)
                        text = r.recognize_google(audio_data)
                        if text:
                            self.transcript.append((t_start, t_end, text, "Neutral"))
                except sr.UnknownValueError:
                    pass  # No speech detected in this chunk
                except (OSError, sr.RequestError) as e:
                    logging.warning("Chunk %d transcription failed: %s", chunk_idx, e)
                finally:
                    clip_temp.close()
                    if os.path.exists(temp_chunk):
                        os.remove(temp_chunk)

            # Classify sentiment per segment before summary
            if self.transcript:
                if progress_callback: progress_callback(92, "Classifying sentiment per segment...")
                self.classify_segment_sentiments()

            if progress_callback: progress_callback(95, "Generating summary...")
            self.generate_summary()
            if progress_callback: progress_callback(100, "Transcription complete")

        except Exception as e:
            logging.error("Transcription failed: %s", e, exc_info=True)
        finally:
            if temp_audio and os.path.exists(temp_audio):
                os.remove(temp_audio)
        logging.debug("transcribe_audio finished")


    def classify_segment_sentiments(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Run a real sentiment classifier on each transcript segment and overwrite
        the placeholder "Neutral" label. Stores numeric score as a 5th tuple element.

        Transcript tuple shape after this call: (start, end, text, label, score_float)
        where score_float is in [-1, 1] (negative ↔ positive).

        Uses CPU by default to avoid contention with whisper/Qwen on the GPU.
        Falls back silently if transformers/model unavailable.
        """
        if not self.transcript:
            return
        try:
            from transformers import pipeline
            import torch
        except ImportError:
            logging.warning("transformers not installed — skipping sentiment classification")
            return

        try:
            device = 0 if torch.cuda.is_available() else -1
            clf = pipeline(
                "sentiment-analysis", model=model_name,
                device=device, top_k=None, truncation=True, max_length=256,
            )
        except Exception as e:
            logging.warning("Could not load sentiment model %s: %s", model_name, e)
            return

        texts = [t[2] for t in self.transcript if len(t) > 2 and t[2]]
        if not texts:
            return

        try:
            results = clf(texts, batch_size=16)
        except Exception as e:
            logging.warning("Sentiment inference failed: %s", e)
            return

        # cardiffnlp output: list of [{label: 'positive'|'neutral'|'negative', score: 0..1}, ...]
        new_transcript = []
        for entry, res in zip(self.transcript, results):
            start, end, text = entry[0], entry[1], entry[2]
            # Pick the top-scoring label
            top = max(res, key=lambda r: r["score"]) if isinstance(res, list) else res
            label = top["label"].lower()
            confidence = float(top["score"])
            # Map to signed score in [-1, 1] weighted by confidence
            if "pos" in label:
                signed = confidence
            elif "neg" in label:
                signed = -confidence
            else:
                signed = 0.0
            new_transcript.append((start, end, text, label, signed))
        self.transcript = new_transcript
        logging.info("Sentiment classification complete (%d segments)", len(new_transcript))


    def generate_summary(self, classifier=None):
        if not self.transcript: return
        
        all_text = " ".join([t[2] for t in self.transcript])
        words = [w.lower() for w in all_text.split() if len(w) > 3]
        common = collections.Counter(words).most_common(5)
        
        summary = f"Total Words: {len(words)}\n"
        
        # Overall Sentiment
        if classifier and all_text:
             try:
                 res = classifier.classify(all_text[:500]) # BERT limit usually 512 tokens
                 if res.classifications:
                     top = res.classifications[0].categories[0]
                     summary += f"Overall Sentiment: {top.category_name} ({top.score:.2f})\n"
             except Exception as e:
                 logging.warning("Sentiment classification failed: %s", e)

        summary += "Top Keywords: " + ", ".join([f"{w}({c})" for w, c in common])
        self.audio_summary = summary

    def get_context_at(self, seconds):
        """Return gesture and transcript context for the given video timestamp (in seconds)."""
        idx = int(seconds * self.fps)
        if idx >= len(self.frames): return None
        
        g_l = self.gesture_history_left[idx][0] if idx < len(self.gesture_history_left) and self.gesture_history_left[idx] and len(self.gesture_history_left[idx]) > 0 else "None"
        g_r = self.gesture_history_right[idx][0] if idx < len(self.gesture_history_right) and self.gesture_history_right[idx] and len(self.gesture_history_right[idx]) > 0 else "None"
        
        spoken = ""
        # Transcript tuple now has 4 elements: (start, end, text, category)
        for t in self.transcript:
            start, end, text = t[0], t[1], t[2]
            # optional category is t[3]
            if start <= seconds < end:
                spoken = text
                break
                
        return {
            "time": seconds,
            "gesture_left": g_l,
            "gesture_right": g_r,
            "spoken": spoken
        }

    def filter_trace(self, raw_points):
        # raw_points is list of (x, y) or None. Coordinates are PIXEL coords.
        if not raw_points: return []

        diag = (self.video_width**2 + self.video_height**2)**0.5
        jump_thresh_px = self.max_jump_pct * diag

        filtered = []
        segment = []

        for i, p in enumerate(raw_points):
            if p is None:
                # Break segment
                if len(segment) >= self.min_trace_len:
                    filtered.extend(segment)
                else:
                    filtered.extend([None] * len(segment)) # Replace short segment with Nones
                
                filtered.append(None)
                segment = []
                continue

            # check jump from previous Valid Point in Segment
            if segment:
                prev = segment[-1]
                dist = ((p[0] - prev[0])**2 + (p[1] - prev[1])**2)**0.5
                if dist > jump_thresh_px:
                    # Jump detected -> End previous segment
                    if len(segment) >= self.min_trace_len:
                        filtered.extend(segment)
                    else:
                        filtered.extend([None] * len(segment))
                    
                    segment = [p]
                else:
                    segment.append(p)
            else:
                segment.append(p)

        # Flush last segment
        if len(segment) >= self.min_trace_len:
            filtered.extend(segment)
        else:
            filtered.extend([None] * len(segment))

        return filtered

    def regenerate_data(self, source="Whole Hand", min_len=5, max_jump_pct=0.10):
        """Recompute heatmaps and trace paths from stored landmarks (called after load or settings change)."""
        self.min_trace_len = min_len
        self.max_jump_pct = max_jump_pct
        
        indices = []
        trace_idx = 0 

        if source == "Whole Hand":
            indices = list(range(21))
            trace_idx = -3 # Palm Center
        elif source == "Wrist":
            indices = [0]
            trace_idx = 0
        elif source == "Thumb":
            indices = [1, 2, 3, 4]
            trace_idx = 4 # Tip
        elif source == "Index":
            indices = [5, 6, 7, 8]
            trace_idx = 8
        elif source == "Middle":
            indices = [9, 10, 11, 12]
            trace_idx = 12
        elif source == "Ring":
            indices = [13, 14, 15, 16]
            trace_idx = 16
        elif source == "Pinky":
            indices = [17, 18, 19, 20]
            trace_idx = 20
        elif source == "Median":
            indices = list(range(21))
            trace_idx = -1 

        self.heatmap = np.zeros((self.video_height, self.video_width), dtype=np.float32)
        self.speed_heatmap = np.zeros((self.video_height, self.video_width), dtype=np.float32)
        self.trace_path_left = []
        self.trace_path_right = []

        for frame_data in self.landmarks_list:
            trace_l, trace_r = None, None

            for hand_landmarks, label in frame_data:
                # Heatmap
                points = []
                for idx in indices:
                    lm = hand_landmarks[idx]
                    x = int(lm.x * self.video_width)
                    y = int(lm.y * self.video_height)
                    points.append((x, y))
                
                if points:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    min_x, max_x = max(0, min(xs)), min(self.video_width, max(xs))
                    min_y, max_y = max(0, min(ys)), min(self.video_height, max(ys))
                    
                    if max_x == min_x: max_x += 1
                    if max_y == min_y: max_y += 1
                    self.heatmap[min_y:max_y, min_x:max_x] += 1

                # Trace
                wrist = hand_landmarks[0]
                mcp = hand_landmarks[9]
                scale = ((wrist.x - mcp.x)**2 + (wrist.y - mcp.y)**2 + (wrist.z - mcp.z)**2)**0.5
                if scale == 0: scale = 0.001
                z_proxy = 1.0 / scale 
                
                if trace_idx == -1:
                    # Median of all points
                    mean_x = np.mean([lm.x for lm in hand_landmarks])
                    mean_y = np.mean([lm.y for lm in hand_landmarks])
                    tx = int(mean_x * self.video_width)
                    ty = int(mean_y * self.video_height)
                elif trace_idx == -2:
                    # Bounding Box Center (Deprecated/Unused now but kept for logic safety?)
                    # Replaced by Palm Center (-3)
                    pass
                elif trace_idx == -3:
                    # Palm Center: Avg of 0, 5, 9, 13, 17
                    palm_indices = [0, 5, 9, 13, 17]
                    px = np.mean([hand_landmarks[i].x for i in palm_indices])
                    py = np.mean([hand_landmarks[i].y for i in palm_indices])
                    tx = int(px * self.video_width)
                    ty = int(py * self.video_height)
                else:
                    t_lm = hand_landmarks[trace_idx]
                    tx = int(t_lm.x * self.video_width)
                    ty = int(t_lm.y * self.video_height)
                
                if label == "Left": trace_l = (tx, ty, z_proxy)
                else: trace_r = (tx, ty, z_proxy)

            self.trace_path_left.append(trace_l)
            self.trace_path_right.append(trace_r)

        self.trace_path_left = self.filter_trace(self.trace_path_left)
        self.trace_path_right = self.filter_trace(self.trace_path_right)

        # Speed Heatmap
        for path in [self.trace_path_left, self.trace_path_right]:
            for i in range(1, len(path)):
                p_prev, p_curr = path[i-1], path[i]
                if p_prev is not None and p_curr is not None:
                    dist = np.sqrt((p_curr[0]-p_prev[0])**2 + (p_curr[1]-p_prev[1])**2)
                    x, y = int(p_curr[0]), int(p_curr[1])
                    if 0 <= x < self.video_width and 0 <= y < self.video_height:
                         # Simple block addition
                         self.speed_heatmap[max(0, y - HEATMAP_BLOCK_RADIUS):min(self.video_height, y + HEATMAP_BLOCK_RADIUS),
                                            max(0, x - HEATMAP_BLOCK_RADIUS):min(self.video_width, x + HEATMAP_BLOCK_RADIUS)] += dist

        if np.max(self.heatmap) > 0:
            self.heatmap = cv2.GaussianBlur(self.heatmap, (HEATMAP_BLUR_KERNEL, HEATMAP_BLUR_KERNEL), 0)
        if np.max(self.speed_heatmap) > 0:
            self.speed_heatmap = cv2.GaussianBlur(self.speed_heatmap, (HEATMAP_BLUR_KERNEL, HEATMAP_BLUR_KERNEL), 0)

    def calculate_metrics(self):
        metrics = {
            'left': {'dist': 0, 'max_speed': 0, 'avg_speed': 0, 'speeds': []},
            'right': {'dist': 0, 'max_speed': 0, 'avg_speed': 0, 'speeds': []}
        }
        
        def smooth_data(data, window_size=5):
            if not data: return []
            result = []
            for i in range(len(data)):
                s, e = max(0, i - window_size // 2), min(len(data), i + window_size // 2 + 1)
                result.append(np.median(data[s:e]))
            return result

        for side, path in [('left', self.trace_path_left), ('right', self.trace_path_right)]:
            total_dist = 0
            speeds = []
            for i in range(1, len(path)):
                p1, p2 = path[i-1], path[i]
                if p1 is None or p2 is None:
                    speeds.append(0)
                    continue
                d = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
                total_dist += d
                speeds.append(d)
                
            if speeds:
                smoothed = smooth_data(speeds)
                metrics[side]['dist'] = total_dist
                metrics[side]['max_speed'] = max(smoothed) if smoothed else 0
                metrics[side]['avg_speed'] = sum(smoothed) / len(smoothed) if smoothed else 0
                metrics[side]['speeds'] = smoothed
                
        return metrics

    def calculate_hmi_metrics(self, side='left'):
        # HMI Analytics: Phase Segmentation, Tortuosity, Tremor
        path = self.trace_path_left if side == 'left' else self.trace_path_right
        if not path or len(path) < 2: return None
        
        # 1. Calculate Velocity & Acceleration & Jerk
        velocities = []
        accelerations = []
        jerks = []
        phases = [] # 0=Rest, 1=Reach, 2=Hover, 3=Touch
        
        dt = 1.0 / self.fps
        
        # Fill missing data filtering Nones
        clean_path = []
        indices = []
        for i, p in enumerate(path):
            if p is not None:
                clean_path.append(np.array(p))
                indices.append(i)
        
        if len(clean_path) < 5: return None

        # Calculate derivatives
        velocities = []
        for i in range(1, len(clean_path)):
            v = np.linalg.norm(clean_path[i] - clean_path[i-1]) / dt
            velocities.append(v)
            
        accelerations = []
        for i in range(1, len(velocities)):
            a = (velocities[i] - velocities[i-1]) / dt
            accelerations.append(a)
            
        jerks = []
        for i in range(1, len(accelerations)):
            j = abs((accelerations[i] - accelerations[i-1]) / dt)
            jerks.append(j)

        # Pad arrays
        velocities = [0] + velocities
        accelerations = [0]*2 + accelerations
        jerks = [0]*3 + jerks
        
        # Phase Segmentation & Hover Extraction
        max_v = max(velocities) if velocities else 1
        hover_points = []
        phases = []
        
        for i, v in enumerate(velocities):
            # Phase Logic:
            # 0=Rest (<10% max speed)
            # 1=Reach (>40% max speed)
            # 2=Hover (In between, active but slow)
            if v < max_v * 0.1:
                p = 0
            elif v > max_v * 0.4:
                p = 1
            else:
                p = 2
                # Store hover coord
                if i < len(clean_path):
                    hover_points.append(clean_path[i])
            phases.append(p)

        # Map phases back to original video timeline
        full_phases = [-1] * len(path) # -1 = No Hand / Unknown
        for k, phase_val in enumerate(phases):
            if k < len(indices):
                original_idx = indices[k]
                full_phases[original_idx] = phase_val
        
        # Path Efficiency (Tortuosity)
        # Calculate for each "Reach" segment
        tortuosity_scores = []
        current_segment_dist = 0
        segment_start_idx = -1
        
        for i in range(1, len(phases)):
            if phases[i] == 1: # Reaching
                if phases[i-1] != 1: # Start of reach
                    segment_start_idx = i-1
                    current_segment_dist = 0
                
                # Add step distance
                step = np.linalg.norm(clean_path[i] - clean_path[i-1])
                current_segment_dist += step
                
            elif phases[i] != 1 and phases[i-1] == 1: # End of reach
                if segment_start_idx >= 0:
                    start_pt = clean_path[segment_start_idx]
                    end_pt = clean_path[i]
                    displacement = np.linalg.norm(end_pt - start_pt)
                    if displacement > 10: # Ignore tiny moves
                        t = current_segment_dist / displacement
                        tortuosity_scores.append(t)
                segment_start_idx = -1

        avg_efficiency = np.mean(tortuosity_scores) if tortuosity_scores else 1.0
        avg_jerk = np.mean(jerks) if jerks else 0

        return {
            'phases': full_phases, # Now correctly aligned with video frames
            'counts': collections.Counter(phases),
            'hover_points': hover_points,
            'efficiency': avg_efficiency,
            'avg_jerk': avg_jerk,
            'clean_path': clean_path # for plotting hover map context
        }

    def save_analysis(self):
        """Serialize the current analysis to a timestamped .pkl file in the Library directory."""
        # Allow saving if we have EITHER hand data OR transcript data
        if not self.landmarks_list and not self.transcript:
            return False, "No analysis data (hands or audio) to save."
            
        try:
            # Create Library directory
            lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Library")
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)
                
            # Generate Filename
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{video_name}_Analysis_{date_str}.pkl"
            filepath = os.path.join(lib_dir, filename)
            
            # Serialize Landmarks (MediaPipe objects not picklable)
            serializable_landmarks = []
            for frame_data in self.landmarks_list:
                # frame_data is list of (landmarks, label)
                s_frame = []
                for lms, label in frame_data:
                    # Convert NormalizedLandmarkList to list of dicts
                    lms_data = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in lms]
                    s_frame.append((lms_data, label))
                serializable_landmarks.append(s_frame)
            
            # Determine flags
            has_hands = len(self.landmarks_list) > 0
            
            # Construct Data Payload
            data = {
                'version': 1.1, # Bump version for has_hands support
                'timestamp': datetime.now().isoformat(),
                'video_path': self.video_path,
                'video_width': self.video_width,
                'video_height': self.video_height,
                'fps': self.fps,
                'total_frames': self.total_frames,
                'landmarks': serializable_landmarks,
                'transcript': self.transcript,
                'audio_summary': self.audio_summary,
                'gesture_counts': dict(self.gesture_counts),
                'has_hands': has_hands 
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
                
            return True, filepath
            
        except Exception as e:
            return False, str(e)

    def load_analysis(self, filepath):
        if not os.path.exists(filepath):
            return False, "File not found"

        # Security: pickle can execute arbitrary code — only load files from the
        # trusted Library directory. Do not expose this path to untrusted input.
        library_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Library"))
        if not os.path.realpath(filepath).startswith(library_dir):
            return False, "Refusing to load analysis from outside the Library directory"

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            # Restore Basic Props
            self.video_path = data.get('video_path', "")
            self.video_width = data.get('video_width', 640)
            self.video_height = data.get('video_height', 480)
            self.fps = data.get('fps', 30.0)
            self.total_frames = data.get('total_frames', 0)
            self.transcript = data.get('transcript', [])
            self.audio_summary = data.get('audio_summary', "")
            self.gesture_counts = collections.defaultdict(int, data.get('gesture_counts', {}))
            
            # Restore Flags
            # If 'has_hands' is missing (old files), assume True if landmarks exist (handled below), 
            # but we set the attribute here to be safe.
            # We'll refine it after loading landmarks.
            explicit_has_hands = data.get('has_hands', None)
            
            # Reconstruct Landmarks
            # Saved as list of list of ( [{'x':...}, label] )
            # Need to convert back to objects with .x, .y attributes
            self.landmarks_list = []
            
            raw_landmarks = data.get('landmarks', [])
            for frame_rows in raw_landmarks:
                reconstructed_frame = []
                for lms_data, label in frame_rows:
                    lms_objs = []
                    for lm_dict in lms_data:
                        lms_objs.append(LandmarkCompat(lm_dict['x'], lm_dict['y'], lm_dict['z']))
                    reconstructed_frame.append((lms_objs, label))
                self.landmarks_list.append(reconstructed_frame)
            
            # Set has_hands flag
            if explicit_has_hands is not None:
                self.has_hands = explicit_has_hands
            else:
                # Fallback for old files: if landmarks exist, it has hands
                self.has_hands = len(self.landmarks_list) > 0
            
            # Rebuild Frames (Video might not be loaded, but we have data)
            # If video exists, try to load it for preview?
            # Ideally yes, but if moved, we just use blank canvas or try
            if os.path.exists(self.video_path):
                 self.load_video(self.video_path) 
                 # This resets everything! Danger. 
                 # load_video clears self.landmarks_list.
                 # We must NOT call load_video fully or we must load video FIRST then override data.
                 # But load_video clears data.
                 
                 # Workaround: Just open capture for frame reading but don't reset data
                 pass
            
            # Regenerate Computed Data (Heatmaps, Traces)
            self.regenerate_data()
            
            return True, "Analysis loaded successfully"
            
        except Exception as e:
            return False, str(e)

    # --- Comparison Helpers ---
    
    def get_clean_words(self):
        """Returns a set of unique keywords from the transcript."""
        text_content = " ".join([entry[2] for entry in self.transcript]).lower()
        # Remove punctuation
        text_content = re.sub(r'[^\w\s]', '', text_content)
        words = text_content.split()
        return {w for w in words if w not in STOP_WORDS and len(w) > 2}

    def get_text_embedding(self):
        """Generates embedding vector for the entire transcript using USE."""
        full_text = " ".join([entry[2] for entry in self.transcript])
        if not full_text: return None
        
        # Load Embedder if possible (Lazy)
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import text
            
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models", "universal_sentence_encoder.tflite")
            if not os.path.exists(model_path):
                logging.warning("universal_sentence_encoder.tflite not found at %s — text embedding unavailable", model_path)
                return None
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = text.TextEmbedderOptions(base_options=base_options)
            embedder = text.TextEmbedder.create_from_options(options)
            
            result = embedder.embed(full_text)
            return result.embeddings[0].embedding.tolist() # List of floats
        except Exception as e:
            logging.warning("Text embedding error: %s", e)
            return None

    @staticmethod
    def cosine_similarity(v1, v2):
        if not v1 or not v2: return 0.0
        dot = sum(a*b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a*a for a in v1))
        norm2 = math.sqrt(sum(b*b for b in v2))
        if norm1 == 0 or norm2 == 0: return 0.0
        return dot / (norm1 * norm2)

    def get_normalized_points(self, source='Whole Hand'):
        """Returns list of (x, y) tuples normalized 0-1 for all active frames."""
        points = []
        # Landmark Indices (approximate mapping)
        # 0=Wrist, 4=ThumbTip, 8=IndexTip, 12=MidTip, 16=RingTip, 20=PinkyTip
        # 9=MiddleMCP (Center of hand roughly)
        
        idx = -1 # Default to center
        if source == "Wrist": idx = 0
        elif source == "Thumb": idx = 4
        elif source == "Index": idx = 8
        elif source == "Median": idx = 12 # Middle finger tip
        
        for frame_data in self.landmarks_list:
            # frame_data: list of (landmarks, label)
            for lms, label in frame_data:
                if not lms: continue
                
                x, y = 0, 0
                if source == "Whole Hand" or idx == -1:
                    # Average of Wrist(0) and MiddleMCP(9)
                    if len(lms) > 9:
                        x = (lms[0].x + lms[9].x) / 2
                        y = (lms[0].y + lms[9].y) / 2
                    elif len(lms) > 0:
                         x, y = lms[0].x, lms[0].y
                else:
                    if len(lms) > idx:
                        x, y = lms[idx].x, lms[idx].y
                        
                points.append((x, y))
        return points

    def get_sentiment_stats(self):
        """Returns sentiment distribution dictionary."""
        stats = {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Total': 0}
        
        for entry in self.transcript:
            # entry: (start, end, text, category_str)
            # category_str example: "joy (0.9)" or "negative (0.8)"
            cat_str = entry[3].lower() if len(entry) > 3 else "neutral"
            
            # Simple keyword matching for common sentiment labels
            if "joy" in cat_str or "positive" in cat_str:
                stats['Positive'] += 1
            elif "sadness" in cat_str or "anger" in cat_str or "negative" in cat_str:
                stats['Negative'] += 1
            else:
                stats['Neutral'] += 1
            stats['Total'] += 1
            
        return stats

    
    # --- Episode Analysis Methods ---
    
    def analyze_episodes(self):
        """Analyze interaction episodes for both hands"""
        # Generate timestamps array
        timestamps = np.array([i / self.fps for i in range(max(len(self.trace_path_left), len(self.trace_path_right)))])
        
        # Prepare transcript for alignment
        transcript_tuples = [(t[0], t[1], t[2]) for t in self.transcript] if self.transcript else None
        
        # Analyze left hand
        if self.trace_path_left and len(self.trace_path_left) > 0:
            try:
                self.episodes_left = self.episode_analyzer.analyze_trace(
                    self.trace_path_left,
                    timestamps[:len(self.trace_path_left)],
                    hand='left',
                    transcript=transcript_tuples
                )
            except Exception as e:
                logging.error("Left hand episode analysis failed: %s", e, exc_info=True)
                self.episodes_left = []

        logging.debug("analyze_episodes finished left hand. Found %d episodes.", len(self.episodes_left) if self.episodes_left else 0)

        # Analyze right hand
        if self.trace_path_right and len(self.trace_path_right) > 0:
            try:
                self.episodes_right = self.episode_analyzer.analyze_trace(
                    self.trace_path_right,
                    timestamps[:len(self.trace_path_right)],
                    hand='right',
                    transcript=transcript_tuples
                )
            except Exception as e:
                logging.error("Right hand episode analysis failed: %s", e, exc_info=True)
                self.episodes_right = []
    
    def get_episode_metrics(self):
        """Calculate aggregate episode-level metrics"""
        metrics = {
            'left': {'count': 0, 'avg_duration': 0, 'avg_confidence': 0, 'types': {}},
            'right': {'count': 0, 'avg_duration': 0, 'avg_confidence': 0, 'types': {}}
        }
        
        for side, episodes in [('left', self.episodes_left), ('right', self.episodes_right)]:
            if not episodes:
                continue
                
            metrics[side]['count'] = len(episodes)
            metrics[side]['avg_duration'] = np.mean([ep.duration for ep in episodes])
            metrics[side]['avg_confidence'] = np.mean([ep.confidence for ep in episodes])
            
            # Count episode types
            type_counts = collections.Counter([ep.episode_type for ep in episodes])
            metrics[side]['types'] = dict(type_counts)
        
        return metrics
    
    
    def get_episode_at(self, time_seconds, hand='left'):
        """Get episode information at specific timestamp"""
        episodes = self.episodes_left if hand == 'left' else self.episodes_right
        
        for ep in episodes:
            if ep.start_time <= time_seconds <= ep.end_time:
                return ep
        return None
    
    def generate_audio_wheel_html(self, output_dir=None):
        """
        Generate interactive audio wheel analyzer HTML.
        
        Args:
            output_dir: Optional directory for output files (uses temp if None)
        
        Returns:
            Path to the generated HTML file, or None if no transcript available
        """
        if not self.transcript or len(self.transcript) == 0:
            return None
        
        
        from audio_wheel_analyzer import generate_audio_wheel_html
        return generate_audio_wheel_html(self.video_path, self.transcript, output_dir)

# --- Multiprocessing Worker ---
def _worker_whisper_transcribe(audio_path, model_size, queue):
    """
    Independent worker function to run faster-whisper in a separate process.
    This isolates CTranslate2/CUDA context to prevent crashes in the main app.
    """
    try:
        # Re-import essential libraries in the subprocess 
        import torch
        import gc
        
        # Check availability again within the process
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            queue.put(("error", "faster-whisper not installed in subprocess"))
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        queue.put(("progress", 20, f"Loading '{model_size}' model on {device}..."))
        
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=4, num_workers=2)
        except Exception as e:
            queue.put(("progress", 20, f"Failed to load {model_size}, trying medium..."))
            model = WhisperModel("medium", device=device, compute_type=compute_type, cpu_threads=4, num_workers=2)

        queue.put(("progress", 25, "Transcribing audio..."))
        
        # Run transcription — batch_size feeds multiple chunks to the GPU simultaneously,
        # significantly increasing utilization on longer audio files.
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            batch_size=16 if device == "cuda" else 1,
        )

        # Process segments (Aggregation)
        all_words = []
        total_duration = info.duration
        
        for i, segment in enumerate(segments):
            if segment.words:
                all_words.extend(segment.words)
            
            # Progress update
            current_time = segment.end
            if total_duration > 0 and i % 5 == 0:
                 prog = 25 + int((current_time / total_duration) * 65)
                 queue.put(("progress", prog, f"Transcribing... {int(current_time)}s / {int(total_duration)}s"))

        queue.put(("progress", 90, f"Aggregating {len(all_words)} words..."))

        # Reconstruct sentences
        transcript_result = []
        current_sentence_words = []
        sentence_start = 0.0
        
        for i, word_data in enumerate(all_words):
            word_text = word_data.word.strip()
            t_start = word_data.start
            t_end = word_data.end
            
            if not word_text: continue
                
            if not current_sentence_words:
                sentence_start = t_start
            
            current_sentence_words.append(word_text)
            
            # Split logic
            ends_sentence = word_text.endswith(('.', '!', '?'))
            
            has_long_gap = False
            if i < len(all_words) - 1:
                next_start = all_words[i+1].start
                if next_start - t_end > 1.0: # Reduced from 2.0
                    has_long_gap = True
            
            current_len = sum(len(w) + 1 for w in current_sentence_words)
            is_too_long = current_len > 250
            
            should_split = False
            if ends_sentence: 
                should_split = True
            elif has_long_gap and current_len > 30: 
                should_split = True
            elif is_too_long: 
                should_split = True # Force split if too long
                
            if should_split or i == len(all_words) - 1:
                final_text = " ".join(current_sentence_words)
                transcript_result.append((sentence_start, t_end, final_text, "Neutral"))
                current_sentence_words = []
                sentence_start = 0.0
        
        # Done!
        queue.put(("result", transcript_result))
        
        # Explicit cleanup (safe here because process will exit)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        queue.put(("error", str(e)))


