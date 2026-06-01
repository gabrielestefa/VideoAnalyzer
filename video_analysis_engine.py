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

# Phase 0/1: pluggable pose backends.  See VISION_UPGRADE_PLAN.md and
# pose_backends.py.  Default still MediaPipe — opt into RTMW via
# VIDEOANALYZER_BACKEND=rtmw (requires `pip install -r requirements-rtmw.txt`).
from pose_backends import BackendBase, FrameResult, make_backend

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
    try:
        # BatchedInferencePipeline is required for batched CUDA inference.
        # Available in faster-whisper >= 1.0.0; older versions degrade gracefully.
        from faster_whisper import BatchedInferencePipeline
    except ImportError:
        BatchedInferencePipeline = None
except ImportError:
    WhisperModel = None
    BatchedInferencePipeline = None
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


def _format_duration(seconds: float) -> str:
    """Human-readable duration, e.g. '45s', '2m 14s', '1h 3m'."""
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


# Whole-body heatmap/trace sources, available when a whole-body backend
# (RTMW) populated body_landmarks_list.  COCO-17 indices:
#   0 nose, 5/6 shoulders, 7/8 elbows, 9/10 wrists,
#   11/12 hips, 13/14 knees, 15/16 ankles.
# Value = (heatmap_indices, left_trace_idx, right_trace_idx).
BODY_HEATMAP_SOURCES = {
    "Body: Full Pose": (list(range(17)), 9, 10),
    "Body: Arms":      ([5, 7, 9, 6, 8, 10], 9, 10),
    "Body: Legs":      ([11, 13, 15, 12, 14, 16], 15, 16),
    "Body: Head":      ([0, 1, 2, 3, 4], 0, 0),
}

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
    def __init__(self, model_path, backend_name: str | None = None):
        self.model_path = model_path
        # Backend selection: explicit arg → env var → "auto".  "auto" prefers
        # MediaPipe unless RTMW is installed and a GPU profile is selected.
        self.backend_name = (
            backend_name
            or os.environ.get("VIDEOANALYZER_BACKEND")
            or "auto"
        )
        self.backend: BackendBase | None = None  # built lazily in process_video
        self.video_path = None
        self.frames = []
        self.heatmap = None
        self.speed_heatmap = None
        self.landmarks_list = []
        # Phase 1: body / face keypoints from whole-body backends (None for
        # MediaPipe).  Same per-frame indexing as landmarks_list.
        self.body_landmarks_list: list = []
        self.face_landmarks_list: list = []
        # Multi-person tracking (RTMW): each frame stores list of per-person
        # keypoint arrays.  Heatmaps/traces still use the single-subject
        # lists above (body_landmarks_list / face_landmarks_list); the
        # _all variants drive the multi-subject UI overlay.
        self.body_landmarks_list_all: list = []
        self.face_landmarks_list_all: list = []
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
        
        # Build the pose backend.  Sensitivity is forwarded — only the
        # MediaPipe backend uses it; RTMW handles thresholds internally.
        if self.backend is not None:
            try: self.backend.close()
            except Exception: pass
        self.backend = make_backend(self.backend_name, self.model_path,
                                    sensitivity=sensitivity)
        self.backend.initialize(self.video_width, self.video_height, self.fps)
        logger.info("Pose backend active: %s", self.backend.name)

        # Initialize storage
        self.frames = []
        self.landmarks_list = []
        self.body_landmarks_list = []
        self.face_landmarks_list = []
        self.body_landmarks_list_all = []
        self.face_landmarks_list_all = []
        self.gesture_history_left = []
        self.gesture_history_right = []
        self.gesture_counts.clear()
        
        # Audio Data
        self.transcript = []
        self.audio_summary = ""

        frame_index = 0
        current_chunk = 0
        sampled_frame_count = 0

        # ETA tracking — rolling window of (wall_time, frame_index) samples so
        # the estimate reflects recent throughput, not a cold-start average.
        eta_samples = collections.deque(maxlen=10)
        eta_samples.append((time.time(), 0))

        def _eta_text() -> str:
            """Estimate remaining analysis time from recent frame throughput."""
            if total_frames <= 0 or len(eta_samples) < 2:
                return ""
            t_old, f_old = eta_samples[0]
            t_new, f_new = eta_samples[-1]
            dt = t_new - t_old
            df = f_new - f_old
            if dt <= 0 or df <= 0:
                return ""
            fps_proc = df / dt
            remaining = (total_frames - f_new) / fps_proc
            return f" — ETA {_format_duration(remaining)}"

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Keep only sampled frames in memory; still run detection on every frame
            if frame_index % FRAME_SAMPLE_RATE == 0:
                self.frames.append(frame.copy())
                sampled_frame_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(frame_index * 1000 / self.fps)
            try:
                result: FrameResult = self.backend.process_frame(rgb, timestamp_ms)
            except Exception as e:
                logging.warning("Frame %d detection error: %s", frame_index, e)
                frame_index += 1
                # Still append a placeholder so list indices stay aligned with
                # frame timestamps for downstream consumers.
                self.landmarks_list.append([])
                self.gesture_history_left.append(None)
                self.gesture_history_right.append(None)
                if self.backend.supports_body:
                    self.body_landmarks_list.append(None)
                    self.body_landmarks_list_all.append([])
                if self.backend.supports_face:
                    self.face_landmarks_list.append(None)
                    self.face_landmarks_list_all.append([])
                continue

            frame_index += 1

            # Progress updates
            if progress_callback and frame_index % 30 == 0:
                eta_samples.append((time.time(), frame_index))
                current_chunk = frame_index // frames_per_chunk
                chunk_prog = (frame_index % frames_per_chunk) / frames_per_chunk
                overall_percent = int((current_chunk / num_chunks) * 85 + (chunk_prog / num_chunks) * 85)
                progress_callback(overall_percent,
                                  f"Processing chunk {current_chunk + 1}/{num_chunks} "
                                  f"({sampled_frame_count} frames stored){_eta_text()}")

            # Chunk boundary progress update
            if frame_index % frames_per_chunk == 0:
                current_chunk = frame_index // frames_per_chunk
                if progress_callback:
                    progress_callback(int((current_chunk / num_chunks) * 85),
                                    f"Completed chunk {current_chunk}/{num_chunks} \n {sampled_frame_count} frames stored")


            # Extract Data — backend already returned a normalized FrameResult.
            # Shape matches the original (landmarks, label) tuples so all
            # downstream code (regenerate_data, save/load, UI) is unchanged.
            self.landmarks_list.append(result.hands)
            self.gesture_history_left.append(result.gesture_left)
            self.gesture_history_right.append(result.gesture_right)

            if result.gesture_left:
                self.gesture_counts[f"Left_{result.gesture_left[0]}"] += 1
            if result.gesture_right:
                self.gesture_counts[f"Right_{result.gesture_right[0]}"] += 1

            # Whole-body backends populate body / face; MediaPipe leaves them None.
            if self.backend.supports_body:
                self.body_landmarks_list.append(result.body)
                self.body_landmarks_list_all.append(result.body_per_person)
            if self.backend.supports_face:
                self.face_landmarks_list.append(result.face)
                self.face_landmarks_list_all.append(result.face_per_person)

        cap.release()
        # Release backend resources (frees CUDA memory for the Whisper worker)
        try:
            self.backend.close()
        except Exception as e:
            logger.warning("Backend close failed: %s", e)
        finally:
            self.backend = None

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

            # Try faster-whisper (chunk-based persistent worker)
            if FASTER_WHISPER_AVAILABLE:
                if progress_callback:
                    progress_callback(15, "Starting Whisper transcription...")
                segments = self._run_chunked_transcription(temp_audio, duration, progress_callback)
                if segments:
                    self.transcript = segments
                    logging.info("[faster-whisper] Transcribed %d segments", len(self.transcript))
                    if progress_callback: progress_callback(85, "Classifying sentiment...")
                    self.classify_segment_sentiments()
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

    def _extract_audio_chunk(self, source_audio_path, chunk_start, chunk_end, output_path):
        """Extract [chunk_start, chunk_end] from source_audio_path into a 16 kHz mono WAV.

        Uses ffmpeg directly (10-100x faster than moviepy's Python audio loop).
        ffmpeg is always available because moviepy depends on imageio-ffmpeg.
        Falls back to moviepy only if ffmpeg lookup fails for any reason.
        """
        duration = max(0.001, chunk_end - chunk_start)
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg = get_ffmpeg_exe()
            import subprocess as _sp
            # -ss before -i = fast seek; -ac 1 = mono; -ar 16000 = 16 kHz
            _sp.run(
                [ffmpeg, "-y", "-loglevel", "error",
                 "-ss", str(chunk_start), "-t", str(duration),
                 "-i", source_audio_path,
                 "-ac", "1", "-ar", "16000",
                 "-c:a", "pcm_s16le", output_path],
                check=True, capture_output=True,
            )
            return
        except Exception as ffmpeg_err:
            logging.warning("ffmpeg extraction failed (%s); falling back to moviepy",
                            ffmpeg_err)

        try:
            from moviepy import AudioFileClip
        except ImportError:
            from moviepy.editor import AudioFileClip
        clip = AudioFileClip(source_audio_path)
        end = min(chunk_end, clip.duration)
        sub = clip.subclipped(chunk_start, end)
        sub.write_audiofile(output_path, fps=16000, nbytes=2, codec="pcm_s16le",
                            logger=None)
        sub.close()
        clip.close()

    def _run_chunked_transcription(self, temp_audio, duration, progress_callback=None):
        """
        Transcribes temp_audio using a persistent Whisper worker that loads the model once.
        Audio is split into 5-minute chunks with 2-second overlap to avoid boundary cuts.
        If the worker stalls or crashes, it is restarted and the failed chunk is retried once.
        Completed chunk results are accumulated in self.transcript as each chunk finishes,
        so partial transcripts are preserved even if later chunks fail.
        Returns the full list of (start, end, text, label) tuples.
        """
        import queue as _queue_mod, tempfile as _tempfile

        CHUNK_SECONDS = 300          # 5-minute chunks
        OVERLAP_SECONDS = 2          # overlap on each side to avoid cutting sentences
        STALL_SECONDS = 600          # kill worker if no message for 10 minutes
        MAX_RETRIES = 1              # retry a failed/stalled chunk once

        num_chunks = max(1, math.ceil(duration / CHUNK_SECONDS))
        completed_segments = []
        worker_proc = None
        cmd_q = res_q = None

        def _spawn():
            nonlocal worker_proc, cmd_q, res_q
            # Shut down existing worker if alive
            if worker_proc and worker_proc.is_alive():
                try: cmd_q.put_nowait(("shutdown",))
                except Exception: pass
                worker_proc.join(timeout=5)
                if worker_proc.is_alive():
                    worker_proc.kill(); worker_proc.join()
            cmd_q = multiprocessing.Queue()
            res_q = multiprocessing.Queue()
            worker_proc = multiprocessing.Process(
                target=_persistent_whisper_worker,
                args=("large-v3", cmd_q, res_q),
                daemon=True,
            )
            worker_proc.start()
            # Wait for "ready". Deadline resets on every message so that long
            # but actively-progressing operations (e.g. first-run download of
            # the 3 GB large-v3 model) don't get killed. The worker emits a
            # heartbeat every 10s during the download to keep this loop alive.
            SILENCE_TIMEOUT = 120  # seconds with no message → assume dead
            deadline = time.time() + SILENCE_TIMEOUT
            while True:
                if not worker_proc.is_alive():
                    logging.error("Whisper worker died before signalling ready")
                    return False
                remaining = deadline - time.time()
                if remaining <= 0:
                    logging.error("Whisper worker silent for %ds; terminating",
                                  SILENCE_TIMEOUT)
                    if worker_proc.is_alive():
                        worker_proc.terminate(); worker_proc.join()
                    return False
                try:
                    msg = res_q.get(timeout=min(remaining, 1.0))
                except _queue_mod.Empty:
                    continue
                # Any message → worker is alive, reset the silence window
                deadline = time.time() + SILENCE_TIMEOUT
                if msg[0] == "ready":
                    return True
                if msg[0] == "fatal_error":
                    logging.error("Whisper worker failed to start: %s", msg[1])
                    worker_proc.terminate(); worker_proc.join()
                    return False
                if msg[0] == "progress" and progress_callback:
                    progress_callback(msg[1], msg[2])

        if not _spawn():
            return completed_segments

        # ETA tracking — wall time of each completed chunk
        chunk_durations: list[float] = []

        def _fmt_eta(seconds: float) -> str:
            seconds = int(max(0, seconds))
            if seconds < 60:
                return f"{seconds}s"
            if seconds < 3600:
                return f"{seconds // 60}m {seconds % 60}s"
            return f"{seconds // 3600}h {(seconds % 3600) // 60}m"

        def _eta_text(idx_done: int) -> str:
            """Return ' — ETA Xm Ys' string, or '' before we have any data."""
            if not chunk_durations:
                return ""
            avg = sum(chunk_durations) / len(chunk_durations)
            remaining = (num_chunks - idx_done) * avg
            return f" — ETA {_fmt_eta(remaining)}"

        for chunk_index in range(num_chunks):
            chunk_start = chunk_index * CHUNK_SECONDS
            chunk_end = min(chunk_start + CHUNK_SECONDS, duration)
            # Extend extraction window by OVERLAP_SECONDS on both sides
            extract_start = max(0.0, chunk_start - OVERLAP_SECONDS)
            extract_end = min(duration, chunk_end + OVERLAP_SECONDS)
            pct_start = 15 + int((chunk_index / num_chunks) * 70)
            pct_end = 15 + int(((chunk_index + 1) / num_chunks) * 70)

            chunk_audio = os.path.join(
                _tempfile.gettempdir(),
                f"whisper_chunk_{os.getpid()}_{chunk_index}.wav",
            )
            chunk_t0 = time.time()
            try:
                if progress_callback:
                    progress_callback(pct_start,
                                      f"Extracting chunk {chunk_index+1}/{num_chunks}"
                                      f"{_eta_text(chunk_index)}")
                self._extract_audio_chunk(temp_audio, extract_start, extract_end, chunk_audio)
            except Exception as e:
                logging.error("Chunk %d audio extraction failed: %s", chunk_index, e)
                continue

            success = False
            for attempt in range(MAX_RETRIES + 1):
                if attempt > 0:
                    # Re-spawn only if worker is dead; otherwise just retry
                    if not worker_proc.is_alive():
                        logging.info("Respawning worker for chunk %d retry", chunk_index)
                        if not _spawn():
                            logging.error("Worker restart failed; aborting transcription")
                            try: os.remove(chunk_audio)
                            except OSError: pass
                            return completed_segments
                    if progress_callback:
                        progress_callback(pct_start,
                                          f"Retrying chunk {chunk_index+1}/{num_chunks}...")

                cmd_q.put(("chunk", chunk_audio, chunk_start))
                last_activity = time.time()
                worker_failed = False

                while True:
                    try:
                        msg = res_q.get(timeout=0.2)
                        last_activity = time.time()

                        if msg[0] == "progress":
                            pct = pct_start + int((msg[1] / 100.0) * (pct_end - pct_start))
                            if progress_callback:
                                progress_callback(pct,
                                                  f"[{chunk_index+1}/{num_chunks}] {msg[2]}"
                                                  f"{_eta_text(chunk_index)}")

                        elif msg[0] == "chunk_result":
                            # Strip overlap regions from both sides before appending
                            filtered = [s for s in msg[2]
                                        if s[1] > chunk_start and s[0] < chunk_end]
                            completed_segments.extend(filtered)
                            self.transcript = list(completed_segments)
                            success = True
                            break

                        elif msg[0] == "chunk_error":
                            logging.warning("Chunk %d error (attempt %d): %s",
                                            chunk_index, attempt + 1, msg[2])
                            break

                        elif msg[0] == "fatal_error":
                            logging.error("Worker fatal on chunk %d: %s", chunk_index, msg[1])
                            worker_failed = True
                            break

                    except _queue_mod.Empty:
                        if not worker_proc.is_alive():
                            logging.error("Worker died on chunk %d (attempt %d)",
                                          chunk_index, attempt + 1)
                            worker_failed = True
                            break
                        if time.time() - last_activity > STALL_SECONDS:
                            logging.error("Worker stalled on chunk %d; terminating", chunk_index)
                            if progress_callback:
                                progress_callback(
                                    pct_start,
                                    f"Chunk {chunk_index+1} stalled; restarting worker...")
                            worker_proc.terminate()
                            worker_proc.join(timeout=5)
                            if worker_proc.is_alive():
                                worker_proc.kill(); worker_proc.join()
                            worker_failed = True
                            break

                if success:
                    break
                # On worker_failed with retries left, the next iteration will re-spawn.
                # On chunk_error the worker is still alive; just retry the send.

            try: os.remove(chunk_audio)
            except OSError: pass

            if success:
                chunk_durations.append(time.time() - chunk_t0)
                # Keep ETA responsive to recent speed — average last 5 chunks only
                if len(chunk_durations) > 5:
                    chunk_durations.pop(0)
                if progress_callback:
                    progress_callback(
                        pct_end,
                        f"Chunk {chunk_index+1}/{num_chunks} done "
                        f"({len(completed_segments)} segments so far)"
                        f"{_eta_text(chunk_index + 1)}")
            else:
                logging.warning("Chunk %d/%d failed after %d attempt(s); skipping",
                                chunk_index + 1, num_chunks, MAX_RETRIES + 1)
                # Ensure worker is alive for the next chunk
                if not worker_proc.is_alive():
                    if not _spawn():
                        return completed_segments

        # Shut down worker cleanly
        if worker_proc and worker_proc.is_alive():
            try: cmd_q.put(("shutdown",))
            except Exception: pass
            worker_proc.join(timeout=10)
            if worker_proc.is_alive():
                worker_proc.kill(); worker_proc.join()

        return completed_segments

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

        # Whole-body source → build heatmap/traces from body keypoints instead
        # of hands, then fall through to the shared filter/speed/blur tail.
        if source in BODY_HEATMAP_SOURCES:
            self._regenerate_from_body(source)
            self.trace_path_left = self.filter_trace(self.trace_path_left)
            self.trace_path_right = self.filter_trace(self.trace_path_right)
            self._finalize_heatmaps()
            return

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
        self._finalize_heatmaps()

    def _finalize_heatmaps(self):
        """Shared tail: build the speed heatmap from traces and blur both maps."""
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

    def _regenerate_from_body(self, source):
        """
        Build heatmap + L/R traces from whole-body (RTMW) keypoints.
        Populates self.heatmap, self.trace_path_left, self.trace_path_right.
        No-op-safe when body_landmarks_list is empty (e.g. MediaPipe backend).
        """
        hm_indices, left_idx, right_idx = BODY_HEATMAP_SOURCES[source]
        W, H = self.video_width, self.video_height

        def _vis(lm):
            return getattr(lm, "visibility", 1.0)

        for body_kps in self.body_landmarks_list:
            trace_l, trace_r = None, None
            if body_kps:
                # Heatmap: bounding box over the visible region keypoints
                pts = []
                for i in hm_indices:
                    if i < len(body_kps) and _vis(body_kps[i]) >= 0.3:
                        pts.append((int(body_kps[i].x * W), int(body_kps[i].y * H)))
                if pts:
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    min_x, max_x = max(0, min(xs)), min(W, max(xs))
                    min_y, max_y = max(0, min(ys)), min(H, max(ys))
                    if max_x == min_x: max_x += 1
                    if max_y == min_y: max_y += 1
                    self.heatmap[min_y:max_y, min_x:max_x] += 1

                # Traces from the chosen left/right keypoints
                if left_idx < len(body_kps) and _vis(body_kps[left_idx]) >= 0.3:
                    lm = body_kps[left_idx]
                    trace_l = (int(lm.x * W), int(lm.y * H), 1.0)
                if right_idx < len(body_kps) and _vis(body_kps[right_idx]) >= 0.3:
                    lm = body_kps[right_idx]
                    trace_r = (int(lm.x * W), int(lm.y * H), 1.0)

            self.trace_path_left.append(trace_l)
            self.trace_path_right.append(trace_r)

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
            def _lm_dict(lm):
                # Tolerant of both MediaPipe landmarks and pose_backends.Landmark
                return {'x': lm.x, 'y': lm.y, 'z': getattr(lm, 'z', 0.0)}

            serializable_landmarks = []
            for frame_data in self.landmarks_list:
                # frame_data is list of (landmarks, label)
                s_frame = []
                for lms, label in frame_data:
                    s_frame.append(([_lm_dict(lm) for lm in lms], label))
                serializable_landmarks.append(s_frame)

            # Phase 1: body / face from whole-body backends.  Stored as lists
            # of lists-of-dicts; each per-frame entry may be None.
            def _serialize_kp_list(per_frame_list):
                out = []
                for kp in per_frame_list:
                    if kp is None:
                        out.append(None)
                    else:
                        out.append([_lm_dict(lm) for lm in kp])
                return out

            serializable_body = _serialize_kp_list(self.body_landmarks_list)
            serializable_face = _serialize_kp_list(self.face_landmarks_list)

            # v1.3: per-person body / face (RTMW multi-person).  Each frame is
            # a list of per-person keypoint lists; empty list means "no people".
            def _serialize_per_person(per_frame_list):
                out = []
                for people in (per_frame_list or []):
                    out.append([[_lm_dict(lm) for lm in p] for p in people])
                return out
            serializable_body_all = _serialize_per_person(self.body_landmarks_list_all)
            serializable_face_all = _serialize_per_person(self.face_landmarks_list_all)

            # Determine flags
            has_hands = len(self.landmarks_list) > 0

            # Construct Data Payload
            data = {
                'version': 1.3,  # 1.3 adds multi-person body/face keypoint lists
                'timestamp': datetime.now().isoformat(),
                'video_path': self.video_path,
                'video_width': self.video_width,
                'video_height': self.video_height,
                'fps': self.fps,
                'total_frames': self.total_frames,
                'landmarks': serializable_landmarks,
                'body_landmarks': serializable_body,
                'face_landmarks': serializable_face,
                'body_landmarks_all': serializable_body_all,
                'face_landmarks_all': serializable_face_all,
                'transcript': self.transcript,
                'audio_summary': self.audio_summary,
                'gesture_counts': dict(self.gesture_counts),
                'has_hands': has_hands,
                'backend': self.backend_name,
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

            # v1.2 additions — body / face keypoints from whole-body backends.
            # Missing in older files → empty lists (downstream code treats them
            # as optional).
            def _deser_kp_list(raw):
                out = []
                for kp in (raw or []):
                    if kp is None:
                        out.append(None)
                    else:
                        out.append([LandmarkCompat(d['x'], d['y'], d.get('z', 0.0))
                                    for d in kp])
                return out
            self.body_landmarks_list = _deser_kp_list(data.get('body_landmarks'))
            self.face_landmarks_list = _deser_kp_list(data.get('face_landmarks'))

            # v1.3 multi-person.  Missing in older files → derive single-person
            # _all from the single-subject lists so the UI still iterates.
            def _deser_per_person(raw):
                out = []
                for people in (raw or []):
                    out.append([[LandmarkCompat(d['x'], d['y'], d.get('z', 0.0))
                                 for d in p] for p in people])
                return out
            body_all_raw = data.get('body_landmarks_all')
            face_all_raw = data.get('face_landmarks_all')
            if body_all_raw is not None:
                self.body_landmarks_list_all = _deser_per_person(body_all_raw)
            else:
                self.body_landmarks_list_all = [
                    [kp] if kp else [] for kp in self.body_landmarks_list
                ]
            if face_all_raw is not None:
                self.face_landmarks_list_all = _deser_per_person(face_all_raw)
            else:
                self.face_landmarks_list_all = [
                    [kp] if kp else [] for kp in self.face_landmarks_list
                ]
            self.backend_name = data.get('backend', self.backend_name)
            
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

# --- Shared word-aggregation helper (used by both worker functions) ---
def _aggregate_words_to_sentences(all_words, time_offset=0.0):
    """Convert flat faster-whisper word objects into (start, end, text, label) tuples."""
    transcript_result = []
    current_sentence_words = []
    sentence_start = 0.0

    for i, word_data in enumerate(all_words):
        word_text = word_data.word.strip()
        t_start = word_data.start + time_offset
        t_end = word_data.end + time_offset

        if not word_text:
            continue

        if not current_sentence_words:
            sentence_start = t_start

        current_sentence_words.append(word_text)

        ends_sentence = word_text.endswith(('.', '!', '?'))

        has_long_gap = False
        if i < len(all_words) - 1:
            if (all_words[i + 1].start + time_offset) - t_end > 1.0:
                has_long_gap = True

        current_len = sum(len(w) + 1 for w in current_sentence_words)
        should_split = (
            ends_sentence
            or (has_long_gap and current_len > 30)
            or current_len > 250
        )

        if should_split or i == len(all_words) - 1:
            transcript_result.append((sentence_start, t_end, " ".join(current_sentence_words), "Neutral"))
            current_sentence_words = []
            sentence_start = 0.0

    return transcript_result


# --- Legacy single-shot worker (kept for reference; superseded by persistent worker) ---
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
            try:
                from faster_whisper import BatchedInferencePipeline
            except ImportError:
                BatchedInferencePipeline = None
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

        # WhisperModel.transcribe() does NOT accept batch_size — that argument
        # belongs to BatchedInferencePipeline. Use batched inference on CUDA
        # for higher GPU utilisation; fall back to plain transcribe on CPU.
        common_kwargs = dict(
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        if device == "cuda" and BatchedInferencePipeline is not None:
            try:
                transcriber = BatchedInferencePipeline(model=model)
                segments, info = transcriber.transcribe(
                    audio_path, batch_size=16, **common_kwargs,
                )
            except Exception as batched_err:
                # Some VAD/batched combinations can fail on certain audio;
                # gracefully fall back to standard inference.
                queue.put(("progress", 25,
                           f"Batched inference failed ({batched_err.__class__.__name__}); using standard"))
                segments, info = model.transcribe(audio_path, **common_kwargs)
        else:
            segments, info = model.transcribe(audio_path, **common_kwargs)

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
        transcript_result = _aggregate_words_to_sentences(all_words)
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


# --- Persistent chunk worker ---
def _persistent_whisper_worker(model_size, cmd_queue, res_queue):
    """
    Loads the Whisper model once then processes chunk commands until shutdown.
    Receives: ("chunk", path, start_offset_seconds) | ("shutdown",)
    Sends:    ("ready",) | ("progress", pct, msg) | ("chunk_result", start, segments)
              | ("chunk_error", start, msg) | ("fatal_error", msg)
    Worker survives individual chunk errors so the parent doesn't have to restart it.
    """
    try:
        import torch, gc
        from faster_whisper import WhisperModel
        try:
            from faster_whisper import BatchedInferencePipeline
        except ImportError:
            BatchedInferencePipeline = None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        res_queue.put(("progress", 5, f"Loading Whisper '{model_size}' on {device}..."))

        # The first launch may download ~3 GB of weights with no API feedback.
        # Run a background heartbeat so the parent's silence-timeout doesn't
        # kill us mid-download. The heartbeat scans the HuggingFace cache to
        # report actual download progress (MiB and %).
        import threading, glob
        load_done = threading.Event()

        # Approximate total sizes (MiB) — used for % calculation only.
        EXPECTED_MIB = {"large-v3": 3090, "large-v2": 3090, "large": 3090,
                        "medium": 1530, "small": 484, "base": 145, "tiny": 75}
        expected = EXPECTED_MIB.get(model_size, 3090)

        # HF cache layout: ~/.cache/huggingface/hub/models--<org>--<name>/blobs/*
        hf_hub = os.path.join(os.path.expanduser("~"), ".cache",
                              "huggingface", "hub")

        def _current_download_mib():
            """Return MiB of the largest in-flight blob, or None if nothing found."""
            try:
                pattern = os.path.join(hf_hub, "models--*whisper*", "blobs", "*")
                files = glob.glob(pattern)
                if not files:
                    return None
                largest = max(files, key=lambda p: os.path.getsize(p))
                return os.path.getsize(largest) / (1024 * 1024)
            except Exception:
                return None

        def _heartbeat():
            secs = 0
            while not load_done.wait(5):
                secs += 5
                mib = _current_download_mib()
                if mib is not None and mib > 1.0:
                    pct = min(99, int(mib / expected * 100))
                    msg = f"Downloading Whisper model: {mib:.0f} / ~{expected} MiB ({pct}%)"
                else:
                    msg = f"Loading Whisper model... ({secs}s)"
                try:
                    res_queue.put(("progress", 5, msg))
                except Exception:
                    return
        threading.Thread(target=_heartbeat, daemon=True).start()

        try:
            try:
                model = WhisperModel(model_size, device=device, compute_type=compute_type,
                                     cpu_threads=4, num_workers=2)
            except Exception:
                res_queue.put(("progress", 5, f"'{model_size}' unavailable; falling back to 'medium'"))
                model = WhisperModel("medium", device=device, compute_type=compute_type,
                                     cpu_threads=4, num_workers=2)
        finally:
            load_done.set()

        use_batched = device == "cuda" and BatchedInferencePipeline is not None
        transcriber = BatchedInferencePipeline(model=model) if use_batched else None

        common_kwargs = dict(
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        res_queue.put(("ready",))

        while True:
            cmd = cmd_queue.get()
            if cmd[0] == "shutdown":
                break

            chunk_path, chunk_start = cmd[1], float(cmd[2])

            try:
                if use_batched:
                    segments, info = transcriber.transcribe(chunk_path, batch_size=16, **common_kwargs)
                else:
                    segments, info = model.transcribe(chunk_path, **common_kwargs)

                all_words = []
                total_dur = max(info.duration, 1e-6)
                for i, seg in enumerate(segments):
                    if seg.words:
                        all_words.extend(seg.words)
                    if i % 5 == 0:
                        pct = int((seg.end / total_dur) * 90)
                        res_queue.put(("progress", pct,
                                       f"Transcribing at {int(seg.end)}s / {int(total_dur)}s"))

                result = _aggregate_words_to_sentences(all_words, time_offset=chunk_start)
                res_queue.put(("chunk_result", chunk_start, result))

            except Exception as exc:
                import traceback; traceback.print_exc()
                res_queue.put(("chunk_error", chunk_start, str(exc)))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    except Exception as exc:
        import traceback; traceback.print_exc()
        res_queue.put(("fatal_error", str(exc)))


