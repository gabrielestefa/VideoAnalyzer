# VideoAnalyzer Whisper Fix Plan

This file describes the fix for the `faster-whisper` error and the recommended design change for preventing transcription hangs while preserving progress.

Target repository:

```text
https://github.com/gabrielestefa/VideoAnalyzer
```

Target file:

```text
video_analysis_engine.py
```

Observed error:

```text
Traceback (most recent call last):
  File "c:\Users\hosseinalipourA\Downloads\VideoAnalyzer-main\video_analysis_engine.py", line 1146, in _worker_whisper_transcribe
    segments, info = model.transcribe(
TypeError: WhisperModel.transcribe() got an unexpected keyword argument 'batch_size'

WARNING [root] Transcription worker error: WhisperModel.transcribe() got an unexpected keyword argument 'batch_size'
```

## Root cause

`batch_size` is being passed directly to:

```python
WhisperModel.transcribe(...)
```

That is not supported by `faster-whisper`'s `WhisperModel.transcribe`.

For batched inference, `faster-whisper` requires:

```python
from faster_whisper import BatchedInferencePipeline

transcriber = BatchedInferencePipeline(model=model)
segments, info = transcriber.transcribe(..., batch_size=16)
```

So the immediate fix is to use `BatchedInferencePipeline` only when running on CUDA, and to avoid passing `batch_size` to `WhisperModel.transcribe`.

---

# Part 1: Immediate fix for the crash

## 1. Update the faster-whisper import

In `video_analysis_engine.py`, find the import near the top:

```python
from faster_whisper import WhisperModel
```

Replace it with:

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline
```

If the file has an import fallback like this:

```python
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None
```

Replace it with:

```python
try:
    from faster_whisper import WhisperModel, BatchedInferencePipeline
except ImportError:
    WhisperModel = None
    BatchedInferencePipeline = None
```

---

## 2. Replace the failing `model.transcribe(...)` call

Find the failing block inside `_worker_whisper_transcribe`, around the line shown in the traceback, approximately line `1146`.

It likely looks like this:

```python
segments, info = model.transcribe(
    audio_path,
    beam_size=5,
    word_timestamps=True,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
    batch_size=16 if device == "cuda" else 1,
)
```

Replace it with this:

```python
if device == "cuda":
    transcriber = BatchedInferencePipeline(model=model)
    segments, info = transcriber.transcribe(
        audio_path,
        batch_size=16,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
else:
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
```

This fixes:

```text
TypeError: WhisperModel.transcribe() got an unexpected keyword argument 'batch_size'
```

---

# Part 2: Basic worker timeout

The current worker can leave the main program waiting indefinitely if the subprocess gets stuck.

Look for code similar to this inside `transcribe_audio`:

```python
msg_queue = multiprocessing.Queue()
p = multiprocessing.Process(target=_worker_whisper_transcribe, args=(temp_audio, "large-v3", msg_queue))
p.start()

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
            break

    except _queue_mod.Empty:
        if not p.is_alive():
            break

p.join()
```

Replace it with:

```python
msg_queue = multiprocessing.Queue()
p = multiprocessing.Process(
    target=_worker_whisper_transcribe,
    args=(temp_audio, "large-v3", msg_queue),
)
p.start()

import queue as _queue_mod

MAX_TRANSCRIBE_SECONDS = 60 * 60  # 1 hour
last_progress_time = time.time()
last_progress_percent = 15

while True:
    try:
        msg = msg_queue.get(timeout=0.1)

        if msg[0] == "progress":
            last_progress_time = time.time()
            last_progress_percent = msg[1]
            if progress_callback:
                progress_callback(msg[1], msg[2])

        elif msg[0] == "result":
            self.transcript = msg[1]

        elif msg[0] == "error":
            logging.warning("Transcription worker error: %s", msg[1])
            self.transcript = []
            break

    except _queue_mod.Empty:
        if not p.is_alive():
            break

        if time.time() - last_progress_time > MAX_TRANSCRIBE_SECONDS:
            logging.error("Transcription worker timed out; terminating process")

            if progress_callback:
                progress_callback(
                    last_progress_percent,
                    "Transcription stalled; terminating worker...",
                )

            p.terminate()
            p.join(timeout=5)

            if p.is_alive():
                p.kill()
                p.join()

            self.transcript = []
            break

p.join(timeout=5)
if p.is_alive():
    logging.error("Transcription worker did not exit cleanly; killing process")
    p.kill()
    p.join()
```

This prevents the parent process from hanging forever.

Important: this basic timeout does **not** preserve partial Whisper output, because the current worker transcribes the entire audio in a single call.

---

# Part 3: Restart worker while preserving progress

## Important design note

With the current design, a restart can only preserve the last displayed progress percentage. It cannot preserve actual transcript content unless the worker sends completed partial results back to the parent process.

To preserve progress correctly, transcription must be chunk-based:

```text
chunk 1 complete -> save result
chunk 2 complete -> save result
chunk 3 stalls -> kill worker
restart from chunk 3
continue from last completed chunk
```

Recommended chunk size:

```python
WHISPER_CHUNK_SECONDS = 5 * 60
MAX_WORKER_STALL_SECONDS = 10 * 60
MAX_CHUNK_RETRIES = 1
```

Use 5-minute chunks for safer progress preservation. Use 10-minute chunks if performance is more important than restart granularity.

---

## Recommended architecture

Add three concepts:

1. A helper that extracts a temporary audio chunk.
2. A worker that transcribes one chunk.
3. A parent loop that retries failed chunks and appends completed segments to `self.transcript`.

Recommended function names:

```python
_extract_audio_chunk(...)
_worker_whisper_transcribe_chunk(...)
_run_whisper_chunk_with_timeout(...)
```

---

## 1. Add a chunk worker

Add this near the current `_worker_whisper_transcribe` function.

```python
def _worker_whisper_transcribe_chunk(audio_chunk_path, model_size, chunk_start_seconds, msg_queue):
    try:
        import torch
        from faster_whisper import WhisperModel, BatchedInferencePipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        msg_queue.put((
            "progress",
            0,
            f"Loading Whisper model on {device}..."
        ))

        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        msg_queue.put((
            "progress",
            10,
            "Transcribing audio chunk..."
        ))

        if device == "cuda":
            transcriber = BatchedInferencePipeline(model=model)
            segments, info = transcriber.transcribe(
                audio_chunk_path,
                batch_size=16,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
        else:
            segments, info = model.transcribe(
                audio_chunk_path,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

        result_segments = []
        for segment in segments:
            result_segments.append({
                "start": float(segment.start) + float(chunk_start_seconds),
                "end": float(segment.end) + float(chunk_start_seconds),
                "text": segment.text.strip(),
            })

        msg_queue.put(("result", result_segments))

    except Exception as exc:
        msg_queue.put(("error", str(exc)))
```

Notes:

- `chunk_start_seconds` is added back to each segment timestamp so that the final transcript is aligned to the original video/audio.
- This worker sends a completed chunk result back to the parent.
- If it crashes, the parent still keeps all previously completed chunks.

---

## 2. Add a helper to run one chunk with timeout

Add this as a method on the main class that owns `transcribe_audio`.

```python
def _run_whisper_chunk_with_timeout(
    self,
    audio_chunk_path,
    model_size,
    chunk_start_seconds,
    overall_progress_start,
    overall_progress_end,
    progress_callback=None,
):
    msg_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_worker_whisper_transcribe_chunk,
        args=(audio_chunk_path, model_size, chunk_start_seconds, msg_queue),
    )

    p.start()

    import queue as _queue_mod

    MAX_WORKER_STALL_SECONDS = 10 * 60
    last_progress_time = time.time()
    chunk_segments = []

    while True:
        try:
            msg = msg_queue.get(timeout=0.1)

            if msg[0] == "progress":
                last_progress_time = time.time()

                chunk_progress = float(msg[1]) / 100.0
                overall_progress = int(
                    overall_progress_start
                    + chunk_progress * (overall_progress_end - overall_progress_start)
                )

                if progress_callback:
                    progress_callback(overall_progress, msg[2])

            elif msg[0] == "result":
                chunk_segments = msg[1]
                break

            elif msg[0] == "error":
                logging.warning("Whisper chunk worker error: %s", msg[1])
                break

        except _queue_mod.Empty:
            if not p.is_alive():
                break

            if time.time() - last_progress_time > MAX_WORKER_STALL_SECONDS:
                logging.error(
                    "Whisper chunk worker stalled for %s seconds; terminating",
                    MAX_WORKER_STALL_SECONDS,
                )

                if progress_callback:
                    progress_callback(
                        overall_progress_start,
                        "Transcription chunk stalled; restarting...",
                    )

                p.terminate()
                p.join(timeout=5)

                if p.is_alive():
                    p.kill()
                    p.join()

                return False, []

    p.join(timeout=5)

    if p.is_alive():
        p.kill()
        p.join()
        return False, []

    return bool(chunk_segments), chunk_segments
```

---

## 3. Add or reuse an audio chunk extraction helper

The exact implementation depends on how the project currently extracts audio.

If MoviePy is already used, a helper can look like this:

```python
def _extract_audio_chunk(self, source_audio_path, chunk_start, chunk_end, output_path):
    from moviepy.editor import AudioFileClip

    audio_clip = AudioFileClip(source_audio_path)
    subclip = audio_clip.subclip(chunk_start, chunk_end)
    subclip.write_audiofile(
        output_path,
        fps=16000,
        nbytes=2,
        codec="pcm_s16le",
        verbose=False,
        logger=None,
    )
    subclip.close()
    audio_clip.close()
```

If the project already has a safer audio extraction method, prefer reusing it.

On newer MoviePy versions, `subclip` may be `subclipped`. If needed, use:

```python
subclip = audio_clip.subclipped(chunk_start, chunk_end)
```

---

## 4. Replace full-file transcription with chunk loop

Inside `transcribe_audio`, after the program has created `temp_audio`, replace the single full-file Whisper process with a chunk loop.

Pseudo-implementation:

```python
WHISPER_CHUNK_SECONDS = 5 * 60
MAX_CHUNK_RETRIES = 1

completed_segments = []

audio_clip = AudioFileClip(temp_audio)
duration = float(audio_clip.duration)
audio_clip.close()

num_chunks = max(1, math.ceil(duration / WHISPER_CHUNK_SECONDS))

for chunk_index in range(num_chunks):
    chunk_start = chunk_index * WHISPER_CHUNK_SECONDS
    chunk_end = min(chunk_start + WHISPER_CHUNK_SECONDS, duration)

    overall_progress_start = 15 + int((chunk_index / num_chunks) * 70)
    overall_progress_end = 15 + int(((chunk_index + 1) / num_chunks) * 70)

    chunk_audio_path = os.path.join(
        tempfile.gettempdir(),
        f"whisper_chunk_{os.getpid()}_{chunk_index}.wav",
    )

    self._extract_audio_chunk(
        temp_audio,
        chunk_start,
        chunk_end,
        chunk_audio_path,
    )

    success = False
    chunk_segments = []

    for attempt in range(MAX_CHUNK_RETRIES + 1):
        if progress_callback:
            progress_callback(
                overall_progress_start,
                f"Transcribing chunk {chunk_index + 1}/{num_chunks}, attempt {attempt + 1}...",
            )

        success, chunk_segments = self._run_whisper_chunk_with_timeout(
            chunk_audio_path,
            "large-v3",
            chunk_start,
            overall_progress_start,
            overall_progress_end,
            progress_callback,
        )

        if success:
            break

        logging.warning(
            "Chunk %s/%s failed on attempt %s",
            chunk_index + 1,
            num_chunks,
            attempt + 1,
        )

    try:
        os.remove(chunk_audio_path)
    except OSError:
        pass

    if success:
        completed_segments.extend(chunk_segments)
        self.transcript = completed_segments

        if progress_callback:
            progress_callback(
                overall_progress_end,
                f"Completed chunk {chunk_index + 1}/{num_chunks}",
            )
    else:
        logging.error(
            "Chunk %s/%s failed after retries; continuing with next chunk",
            chunk_index + 1,
            num_chunks,
        )
```

At the end of the loop:

```python
self.transcript = completed_segments
```

Then continue with the existing downstream logic.

---

# Part 4: Avoiding duplicate or overlapping text at chunk boundaries

Chunking can cut words/sentences at boundaries.

A simple first version can accept this.

A better version uses overlapping chunks:

```python
CHUNK_OVERLAP_SECONDS = 2
```

For chunk extraction:

```python
effective_start = max(0, chunk_start - CHUNK_OVERLAP_SECONDS)
effective_end = min(duration, chunk_end + CHUNK_OVERLAP_SECONDS)
```

Then discard returned segments outside the true chunk window:

```python
filtered_segments = []
for segment in result_segments:
    if segment["end"] < chunk_start:
        continue
    if segment["start"] > chunk_end:
        continue
    filtered_segments.append(segment)
```

This reduces lost words at chunk boundaries.

---

# Part 5: Recommended implementation order

Implement in this order:

1. Fix `BatchedInferencePipeline`.
2. Add basic timeout to prevent indefinite hangs.
3. Refactor to chunk-based transcription.
4. Add retry per chunk.
5. Add optional overlap and deduplication.

Do not start with chunking before fixing the `batch_size` crash.

---

# Part 6: Local test checklist

After editing:

```bash
python -m py_compile video_analysis_engine.py
```

Then test with a short file first:

```text
1-2 minute video/audio file
```

Then test with a longer file:

```text
20-30 minute video/audio file
```

Watch for:

```text
- No TypeError about batch_size
- CUDA selected on the RTX 5090 machine
- Progress updates continue during transcription
- No indefinite hang after worker failure
- Partial transcript survives a failed chunk
```

Useful runtime checks:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

And:

```python
import ctranslate2
print(ctranslate2.__version__)
print(ctranslate2.get_cuda_device_count())
```

---

# Part 7: Recommended dependency update

For a Windows machine with an RTX 5090, use recent versions of `faster-whisper` and `ctranslate2`:

```bash
pip install --upgrade faster-whisper ctranslate2
```

If CUDA DLL errors occur, check that the environment has CUDA 12 and cuDNN 9-compatible runtime libraries available.

---

# Summary for another session

Please modify `video_analysis_engine.py` in the VideoAnalyzer repo.

Primary bug:

```python
WhisperModel.transcribe(..., batch_size=16)
```

is invalid.

Fix:

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline

if device == "cuda":
    transcriber = BatchedInferencePipeline(model=model)
    segments, info = transcriber.transcribe(..., batch_size=16)
else:
    segments, info = model.transcribe(...)
```

Then add a timeout around the multiprocessing worker.

For progress-preserving restart, refactor transcription into chunks and save each completed chunk's segments in the parent process. Restart only the failed chunk instead of restarting the whole transcription.
