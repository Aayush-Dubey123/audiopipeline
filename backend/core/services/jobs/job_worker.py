"""
Single-threaded asynchronous job worker queue for GPU inference safety.

Manages serialized job execution, execution timeouts, stale recovery, and pipeline dispatching.
"""

import os
import queue
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from core import logger
from core.config import get_settings
from core.database.database import SessionLocal, init_db
from core.cruds.job_crud import JobCRUD
from core.services.jobs.cleanup_service import CleanupService

logging = logger(__name__)
settings = get_settings()

import re
from typing import List, Tuple, Set

# Abbreviations that must NOT be treated as sentence-ending periods.
_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc",
    "ie", "eg", "approx", "dept", "est", "govt", "inc", "ltd", "no",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "st", "ave", "blvd",
}
_SENT_END = re.compile(r"[.!?।॥]+(?=\s|$)")


def _is_abbreviation(word: str) -> bool:
    return word.lower().rstrip(".") in _ABBREVIATIONS


_CHUNK_END_PUNCT = tuple(".!?।॥,;:\"')")


def mark_mid_sentence_cuts(chunks: List[str], paragraph_starts: Set[int]) -> List[str]:
    """
    Give chunks that were cut mid-sentence a trailing comma.

    Chatterbox appends '.' to any text without ending punctuation, which makes the voice
    drop into sentence-final intonation in the middle of a sentence. A comma keeps it
    continuing. Paragraph-final chunks are left alone.

    Args:
        chunks (List[str]): Chunk texts from split_text_with_boundaries.
        paragraph_starts (Set[int]): Chunk indices that start a new paragraph.

    Returns:
        List[str]: Chunk texts ready for synthesis (same length and order).
    """
    marked = []
    for idx, chunk in enumerate(chunks):
        paragraph_end = idx == len(chunks) - 1 or (idx + 1) in paragraph_starts
        if not paragraph_end and not chunk.rstrip().endswith(_CHUNK_END_PUNCT):
            chunk = chunk.rstrip() + ","
        marked.append(chunk)
    return marked


def split_text_with_boundaries(text: str, max_chars: int = 1000) -> Tuple[List[str], Set[int]]:
    if not text or not text.strip():
        return [], set()

    chunks: List[str] = []
    paragraph_start_indices = set()
    paragraphs = re.split(r"\n{2,}", text)

    def flush(buf: str) -> List[str]:
        results: List[str] = []
        while len(buf) > max_chars:
            cut = max_chars
            for pat in (r"[,;]\s", r"\s"):
                m = None
                for match in re.finditer(pat, buf[:max_chars]):
                    m = match
                if m:
                    cut = m.end()
                    break
            results.append(buf[:cut].strip())
            buf = buf[cut:].lstrip()
        if buf.strip():
            results.append(buf.strip())
        return results

    for para_idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue

        start_chunk_idx = len(chunks)
        if para_idx > 0 and start_chunk_idx > 0:
            paragraph_start_indices.add(start_chunk_idx)

        positions: List[int] = []
        for m in _SENT_END.finditer(para):
            end = m.end()
            before = para[:m.start()].rstrip()
            last_word = re.split(r"\s", before)[-1] if before else ""
            after_char = para[end:end + 1] if end < len(para) else ""
            if after_char.isdigit():
                continue
            if _is_abbreviation(last_word):
                continue
            positions.append(end)

        segments: List[str] = []
        prev = 0
        for pos in positions:
            seg = para[prev:pos].strip()
            if seg:
                segments.append(seg)
            prev = pos
        tail = para[prev:].strip()
        if tail:
            segments.append(tail)

        if not segments:
            segments = [para]

        current = ""
        for seg in segments:
            if len(seg) > max_chars:
                if current.strip():
                    chunks.extend(flush(current.strip()))
                    current = ""
                chunks.extend(flush(seg))
                continue

            if current and len(current) + 1 + len(seg) > max_chars:
                chunks.extend(flush(current.strip()))
                current = seg
            else:
                current = (current + " " + seg).lstrip() if current else seg

        if current.strip():
            chunks.extend(flush(current.strip()))

    final_chunks = [c for c in chunks if c.strip()]
    return final_chunks, paragraph_start_indices


class QueueFullError(Exception):
    """
    Raised when an incoming job cannot be enqueued due to queue capacity.
    """
    pass


class JobWorker:
    """
    Asynchronous job coordinator enforcing single-worker concurrency for GPU safety.
    """

    def __init__(self, maxsize: Optional[int] = None) -> None:
        """
        Initialize the job worker queue and concurrency locks.

        Args:
            maxsize (Optional[int]): Maximum bounded queue capacity.
        """
        capacity = maxsize if maxsize is not None else settings.QUEUE_MAX_SIZE
        self.queue: queue.Queue = queue.Queue(maxsize=capacity)
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._lock: threading.Lock = threading.Lock()
        self._default_pipeline: Optional[Callable] = None
        self._last_janitor: float = 0.0
        self._last_job_finished: float = time.monotonic()
        self._model_warm: bool = False

    def set_default_pipeline(self, pipeline_fn: Callable) -> None:
        """
        Set fallback pipeline callable when no specific handler is injected.

        Args:
            pipeline_fn (Callable): Pipeline function accepting (job_id, kind, voice_id, payload, job_dir).
        """
        self._default_pipeline = pipeline_fn

    def start(self) -> None:
        """
        Start the background worker thread and recover any stale orphaned jobs.

        Returns:
            None
        """
        with self._lock:
            if self._running:
                return

            logging.info("Starting JobWorker background daemon thread")
            db = SessionLocal()
            try:
                JobCRUD.recover_stale_jobs(db)
            finally:
                db.close()
            self._run_janitor()

            if self._default_pipeline is None:
                self._default_pipeline = production_inference_pipeline

            self._running = True
            self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="JobWorkerThread")
            self._thread.start()

    def stop(self) -> None:
        """
        Signal the worker thread to shut down cleanly.
        """
        with self._lock:
            if not self._running:
                return
            logging.info("Stopping JobWorker background thread")
            self._running = False
            self.queue.put((None, None))
            if hasattr(self, "_thread") and self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)

    def enqueue_job(self, job_id: str, pipeline_fn: Optional[Callable] = None) -> None:
        """
        Add a job to the bounded execution queue.

        Args:
            job_id (str): Unique job identifier to process.
            pipeline_fn (Optional[Callable]): Optional custom or fake pipeline callable.

        Raises:
            QueueFullError: If current queue capacity is exhausted.
        """
        logging.info(f"Executing JobWorker.enqueue_job for job '{job_id}'")
        try:
            self.queue.put_nowait((job_id, pipeline_fn or self._default_pipeline))
        except queue.Full:
            logging.warning(f"Rejected job '{job_id}': inference queue is full (size={self.queue.maxsize})")
            raise QueueFullError("The system is currently handling maximum capacity. Please retry shortly.")

    def _worker_loop(self) -> None:
        """
        Continuously pop and process jobs from the FIFO queue.
        """
        logging.info("JobWorker background loop active and awaiting tasks")
        while self._running:
            try:
                item = self.queue.get(timeout=1.0)
            except queue.Empty:
                self._idle_housekeeping()
                continue

            job_id, pipeline_fn = item
            if job_id is None:
                self.queue.task_done()
                break

            try:
                self._model_warm = True
                self._execute_job(job_id, pipeline_fn)
            except Exception as loop_error:
                logging.error(f"Unexpected error in JobWorker execution loop for job '{job_id}': {loop_error}")
            finally:
                self._last_job_finished = time.monotonic()
                self.queue.task_done()

        logging.info("JobWorker background loop terminated")

    def _run_janitor(self) -> None:
        """Purge expired/failed/orphaned job files and trim caches."""
        self._last_janitor = time.monotonic()
        db = SessionLocal()
        try:
            CleanupService.run_janitor(db)
        except Exception as err:
            logging.warning(f"Janitor pass failed: {err}")
        finally:
            db.close()

    def _idle_housekeeping(self) -> None:
        """Runs only while the queue is empty, so it never competes with a job."""
        now = time.monotonic()
        if now - self._last_janitor >= max(settings.CLEANUP_INTERVAL_SECONDS, 60):
            self._run_janitor()
        idle_limit = settings.MODEL_IDLE_UNLOAD_SECONDS
        if self._model_warm and idle_limit > 0 and now - self._last_job_finished >= idle_limit:
            self._model_warm = False
            try:
                from core.services.pipelines.chatterbox.chatterbox_service import ChatterboxService
                if ChatterboxService._resident_model is not None:
                    logging.info(f"Model idle for {idle_limit}s, releasing GPU/CPU memory")
                    ChatterboxService.unload_all()
            except Exception as err:
                logging.warning(f"Idle model unload failed: {err}")

    def _execute_job(self, job_id: str, pipeline_fn: Optional[Callable]) -> None:
        """
        Perform transactional lifecycle transitions and execute the assigned pipeline.

        Args:
            job_id (str): Identifier of job to execute.
            pipeline_fn (Optional[Callable]): Pipeline function to invoke.
        """
        logging.info(f"Executing JobWorker._execute_job for job '{job_id}'")
        db = SessionLocal()
        try:
            job = JobCRUD.get_job_by_id(db, job_id)
            if not job or job.status != "queued":
                logging.warning(f"Skipping job '{job_id}': state is not queued (status={getattr(job, 'status', None)})")
                return

            started_at = datetime.now(timezone.utc)
            JobCRUD.update_job_status(
                db=db,
                job_id=job_id,
                status="processing",
                progress_message="Loading voice model...",
                stage="Loading voice",
                stage_index=2,
                stage_total=4,
                started_at=started_at,
            )

            job_dir = (settings.tmp_path / job_id).resolve()
            job_dir.mkdir(parents=True, exist_ok=True)

            if pipeline_fn is None:
                raise RuntimeError("No inference pipeline handler configured for this job.")

            timeout_seconds = (
                settings.JOB_MAX_SECONDS
                if job.kind in ("tts", "tts_custom")
                else settings.JOB_TIMEOUT_SECONDS
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    pipeline_fn,
                    job_id,
                    job.kind,
                    job.voice_id,
                    job.payload or {},
                    job_dir,
                )
                try:
                    metadata = future.result(timeout=timeout_seconds)
                except FutureTimeoutError:
                    logging.error(f"Job '{job_id}' exceeded timeout limit of {timeout_seconds}s")
                    JobCRUD.update_job_status(
                        db=db,
                        job_id=job_id,
                        status="failed",
                        error_code="TIMEOUT",
                        error_message=f"Inference job exceeded execution limit of {timeout_seconds} seconds.",
                        progress_message="Job timed out during execution. Finished chunks are preserved for resumption.",
                    )
                    return

            JobCRUD.update_job_status(
                db=db,
                job_id=job_id,
                status="completed",
                progress_message="Inference completed successfully.",
                output_metadata=metadata if isinstance(metadata, dict) else {},
                stage="Completed",
                stage_index=4,
                stage_total=4,
            )
            logging.info(f"Job '{job_id}' completed successfully")

        except Exception as error:
            logging.error(f"Inference execution failed for job '{job_id}': {error}")
            JobCRUD.update_job_status(
                db=db,
                job_id=job_id,
                status="failed",
                error_code="INFERENCE_FAILED",
                error_message=str(error),
                progress_message="Inference failed. Finished chunks are preserved for resumption.",
            )
        finally:
            db.close()


def _run_long_form_tts(
    job_id: str,
    text: str,
    ref_path: Path,
    out_wav: Path,
    speed: float,
    language: str,
    job_dir: Path,
    pipeline_service: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Execute chunked long-form text-to-speech synthesis with pause stitching and progress updates.

    Args:
        job_id (str): Unique job identifier for progress tracking.
        text (str): Input script text (up to 30,000 characters).
        ref_path (Path): Path to reference speaker audio file.
        out_wav (Path): Target destination for concatenated output WAV.
        speed (float): Playback speed multiplier.
        language (str): Target language ('en', 'hi', 'mr').
        job_dir (Path): Working directory for chunk temporary files.
        pipeline_service (Optional[Any]): Optional dedicated synthesis pipeline service instance.

    Returns:
        Dict[str, Any]: Generated audio metadata.

    Raises:
        RuntimeError: If disk space is insufficient, reference audio is missing, or generation fails.
        TimeoutError: If execution exceeds chunk or job timeouts.
    """
    from core.services.audio.audio_service import AudioService
    from core.database.database import SessionLocal

    # 1. Enforce disk space safety before creating chunks
    AudioService.check_free_disk_gb(settings.tmp_path, required_gb=1.0)

    # 1b. Verify reference audio and compute content identity hash
    if not ref_path.exists():
        logging.error(f"TTS synthesis rejected: reference file does not exist at {ref_path}")
        raise RuntimeError(f"Reference audio missing or unreadable: {ref_path.name}")

    import hashlib
    ref_bytes = ref_path.read_bytes()
    ref_hash = hashlib.sha256(ref_bytes).hexdigest()[:16]
    logging.info(
        f"Conditioning TTS job '{job_id}' on reference: hash='{ref_hash}', engine='chatterbox_multilingual', lang='{language}'"
    )

    # 2. Normalize text and split into chunks respecting sentence and paragraph boundaries
    from core.services.llm.llm_service import LLMService
    clean_text = LLMService.rule_based_cleanup(text, language=language)
    chunks, para_boundaries = split_text_with_boundaries(clean_text, max_chars=settings.CHUNK_MAX_CHARS)
    if not chunks:
        raise RuntimeError("No valid text provided for synthesis.")
    chunks = mark_mid_sentence_cuts(chunks, para_boundaries)

    chunk_total = len(chunks)
    chunks_dir = job_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any leftover incomplete .part files from prior interrupted attempts
    for stale_part in chunks_dir.glob("*_part.wav"):
        stale_part.unlink(missing_ok=True)
    for stale_part in chunks_dir.glob("*.part*"):
        stale_part.unlink(missing_ok=True)

    # Check for already completed chunks
    existing_finished = [
        chunks_dir / f"chunk_{i:04d}.wav"
        for i in range(chunk_total)
        if (chunks_dir / f"chunk_{i:04d}.wav").exists() and (chunks_dir / f"chunk_{i:04d}.wav").stat().st_size > 44
    ]
    is_resumed = len(existing_finished) > 0

    # 3. Publish initial progress
    db = SessionLocal()
    try:
        JobCRUD.update_chunk_progress(
            db,
            job_id=job_id,
            chunk_done=len(existing_finished),
            chunk_total=chunk_total,
            eta_seconds=None,
        )
    finally:
        db.close()

    if pipeline_service is not None:
        service = pipeline_service
    else:
        from core.services.pipelines.chatterbox.chatterbox_service import ChatterboxService
        service = ChatterboxService()
    chunk_paths = []
    job_start_time = time.perf_counter()
    # Model-native 24 kHz: no pointless upsampling, ~45% smaller files than 44.1 kHz.
    sample_rate = settings.OUTPUT_SAMPLE_RATE
    import inspect
    try:
        accepts_sample_rate = "target_sample_rate" in inspect.signature(service.generate).parameters
    except (TypeError, ValueError):
        accepts_sample_rate = False

    try:
        for idx, chunk_text in enumerate(chunks):
            # Enforce overall job timeout
            if (time.perf_counter() - job_start_time) > settings.JOB_MAX_SECONDS:
                raise TimeoutError(f"Job exceeded overall timeout limit of {settings.JOB_MAX_SECONDS} seconds.")

            chunk_out = chunks_dir / f"chunk_{idx:04d}.wav"
            chunk_part = chunks_dir / f"chunk_{idx:04d}_part.wav"

            # Skip already finished chunk
            if chunk_out.exists() and chunk_out.stat().st_size > 44:
                logging.info(
                    "Skipping already synthesized chunk %d/%d (resuming)",
                    idx + 1,
                    chunk_total,
                )
                chunk_paths.append(chunk_out)
                chunk_done = idx + 1
                sample_rate = AudioService.probe_audio(chunk_out).sample_rate
                continue

            chunk_t0 = time.perf_counter()
            chunk_lang = "hi" if any("\u0900" <= c <= "\u097F" for c in chunk_text) else language

            # Synthesize chunk to .part file
            gen_kwargs = {
                "text": chunk_text,
                "reference_audio_path": str(ref_path),
                "output_path": str(chunk_part),
                "language": chunk_lang,
            }
            if accepts_sample_rate:
                gen_kwargs["target_sample_rate"] = sample_rate
            res = service.generate(**gen_kwargs)

            # Atomically rename .part to final chunk .wav
            if chunk_part.exists():
                chunk_part.replace(chunk_out)

            chunk_duration = time.perf_counter() - chunk_t0
            if chunk_duration > settings.CHUNK_TIMEOUT_SECONDS:
                logging.warning(
                    "Chunk %d/%d took %.1fs (timeout=%ds)",
                    idx + 1,
                    chunk_total,
                    chunk_duration,
                    settings.CHUNK_TIMEOUT_SECONDS,
                )

            chunk_paths.append(chunk_out)
            sample_rate = getattr(res, "sample_rate", sample_rate) or sample_rate

            # Calculate rolling ETA
            chunk_done = idx + 1
            elapsed = time.perf_counter() - job_start_time
            avg_per_chunk = elapsed / max(chunk_done - len(existing_finished), 1)
            remaining = chunk_total - chunk_done
            eta_seconds = int(avg_per_chunk * remaining) if remaining > 0 else 0

            db = SessionLocal()
            try:
                JobCRUD.update_chunk_progress(
                    db,
                    job_id=job_id,
                    chunk_done=chunk_done,
                    chunk_total=chunk_total,
                    eta_seconds=eta_seconds,
                )
            finally:
                db.close()

        # 4. Join chunks or copy single chunk
        db = SessionLocal()
        try:
            JobCRUD.update_job_status(
                db=db,
                job_id=job_id,
                status="processing",
                progress_message="Polishing audio and exporting formats...",
                stage="Polishing audio",
                stage_index=4,
                stage_total=4,
            )
        finally:
            db.close()

        # Trim, level-match and join over room tone (also cleans the edges of a single chunk)
        AudioService.join_chunks_natural(
            chunk_paths=chunk_paths,
            output_path=out_wav,
            chunk_texts=chunks,
            pause_sentence_ms=settings.CHUNK_PAUSE_SENTENCE_MS,
            pause_paragraph_ms=settings.CHUNK_PAUSE_PARAGRAPH_MS,
            paragraph_boundary_indices=para_boundaries,
            sample_rate=sample_rate,
            seed=int(hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:8], 16),
        )

        # 5. One encode: speed (pitch-preserving) + linear loudness + AI attribution tags
        try:
            AudioService.polish(out_wav, out_wav, speed=speed, sample_rate=sample_rate, ai_label=AudioService.AI_LABEL)
        except Exception as norm_err:
            logging.warning(f"Audio polish skipped or failed: {norm_err}")
            AudioService.add_ai_label_metadata(out_wav)  # the AI label is never skipped

        # 6. Encode the single MP3 deliverable from the master WAV
        format_sizes = AudioService.export_all_formats(out_wav, job_dir)

        # 7. Success: keep exactly output.wav + output.mp3 (drops chunks, reference, temp files)
        CleanupService.keep_only_final_outputs(job_dir)

        final_meta = AudioService.probe_audio(out_wav)
        return {
            "format": "wav",
            "duration_seconds": final_meta.duration_seconds,
            "sample_rate": final_meta.sample_rate,
            "language": language,
            "chunk_total": chunk_total,
            "ai_generated": True,
            "format_sizes": format_sizes,
            "ref_hash": ref_hash,
            "engine": "chatterbox_multilingual",
        }
    except Exception as exc:
        # Keep finished chunks, only purge partial incomplete .part files
        if chunks_dir.exists():
            for p in chunks_dir.glob("*_part.wav"):
                p.unlink(missing_ok=True)
            for p in chunks_dir.glob("*.part*"):
                p.unlink(missing_ok=True)
        raise exc
    finally:
        # Free cached activation memory once per job (not per chunk); the model stays resident
        release = getattr(service, "release_gpu_cache", None)
        if callable(release):
            release()


def production_inference_pipeline(
    job_id: str,
    kind: str,
    voice_id: Optional[str],
    payload: Dict[str, Any],
    job_dir: Path,
) -> Dict[str, Any]:
    """
    Standard production inference pipeline dispatching to Chatterbox TTS.

    Args:
        job_id (str): Unique job identifier token.
        kind (str): Pipeline type ('tts' or 'tts_custom').
        voice_id (Optional[str]): Registered voice identifier slug.
        payload (Dict[str, Any]): Job-specific parameters (text, language, speed, pitch_shift).
        job_dir (Path): Dedicated temporary execution directory for this job.

    Returns:
        Dict[str, Any]: Execution metadata including duration, sample rate, and artifact paths.

    Raises:
        RuntimeError: If execution fails or required assets are missing.
    """
    logging.info(f"Executing production_inference_pipeline for job '{job_id}' (kind={kind})")
    out_wav = job_dir / "output.wav"

    if kind == "tts":
        from core.cruds.voice_crud import VoiceCRUD
        from core.database.database import SessionLocal

        text = payload.get("text", "")
        language = payload.get("language", "en")
        speed = float(payload.get("speed", 1.0))

        db = SessionLocal()
        try:
            voice = VoiceCRUD.get_voice_by_id(db, voice_id) if voice_id else None
            if not voice:
                raise RuntimeError(f"Voice '{voice_id}' not found in catalog.")
            if voice.model_type != "chatterbox_reference":
                raise RuntimeError(f"Voice '{voice_id}' has unsupported model type: {voice.model_type}")
            
            ref_path = (settings.voices_path / voice.asset_file).resolve()
            if not ref_path.exists():
                logging.error(f"Voice '{voice_id}' asset file not found on disk: {ref_path}")
                raise RuntimeError(f"Voice '{voice_id}' asset file missing on disk: {voice.asset_file}")
            from core.services.pipelines.chatterbox.chatterbox_service import ChatterboxService
            pipeline_service = ChatterboxService()
        finally:
            db.close()

        return _run_long_form_tts(
            job_id=job_id,
            text=text,
            ref_path=ref_path,
            out_wav=out_wav,
            speed=speed,
            language=language,
            job_dir=job_dir,
            pipeline_service=pipeline_service,
        )

    elif kind == "tts_custom":
        text = payload.get("text", "")
        language = payload.get("language", "hi")
        speed = float(payload.get("speed", 1.0))
        ref_path = job_dir / "reference.wav"

        if not ref_path.exists():
            raise RuntimeError("Reference audio file missing in custom TTS job directory.")

        from core.services.pipelines.chatterbox.chatterbox_service import ChatterboxService
        pipeline_service = ChatterboxService()

        return _run_long_form_tts(
            job_id=job_id,
            text=text,
            ref_path=ref_path,
            out_wav=out_wav,
            speed=speed,
            language=language,
            job_dir=job_dir,
            pipeline_service=pipeline_service,
        )

    elif kind == "vc":
        raise RuntimeError("Voice-to-voice conversion is currently disabled (coming soon).")

    elif kind in ("preview", "tts_preview"):
        from core.cruds.voice_crud import VoiceCRUD
        from core.database.database import SessionLocal
        from core.services.audio.audio_service import AudioService
        import shutil

        raw_text = (payload.get("text") or "")[:200]
        language = payload.get("language", "hi")
        speed = float(payload.get("speed", 1.0))

        from core.services.llm.llm_service import LLMService
        text = LLMService.rule_based_cleanup(raw_text, language=language)
        ref_path = None
        # 1. Custom reference audio uploaded for this specific preview job takes precedence
        job_ref = job_dir / "reference.wav"
        if job_ref.exists() and job_ref.stat().st_size > 44:
            ref_path = job_ref
        elif voice_id:
            db = SessionLocal()
            try:
                voice = VoiceCRUD.get_voice_by_id(db, voice_id)
                if not voice:
                    logging.error(f"Preview voice '{voice_id}' not found in catalog")
                    raise RuntimeError(f"Voice '{voice_id}' not found in catalog.")
                if hasattr(voice, "asset_file") and voice.asset_file:
                    p = (settings.voices_path / voice.asset_file).resolve()
                    if p.exists():
                        ref_path = p
                    else:
                        logging.error(f"Voice '{voice_id}' asset file missing on disk: {p}")
                        raise RuntimeError(f"Voice '{voice_id}' asset file not found on disk: {voice.asset_file}")
            finally:
                db.close()

        if not ref_path or not ref_path.exists():
            logging.error(f"Preview job '{job_id}' rejected: no valid reference audio found")
            raise RuntimeError(f"Reference audio for preview job '{job_id}' not found. Please provide a valid voice or reference recording.")

        import hashlib
        ref_bytes = ref_path.read_bytes()
        ref_hash = hashlib.sha256(ref_bytes).hexdigest()[:16]
        logging.info(
            f"Conditioning preview job '{job_id}' on reference: hash='{ref_hash}', engine='chatterbox_multilingual', lang='{language}'"
        )

        db = SessionLocal()
        try:
            JobCRUD.update_job_status(
                db=db,
                job_id=job_id,
                status="processing",
                progress_message="Generating preview speech...",
                stage="Generating speech",
                stage_index=3,
                stage_total=4,
            )
        finally:
            db.close()

        from core.services.pipelines.chatterbox.chatterbox_service import ChatterboxService
        pipeline_service = ChatterboxService()

        preview_wav = job_dir / "preview.wav"
        pipeline_service.generate(
            text=text,
            reference_audio_path=str(ref_path),
            output_path=str(preview_wav),
            language=language,
            target_sample_rate=settings.OUTPUT_SAMPLE_RATE,
        )

        db = SessionLocal()
        try:
            JobCRUD.update_job_status(
                db=db,
                job_id=job_id,
                status="processing",
                progress_message="Polishing audio...",
                stage="Polishing audio",
                stage_index=4,
                stage_total=4,
            )
        finally:
            db.close()

        AudioService.polish(
            preview_wav, preview_wav, speed=speed, sample_rate=settings.OUTPUT_SAMPLE_RATE, ai_label=AudioService.AI_LABEL
        )
        meta = AudioService.probe_audio(preview_wav)
        CleanupService.keep_only_final_outputs(job_dir, keep={"preview.wav"})

        return {
            "format": "wav",
            "duration_seconds": round(meta.duration_seconds, 2),
            "sample_rate": meta.sample_rate,
            "language": language,
            "ai_generated": True,
            "confidence_label": "good",
            "text_preview": text,
            "ref_hash": ref_hash,
            "engine": "chatterbox_multilingual",
        }

    else:
        raise ValueError(f"Unknown job kind: '{kind}'")


# Process-wide singleton worker instance
job_worker = JobWorker()



if __name__ == "__main__":
    print("============================================================")
    print("AI Voice Studio - JobWorker Self-Check")
    print("============================================================")

    init_db()
    worker = JobWorker(maxsize=2)
    worker.start()

    def fake_pipeline(job_id: str, kind: str, voice_id: str, payload: dict, job_dir: Path) -> dict:
        """Simulate fake inference pipeline without loading real models."""
        out_wav = job_dir / "output.wav"
        out_wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00")
        time.sleep(0.2)
        return {
            "format": "wav",
            "duration_seconds": 1.0,
            "sample_rate": 24000,
            "size_bytes": out_wav.stat().st_size,
        }

    test_db = SessionLocal()
    try:
        # 1. Enqueue job
        token = secrets.token_urlsafe(16)
        expires = datetime.now(timezone.utc) + timedelta(minutes=settings.JOB_TTL_MINUTES)
        job_data = {
            "id": token,
            "kind": "tts",
            "voice_id": "my_voice",
            "status": "queued",
            "client_key": "fake_hashed_client_key",
            "payload": {"text": "Hello world from self check"},
            "expires_at": expires,
        }
        JobCRUD.create_job(test_db, job_data)
        print(f"Created fake job: {token} (status=queued)")

        worker.enqueue_job(token, fake_pipeline)
        print("Enqueued job to worker queue")

        # 2. Monitor transitions
        final_status = None
        for _ in range(50):
            time.sleep(0.1)
            j = JobCRUD.get_job_by_id(test_db, token)
            if j:
                print(f"State transition: status={j.status}, progress='{j.progress_message}'")
                if j.status in ("completed", "failed"):
                    final_status = j.status
                    break

        assert final_status == "completed", f"Expected completed, got {final_status}"
        print(f"Verified state transition sequence: queued -> processing -> completed")

        # 3. Test queue capacity
        token1 = secrets.token_urlsafe(16)
        token2 = secrets.token_urlsafe(16)
        token3 = secrets.token_urlsafe(16)
        for t in (token1, token2, token3):
            JobCRUD.create_job(test_db, {
                "id": t,
                "kind": "tts",
                "voice_id": "my_voice",
                "status": "queued",
                "client_key": "fake_hashed_client_key",
                "expires_at": expires,
            })

        worker.enqueue_job(token1, lambda *a, **k: time.sleep(1.0))
        worker.enqueue_job(token2, lambda *a, **k: time.sleep(1.0))
        queue_full_hit = False
        try:
            worker.enqueue_job(token3, lambda *a, **k: time.sleep(1.0))
        except QueueFullError:
            queue_full_hit = True
            print("Verified queue-full enforcement (raised QueueFullError as expected)")
        assert queue_full_hit, "Queue capacity check failed!"

        # 4. Test CleanupService expiry
        token_exp = secrets.token_urlsafe(16)
        past_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        JobCRUD.create_job(test_db, {
            "id": token_exp,
            "kind": "tts",
            "voice_id": "my_voice",
            "status": "completed",
            "client_key": "fake_hashed_client_key",
            "expires_at": past_time,
        })
        exp_dir = settings.tmp_path / token_exp
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "output.wav").write_bytes(b"dummy")

        cleaned = CleanupService.clean_expired_jobs(test_db)
        print(f"CleanupService cleaned {cleaned} expired job(s)")
        assert not exp_dir.exists(), "Expired job folder was not removed!"
        j_exp = JobCRUD.get_job_by_id(test_db, token_exp)
        assert j_exp.status == "expired", "Job status not updated to expired!"
        print("Verified expired job cleanup and folder removal successfully")

    finally:
        worker.stop()
        test_db.close()

    print("============================================================")
    print("JobWorker Self-Check PASSED")
    print("============================================================")
