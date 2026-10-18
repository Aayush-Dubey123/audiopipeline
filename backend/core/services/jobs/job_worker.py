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
                continue

            job_id, pipeline_fn = item
            if job_id is None:
                self.queue.task_done()
                break

            try:
                self._execute_job(job_id, pipeline_fn)
            except Exception as loop_error:
                logging.error(f"Unexpected error in JobWorker execution loop for job '{job_id}': {loop_error}")
            finally:
                self.queue.task_done()

        logging.info("JobWorker background loop terminated")

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

            JobCRUD.update_job_status(
                db=db,
                job_id=job_id,
                status="processing",
                progress_message="Starting inference generation...",
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

    Returns:
        Dict[str, Any]: Generated audio metadata.

    Raises:
        RuntimeError: If disk space is insufficient, reference audio is missing, or generation fails.
        TimeoutError: If execution exceeds chunk or job timeouts.
    """
    from core.services.pipelines.openvoice_service import OpenVoiceService, split_text_with_boundaries
    from core.services.audio.audio_service import AudioService
    from core.database.database import SessionLocal
    import shutil

    # 1. Enforce disk space safety before creating chunks
    AudioService.check_free_disk_gb(settings.tmp_path, required_gb=1.0)

    # 2. Split text into chunks respecting sentence and paragraph boundaries
    chunks, para_boundaries = split_text_with_boundaries(text, max_chars=settings.CHUNK_MAX_CHARS)
    if not chunks:
        raise RuntimeError("No valid text provided for synthesis.")

    chunk_total = len(chunks)
    chunks_dir = job_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any leftover incomplete .part files from prior interrupted attempts
    for stale_part in chunks_dir.glob("*.part"):
        stale_part.unlink(missing_ok=True)

    # Check for already completed chunks
    existing_finished = [
        chunks_dir / f"chunk_{i:04d}.wav"
        for i in range(chunk_total)
        if (chunks_dir / f"chunk_{i:04d}.wav").exists() and (chunks_dir / f"chunk_{i:04d}.wav").stat().st_size > 44
    ]
    is_resumed = len(existing_finished) > 0
    force_recompute_se = is_resumed  # Recompute speaker embedding on resume

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

    service = OpenVoiceService()
    chunk_paths = []
    job_start_time = time.perf_counter()
    sample_rate = 22050

    try:
        for idx, chunk_text in enumerate(chunks):
            # Enforce overall job timeout
            if (time.perf_counter() - job_start_time) > settings.JOB_MAX_SECONDS:
                raise TimeoutError(f"Job exceeded overall timeout limit of {settings.JOB_MAX_SECONDS} seconds.")

            chunk_out = chunks_dir / f"chunk_{idx:04d}.wav"
            chunk_part = chunks_dir / f"chunk_{idx:04d}.wav.part"

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
            res = service.generate(
                text=chunk_text,
                reference_audio_path=str(ref_path),
                output_path=str(chunk_part),
                speed=speed,
                language=chunk_lang,
                force_recompute_se=force_recompute_se,
            )
            force_recompute_se = False

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
            sample_rate = res.sample_rate

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
        if chunk_total == 1:
            shutil.copyfile(str(chunk_paths[0]), str(out_wav))
        else:
            AudioService.join_chunks_ffmpeg(
                chunk_paths=chunk_paths,
                output_path=out_wav,
                pause_sentence_ms=settings.CHUNK_PAUSE_SENTENCE_MS,
                pause_paragraph_ms=settings.CHUNK_PAUSE_PARAGRAPH_MS,
                paragraph_boundary_indices=para_boundaries,
                sample_rate=sample_rate,
            )

        # 5. Add synthetic AI attribution metadata
        AudioService.add_ai_label_metadata(out_wav)

        # 6. Purge intermediate chunks ONLY after successful join
        shutil.rmtree(chunks_dir, ignore_errors=True)

        final_meta = AudioService.probe_audio(out_wav)
        return {
            "format": "wav",
            "duration_seconds": final_meta.duration_seconds,
            "sample_rate": final_meta.sample_rate,
            "language": language,
            "chunk_total": chunk_total,
            "ai_generated": True,
        }
    except Exception as exc:
        # Keep finished chunks, only purge partial incomplete .part files
        if chunks_dir.exists():
            for p in chunks_dir.glob("*.part"):
                p.unlink(missing_ok=True)
        raise exc


def production_inference_pipeline(
    job_id: str,
    kind: str,
    voice_id: Optional[str],
    payload: Dict[str, Any],
    job_dir: Path,
) -> Dict[str, Any]:
    """
    Standard production inference pipeline dispatching to OpenVoice or RVC.

    Args:
        job_id (str): Unique job identifier token.
        kind (str): Pipeline type ('tts', 'tts_custom', or 'vc').
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
            if voice.model_type != "openvoice_reference":
                raise RuntimeError(f"Voice '{voice_id}' is not an OpenVoice reference model.")
            
            ref_path = (settings.voices_path / voice.asset_file).resolve()
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
        )

    elif kind == "tts_custom":
        text = payload.get("text", "")
        language = payload.get("language", "en")
        speed = float(payload.get("speed", 1.0))
        ref_path = job_dir / "reference.wav"

        if not ref_path.exists():
            raise RuntimeError("Reference audio file missing in custom TTS job directory.")

        return _run_long_form_tts(
            job_id=job_id,
            text=text,
            ref_path=ref_path,
            out_wav=out_wav,
            speed=speed,
            language=language,
            job_dir=job_dir,
        )

    elif kind == "vc":
        from core.services.pipelines.rvc_service import RVCService
        from core.services.audio.audio_service import AudioService
        from core.cruds.voice_crud import VoiceCRUD
        from core.database.database import SessionLocal

        pitch_shift = int(payload.get("pitch_shift", 0))
        src_path = job_dir / "source.wav"
        if not src_path.exists():
            raise RuntimeError("Source audio file missing in VC job directory.")

        db = SessionLocal()
        try:
            voice = VoiceCRUD.get_voice_by_id(db, voice_id) if voice_id else None
            if not voice:
                raise RuntimeError(f"Voice '{voice_id}' not found in catalog.")
            if voice.model_type != "rvc_model":
                raise RuntimeError(f"Voice '{voice_id}' is not an RVC model.")
            
            model_path = (settings.voices_path / voice.asset_file).resolve()
        finally:
            db.close()

        service = RVCService()
        rvc_res = service.convert(
            source_audio_path=str(src_path),
            voice_model_path=str(model_path),
            output_path=str(out_wav),
            pitch_shift=pitch_shift,
        )
        AudioService.add_ai_label_metadata(out_wav)

        return {
            "format": "wav",
            "duration_seconds": rvc_res.duration_seconds,
            "sample_rate": rvc_res.sample_rate,
            "pitch_shift": pitch_shift,
            "ai_generated": True,
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
