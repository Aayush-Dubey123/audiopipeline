"""
Audio upload and reference quality validation service.

Enforces extension allowlists, streaming file size limits, content probing,
and acoustic quality metrics for voice reference audio.
"""

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from fastapi import UploadFile

from core import logger
from core.config import get_settings
from core.services.audio.audio_service import AudioCorruptError, AudioService

logging = logger(__name__)
settings = get_settings()

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}


class AudioValidationError(Exception):
    """Base exception for audio file and stream validation failures."""
    pass


class InvalidAudioExtensionError(AudioValidationError):
    """Raised when an uploaded file extension is not in the allowed list."""
    pass


class AudioFileSizeExceededError(AudioValidationError):
    """Raised when uploaded file exceeds the configured maximum upload size."""
    pass


class AudioDurationExceededError(AudioValidationError):
    """Raised when audio duration exceeds the configured maximum length."""
    pass


class ReferenceQualityError(AudioValidationError):
    """Raised when reference audio fails minimum acoustic quality thresholds."""
    pass


@dataclass(frozen=True)
class QualityMetrics:
    """
    Diagnostic acoustic measurements for reference voice audio.
    """
    duration_seconds: float
    sample_rate: int
    channels: int
    rms_volume: float
    peak_amplitude: float
    silence_ratio: float
    is_clipped: bool
    is_too_quiet: bool
    quality_score: float
    warnings: List[str] = field(default_factory=list)


class ValidationService:
    """
    Service enforcing security boundaries and audio quality validation rules.
    """

    @staticmethod
    def sanitize_and_validate_extension(filename: Optional[str]) -> str:
        """
        Validate that the uploaded filename has an approved audio extension.

        Args:
            filename (Optional[str]): Original client-supplied filename.

        Returns:
            str: Verified lowercase extension including the leading dot.

        Raises:
            InvalidAudioExtensionError: If extension is missing or unapproved.
        """
        logging.info(f"Executing ValidationService.sanitize_and_validate_extension for '{filename}'")
        if not filename:
            logging.warning("Rejected upload with empty filename")
            raise InvalidAudioExtensionError("Audio file must have a valid name and extension.")

        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_AUDIO_EXTENSIONS:
            logging.warning(f"Rejected upload with forbidden extension: '{ext}'")
            raise InvalidAudioExtensionError(
                f"Unsupported audio format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
            )

        return ext

    @staticmethod
    async def stream_upload_to_disk(
        upload_file: UploadFile,
        destination_dir: Path,
        max_bytes: Optional[int] = None,
    ) -> Path:
        """
        Stream an incoming HTTP multipart file directly to disk with a hard byte cutoff.

        Uses an unguessable UUID filename to prevent path traversal and collision vulnerabilities.

        Args:
            upload_file (UploadFile): FastApi multipart upload file handle.
            destination_dir (Path): Secure server directory where the file will be saved.
            max_bytes (Optional[int]): Maximum allowed byte size before truncation abort.

        Returns:
            Path: Path to the securely saved file on disk.

        Raises:
            AudioFileSizeExceededError: If the upload exceeds the byte limit.
            InvalidAudioExtensionError: If the extension is invalid.
        """
        logging.info("Executing ValidationService.stream_upload_to_disk")
        ext = ValidationService.sanitize_and_validate_extension(upload_file.filename)
        destination_dir.mkdir(parents=True, exist_ok=True)

        safe_filename = f"{uuid.uuid4().hex}{ext}"
        destination_path = (destination_dir / safe_filename).resolve()

        # Enforce that destination resides strictly inside destination_dir
        if not str(destination_path).startswith(str(destination_dir.resolve())):
            logging.error("Security violation: path traversal attempt during upload")
            raise AudioValidationError("Invalid target path.")

        cutoff = max_bytes if max_bytes is not None else settings.MAX_UPLOAD_MB * 1024 * 1024
        total_bytes = 0
        chunk_size = 64 * 1024  # 64KB chunks

        try:
            with open(destination_path, "wb") as buffer:
                while chunk := await upload_file.read(chunk_size):
                    total_bytes += len(chunk)
                    if total_bytes > cutoff:
                        buffer.close()
                        if destination_path.exists():
                            destination_path.unlink()
                        logging.warning(f"Upload exceeded size limit: {total_bytes} > {cutoff} bytes")
                        raise AudioFileSizeExceededError(
                            f"File size exceeds maximum allowed limit of {settings.MAX_UPLOAD_MB} MB."
                        )
                    buffer.write(chunk)
        except Exception:
            if destination_path.exists():
                destination_path.unlink()
            raise

        logging.info(f"Uploaded file safely written: {destination_path.name} ({total_bytes} bytes)")
        return destination_path

    @staticmethod
    def validate_audio_duration(
        file_path: Path,
        max_seconds: Optional[int] = None,
        min_seconds: float = 0.5,
    ) -> float:
        """
        Probe audio file content and enforce minimum and maximum duration boundaries.

        Args:
            file_path (Path): Target audio file path on disk.
            max_seconds (Optional[int]): Upper duration threshold in seconds.
            min_seconds (float): Lower duration threshold in seconds.

        Returns:
            float: Validated duration in seconds.

        Raises:
            AudioCorruptError: If the file cannot be probed as valid audio.
            AudioDurationExceededError: If audio is longer than max_seconds or shorter than min_seconds.
        """
        logging.info(f"Executing ValidationService.validate_audio_duration on {file_path.name}")
        meta = AudioService.probe_audio(file_path)

        upper_bound = max_seconds if max_seconds is not None else settings.MAX_AUDIO_SECONDS
        if meta.duration_seconds > upper_bound:
            logging.warning(
                f"Audio duration {meta.duration_seconds}s exceeds limit {upper_bound}s"
            )
            raise AudioDurationExceededError(
                f"Audio duration ({meta.duration_seconds:.1f}s) exceeds maximum allowed length of {upper_bound} seconds."
            )

        if meta.duration_seconds < min_seconds:
            logging.warning(
                f"Audio duration {meta.duration_seconds}s below minimum {min_seconds}s"
            )
            raise AudioDurationExceededError(
                f"Audio is too short ({meta.duration_seconds:.1f}s). Minimum required duration is {min_seconds} seconds."
            )

        return meta.duration_seconds

    @staticmethod
    def inspect_reference_audio_quality(
        file_path: Path,
        min_duration: float = 3.0,
        max_duration: float = 60.0,
    ) -> QualityMetrics:
        """
        Analyze acoustic quality of reference voice audio and provide actionable guidance.

        Evaluates duration, signal-to-noise ratio / RMS loudness, clipping distortion,
        and silence ratios to ensure successful voice cloning.

        Args:
            file_path (Path): Path to the reference audio WAV file.
            min_duration (float): Minimum duration for reference cloning (default 3.0s).
            max_duration (float): Maximum duration for reference cloning (default 60.0s).

        Returns:
            QualityMetrics: Comprehensive acoustic quality evaluation.

        Raises:
            ReferenceQualityError: If reference audio fails critical usability criteria.
            AudioCorruptError: If audio file cannot be decoded.
        """
        logging.info(f"Executing ValidationService.inspect_reference_audio_quality on {file_path.name}")
        meta = AudioService.probe_audio(file_path)

        warnings: List[str] = []

        if meta.duration_seconds < min_duration:
            raise ReferenceQualityError(
                f"Reference audio is too short ({meta.duration_seconds:.1f}s). Please provide at least {min_duration:.0f} seconds of clear speech (10–30s recommended)."
            )

        if meta.duration_seconds > max_duration:
            warnings.append(
                f"Audio is longer than recommended ({meta.duration_seconds:.1f}s). Only the first {max_duration:.0f}s will be prioritized."
            )

        # Load samples for waveform inspection
        try:
            data, sr = sf.read(str(file_path), always_2d=True)
            mono_data = np.mean(data, axis=1)
        except Exception as read_err:
            logging.error(f"Failed to read audio waveform for quality inspection: {read_err}")
            raise AudioCorruptError(f"Audio waveform decoding failed: {read_err}")

        total_samples = len(mono_data)
        if total_samples == 0:
            raise ReferenceQualityError("Audio file contains no audible samples.")

        # 1. Amplitude metrics
        peak = float(np.max(np.abs(mono_data)))
        rms = float(np.sqrt(np.mean(mono_data ** 2) + 1e-10))

        is_too_quiet = bool(peak < 0.05 or rms < 0.01)
        if is_too_quiet:
            raise ReferenceQualityError(
                "Reference audio is too quiet. Please speak closer to the microphone or boost input gain."
            )

        # 2. Digital clipping check (values hitting near 1.0)
        clipped_samples = int(np.sum(np.abs(mono_data) >= 0.995))
        clipping_ratio = clipped_samples / total_samples
        is_clipped = bool(clipping_ratio > 0.005)
        if is_clipped:
            warnings.append(
                "Noticeable digital clipping detected. Lower your microphone input gain to avoid distortion."
            )

        # 3. Silence ratio check (frame-level RMS below 0.005)
        frame_len = int(sr * 0.02)  # 20ms frames
        if frame_len > 0:
            num_frames = total_samples // frame_len
            silent_frames = 0
            for i in range(num_frames):
                frame = mono_data[i * frame_len : (i + 1) * frame_len]
                frame_rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
                if frame_rms < 0.005:
                    silent_frames += 1
            silence_ratio = silent_frames / max(num_frames, 1)
        else:
            silence_ratio = 0.0

        if silence_ratio > 0.70:
            raise ReferenceQualityError(
                "Audio contains over 70% background silence. Please provide continuous speaking without long pauses."
            )
        elif silence_ratio > 0.40:
            warnings.append("Consider trimming prolonged pauses for optimal tone color extraction.")

        # Compute overall quality score (0.0 to 1.0)
        score = 1.0
        if meta.duration_seconds < 10.0:
            score -= 0.15
        if is_clipped:
            score -= 0.25
        if silence_ratio > 0.30:
            score -= 0.15
        score = max(round(score, 2), 0.1)

        logging.info(
            f"Reference quality: duration={meta.duration_seconds}s, rms={rms:.4f}, peak={peak:.4f}, "
            f"silence={silence_ratio:.2%}, score={score}, warnings={len(warnings)}"
        )

        return QualityMetrics(
            duration_seconds=meta.duration_seconds,
            sample_rate=meta.sample_rate,
            channels=meta.channels,
            rms_volume=round(rms, 4),
            peak_amplitude=round(peak, 4),
            silence_ratio=round(silence_ratio, 3),
            is_clipped=is_clipped,
            is_too_quiet=is_too_quiet,
            quality_score=score,
            warnings=warnings,
        )
