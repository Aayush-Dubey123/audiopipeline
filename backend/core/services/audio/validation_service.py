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


def _looks_like_audio(head: bytes) -> bool:
    """
    Check file magic bytes so a renamed playlist/script (e.g. an HLS .m3u8 saved as
    .wav, which ffmpeg would follow to read local files or fetch URLs) is rejected.
    """
    if len(head) < 12:
        return False
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return True
    if head[:3] == b"ID3" or (head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return True  # MP3 (ID3 tag or MPEG frame sync)
    if head[4:8] == b"ftyp":
        return True  # MP4/M4A
    if head[:4] in (b"OggS", b"fLaC", b"\x1aE\xdf\xa3"):
        return True  # Ogg, FLAC, WebM/Matroska
    return False


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
                    if total_bytes == len(chunk) and not _looks_like_audio(chunk[:16]):
                        raise InvalidAudioExtensionError(
                            "File content is not a recognised audio format."
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
        wav_temp_to_clean: Optional[Path] = None
        try:
            try:
                data, sr = sf.read(str(file_path), always_2d=True)
            except Exception:
                wav_temp = file_path.parent / f"transcoded_{file_path.stem}_{uuid.uuid4().hex[:8]}.wav"
                AudioService.to_wav(file_path, wav_temp)
                wav_temp_to_clean = wav_temp
                data, sr = sf.read(str(wav_temp), always_2d=True)
            mono_data = np.mean(data, axis=1)
        except Exception as read_err:
            logging.error(f"Failed to read audio waveform for quality inspection: {read_err}")
            raise AudioCorruptError(f"Audio waveform decoding failed: {read_err}")
        finally:
            if wav_temp_to_clean and wav_temp_to_clean.exists():
                try:
                    wav_temp_to_clean.unlink(missing_ok=True)
                except Exception:
                    pass

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

    @classmethod
    def check_reference_safety(cls, file_path: Path) -> dict:
        """
        Comprehensive acoustic safety inspection for reference voice cloning clips.

        Evaluates speech duration, loudness, background noise / SNR, digital clipping,
        echo / reverb level, speaker consistency (multi-speaker detection), and background music.
        Returns a 0-100 score, usable flag (hard-blocks unusable audio), and plain advice.

        Args:
            file_path (Path): Path to the candidate reference audio file.

        Returns:
            dict: Diagnostic safety evaluation with score, usability, advice, and acoustic metrics.
        """
        from scipy import signal

        file_path = Path(file_path)
        logging.info(f"Executing ValidationService.check_reference_safety on {file_path.name}")
        meta = AudioService.probe_audio(file_path)

        wav_temp_to_clean: Optional[Path] = None
        try:
            try:
                data, sr = sf.read(str(file_path), always_2d=True)
            except Exception:
                wav_temp = file_path.parent / f"transcoded_{file_path.stem}_{uuid.uuid4().hex[:8]}.wav"
                AudioService.to_wav(file_path, wav_temp)
                wav_temp_to_clean = wav_temp
                data, sr = sf.read(str(wav_temp), always_2d=True)
            mono_data = np.mean(data, axis=1)
        except Exception as read_err:
            logging.error(f"Failed to read audio waveform for safety check: {read_err}")
            raise AudioCorruptError(f"Audio waveform decoding failed: {read_err}")
        finally:
            if wav_temp_to_clean and wav_temp_to_clean.exists():
                try:
                    wav_temp_to_clean.unlink(missing_ok=True)
                except Exception:
                    pass

        total_samples = len(mono_data)
        if total_samples == 0:
            return {
                "score": 0,
                "is_usable": False,
                "advice": ["Audio file contains no audible samples."],
                "metrics": {
                    "speech_length_s": 0.0,
                    "total_duration_s": 0.0,
                    "loudness_dbfs": -99.0,
                    "snr_db": 0.0,
                    "clipping_ratio": 0.0,
                    "echo_reverb": "unknown",
                    "speaker_consistency": 0.0,
                    "multi_speaker_detected": False,
                    "music_detected": False,
                },
            }

        total_duration = total_samples / sr
        peak = float(np.max(np.abs(mono_data)))
        rms = float(np.sqrt(np.mean(mono_data ** 2) + 1e-10))
        loudness_dbfs = float(round(20.0 * np.log10(rms + 1e-9), 1))

        # Clipping ratio
        clipped_samples = int(np.sum(np.abs(mono_data) >= 0.995))
        clipping_ratio = round(clipped_samples / max(total_samples, 1), 4)

        # Frame energy (25ms window, 10ms hop)
        w = int(sr * 0.025)
        hop = int(sr * 0.010)
        frames = [mono_data[i : i + w] for i in range(0, total_samples - w, hop)]
        if not frames:
            return {
                "score": 0,
                "is_usable": False,
                "advice": ["Audio is too short to analyze."],
                "metrics": {
                    "speech_length_s": 0.0,
                    "total_duration_s": round(total_duration, 2),
                    "loudness_dbfs": loudness_dbfs,
                    "snr_db": 0.0,
                    "clipping_ratio": clipping_ratio,
                    "echo_reverb": "unknown",
                    "speaker_consistency": 0.0,
                    "multi_speaker_detected": False,
                    "music_detected": False,
                },
            }

        energies = np.array([np.sqrt(np.mean(f ** 2) + 1e-10) for f in frames])
        en_db = 20.0 * np.log10(energies + 1e-9)
        noise_floor = float(np.percentile(en_db, 10))
        speech_peak = float(np.percentile(en_db, 90))
        snr_db = round(max(0.0, speech_peak - noise_floor), 1)

        # Speech length calculation based on dynamic SNR range
        if snr_db >= 4.0:
            dynamic_thresh = noise_floor + 0.30 * (speech_peak - noise_floor)
            speech_thresh = min(dynamic_thresh, speech_peak - 3.0)
        else:
            speech_thresh = noise_floor + 2.0

        speech_mask = en_db >= speech_thresh
        speech_frames = int(np.sum(speech_mask))
        speech_length = round(speech_frames * (hop / sr), 2)

        # Fallback safeguard: if total duration is >= 2.0s and audio has audible volume,
        # speech length must never drop to 0.0 if frames exceed basic noise floor
        if speech_length < 0.5 and total_duration >= 2.0 and loudness_dbfs > -52.0:
            audible_mask = en_db >= (noise_floor + 2.0)
            speech_length = round(float(np.sum(audible_mask)) * (hop / sr), 2)

        # Reverb / echo estimate
        reverb_level = "low"
        pause_mask = ~speech_mask & (en_db > -55.0)
        if np.sum(pause_mask) > int(sr * 0.2 / hop):
            decay_rates = []
            for i in range(1, len(speech_mask) - 10):
                if speech_mask[i - 1] and not speech_mask[i]:
                    drop = en_db[i - 1] - en_db[min(i + 9, len(en_db) - 1)]
                    decay_rates.append(drop)
            if decay_rates:
                med_decay = float(np.median(decay_rates))
                if med_decay < 9.0:
                    reverb_level = "high"
                elif med_decay < 16.0:
                    reverb_level = "moderate"

        # Multi-speaker detection (spectral consistency across speech chunks)
        multi_speaker = False
        speaker_consistency = 1.0
        speech_indices = np.where(speech_mask)[0]
        if len(speech_indices) > 60:
            chunk_size = len(speech_indices) // 3
            chunks = [speech_indices[k * chunk_size : (k + 1) * chunk_size] for k in range(3)]
            chunk_specs = []
            for c in chunks:
                sig_parts = np.concatenate([frames[idx] for idx in c])
                if len(sig_parts) > int(sr * 0.1):
                    _, _, Sxx = signal.spectrogram(sig_parts, fs=sr, nperseg=int(sr * 0.025))
                    bands = np.array_split(Sxx, 20, axis=0)
                    feat = np.array([np.mean(np.log(np.mean(b, axis=0) + 1e-10)) for b in bands])
                    chunk_specs.append(feat)
            if len(chunk_specs) == 3:
                sims = []
                for i in range(len(chunk_specs)):
                    for j in range(i + 1, len(chunk_specs)):
                        dot = float(np.dot(chunk_specs[i], chunk_specs[j]))
                        norm = float(np.linalg.norm(chunk_specs[i]) * np.linalg.norm(chunk_specs[j]) + 1e-9)
                        sims.append(dot / norm)
                speaker_consistency = round(float(np.min(sims)), 3)
                if speaker_consistency < 0.88:
                    multi_speaker = True

        # Music detection in pause frames
        music_detected = False
        if np.sum(pause_mask) > int(sr * 0.5 / hop):
            pause_samples = np.concatenate([frames[idx] for idx in np.where(pause_mask)[0]])
            if len(pause_samples) > sr:
                corr = np.correlate(pause_samples[: sr * 2], pause_samples[: sr * 2], mode="full")
                mid = len(corr) // 2
                corr = corr[mid:]
                corr = corr / (corr[0] + 1e-9)
                min_lag = int(sr / 800)
                max_lag = int(sr / 60)
                if max_lag < len(corr):
                    max_peak = float(np.max(corr[min_lag:max_lag]))
                    if max_peak > 0.65:
                        music_detected = True

        # Score & plain-language advice
        score = 100
        advice = []
        hard_block = False

        if speech_length < 3.0:
            hard_block = True
            score -= 45
            advice.append(f"Hard block: Speech is too short ({speech_length:.1f}s). Provide at least 5 to 15 seconds of speaking.")
        elif speech_length < 8.0:
            score -= 15
            advice.append(f"Warning: Speech duration ({speech_length:.1f}s) is slightly short. 10–25s yields best cloning.")
        else:
            advice.append(f"Speech length ({speech_length:.1f}s) is good for voice cloning.")

        if loudness_dbfs < -36.0:
            hard_block = True
            score -= 40
            advice.append(f"Hard block: Audio is too quiet ({loudness_dbfs} dBFS). Speak closer to the microphone.")
        elif loudness_dbfs < -26.0:
            score -= 10
            advice.append(f"Warning: Audio is somewhat quiet ({loudness_dbfs} dBFS). Consider increasing recording volume.")
        elif loudness_dbfs > -6.0:
            score -= 10
            advice.append(f"Warning: Audio is very loud ({loudness_dbfs} dBFS) and close to 0 dBFS.")
        else:
            advice.append(f"Loudness level ({loudness_dbfs} dBFS) is optimal.")

        if clipping_ratio > 0.015:
            hard_block = True
            score -= 40
            advice.append(f"Hard block: Severe clipping distortion detected ({clipping_ratio:.1%}). Lower microphone gain.")
        elif clipping_ratio > 0.002:
            score -= 15
            advice.append(f"Warning: Mild clipping detected ({clipping_ratio:.2%}). Slightly reduce input level.")

        if snr_db < 8.0:
            hard_block = True
            score -= 40
            advice.append(f"Hard block: Heavy background noise (SNR {snr_db} dB). Record in a quiet room.")
        elif snr_db < 16.0:
            score -= 15
            advice.append(f"Warning: Moderate background noise (SNR {snr_db} dB). A quieter mic will improve quality.")
        else:
            advice.append(f"Background noise level is low (SNR {snr_db} dB).")

        if reverb_level == "high":
            score -= 15
            advice.append("Warning: Noticeable room echo or reverb detected. Record in a carpeted room or closer to mic.")
        elif reverb_level == "moderate":
            score -= 5
            advice.append("Slight room ambience detected.")

        if multi_speaker:
            score -= 25
            advice.append(f"Warning: Possible multiple speakers or shifting voice detected (consistency: {speaker_consistency:.2f}).")

        if music_detected:
            score -= 25
            advice.append("Warning: Background music or melodic tone detected. Clean speech without background music gives best results.")

        score = max(0, min(100, score))
        is_usable = (not hard_block) and (score >= 40)

        return {
            "score": score,
            "is_usable": is_usable,
            "advice": advice,
            "metrics": {
                "speech_length_s": speech_length,
                "total_duration_s": round(total_duration, 2),
                "loudness_dbfs": loudness_dbfs,
                "snr_db": snr_db,
                "clipping_ratio": clipping_ratio,
                "echo_reverb": reverb_level,
                "speaker_consistency": speaker_consistency,
                "multi_speaker_detected": multi_speaker,
                "music_detected": music_detected,
            },
        }
