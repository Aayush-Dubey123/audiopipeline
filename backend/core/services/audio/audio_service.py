"""
Audio processing service for AI Voice Studio.

Provides robust audio inspection, format conversion, resampling, normalization,
MP3 encoding, and synthetic AI labeling metadata.
"""

import io
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf
from core import logger

logging = logger(__name__)


class AudioProcessingError(Exception):
    """Base exception for all audio processing and transcoding failures."""
    pass


class AudioCorruptError(AudioProcessingError):
    """Raised when an audio file cannot be decoded, is truncated, or has corrupt streams."""
    pass


class FFmpegNotFoundError(AudioProcessingError):
    """Raised when the FFmpeg or ffprobe executable cannot be located in the system PATH."""
    pass


class _RoomTone:
    """
    Continuous room-tone stream built by overlap-adding random slices of recorded noise.

    Uses power-complementary sine windows at 50% overlap, so uncorrelated noise keeps a
    constant level with no clicks across successive take() calls. Falls back to soft
    synthetic noise at the measured floor when too little real room tone is available.
    """

    SEG = 2048

    def __init__(self, bank: np.ndarray, rng: np.random.Generator, level_rms: float) -> None:
        self.rng = rng
        self.hop = self.SEG // 2
        self.win = np.sin(np.pi * (np.arange(self.SEG) + 0.5) / self.SEG).astype(np.float32)
        if bank.size < self.SEG * 2:
            if level_rms < 1e-5:
                bank = np.zeros(self.SEG * 2, dtype=np.float32)
            else:
                noise = np.convolve(rng.standard_normal(48000), [0.25, 0.5, 0.25], mode="same")
                bank = (noise * (level_rms / (np.sqrt(np.mean(noise ** 2)) + 1e-12))).astype(np.float32)
        self.bank = bank.astype(np.float32)
        # Prime with half a window so the stream starts at full level instead of ramping up from zero
        off = int(self.rng.integers(0, self.bank.size - self.SEG + 1))
        self.tail = (self.bank[off:off + self.SEG] * self.win)[self.hop:].copy()
        self.buf = np.zeros(0, dtype=np.float32)

    def take(self, n: int) -> np.ndarray:
        """Return the next n samples of the stream."""
        n = max(int(n), 0)
        pieces, have = [self.buf], self.buf.size
        while have < n:
            off = int(self.rng.integers(0, self.bank.size - self.SEG + 1))
            seg = self.bank[off:off + self.SEG] * self.win
            pieces.append(self.tail + seg[:self.hop])
            self.tail = seg[self.hop:].copy()
            have += self.hop
        out = np.concatenate(pieces)
        self.buf = out[n:]
        return out[:n]


@dataclass(frozen=True)
class AudioMetadata:
    """
    Extracted structural metadata describing an audio stream.
    """
    duration_seconds: float
    sample_rate: int
    channels: int
    format_name: str


class AudioService:
    """
    Unified audio manipulation engine executing verified subprocess routines.
    """

    @staticmethod
    def _check_ffmpeg_tool(tool_name: str) -> None:
        """
        Verify that the requested CLI audio tool exists in PATH.

        If not immediately located in active environment, refreshes PATH from
        local Windows WinGet links directory.

        Args:
            tool_name (str): The executable name (e.g. 'ffmpeg', 'ffprobe').

        Returns:
            None

        Raises:
            FFmpegNotFoundError: If the tool is not discovered in system PATH.
        """
        if shutil.which(tool_name) is not None:
            return

        # Dynamically discover newly installed WinGet links on Windows
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            winget_links = Path(local_app_data) / "Microsoft" / "WinGet" / "Links"
            if winget_links.exists() and str(winget_links) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{winget_links};{os.environ.get('PATH', '')}"

        if shutil.which(tool_name) is None:
            logging.error(f"Required multimedia binary '{tool_name}' was not found in system PATH")
            raise FFmpegNotFoundError(f"'{tool_name}' is not installed or not in system PATH.")

    @classmethod
    def probe_audio(cls, file_path: Path) -> AudioMetadata:
        """
        Extract stream parameters and metadata using ffprobe.

        Args:
            file_path (Path): Path to the target audio file.

        Returns:
            AudioMetadata: Container with duration, sample rate, channels, and container format.

        Raises:
            FFmpegNotFoundError: If ffprobe is not accessible in system PATH.
            AudioCorruptError: If the file is unreadable, invalid, or corrupt.
        """
        cls._check_ffmpeg_tool("ffprobe")

        if not file_path.exists():
            logging.error(f"Target audio file does not exist: {file_path}")
            raise AudioCorruptError(f"Audio file does not exist: {file_path}")

        logging.info(f"Executing AudioService.probe_audio on {file_path.name}")

        cmd = [
            "ffprobe",
            "-v", "error",
            "-protocol_whitelist", "file",
            "-show_entries", "format=duration,format_name:stream=duration,sample_rate,channels",
            "-of", "json",
            str(file_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            data = json.loads(result.stdout)

            format_info = data.get("format", {})
            streams = data.get("streams", [])

            if not streams:
                logging.warning(f"No audio streams detected in file {file_path.name}")
                raise AudioCorruptError(f"No valid audio streams found in {file_path.name}")

            audio_stream = streams[0]
            format_dur = float(format_info.get("duration", 0.0))
            stream_dur = audio_stream.get("duration")
            if stream_dur and stream_dur != "N/A":
                try:
                    parsed_stream_dur = float(stream_dur)
                    duration = parsed_stream_dur if parsed_stream_dur > 0 else format_dur
                except (ValueError, TypeError):
                    duration = format_dur
            else:
                duration = format_dur

            sample_rate = int(audio_stream.get("sample_rate", 0))
            channels = int(audio_stream.get("channels", 0))
            format_name = format_info.get("format_name", "unknown")

            if sample_rate <= 0 or channels <= 0:
                raise AudioCorruptError("Audio stream reports invalid sample rate or zero channels.")

            return AudioMetadata(
                duration_seconds=round(duration, 3),
                sample_rate=sample_rate,
                channels=channels,
                format_name=format_name,
            )

        except subprocess.TimeoutExpired:
            logging.error(f"ffprobe timed out while analyzing {file_path.name}")
            raise AudioCorruptError("Probe command timed out reading audio file.")
        except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as error:
            logging.error(f"Error in AudioService.probe_audio: {error}")
            raise AudioCorruptError(f"Failed to probe audio file: {error}")

    @classmethod
    def read_audio_tags(cls, file_path: Path) -> dict:
        """
        Extract metadata tags (ID3 or container tags) from an audio file.

        Args:
            file_path (Path): Path to the audio file.

        Returns:
            dict: Dictionary of key-value tags found in format metadata.
        """
        cls._check_ffmpeg_tool("ffprobe")
        if not file_path.exists():
            return {}

        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_entries", "format_tags",
            str(file_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=15)
            data = json.loads(result.stdout)
            return data.get("format", {}).get("tags", {})
        except Exception as error:
            logging.warning(f"Failed to read audio tags from {file_path.name}: {error}")
            return {}

    @classmethod
    def to_wav(
        cls, input_path: Path, output_path: Path, sample_rate: Optional[int] = None, mono: bool = False
    ) -> Path:
        """
        Transcode an arbitrary audio file to standard 16-bit PCM WAV (optionally resampled/mono in the same pass).

        Args:
            input_path (Path): Source audio path.
            output_path (Path): Target WAV path.
            sample_rate (Optional[int]): Resample to this rate in the same ffmpeg run.
            mono (bool): Downmix to mono in the same ffmpeg run.

        Returns:
            Path: Path to the generated WAV file.

        Raises:
            FFmpegNotFoundError: If ffmpeg is missing.
            AudioCorruptError: If transcoding fails.
        """
        cls._check_ffmpeg_tool("ffmpeg")
        logging.info(f"Executing AudioService.to_wav from {input_path.name} to {output_path.name}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-protocol_whitelist", "file",
            "-i", str(input_path),
            "-vn",
            *(["-ar", str(int(sample_rate))] if sample_rate else []),
            *(["-ac", "1"] if mono else []),
            "-c:a", "pcm_s16le",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            return output_path
        except subprocess.TimeoutExpired:
            output_path.unlink(missing_ok=True)
            raise AudioCorruptError("Transcoding to WAV timed out.")
        except subprocess.CalledProcessError as error:
            logging.error(f"Error in AudioService.to_wav: {error.stderr}")
            raise AudioCorruptError(f"Transcoding to WAV failed: {error.stderr}")

    @classmethod
    def resample(cls, input_path: Path, output_path: Path, target_sample_rate: int = 24000) -> Path:
        """
        Resample an audio file to a specified target sample rate and mono channel.

        Args:
            input_path (Path): Source audio path.
            output_path (Path): Target resampled path.
            target_sample_rate (int): Sampling rate in Hz (default 24000).

        Returns:
            Path: Path to the resampled output file.

        Raises:
            FFmpegNotFoundError: If ffmpeg is missing.
            AudioProcessingError: If resampling fails.
        """
        cls._check_ffmpeg_tool("ffmpeg")
        logging.info(f"Executing AudioService.resample to {target_sample_rate}Hz on {input_path.name}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        in_place = input_path.resolve() == output_path.resolve()
        target_file = (
            output_path.parent / f"_tmp_{uuid.uuid4().hex[:8]}_{output_path.name}"
            if in_place
            else output_path
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-protocol_whitelist", "file",
            "-i", str(input_path),
            "-ar", str(target_sample_rate),
            "-ac", "1",
            str(target_file),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            if in_place:
                shutil.move(str(target_file), str(output_path))
            return output_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            if in_place and target_file.exists():
                target_file.unlink()
            logging.error(f"Error in AudioService.resample: {getattr(error, 'stderr', error)}")
            raise AudioProcessingError("Resampling failed.")

    @classmethod
    def normalize_loudness(cls, input_path: Path, output_path: Path) -> Path:
        """
        Apply light EBU R128 loudness normalization.

        Args:
            input_path (Path): Input audio file path.
            output_path (Path): Output normalized audio file path.

        Returns:
            Path: Path to the loudness-normalized audio file.

        Raises:
            FFmpegNotFoundError: If ffmpeg is missing.
            AudioProcessingError: If normalization fails.
        """
        cls._check_ffmpeg_tool("ffmpeg")
        logging.info(f"Executing AudioService.normalize_loudness on {input_path.name}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        in_place = input_path.resolve() == output_path.resolve()
        target_file = (
            output_path.parent / f"_tmp_{uuid.uuid4().hex[:8]}_{output_path.name}"
            if in_place
            else output_path
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            str(target_file),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
            if in_place:
                shutil.move(str(target_file), str(output_path))
            return output_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            if in_place and target_file.exists():
                target_file.unlink()
            logging.error(f"Error in AudioService.normalize_loudness: {getattr(error, 'stderr', error)}")
            raise AudioProcessingError("Loudness normalization failed.")

    @classmethod
    def export_mp3(
        cls,
        input_path: Path,
        output_path: Path,
        bitrate: str = "192k",
        mono: bool = False,
        label: str = "AI-generated synthetic voice by AI Voice Studio",
    ) -> Path:
        """
        Encode an audio file to MP3 format with a target bitrate and ID3 AI tags.

        Args:
            input_path (Path): Source audio path.
            output_path (Path): Destination MP3 path.
            bitrate (str): Target MP3 constant bitrate (e.g., '192k', '64k').
            mono (bool): Whether to downmix to single mono channel.
            label (str): Attribution message to record in ID3 tags.

        Returns:
            Path: Path to the generated MP3 file.

        Raises:
            FFmpegNotFoundError: If ffmpeg is missing.
            AudioProcessingError: If MP3 encoding fails.
        """
        cls._check_ffmpeg_tool("ffmpeg")
        logging.info(
            f"Executing AudioService.export_mp3 at {bitrate} (mono={mono}) for {input_path.name}"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-vn",
        ]
        if mono:
            cmd.extend(["-ac", "1"])
        cmd.extend([
            "-b:a", bitrate,
            "-id3v2_version", "3",
            "-write_xing", "1",
            "-metadata", f"comment={label}",
            "-metadata", f"description={label}",
            "-metadata", "artist=AI Voice Studio",
            "-metadata", "publisher=AI Voice Studio",
            str(output_path),
        ])

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
            return output_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            output_path.unlink(missing_ok=True)
            logging.error(f"Error in AudioService.export_mp3: {getattr(error, 'stderr', error)}")
            raise AudioProcessingError("MP3 export failed.")

    @classmethod
    def export_all_formats(cls, wav_path: Path, job_dir: Path) -> dict:
        """
        Encode the single 192 kbps MP3 deliverable from the master WAV.

        Encodes directly from the master WAV without re-synthesis, preserving AI attribution tags.

        Args:
            wav_path (Path): Master normalized and labeled WAV file.
            job_dir (Path): Job root directory where output files reside.

        Returns:
            dict: Mapping of format keys ('wav', 'mp3') to byte sizes on disk.
        """
        mp3_high = job_dir / "output.mp3"
        # Speech is mono; a mono 192k MP3 is transparent for voice and half the size of stereo.
        cls.export_mp3(wav_path, mp3_high, bitrate="192k", mono=True)
        return {
            "wav_bytes": wav_path.stat().st_size if wav_path.is_file() else 0,
            "mp3_bytes": mp3_high.stat().st_size if mp3_high.is_file() else 0,
        }

    AI_LABEL = "AI-generated synthetic voice by AI Voice Studio"

    @staticmethod
    def _ai_label_args(label: Optional[str]) -> list:
        """ffmpeg -metadata arguments carrying the AI attribution label (empty list if no label)."""
        if not label:
            return []
        return [
            "-metadata", f"comment={label}",
            "-metadata", f"description={label}",
            "-metadata", "artist=AI Voice Studio",
        ]

    @classmethod
    def _linear_loudnorm_filter(cls, input_path: Path, pre_filters: list) -> str:
        """
        Measure loudness (analysis only, nothing written) and return a linear-mode loudnorm filter.

        Falls back to the dynamic single-pass filter if measurement fails.
        """
        base = "loudnorm=I=-16:TP=-1.5:LRA=11"
        cmd = [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(input_path),
            "-af", ",".join(pre_filters + [base + ":print_format=json"]),
            "-f", "null", "-",
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            stats = json.loads(res.stderr[res.stderr.rfind("{"):res.stderr.rfind("}") + 1])
            return (
                f"{base}:measured_I={stats['input_i']}:measured_TP={stats['input_tp']}"
                f":measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}"
                f":offset={stats['target_offset']}:linear=true"
            )
        except Exception as exc:
            logging.warning(f"Loudness measurement failed, using single-pass loudnorm: {exc}")
            return base

    @classmethod
    def polish(
        cls,
        input_path: Path,
        output_path: Path,
        speed: float = 1.0,
        sample_rate: int = 24000,
        ai_label: Optional[str] = None,
    ) -> Path:
        """
        Optional tempo change (pitch preserved) + linear loudness normalisation to -16 LUFS.

        A fast analysis-only pass measures loudness, then a single encode applies one fixed
        gain (loudnorm linear mode). Single-pass loudnorm rides the gain like an AGC, which
        pumps the room tone up in every pause; linear mode keeps natural dynamics.

        Args:
            input_path (Path): Source WAV.
            output_path (Path): Destination WAV (may equal input_path).
            speed (float): Tempo multiplier, 0.5-2.0. 1.0 = unchanged.
            sample_rate (int): Output sample rate.
            ai_label (Optional[str]): If set, AI attribution tags are written in the same encode
                (same tags as add_ai_label_metadata, without a second full-file rewrite).

        Returns:
            Path: Polished WAV path.
        """
        cls._check_ffmpeg_tool("ffmpeg")
        speed = max(0.5, min(2.0, float(speed or 1.0)))
        filters = []
        if abs(speed - 1.0) > 0.01:
            filters.append(f"atempo={speed:.3f}")
        filters.append(cls._linear_loudnorm_filter(input_path, list(filters)))

        in_place = input_path.resolve() == output_path.resolve()
        target_file = (
            output_path.parent / f"_tmp_{uuid.uuid4().hex[:8]}_{output_path.name}" if in_place else output_path
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-af", ",".join(filters),
            "-ar", str(sample_rate),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            *cls._ai_label_args(ai_label),
            str(target_file),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
            if in_place:
                shutil.move(str(target_file), str(output_path))
            return output_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            if in_place and target_file.exists():
                target_file.unlink()
            logging.error(f"Error in AudioService.polish: {getattr(error, 'stderr', error)}")
            raise AudioProcessingError("Audio polishing failed.")

    @classmethod
    def add_ai_label_metadata(
        cls, audio_path: Path, label: str = "AI-generated synthetic voice by AI Voice Studio"
    ) -> Path:
        """
        Inject synthetic AI labeling metadata tags into an audio file.

        Args:
            audio_path (Path): File to tag with AI attribution.
            label (str): Attribution message to record in audio tags.

        Returns:
            Path: Path to the tagged audio file.

        Raises:
            FFmpegNotFoundError: If ffmpeg is missing.
            AudioProcessingError: If tagging fails.
        """
        cls._check_ffmpeg_tool("ffmpeg")
        logging.info(f"Executing AudioService.add_ai_label_metadata on {audio_path.name}")

        temp_output = audio_path.with_name(f"tagged_{audio_path.name}")

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(audio_path),
            *cls._ai_label_args(label),
            "-c", "copy",
            str(temp_output),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            # Replace original with tagged version
            shutil.move(str(temp_output), str(audio_path))
            return audio_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            if temp_output.exists():
                temp_output.unlink()
            logging.error(f"Error in AudioService.add_ai_label_metadata: {getattr(error, 'stderr', error)}")
            raise AudioProcessingError("Metadata tagging failed.")

    @classmethod
    def _decode_mono(cls, input_path: Path, sample_rate: int, filters: str = "") -> np.ndarray:
        """
        Decode an audio file to mono float32 PCM in memory via ffmpeg (no temp files).

        Args:
            input_path (Path): Source audio file.
            sample_rate (int): Output sample rate in Hz.
            filters (str): Optional ffmpeg audio filter chain.

        Returns:
            np.ndarray: Mono float32 samples.

        Raises:
            AudioProcessingError: If ffmpeg fails to decode the file.
        """
        cmd = ["ffmpeg", "-v", "error", "-i", str(input_path)]
        if filters:
            cmd += ["-af", filters]
        cmd += ["-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"]
        try:
            res = subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            logging.error(f"Error in AudioService._decode_mono: {getattr(error, 'stderr', error)}")
            raise AudioProcessingError("Audio decoding failed.")
        return np.frombuffer(res.stdout, dtype=np.float32).copy()

    @staticmethod
    def _frame_db(audio: np.ndarray, frame: int) -> np.ndarray:
        """Per-frame RMS level in dBFS (non-overlapping frames)."""
        n = len(audio) // frame
        if n == 0:
            return np.full(1, -120.0)
        frames = audio[: n * frame].reshape(n, frame).astype(np.float64)
        return 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)

    @classmethod
    def prepare_conditioning_clip(cls, ref_path: Path, sample_rate: int = 24000, window_s: float = 10.0) -> np.ndarray:
        """
        Build a cleaned conditioning signal whose first `window_s` seconds are the best part of the reference.

        Chatterbox takes speaking style from the first 6s and timbre/room acoustics from the
        first 10s of the conditioning audio; only the speaker embedding sees all of it. This
        repairs clipping, removes rumble, applies mild denoising, picks the speech-dense,
        unclipped window that starts and ends near a pause, and rotates it to the front so the
        whole reference still feeds the speaker embedding.

        Args:
            ref_path (Path): Reference audio file.
            sample_rate (int): Output sample rate in Hz (Chatterbox S3Gen uses 24 kHz).
            window_s (float): Length of the conditioning window in seconds.

        Returns:
            np.ndarray: Mono float32 conditioning signal peaking at -1 dBFS.

        Raises:
            AudioProcessingError: If the reference cannot be decoded or is empty.
        """
        cls._check_ffmpeg_tool("ffmpeg")
        ref_path = Path(ref_path)
        raw = cls._decode_mono(ref_path, sample_rate)
        if raw.size == 0:
            raise AudioProcessingError("Reference audio is empty.")

        frame = int(0.02 * sample_rate)
        noise_db = float(np.clip(np.percentile(cls._frame_db(raw, frame), 10), -80.0, -20.0))
        clipped = int(np.sum(np.abs(raw) >= 0.99))

        filters = (["adeclip"] if clipped else []) + ["highpass=f=70", f"afftdn=nr=6:nf={noise_db:.1f}"]
        audio = cls._decode_mono(ref_path, sample_rate, ",".join(filters))
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio * (0.89 / peak)  # -1 dBFS

        win_f = int(window_s * sample_rate) // frame
        db = cls._frame_db(audio, frame)
        n = min(len(db), len(raw) // frame)
        if n <= win_f + int(1.0 / 0.02):
            logging.info(f"Conditioning clip: reference too short for window search (clipped={clipped})")
            return audio.astype(np.float32)

        db = db[:n]
        speech = (db > noise_db + 12.0).astype(np.float64)
        clip_frames = (np.abs(raw[: n * frame]).reshape(n, frame).max(axis=1) >= 0.99).astype(np.float64)
        cs_speech = np.concatenate([[0.0], np.cumsum(speech)])
        cs_clip = np.concatenate([[0.0], np.cumsum(clip_frames)])

        starts = np.arange(0, n - win_f + 1)
        speech_ratio = (cs_speech[starts + win_f] - cs_speech[starts]) / win_f
        clip_ratio = (cs_clip[starts + win_f] - cs_clip[starts]) / win_f
        span = max(float(np.percentile(db, 95)) - noise_db, 1.0)
        edge = (
            np.clip(db[starts] - noise_db, 0, span) + np.clip(db[starts + win_f - 1] - noise_db, 0, span)
        ) / (2 * span)
        score = speech_ratio - 4.0 * clip_ratio - 0.15 * edge
        best_idx = int(np.argmax(score))
        best = int(starts[best_idx]) * frame

        logging.info(
            "Conditioning clip: window %.2fs-%.2fs, speech=%.0f%%, clipped_frames=%.1f%%, repaired_samples=%d, noise=%.1f dB",
            best / sample_rate, best / sample_rate + window_s,
            speech_ratio[best_idx] * 100, clip_ratio[best_idx] * 100, clipped, noise_db,
        )
        return np.concatenate([audio[best:], audio[:best]]).astype(np.float32)

    @classmethod
    def conditioning_clip_buffer(cls, ref_path: Path, sample_rate: int = 24000) -> io.BytesIO:
        """
        Return the prepared conditioning clip as an in-memory WAV (readable by librosa/soundfile).

        Args:
            ref_path (Path): Reference audio file.
            sample_rate (int): Output sample rate in Hz.

        Returns:
            io.BytesIO: WAV buffer positioned at the start.
        """
        clip = cls.prepare_conditioning_clip(ref_path, sample_rate=sample_rate)
        buf = io.BytesIO()
        sf.write(buf, clip, sample_rate, format="WAV", subtype="FLOAT")
        buf.seek(0)
        return buf

    @staticmethod
    def _pause_ms_for(text: str, is_paragraph: bool, sentence_ms: int, paragraph_ms: int) -> float:
        """Pause after a chunk, scaled by how the chunk ends."""
        if is_paragraph:
            return float(paragraph_ms)
        end = (text or "").rstrip().rstrip("\"')")[-1:]
        if not end:
            return sentence_ms * 0.7
        if end in "?!":
            return sentence_ms * 1.15
        if end in ".।॥":
            return float(sentence_ms)
        if end in ",;:":
            return sentence_ms * 0.45
        return sentence_ms * 0.7

    @classmethod
    def assess_chunk(cls, audio: np.ndarray, sample_rate: int, text: str) -> dict:
        """
        Cheap sanity check of one synthesized chunk (no extra model, no GPU).

        Flags the usual zero-shot failures: babble/repeats (speech far too slow for the text,
        or the 40s token cap reached), truncation (far too fast), and long dead air inside speech.

        Args:
            audio (np.ndarray): Mono samples of the generated chunk.
            sample_rate (int): Sample rate in Hz.
            text (str): Text the chunk was generated from.

        Returns:
            dict: {"ok": bool, "reason": str, "rate": chars per second of speech}.
        """
        chars = len("".join((text or "").split()))
        info = cls._analyse_chunk(audio, sample_rate)
        speech_s = max((info["end"] - info["start"]) / sample_rate, 1e-3)
        rate = chars / speech_s
        reason = ""
        if info["active_rms"] <= 0 or speech_s < 0.25:
            reason = "no speech"
        elif len(audio) / sample_rate >= 39.0:
            reason = "hit the generation length cap"
        elif chars >= 12 and rate < 5.0:
            reason = f"too slow for the text ({rate:.1f} chars/s)"
        elif rate > 30.0:
            reason = f"too fast for the text ({rate:.1f} chars/s), likely truncated"
        else:
            frame = max(int(0.01 * sample_rate), 1)
            seg = audio[info["start"]:info["end"]]
            db = cls._frame_db(seg, frame)
            thr = max(float(np.percentile(db, 95)) - 35.0, -60.0)
            quiet = np.concatenate([[0], (db < thr).astype(np.int8), [0]])
            edges = np.flatnonzero(np.diff(quiet))
            longest = int(np.max(edges[1::2] - edges[::2])) if edges.size else 0
            if longest * frame / sample_rate > 1.2:
                reason = f"{longest * frame / sample_rate:.1f}s of dead air inside speech"
        return {"ok": not reason, "reason": reason, "rate": rate}

    @classmethod
    def _analyse_chunk(cls, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Find speech bounds, speech level and room-tone runs of one synthesized chunk.

        Returns:
            dict: start/end sample bounds, active_rms, floor_rms and quiet (s, e) sample runs.
        """
        frame = max(int(0.01 * sample_rate), 1)
        db = cls._frame_db(audio, frame)
        thr = max(float(np.percentile(db, 95)) - 35.0, -60.0)
        active = np.nonzero(db > thr)[0]
        if active.size == 0:
            return {"start": 0, "end": len(audio), "active_rms": 0.0, "floor_rms": 0.0, "runs": []}

        speech_start = int(active[0]) * frame
        speech_end = (int(active[-1]) + 1) * frame
        start = max(0, speech_start - int(0.04 * sample_rate))
        end = min(len(audio), speech_end + int(0.06 * sample_rate))
        active_rms = float(np.sqrt(np.mean(10.0 ** (db[active] / 10.0))))

        # Room tone = the real noise between words. Frames near digital silence (the model's
        # fade-in/out edges) are excluded, otherwise the "room tone" would itself be silence.
        live = db > -75.0
        if not np.any(live & (db < thr)):
            return {"start": start, "end": end, "active_rms": active_rms, "floor_rms": 0.0, "runs": [],
                    "fade_in": speech_start - start, "fade_out": end - speech_end}
        floor_db = float(np.percentile(db[live & (db < thr)], 10))

        quiet = np.concatenate([[False], live & (db < min(floor_db + 6.0, thr - 10.0)), [False]])
        edges = np.flatnonzero(np.diff(quiet.astype(np.int8)))
        runs = [
            ((s + 1) * frame, (e - 1) * frame)
            for s, e in zip(edges[::2], edges[1::2])
            if e - s >= 7  # >= 70 ms, minus one guard frame each side
        ]
        return {
            "start": start,
            "end": end,
            "active_rms": active_rms,
            "floor_rms": float(10.0 ** (floor_db / 20.0)),
            "runs": runs,
            "fade_in": speech_start - start,
            "fade_out": end - speech_end,
        }

    @classmethod
    def join_chunks_natural(
        cls,
        chunk_paths: list,
        output_path: Path,
        chunk_texts: Optional[list] = None,
        pause_sentence_ms: int = 350,
        pause_paragraph_ms: int = 750,
        paragraph_boundary_indices: Optional[set] = None,
        sample_rate: int = 24000,
        seed: int = 0,
    ) -> Path:
        """
        Join synthesized chunks into one natural-sounding take, streamed straight to disk.

        Each chunk's own leading/trailing silence is trimmed, its speech level is matched to
        the median of all chunks, and pauses are filled with the chunks' own room tone (never
        digital silence) with equal-power fades at every boundary. Pause length follows the
        punctuation the chunk ends with, with slight deterministic variation.

        Args:
            chunk_paths (list): Ordered chunk WAV paths (same sample rate).
            output_path (Path): Destination 16-bit PCM WAV.
            chunk_texts (Optional[list]): Text of each chunk, used to size pauses.
            pause_sentence_ms (int): Base pause after a sentence (ms).
            pause_paragraph_ms (int): Pause before a new paragraph (ms).
            paragraph_boundary_indices (Optional[set]): Chunk indices that start a paragraph.
            sample_rate (int): Expected sample rate of every chunk.
            seed (int): Seed for pause variation and room-tone texture (deterministic per job).

        Returns:
            Path: Path to the joined WAV file.

        Raises:
            AudioProcessingError: If the chunk list is empty or sample rates differ.
        """
        logging.info("Executing AudioService.join_chunks_natural: %d chunks -> %s", len(chunk_paths), output_path.name)
        if not chunk_paths:
            raise AudioProcessingError("Cannot join an empty chunk list.")

        para = paragraph_boundary_indices or set()
        texts = list(chunk_texts or [])
        rng = np.random.default_rng(seed)

        def _read(path) -> np.ndarray:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            if sr != sample_rate:
                raise AudioProcessingError(f"Chunk {Path(path).name} is {sr} Hz, expected {sample_rate} Hz.")
            return data.mean(axis=1)

        # Pass 1: analyse chunks and collect up to 10s of room tone
        infos, bank_parts, bank_len = [], [], 0
        bank_cap = 10 * sample_rate
        for idx, path in enumerate(chunk_paths):
            audio = _read(path)
            info = cls._analyse_chunk(audio, sample_rate)
            seg = audio[info["start"]:info["end"]]
            info["peak"] = float(np.max(np.abs(seg))) if seg.size else 0.0
            for s, e in info["runs"]:
                if bank_len >= bank_cap:
                    break
                bank_parts.append((idx, audio[s:e].copy()))
                bank_len += e - s
            infos.append(info)

        levels = [i["active_rms"] for i in infos if i["active_rms"] > 0]
        target = float(np.median(levels)) if levels else 0.0
        for info in infos:
            gain = 1.0
            if target > 0 and info["active_rms"] > 0:
                gain = float(np.clip(target / info["active_rms"], 0.5, 2.0))  # at most +/-6 dB
            if info["peak"] * gain > 0.97:
                gain = 0.97 / info["peak"]
            info["gain"] = gain

        bank = (
            np.concatenate([part * infos[i]["gain"] for i, part in bank_parts]).astype(np.float32)
            if bank_parts else np.zeros(0, dtype=np.float32)
        )
        floor = float(np.median([i["floor_rms"] * i["gain"] for i in infos]))
        tone = _RoomTone(bank, rng, floor)

        # Pass 2: stream trimmed, level-matched chunks over a continuous room-tone bed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        last = len(chunk_paths) - 1
        with sf.SoundFile(str(output_path), "w", samplerate=sample_rate, channels=1, subtype="PCM_16") as out:
            out.write(tone.take(int(0.15 * sample_rate)))
            for idx, (path, info) in enumerate(zip(chunk_paths, infos)):
                seg = _read(path)[info["start"]:info["end"]] * info["gain"]
                if seg.size:
                    # Fades span the kept padding, so the bed covers the model's near-silent edges
                    w = np.ones(seg.size, dtype=np.float32)
                    min_fade = int(0.012 * sample_rate)
                    for fade, head in ((info.get("fade_in", 0), True), (info.get("fade_out", 0), False)):
                        fade = min(max(int(fade), min_fade), seg.size // 2)
                        if fade > 0:
                            ramp = np.sin(0.5 * np.pi * (np.arange(fade) + 0.5) / fade).astype(np.float32)
                            if head:
                                w[:fade] = ramp
                            else:
                                w[-fade:] = ramp[::-1]
                    out.write(tone.take(seg.size) * np.sqrt(1.0 - w ** 2) + seg * w)
                if idx < last:
                    text = texts[idx] if idx < len(texts) else ""
                    pause = cls._pause_ms_for(text, (idx + 1) in para, pause_sentence_ms, pause_paragraph_ms)
                    out.write(tone.take(int(pause * rng.uniform(0.85, 1.15) * sample_rate / 1000)))
            out.write(tone.take(int(0.3 * sample_rate)))

        logging.info(
            "AudioService.join_chunks_natural completed: %s (%d bytes, room-tone bank %.2fs)",
            output_path.name, output_path.stat().st_size, bank.size / sample_rate,
        )
        return output_path

    @staticmethod
    def check_free_disk_gb(path: Path, required_gb: float = 1.0) -> None:
        """
        Verify that sufficient free disk space is available before starting a long job.

        Args:
            path (Path): Directory to check for free space.
            required_gb (float): Minimum required free gigabytes (default 1.0).

        Returns:
            None

        Raises:
            AudioProcessingError: If free space is below required_gb.
        """
        import shutil as _shutil
        total, used, free = _shutil.disk_usage(str(path))
        free_gb = free / (1024 ** 3)
        logging.info(
            "Disk space check at %s: %.2f GB free (required %.2f GB)",
            path, free_gb, required_gb,
        )
        if free_gb < required_gb:
            raise AudioProcessingError(
                f"Insufficient disk space: {free_gb:.2f} GB free, "
                f"{required_gb:.2f} GB required. Free up space and retry."
            )


if __name__ == "__main__":
    """
    Self-check harness: generates a 1s synthetic sine wave tone, probes, normalizes,
    and exports MP3 with AI metadata tags to storage/tmp.
    """
    print("=== Running AudioService Self-Check Harness ===")
    from core.config import get_settings
    tmp_dir = get_settings().tmp_path
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Synthesize 1-second 440 Hz test tone at 22050 Hz
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)

    raw_wav = tmp_dir / "self_check_tone.wav"
    sf.write(str(raw_wav), tone, sr)
    print(f"[1/5] Created synthetic tone: {raw_wav}")

    # 2. Probe audio
    meta = AudioService.probe_audio(raw_wav)
    print(f"[2/5] Probed: {meta.duration_seconds}s, {meta.sample_rate}Hz, {meta.channels}ch, format={meta.format_name}")
    assert meta.sample_rate == 22050
    assert meta.channels == 1

    # 3. Resample to 24000 Hz
    resampled_wav = tmp_dir / "self_check_resampled.wav"
    AudioService.resample(raw_wav, resampled_wav, target_sample_rate=24000)
    meta_resampled = AudioService.probe_audio(resampled_wav)
    print(f"[3/5] Resampled: {meta_resampled.sample_rate}Hz")
    assert meta_resampled.sample_rate == 24000

    # 4. Normalize loudness
    normalized_wav = tmp_dir / "self_check_normalized.wav"
    AudioService.normalize_loudness(resampled_wav, normalized_wav)
    print(f"[4/5] Normalized loudness successfully")

    # 5. Export MP3 & add AI label
    mp3_output = tmp_dir / "self_check_output.mp3"
    AudioService.export_mp3(normalized_wav, mp3_output, bitrate="192k")
    AudioService.add_ai_label_metadata(mp3_output, label="AI-generated synthetic audio test")
    print(f"[5/5] Exported tagged MP3: {mp3_output} ({mp3_output.stat().st_size} bytes)")

    print("=== Self-Check Completed Successfully! ===")
