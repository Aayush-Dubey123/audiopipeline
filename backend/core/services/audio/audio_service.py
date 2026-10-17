"""
Audio processing service for AI Voice Studio.

Provides robust audio inspection, format conversion, resampling, normalization,
MP3 encoding, and synthetic AI labeling metadata.
"""

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
            "-show_entries", "format=duration,format_name:stream=sample_rate,channels",
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
            duration = float(format_info.get("duration", 0.0))
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
    def to_wav(cls, input_path: Path, output_path: Path) -> Path:
        """
        Transcode an arbitrary audio file to standard 16-bit PCM WAV.

        Args:
            input_path (Path): Source audio path.
            output_path (Path): Target WAV path.

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
            "-i", str(input_path),
            "-vn",
            "-c:a", "pcm_s16le",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            return output_path
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
        except subprocess.CalledProcessError as error:
            if in_place and target_file.exists():
                target_file.unlink()
            logging.error(f"Error in AudioService.resample: {error.stderr}")
            raise AudioProcessingError(f"Resampling failed: {error.stderr}")

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
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            if in_place:
                shutil.move(str(target_file), str(output_path))
            return output_path
        except subprocess.CalledProcessError as error:
            if in_place and target_file.exists():
                target_file.unlink()
            logging.error(f"Error in AudioService.normalize_loudness: {error.stderr}")
            raise AudioProcessingError(f"Loudness normalization failed: {error.stderr}")

    @classmethod
    def export_mp3(cls, input_path: Path, output_path: Path, bitrate: str = "192k") -> Path:
        """
        Encode an audio file to MP3 format with a target bitrate.

        Args:
            input_path (Path): Source audio path.
            output_path (Path): Destination MP3 path.
            bitrate (str): Target MP3 constant bitrate (e.g., '192k', '128k').

        Returns:
            Path: Path to the generated MP3 file.

        Raises:
            FFmpegNotFoundError: If ffmpeg is missing.
            AudioProcessingError: If MP3 encoding fails.
        """
        cls._check_ffmpeg_tool("ffmpeg")
        logging.info(f"Executing AudioService.export_mp3 at {bitrate} for {input_path.name}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-vn",
            "-b:a", bitrate,
            str(output_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            return output_path
        except subprocess.CalledProcessError as error:
            logging.error(f"Error in AudioService.export_mp3: {error.stderr}")
            raise AudioProcessingError(f"MP3 export failed: {error.stderr}")

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
            "-metadata", f"comment={label}",
            "-metadata", f"description={label}",
            "-metadata", "artist=AI Voice Studio",
            "-c", "copy",
            str(temp_output),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            # Replace original with tagged version
            shutil.move(str(temp_output), str(audio_path))
            return audio_path
        except subprocess.CalledProcessError as error:
            if temp_output.exists():
                temp_output.unlink()
            logging.error(f"Error in AudioService.add_ai_label_metadata: {error.stderr}")
            raise AudioProcessingError(f"Metadata tagging failed: {error.stderr}")


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
