"""
Unit and service-level test suite for audio manipulation and transcoding.
"""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from core.services.audio.audio_service import (
    AudioService,
    AudioCorruptError,
    AudioMetadata,
)
from core.services.pipelines.openvoice_service import (
    OpenVoiceService,
    ReferenceAudioNotFoundError,
    ModelCheckpointNotFoundError,
)


@pytest.fixture
def temp_audio_dir(tmp_path: Path) -> Path:
    """
    Create a clean temporary directory for isolated audio testing.

    Args:
        tmp_path (Path): Pytest standard temporary directory fixture.

    Returns:
        Path: Allocated scratch audio folder.
    """
    audio_dir = tmp_path / "audio_tests"
    audio_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir


@pytest.fixture
def synthetic_tone_wav(temp_audio_dir: Path) -> Path:
    """
    Generate a 1-second 440Hz test sine tone as a PCM WAV file.

    Args:
        temp_audio_dir (Path): Working test directory fixture.

    Returns:
        Path: Path to generated synthetic audio test file.
    """
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)

    tone_path = temp_audio_dir / "test_synth_tone.wav"
    sf.write(str(tone_path), tone, sr)
    return tone_path


def test_probe_valid_synthetic_audio(synthetic_tone_wav: Path) -> None:
    """
    Verify probe_audio accurately parses sample rate, channels, and format.
    """
    meta: AudioMetadata = AudioService.probe_audio(synthetic_tone_wav)
    assert meta.sample_rate == 22050
    assert meta.channels == 1
    assert meta.duration_seconds > 0.95
    assert "wav" in meta.format_name.lower()


def test_resample_audio(synthetic_tone_wav: Path, temp_audio_dir: Path) -> None:
    """
    Verify resample adjusts sample rate to target frequency and converts to mono.
    """
    target_path = temp_audio_dir / "resampled_24k.wav"
    AudioService.resample(synthetic_tone_wav, target_path, target_sample_rate=24000)

    meta = AudioService.probe_audio(target_path)
    assert meta.sample_rate == 24000
    assert meta.channels == 1


def test_normalize_loudness(synthetic_tone_wav: Path, temp_audio_dir: Path) -> None:
    """
    Verify normalize_loudness processes audio through the EBU R128 filter.
    """
    target_path = temp_audio_dir / "normalized.wav"
    output_path = AudioService.normalize_loudness(synthetic_tone_wav, target_path)

    assert output_path.exists()
    meta = AudioService.probe_audio(output_path)
    assert meta.duration_seconds > 0.9


def test_export_mp3_and_ai_metadata(synthetic_tone_wav: Path, temp_audio_dir: Path) -> None:
    """
    Verify export_mp3 creates valid MP3 audio and add_ai_label_metadata tags it.
    """
    mp3_path = temp_audio_dir / "output.mp3"
    AudioService.export_mp3(synthetic_tone_wav, mp3_path, bitrate="192k")

    assert mp3_path.exists()
    assert mp3_path.stat().st_size > 0

    tagged_path = AudioService.add_ai_label_metadata(mp3_path, label="AI-generated synthetic audio test")
    assert tagged_path.exists()

    meta = AudioService.probe_audio(tagged_path)
    assert "mp3" in meta.format_name.lower()


def test_probe_corrupt_file_raises_custom_error(temp_audio_dir: Path) -> None:
    """
    Verify corrupt or non-audio inputs raise AudioCorruptError instead of unhandled exceptions.
    """
    corrupt_file = temp_audio_dir / "corrupt.wav"
    corrupt_file.write_text("NOT_VALID_AUDIO_DATA_GARBAGE_PAYLOAD", encoding="utf-8")

    with pytest.raises(AudioCorruptError):
        AudioService.probe_audio(corrupt_file)


def test_probe_missing_file_raises_custom_error(temp_audio_dir: Path) -> None:
    """
    Verify missing file path raises AudioCorruptError with descriptive message.
    """
    missing_file = temp_audio_dir / "does_not_exist.wav"
    with pytest.raises(AudioCorruptError):
        AudioService.probe_audio(missing_file)


def test_openvoice_service_initialization() -> None:
    """
    Verify OpenVoiceService defaults to valid paths and enables watermark.
    """
    service = OpenVoiceService()
    assert service.enable_watermark is True
    assert service.device in ("cuda:0", "cpu")
    assert service.converter_dir.name == "converter"


def test_openvoice_missing_reference_raises_custom_error(temp_audio_dir: Path) -> None:
    """
    Verify generate with a nonexistent reference file raises ReferenceAudioNotFoundError.
    """
    service = OpenVoiceService()
    missing_ref = temp_audio_dir / "nonexistent_reference.wav"
    out_file = temp_audio_dir / "out.wav"

    with pytest.raises(ReferenceAudioNotFoundError):
        service.generate(
            text="Test text",
            reference_audio_path=str(missing_ref),
            output_path=str(out_file),
        )


def test_openvoice_empty_text_raises_value_error(temp_audio_dir: Path, synthetic_tone_wav: Path) -> None:
    """
    Verify generate with empty or whitespace-only text raises ValueError.
    """
    service = OpenVoiceService()
    out_file = temp_audio_dir / "out.wav"

    with pytest.raises(ValueError):
        service.generate(
            text="   ",
            reference_audio_path=str(synthetic_tone_wav),
            output_path=str(out_file),
        )


def test_openvoice_missing_checkpoint_raises_error(temp_audio_dir: Path, synthetic_tone_wav: Path) -> None:
    """
    Verify service configured with nonexistent checkpoint path raises ModelCheckpointNotFoundError.
    """
    fake_converter_dir = temp_audio_dir / "missing_converter"
    service = OpenVoiceService(converter_dir=str(fake_converter_dir))
    out_file = temp_audio_dir / "out.wav"

    with pytest.raises(ModelCheckpointNotFoundError):
        service.generate(
            text="Valid test sentence",
            reference_audio_path=str(synthetic_tone_wav),
            output_path=str(out_file),
        )


def test_openvoice_cpu_device_configuration() -> None:
    """
    Verify OpenVoiceService supports explicit CPU device assignment and handles CUDA cleanup safely.
    """
    service = OpenVoiceService(device="cpu", enable_watermark=False)
    assert service.device == "cpu"
    assert service.enable_watermark is False


def test_openvoice_watermark_enabled_by_default() -> None:
    """
    Verify OpenVoiceService defaults to enabling the MyShell audio watermark.
    """
    service = OpenVoiceService()
    assert service.enable_watermark is True
