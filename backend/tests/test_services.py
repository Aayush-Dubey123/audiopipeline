"""
Unit and service-level test suite for audio manipulation and transcoding.
"""

import hashlib
import os
import pickle
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F

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
from core.services.pipelines.rvc_service import (
    RVCService,
    RVCResult,
    ModelCheckpointNotFoundError as RVCModelNotFoundError,
    InputAudioNotFoundError,
    SecurityValidationError,
    verify_checkpoint_hash,
)
from core.services.pipelines.rvc_core.synthesizer import SynthesizerTrn
from core.services.pipelines.rvc_core.rmvpe import MelSpectrogram, E2E
from core.services.llm.llm_service import LLMService
from core.services.audio.validation_service import (
    ValidationService,
    InvalidAudioExtensionError,
    ReferenceQualityError,
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


# ============================================================================
# RVC Pipeline Tests (Phase 4)
# ============================================================================


def test_rvc_service_initialization() -> None:
    """
    Verify RVCService initializes with default paths and unloaded model state.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If default paths or initial model states are invalid.
    """
    service = RVCService(device="cpu")
    assert service.device == "cpu"
    assert service.is_half is False
    assert service.models_dir.name == "rvc_models"
    assert service.rmvpe_path.name == "rmvpe.pt"
    assert service._net_g is None
    assert service._rmvpe is None
    assert service._hubert_model is None


def test_rvc_pitch_shift_bounds_validation(synthetic_tone_wav: Path, temp_audio_dir: Path) -> None:
    """
    Verify internal pitch shift bounds enforce [-24, +24] semitones strictly.

    Args:
        synthetic_tone_wav (Path): 1-second test WAV audio fixture.
        temp_audio_dir (Path): Temporary output directory fixture.

    Returns:
        None

    Raises:
        AssertionError: If pitch bounds validation allows out-of-range values.
    """
    service = RVCService(device="cpu")
    out_file = temp_audio_dir / "out_rvc.wav"
    fake_model = temp_audio_dir / "fake.pth"

    # Out-of-bounds high (+25)
    with pytest.raises(ValueError, match=r"between -24 and \+24"):
        service.convert(
            source_audio_path=synthetic_tone_wav,
            voice_model_path=fake_model,
            output_path=out_file,
            pitch_shift=25,
        )

    # Out-of-bounds low (-25)
    with pytest.raises(ValueError, match=r"between -24 and \+24"):
        service.convert(
            source_audio_path=synthetic_tone_wav,
            voice_model_path=fake_model,
            output_path=out_file,
            pitch_shift=-25,
        )


def test_rvc_missing_input_audio_raises_error(temp_audio_dir: Path) -> None:
    """
    Verify convert raises InputAudioNotFoundError when input audio file is missing.

    Args:
        temp_audio_dir (Path): Temporary directory fixture.

    Returns:
        None

    Raises:
        AssertionError: If InputAudioNotFoundError is not raised.
    """
    service = RVCService(device="cpu")
    missing_audio = temp_audio_dir / "nonexistent_source.wav"
    fake_model = temp_audio_dir / "fake.pth"
    out_file = temp_audio_dir / "out_rvc.wav"

    with pytest.raises(InputAudioNotFoundError):
        service.convert(
            source_audio_path=missing_audio,
            voice_model_path=fake_model,
            output_path=out_file,
            pitch_shift=0,
        )


def test_rvc_missing_model_checkpoint_raises_error(synthetic_tone_wav: Path, temp_audio_dir: Path) -> None:
    """
    Verify convert raises ModelCheckpointNotFoundError when .pth model is missing.

    Args:
        synthetic_tone_wav (Path): Test audio input fixture.
        temp_audio_dir (Path): Temporary directory fixture.

    Returns:
        None

    Raises:
        AssertionError: If ModelCheckpointNotFoundError is not raised.
    """
    service = RVCService(device="cpu")
    missing_model = temp_audio_dir / "nonexistent_model.pth"

    with pytest.raises(RVCModelNotFoundError):
        service.load_voice_model(missing_model)


def test_rvc_security_validation_weights_only(temp_audio_dir: Path) -> None:
    """
    Verify checkpoint loading strictly enforces weights_only=True and rejects arbitrary code execution.

    Args:
        temp_audio_dir (Path): Temporary directory fixture.

    Returns:
        None

    Raises:
        AssertionError: If SecurityValidationError is not raised on unsafe checkpoints.
    """
    exploit_file = temp_audio_dir / "exploit.pth"

    class ExploitPayload:
        def __reduce__(self):
            return (os.system, ("dir",))

    with open(exploit_file, "wb") as f:
        pickle.dump(ExploitPayload(), f)

    service = RVCService(device="cpu")
    with pytest.raises(SecurityValidationError):
        service.load_voice_model(exploit_file)


def test_rvc_verify_checkpoint_hash(temp_audio_dir: Path) -> None:
    """
    Verify verify_checkpoint_hash validates matching hashes and rejects mismatches with SecurityValidationError.

    Args:
        temp_audio_dir (Path): Temporary directory fixture.

    Returns:
        None

    Raises:
        AssertionError: If hash verification behaves unexpectedly.
    """
    test_file = temp_audio_dir / "hash_test.bin"
    test_file.write_bytes(b"TEST_MODEL_BINARY_DATA_FOR_HASH_CHECK")

    expected = hashlib.sha256(b"TEST_MODEL_BINARY_DATA_FOR_HASH_CHECK").hexdigest()
    assert verify_checkpoint_hash(test_file, expected) is True

    with pytest.raises(SecurityValidationError):
        verify_checkpoint_hash(test_file, "0" * 64)


def test_rvc_mutual_unloading() -> None:
    """
    Verify mutual model unloading between OpenVoice and RVC resets GPU caches and releases models.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If model unload methods fail to clear cached references.
    """
    ov_service = OpenVoiceService(device="cpu")
    rvc_service = RVCService(device="cpu")

    # Set mock references
    rvc_service._net_g = "mock_net_g"  # type: ignore
    rvc_service._rmvpe = "mock_rmvpe"  # type: ignore
    rvc_service._hubert_model = "mock_hubert"  # type: ignore

    rvc_service.unload_models()
    assert rvc_service._net_g is None
    assert rvc_service._rmvpe is None
    assert rvc_service._hubert_model is None

    # Test class-level unload_all
    rvc_service._net_g = "mock_net_g"  # type: ignore
    RVCService.unload_all()
    assert rvc_service._net_g is None

    ov_service._converter = "mock_converter"  # type: ignore
    OpenVoiceService.unload_all()
    assert ov_service._converter is None


def test_rvc_synthesizer_trn_forward_cpu() -> None:
    """
    Verify forward inference through SynthesizerTrn generates audio tensor on CPU.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If synthesis output shape or dimensions are invalid.
    """
    net_g = SynthesizerTrn(
        spec_channels=1025,
        segment_size=32,
        inter_channels=64,
        hidden_channels=64,
        filter_channels=128,
        n_heads=2,
        n_layers=2,
        kernel_size=3,
        p_dropout=0.0,
        resblock="1",
        resblock_kernel_sizes=[3, 7],
        resblock_dilation_sizes=[[1, 3], [1, 3]],
        upsample_rates=[4, 4],
        upsample_initial_channel=64,
        upsample_kernel_sizes=[8, 8],
        spk_embed_dim=1,
        gin_channels=64,
        sr=40000,
        is_half=False,
        version="v2",
    )
    net_g.eval()

    phone = torch.randn(1, 8, 768)
    phone_lengths = torch.tensor([8], dtype=torch.long)
    pitch = torch.randint(1, 100, (1, 8), dtype=torch.long)
    nsff0 = torch.randn(1, 8)
    sid = torch.tensor([0], dtype=torch.long)

    with torch.no_grad():
        out = net_g.infer(phone, phone_lengths, pitch, nsff0, sid)

    assert out is not None
    assert out.dim() == 3
    assert out.shape[0] == 1
    assert out.shape[1] == 1
    assert out.shape[2] == 8 * 16  # upsample_rates [4, 4] -> 16x


def test_rvc_rmvpe_mel_and_hidden_forward_cpu() -> None:
    """
    Verify RMVPE MelSpectrogram and E2E feature extraction forward pass on CPU.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If mel spectrogram or hidden pitch features have incorrect dimensions.
    """
    mel_extractor = MelSpectrogram(128, 16000, 1024, 160, 30.0, 8000.0)
    e2e = E2E(4, 1, (2, 2))
    e2e.eval()

    audio = torch.randn(1, 16000)
    with torch.no_grad():
        mel = mel_extractor(audio)
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        mel_padded = F.pad(mel, (0, n_pad), mode="constant") if n_pad > 0 else mel
        hidden = e2e(mel_padded)[:, :n_frames]

    assert mel.dim() == 3
    assert mel.shape[1] == 128
    assert hidden.dim() == 3
    assert hidden.shape[-1] == 360  # 360 pitch frequency bins


@pytest.mark.slow
def test_rvc_real_inference_slow(temp_audio_dir: Path, synthetic_tone_wav: Path) -> None:
    """
    Perform complete real RVC inference using available disk checkpoints.

    Args:
        temp_audio_dir (Path): Working test directory fixture.
        synthetic_tone_wav (Path): Test input audio fixture.

    Returns:
        None

    Raises:
        pytest.skip: When real checkpoints are not present on disk.
        AssertionError: If inference outcome metadata is invalid.
    """
    service = RVCService()
    checkpoint_candidates = [
        service.models_dir / "checkpoints" / "test_voice.pth",
        service.models_dir / "checkpoints" / "NttsAI.pth",
    ]
    test_voice_checkpoint = next((p for p in checkpoint_candidates if p.exists()), None)

    if not service.rmvpe_path.exists() or test_voice_checkpoint is None:
        pytest.skip("RVC real checkpoints (rmvpe.pt or voice .pth) not present on disk")

    out_file = temp_audio_dir / "real_rvc_output.wav"
    result: RVCResult = service.convert(
        source_audio_path=synthetic_tone_wav,
        voice_model_path=test_voice_checkpoint,
        output_path=out_file,
        pitch_shift=0,
    )

    assert out_file.exists()
    assert result.duration_seconds > 0.0
    assert result.sample_rate > 0
    assert result.file_size_bytes > 0


# ==============================================================================
# SECTION 5: VOICE CATALOG REPOSITORY TESTS
# ==============================================================================


def test_voice_crud_sync_and_filtering(tmp_path: Path) -> None:
    """
    Verify VoiceCRUD synchronization handles valid entries and rejects invalid ones.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.

    Returns:
        None
    """
    import json
    from core.database.database import SessionLocal
    from core.cruds.voice_crud import VoiceCRUD

    db = SessionLocal()
    try:
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        ref_dir = voices_dir / "reference"
        ref_dir.mkdir()
        valid_wav = ref_dir / "sample.wav"
        valid_wav.write_bytes(b"RIFFdummydataWAVEfmt ")

        manifest = [
            {
                "id": "test_valid_voice",
                "name": "Test Valid Voice",
                "model_type": "openvoice_reference",
                "asset_file": "reference/sample.wav",
                "license": "MIT",
                "usage_permission": "Public",
                "status": "available",
            },
            {
                "id": "test_traversal",
                "name": "Bad Traversal",
                "model_type": "openvoice_reference",
                "asset_file": "../outside.wav",
                "license": "MIT",
                "usage_permission": "Public",
                "status": "available",
            },
            {
                "id": "test_missing_file",
                "name": "Missing File",
                "model_type": "openvoice_reference",
                "asset_file": "reference/missing.wav",
                "license": "MIT",
                "usage_permission": "Public",
                "status": "available",
            },
            {
                "id": "test_invalid_type",
                "name": "Invalid Engine",
                "model_type": "unsupported_engine",
                "asset_file": "reference/sample.wav",
                "license": "MIT",
                "usage_permission": "Public",
                "status": "available",
            },
        ]
        manifest_file = voices_dir / "voices.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        synced = VoiceCRUD.sync_voices_from_json(db=db, json_path=manifest_file, voices_base_dir=voices_dir)
        assert synced == 1

        voice = VoiceCRUD.get_voice_by_id(db, "test_valid_voice")
        assert voice is not None
        assert voice.name == "Test Valid Voice"

        missing = VoiceCRUD.get_voice_by_id(db, "non_existent")
        assert missing is None
    finally:
        db.close()


def test_llm_service_rule_based_cleanup() -> None:
    """
    Verify rule-based script cleanup normalizes abbreviations, currency, and markdown.
    """
    raw_en = "Hello! Dr. Smith paid $5.00 for the **awesome** app... cool!!!"
    cleaned_en = LLMService.rule_based_cleanup(raw_en, language="en")
    assert "doctor Smith" in cleaned_en
    assert "5.00 dollars" in cleaned_en
    assert "**" not in cleaned_en
    assert "cool!" in cleaned_en
    assert "!!!" not in cleaned_en

    # Devanagari numerals
    raw_hi = "आज १२३ रुपये खर्च झाले."
    cleaned_hi = LLMService.rule_based_cleanup(raw_hi, language="hi")
    assert "123" in cleaned_hi
    assert "१२३" not in cleaned_hi


def test_validation_service_extension_allowlist() -> None:
    """
    Verify audio extension allowlist accepts valid audio and rejects dangerous formats.
    """
    assert ValidationService.sanitize_and_validate_extension("sample.wav") == ".wav"
    assert ValidationService.sanitize_and_validate_extension("recording.MP3") == ".mp3"
    assert ValidationService.sanitize_and_validate_extension("track.flac") == ".flac"

    with pytest.raises(InvalidAudioExtensionError):
        ValidationService.sanitize_and_validate_extension("malicious.exe")

    with pytest.raises(InvalidAudioExtensionError):
        ValidationService.sanitize_and_validate_extension("script.py")

    with pytest.raises(InvalidAudioExtensionError):
        ValidationService.sanitize_and_validate_extension("")


def test_validation_service_inspect_reference_quality(temp_audio_dir: Path) -> None:
    """
    Verify reference quality inspection validates good audio and flags short audio.
    """
    # 1. Valid 4-second sine wave
    sr = 22050
    t = np.linspace(0, 4.0, int(sr * 4.0), endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    valid_wav = temp_audio_dir / "valid_ref.wav"
    sf.write(str(valid_wav), sine, sr)

    metrics = ValidationService.inspect_reference_audio_quality(valid_wav, min_duration=3.0)
    assert metrics.duration_seconds >= 3.9
    assert metrics.peak_amplitude > 0.4
    assert metrics.is_clipped is False
    assert metrics.is_too_quiet is False
    assert metrics.quality_score > 0.7

    # 2. Too short audio (< 3.0s) raises ReferenceQualityError
    short_sine = 0.5 * np.sin(2 * np.pi * 440 * t[: int(sr * 1.5)])
    short_wav = temp_audio_dir / "short_ref.wav"
    sf.write(str(short_wav), short_sine, sr)

    with pytest.raises(ReferenceQualityError, match="too short"):
        ValidationService.inspect_reference_audio_quality(short_wav, min_duration=3.0)

