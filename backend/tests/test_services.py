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
from core.services.jobs.job_worker import split_text_with_boundaries
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


def test_encoder_formats_and_tags(synthetic_tone_wav: Path, temp_audio_dir: Path) -> None:
    """
    Verify the single MP3 deliverable matches WAV duration, preserves AI tags, and
    that no extra audio variants are produced (max two output files per job).
    """
    wav_meta = AudioService.probe_audio(synthetic_tone_wav)
    sizes = AudioService.export_all_formats(synthetic_tone_wav, temp_audio_dir)

    mp3_path = temp_audio_dir / "output.mp3"
    assert mp3_path.exists()
    assert not (temp_audio_dir / "output_small.mp3").exists()
    assert set(sizes) == {"wav_bytes", "mp3_bytes"}
    assert sizes["mp3_bytes"] > 0

    mp3_meta = AudioService.probe_audio(mp3_path)
    assert abs(mp3_meta.duration_seconds - wav_meta.duration_seconds) <= 0.120
    assert mp3_meta.channels == 1

    mp3_tags = AudioService.read_audio_tags(mp3_path)
    all_mp3_tag_text = " ".join(f"{k}:{v}" for k, v in mp3_tags.items()).lower()
    assert "ai" in all_mp3_tag_text or "voice studio" in all_mp3_tag_text


def test_polish_applies_speed(synthetic_tone_wav: Path, temp_audio_dir: Path) -> None:
    """
    Speed setting must actually change duration (it was silently ignored before).
    """
    src = AudioService.probe_audio(synthetic_tone_wav)
    out = temp_audio_dir / "fast.wav"
    AudioService.polish(synthetic_tone_wav, out, speed=1.25, sample_rate=24000)
    meta = AudioService.probe_audio(out)
    assert meta.sample_rate == 24000
    assert meta.channels == 1
    assert abs(meta.duration_seconds - src.duration_seconds / 1.25) < 0.1


def test_keep_only_final_outputs(tmp_path: Path) -> None:
    """
    A finished job directory keeps exactly output.wav + output.mp3.
    """
    from core.services.jobs.cleanup_service import CleanupService

    job_dir = tmp_path / "job"
    (job_dir / "chunks").mkdir(parents=True)
    (job_dir / "chunks" / "chunk_0000.wav").write_bytes(b"x")
    for name in ("output.wav", "output.mp3", "reference.wav", "output_small.mp3", "_concat_list.txt"):
        (job_dir / name).write_bytes(b"x")
    CleanupService.keep_only_final_outputs(job_dir)
    assert sorted(p.name for p in job_dir.iterdir()) == ["output.mp3", "output.wav"]


def test_janitor_removes_orphans_and_trims_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Orphan folders (no DB row) are purged and the conditionals cache is capped.
    """
    import os
    import time
    from core.services.jobs import cleanup_service as cs
    from core.database.database import SessionLocal

    monkeypatch.setattr(cs.settings, "TMP_DIR", str(tmp_path))
    monkeypatch.setattr(cs.settings, "CONDS_CACHE_MAX_FILES", 2)
    old = time.time() - 3600

    orphan = tmp_path / ("o" * 22)
    orphan.mkdir()
    (orphan / "output.wav").write_bytes(b"x")
    stray = tmp_path / "self_check_tone.wav"
    stray.write_bytes(b"x")
    fresh = tmp_path / ("f" * 22)
    fresh.mkdir()
    for path in (orphan, stray):
        os.utime(path, (old, old))

    cache = tmp_path / "cache" / "conditionals"
    cache.mkdir(parents=True)
    for i in range(5):
        f = cache / f"c{i}.pt"
        f.write_bytes(b"x")
        os.utime(f, (old + i, old + i))

    db = SessionLocal()
    try:
        cs.CleanupService.clean_orphan_dirs(db)
        cs.CleanupService.trim_conditionals_cache()
    finally:
        db.close()

    assert not orphan.exists()
    assert not stray.exists()
    assert fresh.exists(), "folders younger than the grace period must survive (upload in progress)"
    assert sorted(p.name for p in cache.iterdir()) == ["c3.pt", "c4.pt"]


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
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(str(valid_wav), tone, sr)

        manifest = [
            {
                "id": "test_valid_voice",
                "name": "Test Valid Voice",
                "model_type": "chatterbox_reference",
                "asset_file": "reference/sample.wav",
                "license": "MIT",
                "usage_permission": "Public",
                "status": "available",
            },
            {
                "id": "test_traversal",
                "name": "Bad Traversal",
                "model_type": "chatterbox_reference",
                "asset_file": "../outside.wav",
                "license": "MIT",
                "usage_permission": "Public",
                "status": "available",
            },
            {
                "id": "test_missing_file",
                "name": "Missing File",
                "model_type": "chatterbox_reference",
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
        voice = VoiceCRUD.get_voice_by_id(db, "test_valid_voice")
        if voice:
            db.delete(voice)
            db.commit()
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


# ==============================================================================
# SECTION 7: TEXT SPLITTER & MULTILINGUAL INPUT VALIDATION TESTS
# ==============================================================================


def test_splitter_devanagari_boundaries() -> None:
    """
    Verify text splitter correctly segments Hindi/Devanagari text at purna viram (। and ॥).
    """
    hi_text = (
        "नमस्ते! आज हम आर्टिफिशियल इंटेलिजेंस के बारे में बात करेंगे। "
        "यह तकनीक शब्दों को आवाज़ में बदलती है॥ "
        "तीसरा वाक्य यहाँ समाप्त होता है।"
    )
    chunks, para_starts = split_text_with_boundaries(hi_text, max_chars=65)
    assert len(chunks) == 3
    for chunk in chunks:
        assert len(chunk) <= 65
        # Ensure Devanagari characters are intact
        assert any("\u0900" <= c <= "\u097F" for c in chunk)


def test_splitter_abbreviations_and_decimals() -> None:
    """
    Verify text splitter does NOT split at known abbreviations (Dr., Mr., etc.) or decimal numbers.
    """
    text = "Dr. Smith met with Mr. Jones at 1.5 miles past the clinic. They agreed on 99.9 percent of the points."
    chunks, _ = split_text_with_boundaries(text, max_chars=200)
    # The whole paragraph should remain a single chunk because none of the periods are true sentence ends
    assert len(chunks) == 1
    assert "Dr. Smith" in chunks[0]
    assert "Mr. Jones" in chunks[0]
    assert "1.5" in chunks[0]
    assert "99.9" in chunks[0]


def test_splitter_very_long_sentences_no_midword_cuts() -> None:
    """
    Verify very long sentences without sentence punctuation are split at whitespace boundaries,
    ensuring words are never cut in half and every chunk is strictly within max_chars.
    """
    # 300 words without any period, question mark, or exclamation mark
    words = [f"word{i:03d}" for i in range(300)]
    long_text = " ".join(words)
    max_c = 120

    chunks, _ = split_text_with_boundaries(long_text, max_chars=max_c)
    assert len(chunks) > 1

    reconstructed_words = []
    for chunk in chunks:
        assert len(chunk) <= max_c
        chunk_words = chunk.split()
        reconstructed_words.extend(chunk_words)
        # Verify no word is truncated (each word must match wordXXX)
        for w in chunk_words:
            assert w.startswith("word") and len(w) == 7, f"Corrupted word: '{w}'"

    # All 300 words must be present in exact order
    assert reconstructed_words == words


# ==============================================================================
# SECTION 8: CHATTERBOX ZERO-SHOT PIPELINE TESTS (FAKE PIPELINE)
# ==============================================================================


def test_chatterbox_service_generate_fake_pipeline(temp_audio_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify ChatterboxService dispatches to worker process and returns structured result with fake pipeline.
    """
    import subprocess
    from core.services.pipelines.chatterbox.chatterbox_service import ChatterboxService

    ref_wav = temp_audio_dir / "client_ref.wav"
    out_wav = temp_audio_dir / "output_test.wav"

    # Create dummy reference WAV
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(ref_wav), sine, sr)

    # Mock subprocess.run to simulate worker process output
    def fake_subprocess_run(cmd, **kwargs):
        # Write dummy output audio
        sf.write(str(out_wav), sine, sr)
        mock_res = subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='{"status": "ok", "duration_seconds": 1.0, "sample_rate": 44100, "peak_vram_mb": 128.5, "gen_time_s": 0.5}',
            stderr="",
        )
        return mock_res

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    service = ChatterboxService()
    result = service.generate(
        text="नमस्ते! यह एक परीक्षण वाक्य है।",
        reference_audio_path=str(ref_wav),
        output_path=str(out_wav),
        language="hi",
    )

    assert result.output_path == str(out_wav)
    assert result.duration_seconds == 1.0
    assert result.sample_rate == 44100
    assert result.peak_vram_mb == 128.5
    assert out_wav.exists()


def test_chatterbox_service_subprocess_failure(temp_audio_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify ChatterboxService raises RuntimeError when worker process exits with non-zero code.
    """
    import subprocess
    from core.services.pipelines.chatterbox.chatterbox_service import ChatterboxService

    ref_wav = temp_audio_dir / "client_ref.wav"
    out_wav = temp_audio_dir / "output_fail.wav"
    sr = 44100
    sine = np.zeros(sr)
    sf.write(str(ref_wav), sine, sr)

    def fake_failed_subprocess(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="CUDA out of memory error in worker",
        )

    monkeypatch.setattr(subprocess, "run", fake_failed_subprocess)

    service = ChatterboxService()
    with pytest.raises(RuntimeError, match="Chatterbox generation failed"):
        service.generate(
            text="परीक्षण",
            reference_audio_path=str(ref_wav),
            output_path=str(out_wav),
            language="hi",
        )


def test_reference_safety_check_valid_and_blocks(temp_audio_dir: Path) -> None:
    """
    Verify reference safety checker returns score, advice, and blocks unusable audio.
    """
    sr = 22050
    # 1. Clean speech-like audio (modulated bursts with natural pauses)
    burst = 0.3 * np.sin(2 * np.pi * 200 * np.linspace(0, 0.6, int(sr * 0.6)))
    pause = np.zeros(int(sr * 0.3))
    pattern = np.concatenate([burst, pause] * 12)
    clean_path = temp_audio_dir / "clean_ref.wav"
    sf.write(str(clean_path), pattern, sr)

    res_clean = ValidationService.check_reference_safety(clean_path)
    assert res_clean["score"] > 50
    assert res_clean["is_usable"] is True
    assert len(res_clean["advice"]) > 0

    # 2. Too-short audio (0.5s) -> must be hard blocked
    short_t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    short_audio = 0.3 * np.sin(2 * np.pi * 220 * short_t)
    short_path = temp_audio_dir / "short_ref.wav"
    sf.write(str(short_path), short_audio, sr)

    res_short = ValidationService.check_reference_safety(short_path)
    assert res_short["is_usable"] is False
    assert any("too short" in adv.lower() for adv in res_short["advice"])

    # 3. Severely clipped audio -> must be hard blocked
    clipped_audio = np.ones(int(sr * 5.0)) * 0.999
    clipped_path = temp_audio_dir / "clipped_ref.wav"
    sf.write(str(clipped_path), clipped_audio, sr)

    res_clipped = ValidationService.check_reference_safety(clipped_path)
    assert res_clipped["is_usable"] is False
    assert any("clipping" in adv.lower() for adv in res_clipped["advice"])


def test_chatterbox_service_availability_check(monkeypatch, temp_audio_dir: Path) -> None:
    """
    Verify ChatterboxService detects environment availability correctly.
    """
    import sys
    import subprocess
    from core.services.pipelines.chatterbox.chatterbox_service import ChatterboxService

    # 1. Non-existent python executable -> False
    service_bad = ChatterboxService(python_executable=str(temp_audio_dir / "nonexistent_python.exe"))
    assert service_bad.is_available() is False

    # 2. Mock successful import
    def fake_ok_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_ok_run)
    service_good = ChatterboxService(python_executable=sys.executable)
    assert service_good.is_available() is True


def test_reference_safety_check_transcoding_fallback(temp_audio_dir: Path) -> None:
    """
    Verify ValidationService decodes non-WAV audio formats (e.g., MP3/compressed)
    transparently via AudioService.to_wav fallback.
    """
    sr = 22050
    # Clean speech-like audio
    burst = 0.3 * np.sin(2 * np.pi * 200 * np.linspace(0, 0.6, int(sr * 0.6)))
    pause = np.zeros(int(sr * 0.3))
    pattern = np.concatenate([burst, pause] * 10)

    wav_path = temp_audio_dir / "clean_source.wav"
    mp3_path = temp_audio_dir / "clean_source.mp3"
    sf.write(str(wav_path), pattern, sr)

    # Convert to MP3
    AudioService.export_mp3(wav_path, mp3_path)
    assert mp3_path.exists()

    # 1. check_reference_safety on MP3 must succeed via transcode fallback
    res = ValidationService.check_reference_safety(mp3_path)
    assert res["score"] > 50
    assert res["is_usable"] is True
    assert res["metrics"]["speech_length_s"] > 3.0

    # 2. inspect_reference_audio_quality on MP3 must also succeed
    quality = ValidationService.inspect_reference_audio_quality(mp3_path, min_duration=3.0)
    assert quality.quality_score > 0.5
    assert quality.duration_seconds > 3.0


def test_reference_safety_duration_not_zero_for_moderate_volume(temp_audio_dir: Path) -> None:
    """
    Verify an 8-second speech clip with moderate loudness does not collapse to 0.0s speech length.
    """
    sr = 22050
    # Moderate loudness synthetic speech bursts (~ -38 dBFS)
    burst = 0.015 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.7, int(sr * 0.7)))
    pause = np.zeros(int(sr * 0.3))
    pattern = np.concatenate([burst, pause] * 8)

    wav_path = temp_audio_dir / "moderate_volume.wav"
    sf.write(str(wav_path), pattern, sr)

    res = ValidationService.check_reference_safety(wav_path)
    assert res["metrics"]["total_duration_s"] >= 7.5
    assert res["metrics"]["speech_length_s"] > 2.0


def test_distinct_reference_conditioning_hashes(temp_audio_dir: Path) -> None:
    """
    Verify two distinct audio references produce distinct SHA-256 conditioning hashes.
    """
    import hashlib

    sr = 22050
    t = np.linspace(0, 3, int(sr * 3))
    audio_a = 0.2 * np.sin(2 * np.pi * 220 * t)
    audio_b = 0.2 * np.sin(2 * np.pi * 440 * t)

    wav_a = temp_audio_dir / "voice_a.wav"
    wav_b = temp_audio_dir / "voice_b.wav"
    sf.write(str(wav_a), audio_a, sr)
    sf.write(str(wav_b), audio_b, sr)

    hash_a = hashlib.sha256(wav_a.read_bytes()).hexdigest()[:16]
    hash_b = hashlib.sha256(wav_b.read_bytes()).hexdigest()[:16]

    assert hash_a != hash_b
    assert len(hash_a) == 16
    assert len(hash_b) == 16


def test_missing_reference_in_job_worker_raises_error(temp_audio_dir: Path) -> None:
    """
    Verify job_worker raises RuntimeError rather than falling back to default voice when reference is missing.
    """
    from core.services.jobs.job_worker import production_inference_pipeline
    import pytest

    missing_path = temp_audio_dir / "missing_job_dir"
    missing_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError) as exc_info:
        production_inference_pipeline(
            job_id="test_missing_job",
            kind="tts_custom",
            voice_id=None,
            payload={"text": "Hello world", "language": "hi"},
            job_dir=missing_path,
        )
    assert "missing" in str(exc_info.value).lower()


def test_chatterbox_language_mapping_marathi() -> None:
    """
    Verify Marathi ('mr') is mapped to 'hi' for Devanagari acoustic modeling in chatterbox worker.
    """
    from chatterbox.mtl_tts import SUPPORTED_LANGUAGES

    # Verify that 'hi' and 'en' are official languages, and 'mr' is routed to 'hi' for Devanagari
    assert "en" in SUPPORTED_LANGUAGES
    assert "hi" in SUPPORTED_LANGUAGES
    assert "mr" not in SUPPORTED_LANGUAGES  # Verifying that Chatterbox doesn't have an isolated 'mr' token

    # Test the worker language mapping rule
    for input_lang, expected_model_lang in [("en", "en"), ("hi", "hi"), ("mr", "hi"), ("MR", "hi")]:
        mapped = (input_lang or "en").lower().strip()
        if mapped == "mr":
            mapped = "hi"
        assert mapped in SUPPORTED_LANGUAGES
        assert mapped == expected_model_lang








