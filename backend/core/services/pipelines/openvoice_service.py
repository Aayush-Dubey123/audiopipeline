"""
OpenVoice V2 and MeloTTS tone color conversion inference pipeline.

Provides structured text-to-speech synthesis and tone color transfer using
MeloTTS English base speaker models and OpenVoice V2 tone conversion neural networks.
"""

import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import soundfile as sf
import torch

from commons.logger import logger

log = logger(__name__)


class OpenVoiceError(Exception):
    """Base exception for OpenVoice pipeline and inference operations."""
    pass


class ReferenceAudioNotFoundError(OpenVoiceError):
    """Raised when the reference audio file cannot be found on disk."""
    pass


class ModelCheckpointNotFoundError(OpenVoiceError):
    """Raised when required OpenVoice or MeloTTS model weights are missing."""
    pass


class OpenVoiceInferenceError(OpenVoiceError):
    """Raised when an error occurs during model evaluation or voice conversion."""
    pass


@dataclass(frozen=True)
class GenerationResult:
    """
    Structured outcome metadata from an OpenVoice V2 synthesis run.
    """
    output_path: str
    duration_seconds: float
    file_size_bytes: int
    sample_rate: int
    base_tts_duration_seconds: float
    tone_conversion_duration_seconds: float
    total_duration_seconds: float
    peak_vram_allocated_mb: float
    peak_vram_reserved_mb: float
    watermark_applied: bool


class OpenVoiceService:
    """
    Inference service managing MeloTTS base synthesis and OpenVoice V2 tone conversion.
    """

    _instances: set = set()

    def __init__(
        self,
        converter_dir: Optional[str] = None,
        base_speakers_dir: Optional[str] = None,
        device: Optional[str] = None,
        enable_watermark: bool = True,
    ) -> None:
        """
        Initialize the OpenVoice service configuration.

        Args:
            converter_dir (Optional[str]): Directory containing converter config.json
                and checkpoint.pth. Defaults to backend/voices/checkpoints_v2/converter.
            base_speakers_dir (Optional[str]): Directory containing base speaker embedding
                files. Defaults to backend/voices/checkpoints_v2/base_speakers/ses.
            device (Optional[str]): Device identifier ('cuda:0' or 'cpu'). Defaults
                to CUDA if available, otherwise CPU.
            enable_watermark (bool): Whether to embed the MyShell audio watermark.
                Defaults to True.

        Returns:
            None

        Raises:
            None
        """
        # Resolve backend root directory
        self.backend_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.default_checkpoints_dir = self.backend_dir / "voices" / "checkpoints_v2"

        self.converter_dir = (
            Path(converter_dir)
            if converter_dir
            else self.default_checkpoints_dir / "converter"
        )
        self.base_speakers_dir = (
            Path(base_speakers_dir)
            if base_speakers_dir
            else self.default_checkpoints_dir / "base_speakers" / "ses"
        )

        if device:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.enable_watermark = enable_watermark

        # Lazy model instances and caches
        self._converter = None
        self._tts_model = None
        self._source_se_cache: Dict[str, torch.Tensor] = {}
        self._target_se_cache: Dict[str, Tuple[float, torch.Tensor]] = {}

        OpenVoiceService._instances.add(self)

        log.info(
            "Initialized OpenVoiceService on device '%s' (converter_dir: %s, watermark: %s)",
            self.device,
            self.converter_dir,
            self.enable_watermark,
        )

    def unload_models(self) -> None:
        """
        Unload neural network models from GPU memory and invoke garbage collection.

        Returns:
            None

        Raises:
            None
        """
        self._converter = None
        self._tts_model = None
        self._source_se_cache.clear()
        self._target_se_cache.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("OpenVoice models successfully unloaded from memory.")

    @classmethod
    def unload_all(cls) -> None:
        """
        Unload all cached OpenVoice models across any active instances.

        Returns:
            None

        Raises:
            None
        """
        for instance in list(cls._instances):
            instance.unload_models()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_converter(self) -> Any:
        """
        Lazily initialize and load the OpenVoice ToneColorConverter model.

        Returns:
            ToneColorConverter: Instantiated and loaded converter model.

        Raises:
            ModelCheckpointNotFoundError: If config.json or checkpoint.pth is missing.
            OpenVoiceInferenceError: If model instantiation fails.
        """
        if self._converter is not None:
            return self._converter

        config_path = self.converter_dir / "config.json"
        checkpoint_path = self.converter_dir / "checkpoint.pth"

        if not config_path.exists():
            raise ModelCheckpointNotFoundError(
                f"ToneColorConverter config not found at: {config_path}"
            )
        if not checkpoint_path.exists():
            raise ModelCheckpointNotFoundError(
                f"ToneColorConverter checkpoint not found at: {checkpoint_path}"
            )

        if "cuda" in self.device:
            try:
                from core.services.pipelines.rvc_service import RVCService
                RVCService.unload_all()
            except Exception:
                pass

        try:
            from openvoice.api import ToneColorConverter

            log.info("Loading ToneColorConverter on %s...", self.device)
            converter = ToneColorConverter(
                str(config_path),
                device=self.device,
            )
            if not self.enable_watermark:
                converter.watermark_model = None
            converter.load_ckpt(str(checkpoint_path))
            self._converter = converter
            log.info("ToneColorConverter successfully loaded.")
            return self._converter
        except Exception as exc:
            log.error("Failed to load ToneColorConverter: %s", exc)
            raise OpenVoiceInferenceError(f"Converter initialization failed: {exc}") from exc

    def _load_tts(self) -> Any:
        """
        Lazily initialize and load the MeloTTS English synthesis model.

        Returns:
            TTS: Instantiated MeloTTS English model.

        Raises:
            OpenVoiceInferenceError: If MeloTTS initialization fails.
        """
        if self._tts_model is not None:
            return self._tts_model

        if "cuda" in self.device:
            try:
                from core.services.pipelines.rvc_service import RVCService
                RVCService.unload_all()
            except Exception:
                pass

        try:
            from melo.api import TTS

            log.info("Loading MeloTTS English model on %s...", self.device)
            tts_model = TTS(language="EN", device=self.device)
            self._tts_model = tts_model
            log.info("MeloTTS English model successfully loaded.")
            return self._tts_model
        except Exception as exc:
            log.error("Failed to load MeloTTS model: %s", exc)
            raise OpenVoiceInferenceError(f"MeloTTS initialization failed: {exc}") from exc

    def get_source_speaker_embedding(self, speaker_key: str = "en-default.pth") -> torch.Tensor:
        """
        Load or retrieve cached source speaker embedding for base MeloTTS synthesis.

        Args:
            speaker_key (str): Embedding filename inside base_speakers_dir.

        Returns:
            torch.Tensor: Source speaker embedding tensor on target device.

        Raises:
            ModelCheckpointNotFoundError: If source speaker embedding file is missing.
        """
        if speaker_key in self._source_se_cache:
            return self._source_se_cache[speaker_key]

        se_path = self.base_speakers_dir / speaker_key
        if not se_path.exists():
            raise ModelCheckpointNotFoundError(
                f"Base speaker embedding not found at: {se_path}"
            )

        embedding = torch.load(str(se_path), map_location=self.device, weights_only=True)
        self._source_se_cache[speaker_key] = embedding
        return embedding

    def extract_target_speaker_embedding(
        self,
        reference_wav_path: str,
        cache_on_disk: bool = True,
    ) -> torch.Tensor:
        """
        Extract or retrieve cached speaker embedding (SE) from reference audio.

        Args:
            reference_wav_path (str): Filepath to the clean 24kHz reference WAV.
            cache_on_disk (bool): Whether to persist extracted tensor to disk.

        Returns:
            torch.Tensor: Target speaker embedding tensor.

        Raises:
            ReferenceAudioNotFoundError: If reference file does not exist on disk.
            OpenVoiceInferenceError: If feature extraction fails.
        """
        ref_path = Path(reference_wav_path).resolve()
        if not ref_path.exists() or not ref_path.is_file():
            raise ReferenceAudioNotFoundError(
                f"Reference audio not found: {reference_wav_path}"
            )

        current_mtime = ref_path.stat().st_mtime
        cache_key = str(ref_path)

        if cache_key in self._target_se_cache:
            cached_mtime, tensor = self._target_se_cache[cache_key]
            if cached_mtime == current_mtime:
                log.debug("Using in-memory target speaker embedding for: %s", ref_path.name)
                return tensor

        disk_cache_path = ref_path.with_suffix(".se.pth")
        if cache_on_disk and disk_cache_path.exists():
            if disk_cache_path.stat().st_mtime >= current_mtime:
                log.info("Loading cached target speaker embedding from: %s", disk_cache_path.name)
                tensor = torch.load(str(disk_cache_path), map_location=self.device, weights_only=True)
                self._target_se_cache[cache_key] = (current_mtime, tensor)
                return tensor

        converter = self._load_converter()
        try:
            log.info("Extracting target speaker embedding from: %s", ref_path.name)
            target_se = converter.extract_se(
                str(ref_path),
                se_save_path=str(disk_cache_path) if cache_on_disk else None,
            )
            self._target_se_cache[cache_key] = (current_mtime, target_se)
            return target_se
        except Exception as exc:
            log.error("Failed to extract target speaker embedding: %s", exc)
            raise OpenVoiceInferenceError(f"Embedding extraction failed: {exc}") from exc

    def generate(
        self,
        text: str,
        reference_audio_path: str,
        output_path: str,
        speaker_id: str = "EN-Default",
        speed: float = 1.0,
        tau: float = 0.3,
    ) -> GenerationResult:
        """
        Generate synthesized speech transferred to the tone of reference audio.

        Args:
            text (str): Script text to synthesize in English.
            reference_audio_path (str): Filepath to reference audio WAV clip.
            output_path (str): Destination path for final converted WAV.
            speaker_id (str): Base MeloTTS speaker profile name ('EN-Default', 'EN-US', etc.).
            speed (float): Playback rate multiplier. Defaults to 1.0.
            tau (float): OpenVoice conversion temperature. Defaults to 0.3.

        Returns:
            GenerationResult: Comprehensive metrics and output file metadata.

        Raises:
            ValueError: If text is empty or invalid.
            ReferenceAudioNotFoundError: If reference audio file does not exist.
            OpenVoiceInferenceError: If synthesis or conversion fails.
        """
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Input text cannot be empty or whitespace.")

        ref_file = Path(reference_audio_path).resolve()
        if not ref_file.exists():
            raise ReferenceAudioNotFoundError(
                f"Reference audio file does not exist: {reference_audio_path}"
            )

        out_file = Path(output_path)
        if not out_file.is_absolute():
            out_file = self.backend_dir / out_file
        out_file = out_file.resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Upfront validation of checkpoint files
        config_path = self.converter_dir / "config.json"
        checkpoint_path = self.converter_dir / "checkpoint.pth"
        if not config_path.exists():
            raise ModelCheckpointNotFoundError(
                f"ToneColorConverter config not found at: {config_path}"
            )
        if not checkpoint_path.exists():
            raise ModelCheckpointNotFoundError(
                f"ToneColorConverter checkpoint not found at: {checkpoint_path}"
            )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        temp_src_file = None

        try:
            # 1. Base TTS Synthesis with MeloTTS
            tts_model = self._load_tts()
            spk_map = getattr(tts_model.hps.data, "spk2id", {})
            try:
                numeric_spk_id = spk_map[speaker_id]
            except (KeyError, TypeError):
                numeric_spk_id = getattr(spk_map, speaker_id, 0)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_src_file = tmp.name

            tts_start = time.time()
            tts_model.tts_to_file(
                clean_text,
                numeric_spk_id,
                temp_src_file,
                speed=speed,
                quiet=True,
            )
            base_tts_duration = time.time() - tts_start
            log.info("MeloTTS base audio generated in %.2fs", base_tts_duration)

            # 2. Extract / Retrieve Speaker Embeddings
            converter = self._load_converter()
            source_se = self.get_source_speaker_embedding("en-default.pth")
            target_se = self.extract_target_speaker_embedding(str(ref_file))

            # 3. Tone Color Conversion
            conv_start = time.time()
            converter.convert(
                audio_src_path=temp_src_file,
                src_se=source_se,
                tgt_se=target_se,
                output_path=str(out_file),
                tau=tau,
                message="@MyShell",
            )
            tone_conv_duration = time.time() - conv_start
            log.info("Tone conversion completed in %.2fs", tone_conv_duration)

            # Safely release temporary conversion tensors
            del source_se, target_se

            total_duration = time.time() - start_time

            # 4. Measure Output Characteristics
            info = sf.info(str(out_file))
            file_size = out_file.stat().st_size

            peak_vram_alloc = 0.0
            peak_vram_res = 0.0
            if torch.cuda.is_available() and "cuda" in self.device:
                peak_vram_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
                peak_vram_res = torch.cuda.max_memory_reserved() / (1024 * 1024)

            log.info(
                "Generation successful: output=%s, size=%d bytes, duration=%.2fs, peak_vram=%.2f MB",
                out_file.name,
                file_size,
                info.duration,
                peak_vram_alloc,
            )

            return GenerationResult(
                output_path=str(out_file),
                duration_seconds=float(info.duration),
                file_size_bytes=file_size,
                sample_rate=info.samplerate,
                base_tts_duration_seconds=base_tts_duration,
                tone_conversion_duration_seconds=tone_conv_duration,
                total_duration_seconds=total_duration,
                peak_vram_allocated_mb=peak_vram_alloc,
                peak_vram_reserved_mb=peak_vram_res,
                watermark_applied=self.enable_watermark,
            )

        except (ModelCheckpointNotFoundError, ReferenceAudioNotFoundError, ValueError):
            raise
        except torch.cuda.OutOfMemoryError as oom_err:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.error("CUDA Out of Memory during OpenVoice generation: %s", oom_err)
            raise OpenVoiceInferenceError(
                f"GPU VRAM exhausted during inference. Details: {oom_err}"
            ) from oom_err
        except Exception as exc:
            log.error("OpenVoice generation failed: %s", exc)
            raise OpenVoiceInferenceError(f"Inference pipeline failed: {exc}") from exc
        finally:
            if temp_src_file and os.path.exists(temp_src_file):
                try:
                    os.remove(temp_src_file)
                except OSError:
                    pass
            if torch.cuda.is_available() and "cuda" in self.device:
                torch.cuda.empty_cache()


def main() -> None:
    """
    CLI execution entry point for standalone verification and benchmarking.
    """
    parser = argparse.ArgumentParser(
        description="AI Voice Studio: OpenVoice V2 + MeloTTS CLI Inference Harness"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="This is an automated verification test for OpenVoice V2 voice cloning.",
        help="Text script to synthesize",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="voices/reference/my_voice.wav",
        help="Path to reference audio file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="voices/output_cli_test.wav",
        help="Path to output WAV file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'cuda:0', 'cpu', or None for auto",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speech speed multiplier",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.3,
        help="Tone color transfer temperature",
    )

    args = parser.parse_args()

    service = OpenVoiceService(device=args.device)

    ref_path = Path(args.reference)
    if not ref_path.is_absolute():
        ref_path = service.backend_dir / ref_path

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = service.backend_dir / out_path

    print("=" * 60)
    print("AI Voice Studio - OpenVoice V2 Inference Test")
    print("=" * 60)
    print(f"Device:           {service.device}")
    print(f"Input text:       {args.text}")
    print(f"Reference audio:  {ref_path}")
    print(f"Output path:      {out_path}")
    print(f"Speech speed:     {args.speed}")
    print(f"Tau:              {args.tau}")
    print("-" * 60)

    try:
        result = service.generate(
            text=args.text,
            reference_audio_path=str(ref_path),
            output_path=str(out_path),
            speed=args.speed,
            tau=args.tau,
        )
        print("RESULT: SUCCESS")
        print(f"Generated Audio Duration: {result.duration_seconds:.2f}s")
        print(f"Audio Sample Rate:        {result.sample_rate} Hz")
        print(f"Output File Size:         {result.file_size_bytes:,} bytes")
        print(f"Base TTS Time:            {result.base_tts_duration_seconds:.2f}s")
        print(f"Tone Conversion Time:     {result.tone_conversion_duration_seconds:.2f}s")
        print(f"Total Pipeline Time:      {result.total_duration_seconds:.2f}s")
        print(f"Peak VRAM Allocated:      {result.peak_vram_allocated_mb:.2f} MB")
        print(f"Peak VRAM Reserved:       {result.peak_vram_reserved_mb:.2f} MB")
        print(f"Watermark Applied:        {result.watermark_applied}")
        print("=" * 60)
    except Exception as exc:
        print(f"RESULT: FAILED - {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
