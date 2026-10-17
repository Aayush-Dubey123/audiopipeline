"""
Retrieval-based Voice Conversion (RVC) orchestration pipeline service.

Provides high-fidelity voice conversion using HuBERT/ContentVec acoustic feature
representations, RMVPE robust vocal pitch estimation, and VITS-based acoustic synthesis.
"""

import argparse
import hashlib
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from core import logger
from core.services.pipelines.rvc_core.rmvpe import RMVPE, SecurityValidationError
from core.services.pipelines.rvc_core.synthesizer import SynthesizerTrn

logging = logger(__name__)


class RVCError(Exception):
    """Base exception for RVC voice conversion operations."""
    pass


class ModelCheckpointNotFoundError(RVCError):
    """Raised when an RVC generator, RMVPE, or ContentVec checkpoint is missing."""
    pass


class InputAudioNotFoundError(RVCError):
    """Raised when the source audio file cannot be found on disk."""
    pass


class RVCInferenceError(RVCError):
    """Raised when an error occurs during feature extraction, pitch tracking, or synthesis."""
    pass


@dataclass(frozen=True)
class RVCResult:
    """
    Structured outcome metadata from an RVC voice conversion inference run.
    """
    output_path: str
    duration_seconds: float
    file_size_bytes: int
    sample_rate: int
    pitch_shift_semitones: int
    f0_extraction_duration_seconds: float
    feature_extraction_duration_seconds: float
    synthesis_duration_seconds: float
    total_duration_seconds: float
    peak_vram_allocated_mb: float
    peak_vram_reserved_mb: float


def verify_checkpoint_hash(
    file_path: Union[str, Path],
    expected_sha256: Optional[str] = None,
) -> bool:
    """
    Verify the cryptographic SHA-256 hash of a model checkpoint file.

    Ensures that downloaded model binaries have not been corrupted or tampered with.

    Args:
        file_path (Union[str, Path]): Path to target model checkpoint file.
        expected_sha256 (Optional[str]): Expected 64-character lowercase hexadecimal hash.

    Returns:
        bool: True if hash matches or if expected_sha256 is None.

    Raises:
        SecurityValidationError: If the computed hash does not match expected_sha256.
        FileNotFoundError: If the target file does not exist on disk.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    if expected_sha256 is None:
        return True

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    computed = sha256.hexdigest().lower()
    expected = expected_sha256.strip().lower()

    if computed != expected:
        raise SecurityValidationError(
            f"SHA-256 verification failed for '{path}'. Expected {expected}, got {computed}"
        )
    return True


class RVCService:
    """
    Orchestration service for Retrieval-based Voice Conversion (RVC v2).
    """

    _instances: set = set()

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        is_half: bool = False,
    ) -> None:
        """
        Initialize the RVC pipeline service configuration and paths.

        Args:
            models_dir (Optional[Union[str, Path]]): Base directory for RVC checkpoints.
                Defaults to backend/voices/rvc_models.
            device (Optional[str]): Target execution device ('cuda:0' or 'cpu').
                Defaults to CUDA if available, otherwise CPU.
            is_half (bool): Whether to use FP16 half-precision for synthesis.
                Defaults to False.

        Returns:
            None

        Raises:
            None
        """
        self.backend_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.default_models_dir = self.backend_dir / "voices" / "rvc_models"

        self.models_dir = Path(models_dir) if models_dir else self.default_models_dir
        self.rmvpe_path = self.models_dir / "predictors" / "rmvpe.pt"
        self.contentvec_dir = self.models_dir / "embedders" / "contentvec"

        if device:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.is_half = is_half and ("cuda" in self.device)

        # Lazy model instances
        self._net_g: Optional[SynthesizerTrn] = None
        self._rmvpe: Optional[RMVPE] = None
        self._hubert_model: Optional[Any] = None
        self._final_proj_weight: Optional[torch.Tensor] = None
        self._final_proj_bias: Optional[torch.Tensor] = None
        self._current_model_path: Optional[str] = None
        self._model_sr: int = 40000
        self._model_version: str = "v2"

        RVCService._instances.add(self)

        logging.info(
            "Initialized RVCService on device '%s' (models_dir: %s, is_half: %s)",
            self.device,
            self.models_dir,
            self.is_half,
        )

    def unload_models(self) -> None:
        """
        Unload RVC generator, RMVPE, and HuBERT networks from memory.

        Frees GPU VRAM and performs explicit garbage collection.

        Returns:
            None

        Raises:
            None
        """
        self._net_g = None
        self._rmvpe = None
        self._hubert_model = None
        self._final_proj_weight = None
        self._final_proj_bias = None
        self._current_model_path = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("RVC models successfully unloaded from memory.")

    @classmethod
    def unload_all(cls) -> None:
        """
        Unload all cached RVC models across any active instances.

        Enforces mutual GPU exclusion so OpenVoice can run without VRAM contention.

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

    def _unload_competing_services(self) -> None:
        """
        Unload active OpenVoice models when executing on GPU.

        Guarantees that only one heavy neural synthesis model resides in VRAM.

        Returns:
            None

        Raises:
            None
        """
        if "cuda" in self.device:
            try:
                from core.services.pipelines.openvoice_service import OpenVoiceService
                OpenVoiceService.unload_all()
            except Exception:
                pass

    def load_voice_model(
        self,
        voice_model_path: Union[str, Path],
        expected_sha256: Optional[str] = None,
    ) -> SynthesizerTrn:
        """
        Lazily load an RVC voice checkpoint using strict safe weights_only deserialization.

        Args:
            voice_model_path (Union[str, Path]): Path to target .pth voice checkpoint.
            expected_sha256 (Optional[str]): Expected cryptographic SHA-256 hash.

        Returns:
            SynthesizerTrn: Loaded and initialized synthesizer network.

        Raises:
            ModelCheckpointNotFoundError: If checkpoint file does not exist on disk.
            SecurityValidationError: If checkpoint fails safe weights_only validation or hash check.
            RVCInferenceError: If model instantiation or state dictionary loading fails.
        """
        path = Path(voice_model_path)
        if not path.is_absolute() and not path.exists():
            path = self.models_dir / path

        if not path.exists():
            raise ModelCheckpointNotFoundError(f"Voice model checkpoint not found: {path}")

        # Check cryptographic hash if provided
        verify_checkpoint_hash(path, expected_sha256)

        if self._net_g is not None and self._current_model_path == str(path):
            return self._net_g

        self._unload_competing_services()

        logging.info("Loading RVC voice checkpoint safely from %s...", path)
        try:
            # Enforce strict safe deserialization
            cpt = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise SecurityValidationError(
                f"Checkpoint at '{path}' failed safe weights_only validation: {exc}"
            ) from exc

        try:
            if not isinstance(cpt, dict) or "config" not in cpt or "weight" not in cpt:
                raise SecurityValidationError(
                    f"Invalid checkpoint format in '{path}': missing config or weight keys."
                )

            config = cpt["config"]
            weight = cpt["weight"]

            # Robustly parse sample rate (e.g. 40000 or "40k")
            sr_val = cpt.get("sr", 40000)
            if isinstance(sr_val, str):
                if sr_val.lower().endswith("k"):
                    sr_val = int(float(sr_val[:-1]) * 1000)
                else:
                    sr_val = int(sr_val)
            self._model_sr = int(sr_val)

            # Detect model version and input representation channels
            emb_phone = weight.get("enc_p.emb_phone.weight")
            if emb_phone is not None and emb_phone.shape[1] == 256:
                self._model_version = "v1"
            elif cpt.get("version") == "v1":
                self._model_version = "v1"
            else:
                self._model_version = cpt.get("version", "v2") or "v2"

            # Instantiate SynthesizerTrn
            net_g = SynthesizerTrn(
                *config,
                is_half=self.is_half,
                version=self._model_version,
            )
            net_g.load_state_dict(weight, strict=False)
            net_g.eval()

            if self.is_half:
                net_g = net_g.half()
            net_g = net_g.to(self.device)
            net_g.remove_weight_norm()

            self._net_g = net_g
            self._current_model_path = str(path)
            logging.info("RVC voice model successfully loaded (version: %s, sr: %d Hz).", self._model_version, self._model_sr)
            return self._net_g
        except Exception as exc:
            raise RVCInferenceError(f"Failed to initialize SynthesizerTrn: {exc}") from exc

    def load_rmvpe(self, expected_sha256: Optional[str] = None) -> RMVPE:
        """
        Lazily load the RMVPE vocal pitch estimation network.

        Args:
            expected_sha256 (Optional[str]): Expected cryptographic SHA-256 hash.

        Returns:
            RMVPE: Loaded pitch estimator instance.

        Raises:
            ModelCheckpointNotFoundError: If rmvpe.pt file does not exist on disk.
            SecurityValidationError: If checkpoint fails safe weights_only validation or hash check.
        """
        if self._rmvpe is not None:
            return self._rmvpe

        if not self.rmvpe_path.exists():
            raise ModelCheckpointNotFoundError(
                f"RMVPE checkpoint not found at: {self.rmvpe_path}"
            )

        verify_checkpoint_hash(self.rmvpe_path, expected_sha256)
        self._unload_competing_services()

        logging.info("Loading RMVPE pitch estimator from %s...", self.rmvpe_path)
        self._rmvpe = RMVPE(str(self.rmvpe_path), device=self.device)
        return self._rmvpe

    def load_contentvec(self) -> Any:
        """
        Lazily load the HuBERT/ContentVec feature extraction model.

        Uses safetensors format to prevent arbitrary code execution during deserialization.

        Returns:
            Any: Instantiated Hugging Face HubertModel.

        Raises:
            ModelCheckpointNotFoundError: If local ContentVec model directory is missing.
            RVCInferenceError: If model loading fails.
        """
        if self._hubert_model is not None:
            return self._hubert_model

        self._unload_competing_services()

        try:
            from transformers import HubertModel

            model_identifier = str(self.contentvec_dir)
            if not self.contentvec_dir.exists():
                # Fallback to standard HuggingFace Hub repository identifier
                model_identifier = "lengyue/contentvec"
                logging.info("Local ContentVec directory not found, using Hugging Face hub: %s", model_identifier)

            logging.info("Loading ContentVec from '%s' with safetensors...", model_identifier)
            hubert = HubertModel.from_pretrained(model_identifier, use_safetensors=True)
            hubert.eval()
            if self.is_half:
                hubert = hubert.half()
            hubert = hubert.to(self.device)
            self._hubert_model = hubert
            return self._hubert_model
        except Exception as exc:
            raise RVCInferenceError(f"Failed to load ContentVec model: {exc}") from exc

    def _extract_f0(
        self,
        audio_16k: np.ndarray,
        pitch_shift: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract and pitch-shift fundamental frequencies using RMVPE.

        Args:
            audio_16k (np.ndarray): 1D audio sample array at 16,000 Hz.
            pitch_shift (int): Pitch shift in semitones (-24 to +24).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Continuous F0 contour (Hz) and quantized coarse pitch tokens.

        Raises:
            RVCInferenceError: If pitch extraction fails.
        """
        rmvpe = self.load_rmvpe()
        f0 = rmvpe.infer_from_audio(audio_16k, thred=0.03)

        # Apply pitch shift in semitones: f_new = f_old * 2^(shift / 12)
        if pitch_shift != 0:
            f0 = f0 * (2.0 ** (pitch_shift / 12.0))

        # Quantize continuous F0 to 1-255 coarse pitch bins (0 reserved for unvoiced frames)
        pitch_coarse = np.zeros_like(f0, dtype=np.int64)
        voiced = f0 > 1.0
        if np.any(voiced):
            pitch_bins = 1.0 + 1120.0 * np.log2(f0[voiced] / 10.0)
            pitch_coarse[voiced] = np.clip(np.round(pitch_bins), 1, 255).astype(np.int64)

        return f0, pitch_coarse

    def _extract_contentvec_features(
        self,
        audio_16k: np.ndarray,
        version: str = "v2",
    ) -> torch.Tensor:
        """
        Extract acoustic token features from ContentVec.

        Args:
            audio_16k (np.ndarray): 1D audio sample array at 16,000 Hz.
            version (str): Target model architecture version ("v1" or "v2"). Defaults to "v2".

        Returns:
            torch.Tensor: Extracted feature tensor [1, frames, 256|768] repeated to 100 Hz.

        Raises:
            RVCInferenceError: If feature extraction fails.
        """
        hubert = self.load_contentvec()
        with torch.no_grad():
            audio_t = torch.from_numpy(audio_16k).float().unsqueeze(0).to(self.device)
            if self.is_half:
                audio_t = audio_t.half()

            outputs = hubert(audio_t, output_hidden_states=True)
            # In RVC v1, representations are sourced from layer 9 (then projected to 256-dim via final_proj)
            # In RVC v2, representations are sourced from layer 12 (768-dim)
            layer_idx = 9 if version == "v1" else 12
            if hasattr(outputs, "hidden_states") and len(outputs.hidden_states) > layer_idx:
                feats = outputs.hidden_states[layer_idx]
            else:
                feats = outputs.last_hidden_state

            # If v1, project from 768-dim to 256-dim using verified safetensors weights
            if version == "v1":
                if self._final_proj_weight is None:
                    safetensors_path = self.contentvec_dir / "model.safetensors"
                    if safetensors_path.exists():
                        from safetensors import safe_open
                        with safe_open(str(safetensors_path), framework="pt", device=str(self.device)) as f:
                            self._final_proj_weight = f.get_tensor("final_proj.weight")
                            self._final_proj_bias = f.get_tensor("final_proj.bias")
                            if self.is_half:
                                self._final_proj_weight = self._final_proj_weight.half()
                                self._final_proj_bias = self._final_proj_bias.half()
                    else:
                        raise RVCInferenceError(
                            f"ContentVec safetensors not found at {safetensors_path} for v1 projection"
                        )
                feats = F.linear(feats, self._final_proj_weight, self._final_proj_bias)

            # ContentVec outputs 50 fps (stride 320 at 16kHz); repeat 2x to align with RMVPE 100 fps
            feats = feats.repeat_interleave(2, dim=1)
            return feats

    def convert(
        self,
        source_audio_path: Union[str, Path],
        voice_model_path: Union[str, Path],
        output_path: Union[str, Path],
        pitch_shift: int = 0,
        index_path: Optional[Union[str, Path]] = None,
        index_rate: float = 0.75,
        rms_mix_rate: float = 0.25,
        expected_sha256: Optional[str] = None,
    ) -> RVCResult:
        """
        Perform complete end-to-end RVC voice conversion on an input audio file.

        Args:
            source_audio_path (Union[str, Path]): Path to source input audio file.
            voice_model_path (Union[str, Path]): Path to target .pth voice checkpoint.
            output_path (Union[str, Path]): Destination path for synthesized WAV.
            pitch_shift (int): Pitch modification in semitones. Must be within [-24, +24].
            index_path (Optional[Union[str, Path]]): Optional FAISS retrieval index file.
            index_rate (float): Blend ratio for FAISS retrieval features (0.0 to 1.0).
            rms_mix_rate (float): Energy loudness envelope blend ratio with original audio.
            expected_sha256 (Optional[str]): Expected SHA-256 hash for the voice checkpoint.

        Returns:
            RVCResult: Detailed timing, audio metadata, and resource consumption report.

        Raises:
            ValueError: If pitch_shift is outside the valid range [-24, +24].
            InputAudioNotFoundError: If source audio file cannot be found.
            ModelCheckpointNotFoundError: If model checkpoint cannot be found.
            SecurityValidationError: If checkpoint deserialization fails safe validation.
            RVCInferenceError: If an error occurs during conversion pipeline execution.
        """
        # Validate internal pitch range: -24 to +24 semitones
        if pitch_shift < -24 or pitch_shift > 24:
            raise ValueError(
                f"pitch_shift must be between -24 and +24 semitones, got {pitch_shift}"
            )

        src_path = Path(source_audio_path)
        if not src_path.exists():
            raise InputAudioNotFoundError(f"Source audio file not found: {src_path}")

        out_path = Path(output_path)
        if not out_path.is_absolute():
            out_path = self.backend_dir / out_path
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.reset_peak_memory_stats()

        try:
            # 1. Load Voice Checkpoint (SynthesizerTrn) early to establish version & sample rate
            net_g = self.load_voice_model(voice_model_path, expected_sha256=expected_sha256)

            # 2. Load and resample input audio to 16kHz mono for extraction
            import librosa
            audio_16k, _ = librosa.load(str(src_path), sr=16000, mono=True)

            # 3. Extract Vocal Pitch Contour (RMVPE)
            f0_start = time.time()
            f0, pitch_coarse = self._extract_f0(audio_16k, pitch_shift=pitch_shift)
            f0_duration = time.time() - f0_start
            logging.info("F0 pitch extraction completed in %.2fs", f0_duration)

            # 4. Extract Acoustic Feature Tokens (ContentVec) matching model version
            feat_start = time.time()
            feats = self._extract_contentvec_features(audio_16k, version=self._model_version)
            feat_duration = time.time() - feat_start
            logging.info(
                "ContentVec feature extraction completed in %.2fs (dim: %d)",
                feat_duration,
                feats.shape[-1],
            )

            # 5. Optional FAISS Index Feature Retrieval
            if index_path and index_rate > 0.0:
                idx_path = Path(index_path)
                if idx_path.exists():
                    try:
                        import faiss
                        index = faiss.read_index(str(idx_path))
                        feats_np = feats.squeeze(0).cpu().numpy().astype("float32")
                        if index.d == feats_np.shape[1]:
                            _, indices = index.search(feats_np, k=8)
                            retrieved_feats = np.take(index.reconstruct_n(0, index.ntotal), indices[:, 0], axis=0)
                            retrieved_t = torch.from_numpy(retrieved_feats).unsqueeze(0).to(self.device)
                            if self.is_half:
                                retrieved_t = retrieved_t.half()
                            feats = (1.0 - index_rate) * feats + index_rate * retrieved_t
                            logging.info("Applied FAISS index feature blending (rate: %.2f)", index_rate)
                        else:
                            logging.warning(
                                "FAISS index dimension %d does not match feature dimension %d",
                                index.d,
                                feats_np.shape[1],
                            )
                    except Exception as idx_err:
                        logging.warning("FAISS index retrieval failed, continuing without index: %s", idx_err)

            # 6. Temporal Alignment between Features and Pitch
            min_len = min(feats.shape[1], pitch_coarse.shape[0])
            feats = feats[:, :min_len, :]
            pitch_coarse = pitch_coarse[:min_len]
            f0 = f0[:min_len]

            # 7. Acoustic Synthesis (SynthesizerTrn)
            synth_start = time.time()
            phone = feats
            phone_lengths = torch.tensor([min_len], device=self.device)
            pitch_t = torch.from_numpy(pitch_coarse).unsqueeze(0).long().to(self.device)
            nsff0_t = torch.from_numpy(f0).unsqueeze(0).float().to(self.device)
            sid_t = torch.zeros(1, dtype=torch.long, device=self.device)

            with torch.no_grad():
                out = net_g.infer(phone, phone_lengths, pitch_t, nsff0_t, sid_t)
                audio_out_t = out[0] if isinstance(out, tuple) else out
                audio_out = audio_out_t.squeeze().float().cpu().numpy()

            synth_duration = time.time() - synth_start
            logging.info("VITS acoustic synthesis completed in %.2fs", synth_duration)

            # 7. Energy Envelope Matching (RMS)
            if rms_mix_rate > 0.0 and len(audio_16k) > 0 and len(audio_out) > 0:
                orig_rms = np.sqrt(np.mean(audio_16k ** 2) + 1e-8)
                out_rms = np.sqrt(np.mean(audio_out ** 2) + 1e-8)
                if out_rms > 0:
                    scale = (orig_rms / out_rms) ** rms_mix_rate
                    audio_out = audio_out * scale

            # 8. Export Audio Waveform
            sf.write(str(out_path), audio_out, self._model_sr)

            total_duration = time.time() - start_time
            info = sf.info(str(out_path))
            file_size = out_path.stat().st_size

            peak_vram_alloc = 0.0
            peak_vram_res = 0.0
            if torch.cuda.is_available() and "cuda" in self.device:
                peak_vram_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
                peak_vram_res = torch.cuda.max_memory_reserved() / (1024 * 1024)

            logging.info(
                "RVC conversion successful: %s, size=%d bytes, duration=%.2fs, peak_vram=%.2f MB",
                out_path.name,
                file_size,
                info.duration,
                peak_vram_alloc,
            )

            return RVCResult(
                output_path=str(out_path),
                duration_seconds=float(info.duration),
                file_size_bytes=file_size,
                sample_rate=info.samplerate,
                pitch_shift_semitones=pitch_shift,
                f0_extraction_duration_seconds=f0_duration,
                feature_extraction_duration_seconds=feat_duration,
                synthesis_duration_seconds=synth_duration,
                total_duration_seconds=total_duration,
                peak_vram_allocated_mb=peak_vram_alloc,
                peak_vram_reserved_mb=peak_vram_res,
            )

        except (InputAudioNotFoundError, ModelCheckpointNotFoundError, SecurityValidationError, ValueError):
            raise
        except Exception as exc:
            logging.error("RVC inference failed: %s", exc)
            raise RVCInferenceError(f"RVC conversion pipeline failed: {exc}") from exc
        finally:
            if torch.cuda.is_available() and "cuda" in self.device:
                torch.cuda.empty_cache()


def main() -> None:
    """
    CLI execution harness for standalone RVC model conversion and benchmarking.
    """
    parser = argparse.ArgumentParser(
        description="AI Voice Studio: RVC (Retrieval-based Voice Conversion) CLI Harness"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input audio file",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to RVC .pth model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="voices/output_rvc_test.wav",
        help="Path to output WAV file",
    )
    parser.add_argument(
        "--pitch",
        type=int,
        default=0,
        help="Pitch shift in semitones (-24 to +24)",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Path to optional FAISS .index file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'cuda:0', 'cpu', or None for auto",
    )

    args = parser.parse_args()
    service = RVCService(device=args.device)

    src_path = Path(args.source)
    if not src_path.is_absolute():
        src_path = service.backend_dir / src_path

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = service.backend_dir / model_path

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = service.backend_dir / out_path

    idx_path = None
    if args.index:
        idx_path = Path(args.index)
        if not idx_path.is_absolute() and not idx_path.exists():
            idx_path = service.backend_dir / idx_path

    print("=" * 60)
    print("AI Voice Studio - RVC Inference Test")
    print("=" * 60)
    print(f"Device:           {service.device}")
    print(f"Source audio:     {src_path}")
    print(f"Voice model:      {model_path}")
    print(f"Output path:      {out_path}")
    print(f"Pitch shift:      {args.pitch} semitones")
    if idx_path:
        print(f"FAISS index:      {idx_path}")
    print("-" * 60)

    try:
        result = service.convert(
            source_audio_path=str(src_path),
            voice_model_path=str(model_path),
            output_path=str(out_path),
            pitch_shift=args.pitch,
            index_path=str(idx_path) if idx_path else None,
        )
        print("RESULT: SUCCESS")
        print(f"Generated Audio Duration: {result.duration_seconds:.2f}s")
        print(f"Audio Sample Rate:        {result.sample_rate} Hz")
        print(f"Output File Size:         {result.file_size_bytes:,} bytes")
        print(f"Pitch Shift:              {result.pitch_shift_semitones} semitones")
        print(f"F0 Extraction Time:       {result.f0_extraction_duration_seconds:.2f}s")
        print(f"Feature Extraction Time:  {result.feature_extraction_duration_seconds:.2f}s")
        print(f"Synthesis Time:           {result.synthesis_duration_seconds:.2f}s")
        print(f"Total Pipeline Time:      {result.total_duration_seconds:.2f}s")
        print(f"Peak VRAM Allocated:      {result.peak_vram_allocated_mb:.2f} MB")
        print(f"Peak VRAM Reserved:       {result.peak_vram_reserved_mb:.2f} MB")
        print("=" * 60)
    except Exception as exc:
        print(f"RESULT: FAILED - {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
