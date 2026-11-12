"""
Chatterbox Multilingual service for Hindi zero-shot voice cloning.

Coordinates high-fidelity multilingual synthesis with resident in-memory model execution,
persistent reference conditionals caching, and isolated worker process fallback.
"""

import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from core import logger
from core.config import get_settings

logging = logger(__name__)
settings = get_settings()


@dataclass
class ChatterboxResult:
    """
    Structured outcome of a Chatterbox synthesis run.
    """

    output_path: str
    duration_seconds: float
    sample_rate: int
    peak_vram_mb: float
    gen_time_s: float


class ChatterboxService:
    """
    Coordinates high-fidelity speech generation via Chatterbox Multilingual.
    """

    _resident_model: Optional[Any] = None
    _model_lock: threading.Lock = threading.Lock()
    # Bounded LRU: each entry holds GPU tensors, so an unbounded dict leaks VRAM
    _conds_cache: "OrderedDict[str, Any]" = OrderedDict()
    _CONDS_MEMORY_MAX = 4
    _ref_hash_memo: Dict[tuple, str] = {}
    _MAX_CHUNK_RETRIES = 1

    @staticmethod
    def resolve_device() -> str:
        """Pick the inference device from CHATTERBOX_DEVICE (auto | cuda | cpu)."""
        import torch

        pref = (getattr(settings, "CHATTERBOX_DEVICE", "auto") or "auto").lower().strip()
        if pref == "cpu":
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if pref == "cuda":
            logging.warning("CHATTERBOX_DEVICE=cuda but CUDA is unavailable; falling back to CPU")
        return "cpu"

    @staticmethod
    def conditionals_cache_dir() -> Path:
        """Anchored under backend/storage/tmp regardless of the launch folder."""
        return settings.tmp_path / "cache" / "conditionals"

    def __init__(
        self,
        python_executable: Optional[str] = None,
        use_in_process: Optional[bool] = None,
    ) -> None:
        """
        Initialize the Chatterbox service with python worker path resolution.

        Args:
            python_executable (Optional[str]): Path to Python binary with chatterbox installed.
            use_in_process (Optional[bool]): Explicitly enable or disable resident in-process inference.
        """
        logging.info("Initializing ChatterboxService")
        if use_in_process is not None:
            self.use_in_process = use_in_process
        elif os.environ.get("PYTEST_CURRENT_TEST"):
            self.use_in_process = False
        else:
            self.use_in_process = getattr(settings, "CHATTERBOX_IN_PROCESS", True)

        if python_executable:
            self.python_path = python_executable
        elif getattr(settings, "CHATTERBOX_PYTHON_PATH", "") and os.path.exists(settings.CHATTERBOX_PYTHON_PATH):
            self.python_path = settings.CHATTERBOX_PYTHON_PATH
        else:
            # Single project environment: backend/.venv
            backend_venv = Path(settings.backend_dir) / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            if backend_venv.exists():
                self.python_path = str(backend_venv)
            else:
                self.python_path = sys.executable

        self.worker_script = (
            Path(__file__).resolve().parent / "chatterbox_worker.py"
        )

    def is_available(self) -> bool:
        """
        Check if the Chatterbox Python executable and dependencies are available.

        Returns:
            bool: True if python executable exists and chatterbox package can be imported.
        """
        if not self.python_path or not os.path.exists(self.python_path):
            return False
        try:
            if Path(self.python_path).resolve() == Path(sys.executable).resolve():
                import importlib.util
                return importlib.util.find_spec("chatterbox") is not None
            res = subprocess.run(
                [str(self.python_path), "-c", "import chatterbox"],
                capture_output=True,
                timeout=60,
                check=False,
            )
            return res.returncode == 0
        except Exception:
            return False

    @classmethod
    def unload_all(cls) -> None:
        """
        Clear any resident models and lingering GPU cache to guarantee single-model GPU concurrency.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        logging.info("Executing ChatterboxService.unload_all")
        with cls._model_lock:
            cls._resident_model = None
            cls._conds_cache.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logging.warning(f"Failed to clear torch cache: {exc}")

    @classmethod
    def _get_resident_model(cls) -> Any:
        """
        Get or initialize the resident ChatterboxMultilingualTTS model singleton.

        Returns:
            Any: Loaded ChatterboxMultilingualTTS model.
        """
        with cls._model_lock:
            if cls._resident_model is not None:
                return cls._resident_model

            from chatterbox import ChatterboxMultilingualTTS

            device = cls.resolve_device()
            logging.info(f"Loading resident ChatterboxMultilingualTTS on {device}")
            cls._resident_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            return cls._resident_model

    @classmethod
    def _get_or_load_conditionals(
        cls,
        model: Any,
        ref_path: str,
        exaggeration: float,
        device: str,
    ) -> Any:
        """
        Retrieve or compute speaker conditioning for a reference audio file with persistent caching.

        Args:
            model (Any): Loaded Chatterbox model.
            ref_path (str): Path to reference audio WAV file.
            exaggeration (float): Exaggeration multiplier.
            device (str): Inference device ('cuda' or 'cpu').

        Returns:
            Any: Conditionals object ready for synthesis.
        """
        ref_hash = cls._reference_hash(ref_path)
        # "c2": conditionals built from the cleaned best-window clip (older raw-reference caches are ignored)
        cache_key = f"{ref_hash}_{int(exaggeration * 100)}_c2"

        # 1. In-memory cache hit
        if cache_key in cls._conds_cache:
            cls._conds_cache.move_to_end(cache_key)
            return cls._conds_cache[cache_key]

        # 2. Disk cache hit
        cache_dir = cls.conditionals_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{cache_key}.pt"

        from chatterbox.mtl_tts import Conditionals

        if cache_file.exists():
            try:
                conds = Conditionals.load(cache_file, map_location=device)
                cls._remember_conds(cache_key, conds)
                try:
                    os.utime(cache_file)  # mark as recently used for the janitor's LRU trim
                except OSError:
                    pass
                logging.info(f"Loaded speaker conditioning from disk cache: '{cache_file.name}'")
                return conds
            except Exception as load_err:
                logging.warning(f"Failed to load cached conditionals from {cache_file}: {load_err}")

        # 3. Compute and cache
        logging.info(f"Extracting new speaker conditioning for ref_hash='{ref_hash}'")
        model.prepare_conditionals(cls.conditioning_source(ref_path), exaggeration=exaggeration)
        conds = model.conds
        cls._remember_conds(cache_key, conds)
        try:
            conds.save(cache_file)
            logging.info(f"Saved speaker conditioning to disk cache: '{cache_file.name}'")
        except Exception as save_err:
            logging.warning(f"Failed to save conditionals cache to {cache_file}: {save_err}")

        return conds

    @staticmethod
    def conditioning_source(ref_path: str) -> Any:
        """
        Cleaned best-window conditioning clip (in-memory WAV), or the raw path if preparation fails.

        Args:
            ref_path (str): Path to reference audio.

        Returns:
            Any: io.BytesIO WAV buffer or the original path string.
        """
        try:
            from core.services.audio.audio_service import AudioService

            return AudioService.conditioning_clip_buffer(Path(ref_path))
        except Exception as prep_err:
            logging.warning(f"Conditioning clip preparation failed, using raw reference: {prep_err}")
            return str(ref_path)

    @staticmethod
    def resolve_params(
        temperature: Optional[float], cfg_weight: Optional[float], exaggeration: Optional[float]
    ) -> tuple:
        """Fill unset generation parameters (temperature, cfg_weight, exaggeration) from settings."""
        return (
            float(settings.CHATTERBOX_TEMPERATURE if temperature is None else temperature),
            float(settings.CHATTERBOX_CFG_WEIGHT if cfg_weight is None else cfg_weight),
            float(settings.CHATTERBOX_EXAGGERATION if exaggeration is None else exaggeration),
        )

    @classmethod
    def _reference_hash(cls, ref_path: str) -> str:
        """
        Content hash of the reference, computed once per file version instead of on every chunk.

        Args:
            ref_path (str): Path to reference audio.

        Returns:
            str: 16-hex-char SHA-256 prefix.
        """
        st = os.stat(ref_path)
        key = (str(Path(ref_path).resolve()), st.st_mtime_ns, st.st_size)
        cached = cls._ref_hash_memo.get(key)
        if cached is None:
            cached = hashlib.sha256(Path(ref_path).read_bytes()).hexdigest()[:16]
            if len(cls._ref_hash_memo) >= 64:
                cls._ref_hash_memo.clear()
            cls._ref_hash_memo[key] = cached
        return cached

    @classmethod
    def release_gpu_cache(cls) -> None:
        """Return cached activation memory to the driver once a job is finished (model stays resident)."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logging.warning(f"Failed to release GPU cache: {exc}")

    @classmethod
    def _remember_conds(cls, key: str, conds: Any) -> None:
        cls._conds_cache[key] = conds
        cls._conds_cache.move_to_end(key)
        while len(cls._conds_cache) > cls._CONDS_MEMORY_MAX:
            cls._conds_cache.popitem(last=False)

    @classmethod
    def _generate_in_process(
        cls,
        text: str,
        reference_audio_path: str,
        output_path: str,
        speed: float = 1.0,
        language: str = "hi",
        temperature: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        exaggeration: Optional[float] = None,
        target_sample_rate: Optional[int] = None,
    ) -> ChatterboxResult:
        """
        Synthesize speech directly in-process using the resident model and cached conditionals.

        Args:
            text (str): Input text to synthesize.
            reference_audio_path (str): Path to reference audio.
            output_path (str): Output destination path.
            speed (float): Playback rate multiplier.
            language (str): Target language identifier code.
            temperature (float): Sampling randomness parameter.
            cfg_weight (float): Classifier-free guidance scale.
            exaggeration (float): Expressiveness exaggeration parameter.
            target_sample_rate (int): Output audio sample rate in Hz.

        Returns:
            ChatterboxResult: Synthesis outcome metadata.
        """
        import torch
        import torchaudio
        import torchaudio.functional as AF

        target_sample_rate = int(target_sample_rate or settings.OUTPUT_SAMPLE_RATE)
        temperature, cfg_weight, exaggeration = cls.resolve_params(temperature, cfg_weight, exaggeration)
        t0 = time.perf_counter()
        model = cls._get_resident_model()
        device = str(getattr(model, "device", cls.resolve_device()))
        device = "cuda" if device.startswith("cuda") else "cpu"
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        conds = cls._get_or_load_conditionals(model, reference_audio_path, exaggeration, device)
        model.conds = conds

        clean_text = text.strip()
        if not clean_text:
            raise RuntimeError("Empty text provided for synthesis.")

        model_lang = (language or "en").lower().strip()
        if model_lang == "mr":
            model_lang = "hi"

        from core.services.audio.audio_service import AudioService

        native_sr = model.sr
        best_wav, best_key = None, None
        with cls._model_lock, torch.inference_mode():
            model.conds = conds
            for attempt in range(cls._MAX_CHUNK_RETRIES + 1):
                if attempt:
                    torch.manual_seed(int.from_bytes(os.urandom(4), "little"))
                # Full precision on purpose: fp16/bf16 autocast measured 2-2.4x slower and +1 GB VRAM on an RTX 3050
                wav = model.generate(
                    text=clean_text,
                    language_id=model_lang,
                    audio_prompt_path=None,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )
                check = AudioService.assess_chunk(wav.squeeze(0).numpy(), native_sr, clean_text)
                key = (not check["ok"], abs(math.log(max(check["rate"], 1e-3) / 14.0)))
                if best_key is None or key < best_key:
                    best_wav, best_key = wav, key
                if check["ok"]:
                    break
                logging.warning(
                    f"Chunk check failed (attempt {attempt + 1}): {check['reason']}"
                    + ("; regenerating with a new seed" if attempt < cls._MAX_CHUNK_RETRIES else "; keeping best take")
                )
        wav = best_wav

        gen_time = time.perf_counter() - t0
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0

        duration = float(wav.shape[-1]) / float(native_sr)

        if native_sr != target_sample_rate:
            wav = AF.resample(wav, orig_freq=native_sr, new_freq=target_sample_rate)

        max_amp = torch.max(torch.abs(wav)).item()
        if max_amp > 0.95:
            wav = (wav / max_amp) * 0.95

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torchaudio.save(output_path, wav.cpu(), target_sample_rate)
        del wav, best_wav
        # No per-chunk empty_cache(): reusing the allocator's blocks is faster; the job
        # worker releases the cache once per job and the idle janitor unloads the model.
        logging.info(
            f"Chunk synthesized: {duration:.1f}s audio in {gen_time:.1f}s (RTF {gen_time / max(duration, 1e-3):.2f}, "
            f"peak VRAM {peak_vram:.0f} MB)"
        )

        return ChatterboxResult(
            output_path=str(output_path),
            duration_seconds=duration,
            sample_rate=target_sample_rate,
            peak_vram_mb=peak_vram,
            gen_time_s=gen_time,
        )

    def _generate_via_subprocess(
        self,
        text: str,
        reference_audio_path: str,
        output_path: str,
        speed: float = 1.0,
        language: str = "hi",
        temperature: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        exaggeration: Optional[float] = None,
        target_sample_rate: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
    ) -> ChatterboxResult:
        """
        Synthesize speech using an isolated subprocess worker.

        Args:
            text (str): Input text to synthesize.
            reference_audio_path (str): Path to conditioning voice reference audio.
            output_path (str): Destination WAV file path.
            speed (float): Playback speed modifier.
            language (str): Target language identifier.
            temperature (float): Sampling randomness parameter.
            cfg_weight (float): Classifier-free guidance scale.
            exaggeration (float): Expressiveness exaggeration parameter.
            target_sample_rate (int): Output audio sample rate in Hz.
            timeout_seconds (Optional[int]): Execution timeout in seconds.

        Returns:
            ChatterboxResult: Synthesis outcome metadata.
        """
        timeout = timeout_seconds or settings.CHUNK_TIMEOUT_SECONDS
        temperature, cfg_weight, exaggeration = self.resolve_params(temperature, cfg_weight, exaggeration)

        target_sample_rate = int(target_sample_rate or settings.OUTPUT_SAMPLE_RATE)
        cmd = [
            str(self.python_path),
            str(self.worker_script),
            "--ref",
            str(reference_audio_path),
            f"--text={text}",
            "--out",
            str(output_path),
            "--language",
            language,
            "--temperature",
            str(temperature),
            "--cfg",
            str(cfg_weight),
            "--exaggeration",
            str(exaggeration),
            "--sample-rate",
            str(target_sample_rate),
            "--device",
            self.resolve_device(),
            "--cache-dir",
            str(self.conditionals_cache_dir()),
        ]

        t0 = time.perf_counter()
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            logging.error(f"Chatterbox generation timed out after {timeout}s: {exc}")
            raise RuntimeError(f"Chatterbox worker process timed out after {timeout} seconds.")

        if res.returncode != 0:
            err_msg = res.stderr.strip() or res.stdout.strip() or f"Exited with code {res.returncode}"
            logging.error(f"Error in ChatterboxService._generate_via_subprocess: {err_msg}")
            raise RuntimeError("Chatterbox generation failed (see server log for details).")

        try:
            out_lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
            payload: Dict[str, Any] = {}
            for line in reversed(out_lines):
                if line.startswith("{") and line.endswith("}"):
                    payload = json.loads(line)
                    break

            if not payload or payload.get("status") != "ok":
                raise ValueError("Worker did not emit a valid completion status.")

            dur = float(payload.get("duration_seconds", 0.0))
            sr = int(payload.get("sample_rate", target_sample_rate))
            peak_vram = float(payload.get("peak_vram_mb", 0.0))
            gen_time = float(payload.get("gen_time_s", time.perf_counter() - t0))

            return ChatterboxResult(
                output_path=str(output_path),
                duration_seconds=dur,
                sample_rate=sr,
                peak_vram_mb=peak_vram,
                gen_time_s=gen_time,
            )
        except Exception as parse_err:
            logging.error(f"Failed to parse Chatterbox worker output: {parse_err}")
            raise RuntimeError(f"Failed to parse Chatterbox output: {parse_err}")

    def generate(
        self,
        text: str,
        reference_audio_path: str,
        output_path: str,
        speed: float = 1.0,
        language: str = "hi",
        temperature: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        exaggeration: Optional[float] = None,
        target_sample_rate: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
    ) -> ChatterboxResult:
        """
        Synthesize speech matching the reference voice using Chatterbox Multilingual.

        Tries fast resident in-memory inference first, falling back to subprocess if necessary.

        Args:
            text (str): Input text to synthesize.
            reference_audio_path (str): Path to conditioning voice reference audio.
            output_path (str): Destination WAV file path.
            speed (float): Playback speed modifier (default: 1.0).
            language (str): ISO language identifier code (default: 'hi').
            temperature (Optional[float]): Sampling randomness (default: settings.CHATTERBOX_TEMPERATURE).
            cfg_weight (Optional[float]): Guidance scale (default: settings.CHATTERBOX_CFG_WEIGHT).
            exaggeration (Optional[float]): Expressiveness (default: settings.CHATTERBOX_EXAGGERATION).
            target_sample_rate (int): Output audio sample rate in Hz (default: 44100).
            timeout_seconds (Optional[int]): Execution timeout in seconds.

        Returns:
            ChatterboxResult: Synthesis metadata including output path and duration.

        Raises:
            RuntimeError: If synthesis fails or output cannot be produced.
        """
        logging.info(f"Executing ChatterboxService.generate (text_len={len(text)}, lang={language})")
        target_sample_rate = int(target_sample_rate or settings.OUTPUT_SAMPLE_RATE)

        if not os.path.exists(reference_audio_path):
            logging.error(f"Reference audio not found: {reference_audio_path}")
            raise RuntimeError(f"Reference audio not found: {reference_audio_path}")

        # Try fast in-process resident inference if enabled and available in current process
        if self.use_in_process:
            try:
                import chatterbox  # noqa: F401
                return self._generate_in_process(
                    text=text,
                    reference_audio_path=reference_audio_path,
                    output_path=output_path,
                    speed=speed,
                    language=language,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    exaggeration=exaggeration,
                    target_sample_rate=target_sample_rate,
                )
            except Exception as in_proc_err:
                logging.warning(f"In-process generation failed, falling back to isolated worker: {in_proc_err}")

        return self._generate_via_subprocess(
                text=text,
                reference_audio_path=reference_audio_path,
                output_path=output_path,
                speed=speed,
                language=language,
                temperature=temperature,
                cfg_weight=cfg_weight,
                exaggeration=exaggeration,
                target_sample_rate=target_sample_rate,
                timeout_seconds=timeout_seconds,
            )
