"""
Chatterbox Multilingual service for Hindi zero-shot voice cloning.

Coordinates high-fidelity multilingual synthesis with resident in-memory model execution,
persistent reference conditionals caching, and isolated worker process fallback.
"""

import gc
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
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
    _conds_cache: Dict[str, Any] = {}

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
            repo_venv = Path(settings.backend_dir).parent / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            backend_venv = Path(settings.backend_dir) / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            backend_venv_cb = Path(settings.backend_dir) / ".venv_chatterbox" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            if repo_venv.exists():
                self.python_path = str(repo_venv)
            elif backend_venv.exists():
                self.python_path = str(backend_venv)
            elif backend_venv_cb.exists():
                self.python_path = str(backend_venv_cb)
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
                timeout=30,
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

            import torch
            from chatterbox import ChatterboxMultilingualTTS

            device = "cuda" if torch.cuda.is_available() else "cpu"
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
        ref_p = Path(ref_path)
        ref_bytes = ref_p.read_bytes()
        ref_hash = hashlib.sha256(ref_bytes).hexdigest()[:16]
        cache_key = f"{ref_hash}_{int(exaggeration * 100)}"

        # 1. In-memory cache hit
        if cache_key in cls._conds_cache:
            return cls._conds_cache[cache_key]

        # 2. Disk cache hit
        cache_dir = Path(settings.TMP_DIR) / "cache" / "conditionals"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{cache_key}.pt"

        from chatterbox.mtl_tts import Conditionals

        if cache_file.exists():
            try:
                conds = Conditionals.load(cache_file, map_location=device)
                cls._conds_cache[cache_key] = conds
                logging.info(f"Loaded speaker conditioning from disk cache: '{cache_file.name}'")
                return conds
            except Exception as load_err:
                logging.warning(f"Failed to load cached conditionals from {cache_file}: {load_err}")

        # 3. Compute and cache
        logging.info(f"Extracting new speaker conditioning for ref_hash='{ref_hash}'")
        model.prepare_conditionals(str(ref_path), exaggeration=exaggeration)
        conds = model.conds
        cls._conds_cache[cache_key] = conds
        try:
            conds.save(cache_file)
            logging.info(f"Saved speaker conditioning to disk cache: '{cache_file.name}'")
        except Exception as save_err:
            logging.warning(f"Failed to save conditionals cache to {cache_file}: {save_err}")

        return conds

    @classmethod
    def _generate_in_process(
        cls,
        text: str,
        reference_audio_path: str,
        output_path: str,
        speed: float = 1.0,
        language: str = "hi",
        temperature: float = 0.6,
        cfg_weight: float = 0.7,
        exaggeration: float = 0.3,
        target_sample_rate: int = 44100,
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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        model = cls._get_resident_model()
        conds = cls._get_or_load_conditionals(model, reference_audio_path, exaggeration, device)
        model.conds = conds

        clean_text = text.strip()
        if not clean_text:
            raise RuntimeError("Empty text provided for synthesis.")

        model_lang = (language or "en").lower().strip()
        if model_lang == "mr":
            model_lang = "hi"
        elif model_lang == "en":
            # Tuned natural prosody parameters for English
            temperature = 0.7 if temperature == 0.6 else temperature
            cfg_weight = 0.6 if cfg_weight == 0.7 else cfg_weight
            exaggeration = 0.35 if exaggeration == 0.3 else exaggeration

        with cls._model_lock:
            wav = model.generate(
                text=clean_text,
                language_id=model_lang,
                audio_prompt_path=None,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )

        gen_time = time.perf_counter() - t0
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0

        native_sr = model.sr
        duration = float(wav.shape[-1]) / float(native_sr)

        if native_sr != target_sample_rate:
            wav = AF.resample(wav, orig_freq=native_sr, new_freq=target_sample_rate)

        max_amp = torch.max(torch.abs(wav)).item()
        if max_amp > 0.95:
            wav = (wav / max_amp) * 0.95

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torchaudio.save(output_path, wav, target_sample_rate)

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
        temperature: float = 0.6,
        cfg_weight: float = 0.7,
        exaggeration: float = 0.3,
        target_sample_rate: int = 44100,
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

        cmd = [
            str(self.python_path),
            str(self.worker_script),
            "--ref",
            str(reference_audio_path),
            "--text",
            text,
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
            raise RuntimeError(f"Chatterbox generation failed: {err_msg}")

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
        temperature: float = 0.6,
        cfg_weight: float = 0.7,
        exaggeration: float = 0.3,
        target_sample_rate: int = 44100,
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
            temperature (float): Sampling randomness parameter (default: 0.6).
            cfg_weight (float): Classifier-free guidance scale (default: 0.7).
            exaggeration (float): Expressiveness exaggeration parameter (default: 0.3).
            target_sample_rate (int): Output audio sample rate in Hz (default: 44100).
            timeout_seconds (Optional[int]): Execution timeout in seconds.

        Returns:
            ChatterboxResult: Synthesis metadata including output path and duration.

        Raises:
            RuntimeError: If synthesis fails or output cannot be produced.
        """
        logging.info(f"Executing ChatterboxService.generate (text_len={len(text)}, lang={language})")

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
