"""
Chatterbox Multilingual service for Hindi zero-shot voice cloning.

Dispatches generation to an isolated worker process and coordinates GPU memory safety.
"""

import json
import os
import subprocess
import sys
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
    Coordinates high-fidelity Hindi speech generation via Chatterbox Multilingual.
    """

    def __init__(self, python_executable: Optional[str] = None) -> None:
        """
        Initialize the Chatterbox service with python worker path resolution.

        Args:
            python_executable (Optional[str]): Path to Python binary with chatterbox installed.
        """
        logging.info("Initializing ChatterboxService")
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
        Clear any lingering GPU cache to guarantee single-model GPU concurrency.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        logging.info("Executing ChatterboxService.unload_all")
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logging.warning(f"Failed to clear torch cache: {exc}")

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
            RuntimeError: If subprocess fails, times out, or output cannot be produced.
        """
        logging.info(f"Executing ChatterboxService.generate (text_len={len(text)}, lang={language})")

        # 1. Enforce single heavy model on GPU at a time
        self.unload_all()

        if not os.path.exists(reference_audio_path):
            logging.error(f"Reference audio not found: {reference_audio_path}")
            raise RuntimeError(f"Reference audio not found: {reference_audio_path}")

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
            logging.error(f"Error in ChatterboxService.generate: {err_msg}")
            raise RuntimeError(f"Chatterbox generation failed: {err_msg}")

        # Parse worker JSON output
        try:
            # Look for JSON payload in stdout lines
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
