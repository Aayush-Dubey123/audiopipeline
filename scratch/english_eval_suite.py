"""
Repeatable English Speech Quality Evaluation Suite for AudioPipeline V1.

Evaluates text normalization, synthesis latency, audio duration, RMS levels,
peak amplitude, and clipping across representative English test cases.
Supports standalone CLI invocation with custom text or curated test cases.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Resolve repository paths robustly regardless of invocation working directory
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = REPO_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Ensure working directory allows backend configuration loading
os.chdir(BACKEND_DIR)

# pyrefly: ignore[import, missing-import]
import numpy as np
# pyrefly: ignore[import, missing-import]
import soundfile as sf

# pyrefly: ignore[import, missing-import]
from core.services.llm.llm_service import LLMService
# pyrefly: ignore[import, missing-import]
from core.services.pipelines.chatterbox.chatterbox_service import ChatterboxService

# Curated evaluation test cases covering distinct phonological challenges
EVAL_CASES: List[Dict[str, str]] = [
    {
        "id": "conversational",
        "name": "Conversational Dialogue",
        "raw_text": "Hello there! I'm really excited to help you test AudioPipeline today. Let's make something amazing.",
    },
    {
        "id": "numbers_currencies_dates",
        "name": "Numbers, Currencies & Dates",
        "raw_text": "On Jan. 15th, Dr. Smith reported that the new GPU server cost $45.50 per hour, a 20% discount.",
    },
    {
        "id": "abbreviations_acronyms",
        "name": "Abbreviations & Acronyms",
        "raw_text": "The AI model uses our REST API with PyTorch on CUDA for high-speed TTS generation.",
    },
    {
        "id": "complex_punctuation",
        "name": "Complex Punctuation & Pauses",
        "raw_text": "Wait... did you hear that? Yes, it worked; however, we should check the preview first!",
    },
]


def resolve_reference_audio(custom_ref: Optional[str] = None) -> Path:
    """
    Locate valid reference audio file from custom argument or default locations.

    Args:
        custom_ref (Optional[str]): User-specified path to reference audio.

    Returns:
        Path: Absolute path to verified reference audio file.

    Raises:
        FileNotFoundError: If no valid reference audio file is found.
    """
    candidate_paths: List[Path] = []
    if custom_ref:
        candidate_paths.append(Path(custom_ref).resolve())

    candidate_paths.extend([
        BACKEND_DIR / "voices" / "reference" / "client_ref.wav",
        BACKEND_DIR / "voices" / "client_ref.wav",
        REPO_ROOT / "voices" / "reference" / "client_ref.wav",
    ])

    for path in candidate_paths:
        if path.exists() and path.is_file() and path.stat().st_size > 44:
            return path.resolve()

    raise FileNotFoundError(
        f"Reference audio not found. Checked: {[str(p) for p in candidate_paths]}"
    )


def evaluate_text(
    service: ChatterboxService,
    ref_audio: Path,
    raw_text: str,
    out_wav: Path,
    temperature: float = 0.70,
    cfg_weight: float = 0.60,
    exaggeration: float = 0.35,
) -> Dict[str, Any]:
    """
    Synthesize and extract objective audio quality metrics for a text snippet.

    Args:
        service (ChatterboxService): Active synthesis service instance.
        ref_audio (Path): Path to reference conditioning audio.
        raw_text (str): Input un-normalized text.
        out_wav (Path): Output WAV destination path.
        temperature (float): Sampling randomness.
        cfg_weight (float): Classifier-free guidance weight.
        exaggeration (float): Prosodic expressiveness multiplier.

    Returns:
        Dict[str, Any]: Synthesis latency and acoustic metrics.
    """
    # 1. Deterministic text normalization
    normalized_text = LLMService.rule_based_cleanup(raw_text, language="en")

    # 2. Synthesis execution
    t0 = time.perf_counter()
    result = service.generate(
        text=normalized_text,
        reference_audio_path=str(ref_audio),
        output_path=str(out_wav),
        language="en",
        temperature=temperature,
        cfg_weight=cfg_weight,
        exaggeration=exaggeration,
    )
    wall_latency = time.perf_counter() - t0

    # 3. Audio inspection & metrics extraction
    data, sr = sf.read(str(out_wav))
    duration_s = len(data) / sr
    peak_amp = float(np.max(np.abs(data)))
    rms_level = float(np.sqrt(np.mean(data**2)))
    clipped = bool(np.any(np.abs(data) >= 0.999))
    silence_pct = float(np.mean(np.abs(data) < 1e-4) * 100)

    return {
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "latency_s": round(wall_latency, 3),
        "engine_gen_time_s": round(result.gen_time_s, 3),
        "audio_duration_s": round(duration_s, 2),
        "sample_rate": sr,
        "peak_amplitude": round(peak_amp, 3),
        "rms_level": round(rms_level, 3),
        "silence_percent": round(silence_pct, 1),
        "clipped": clipped,
        "output_path": str(out_wav),
    }


def run_suite(
    case_filter: str = "all",
    custom_text: Optional[str] = None,
    custom_ref: Optional[str] = None,
    temperature: float = 0.70,
    cfg_weight: float = 0.60,
    exaggeration: float = 0.35,
    out_dir_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Execute English evaluation test suite with objective audio metrics.

    Args:
        case_filter (str): Filter for specific test case or 'all'.
        custom_text (Optional[str]): User-provided text string to evaluate.
        custom_ref (Optional[str]): Custom reference audio path.
        temperature (float): Sampling randomness.
        cfg_weight (float): Guidance scale.
        exaggeration (float): Prosody multiplier.
        out_dir_path (Optional[str]): Target output directory.

    Returns:
        List[Dict[str, Any]]: Results collection.
    """
    ref_audio = resolve_reference_audio(custom_ref)
    out_dir = Path(out_dir_path).resolve() if out_dir_path else (SCRIPT_DIR / "eval_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    service = ChatterboxService(use_in_process=True)
    results: List[Dict[str, Any]] = []

    # Assemble test cases to run
    test_cases: List[Dict[str, str]] = []
    if custom_text:
        test_cases.append({
            "id": "custom",
            "name": "Custom User Input",
            "raw_text": custom_text.strip(),
        })
    else:
        for case in EVAL_CASES:
            if case_filter == "all" or case["id"] == case_filter:
                test_cases.append(case)

    if not test_cases:
        print(f"No test cases matched filter: '{case_filter}'")
        return []

    print("=" * 80)
    print("AUDIO PIPELINE V1 — ENGLISH SPEECH QUALITY EVALUATION SUITE")
    print(f"Reference Audio: {ref_audio.name}")
    print(f"Output Directory: {out_dir}")
    print(f"Prosody Tuning:  temp={temperature}, cfg={cfg_weight}, exag={exaggeration}")
    print("=" * 80)

    for item in test_cases:
        case_id = item["id"]
        out_wav = out_dir / f"{case_id}.wav"
        print(f"\n[Case: {item['name']}]")
        print(f"  Raw Text:   {item['raw_text']}")

        try:
            metrics = evaluate_text(
                service=service,
                ref_audio=ref_audio,
                raw_text=item["raw_text"],
                out_wav=out_wav,
                temperature=temperature,
                cfg_weight=cfg_weight,
                exaggeration=exaggeration,
            )
            metrics["id"] = case_id
            metrics["name"] = item["name"]
            results.append(metrics)

            print(f"  Normalized: {metrics['normalized_text']}")
            print(
                f"  Latency:    {metrics['latency_s']:.2f}s | "
                f"Duration: {metrics['audio_duration_s']:.2f}s | "
                f"SR: {metrics['sample_rate']} Hz"
            )
            print(
                f"  Peak Amp:   {metrics['peak_amplitude']:.3f} | "
                f"RMS: {metrics['rms_level']:.3f} | "
                f"Silence: {metrics['silence_percent']:.1f}% | "
                f"Clipped: {metrics['clipped']}"
            )
        except Exception as exc:
            print(f"  ERROR: Synthesis failed for '{case_id}': {exc}")
            results.append({
                "id": case_id,
                "name": item["name"],
                "error": str(exc),
            })

    summary_file = out_dir / "eval_results.json"
    summary_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"Evaluation complete: {len(results)} case(s) evaluated.")
    print(f"Summary metrics saved to: {summary_file}")
    print("=" * 80)

    return results


def main() -> None:
    """CLI entrypoint for standalone test execution."""
    parser = argparse.ArgumentParser(
        description="Run repeatable English audio quality evaluation suite for AudioPipeline V1."
    )
    parser.add_argument(
        "--case",
        choices=["all", "conversational", "numbers_currencies_dates", "abbreviations_acronyms", "complex_punctuation"],
        default="all",
        help="Specify evaluation case to run (default: all).",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Custom text string to synthesize and evaluate.",
    )
    parser.add_argument(
        "--ref",
        type=str,
        default=None,
        help="Path to custom reference audio file.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.70,
        help="Sampling temperature (default: 0.70).",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=0.60,
        help="Classifier-free guidance scale (default: 0.60).",
    )
    parser.add_argument(
        "--exag",
        type=float,
        default=0.35,
        help="Exaggeration multiplier (default: 0.35).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Custom output directory for generated audio and JSON summary.",
    )

    args = parser.parse_args()
    run_suite(
        case_filter=args.case,
        custom_text=args.text,
        custom_ref=args.ref,
        temperature=args.temp,
        cfg_weight=args.cfg,
        exaggeration=args.exag,
        out_dir_path=args.out_dir,
    )


if __name__ == "__main__":
    main()
