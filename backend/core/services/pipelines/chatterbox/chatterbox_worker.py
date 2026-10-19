"""
Standalone worker process for Chatterbox Multilingual zero-shot voice cloning.

Executed as an isolated worker process using the dedicated chatterbox virtual environment.
"""

import argparse
import json
import os
import sys
import time
import torch
import torchaudio
import torchaudio.functional as AF


def synthesize(
    ref_path: str,
    text: str,
    out_path: str,
    language: str = "hi",
    temperature: float = 0.6,
    cfg_weight: float = 0.7,
    exaggeration: float = 0.3,
    target_sample_rate: int = 44100,
) -> dict:
    """
    Run Chatterbox Multilingual synthesis on GPU and export high-fidelity audio.

    Args:
        ref_path (str): Path to conditioning reference audio file.
        text (str): Input text to synthesize.
        out_path (str): Destination path for the generated WAV file.
        language (str): Target language identifier code (default: 'hi').
        temperature (float): Sampling temperature (default: 0.6).
        cfg_weight (float): Classifier-free guidance weight (default: 0.7).
        exaggeration (float): Expressiveness exaggeration parameter (default: 0.3).
        target_sample_rate (int): Output audio sample rate in Hz (default: 44100).

    Returns:
        dict: Execution metadata including duration, sample rate, peak VRAM, and timing.

    Raises:
        RuntimeError: If synthesis fails or output cannot be written.
    """
    from chatterbox import ChatterboxMultilingualTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    # Prepare speaker conditioning from reference audio
    model.prepare_conditionals(ref_path, exaggeration=exaggeration)

    # Clean text
    clean_text = text.strip()
    if not clean_text:
        raise RuntimeError("Empty text provided for synthesis.")

    # Map language to supported model language code
    model_lang = (language or "en").lower().strip()
    if model_lang == "mr":
        # Chatterbox Multilingual model supports Devanagari script (shared between Hindi and Marathi)
        # under the 'hi' acoustic grapheme vocabulary.
        model_lang = "hi"

    # Generate audio waveform
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

    native_sr = model.sr  # typically 24000 Hz
    duration = float(wav.shape[-1]) / float(native_sr)

    # Resample to target sample rate (44.1 kHz) if needed
    if native_sr != target_sample_rate:
        wav = AF.resample(wav, orig_freq=native_sr, new_freq=target_sample_rate)

    # Peak normalization to prevent digital clipping
    max_amp = torch.max(torch.abs(wav)).item()
    if max_amp > 0.95:
        wav = (wav / max_amp) * 0.95

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torchaudio.save(out_path, wav, target_sample_rate)

    # Clean GPU memory
    del model
    del wav
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "status": "ok",
        "duration_seconds": duration,
        "sample_rate": target_sample_rate,
        "peak_vram_mb": peak_vram,
        "gen_time_s": gen_time,
        "output_path": out_path,
    }


def main() -> None:
    """
    Main CLI entrypoint parsing arguments and emitting JSON results.

    Args:
        None

    Returns:
        None

    Raises:
        SystemExit: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(description="Chatterbox Multilingual worker process")
    parser.add_argument("--ref", required=True, help="Path to reference audio")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--out", required=True, help="Path to output WAV file")
    parser.add_argument("--language", default="hi", help="Target language code (default: hi)")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--cfg", type=float, default=0.7, help="CFG weight")
    parser.add_argument("--exaggeration", type=float, default=0.3, help="Exaggeration strength")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Target sample rate (Hz)")

    args = parser.parse_args()

    try:
        res = synthesize(
            ref_path=args.ref,
            text=args.text,
            out_path=args.out,
            language=args.language,
            temperature=args.temperature,
            cfg_weight=args.cfg,
            exaggeration=args.exaggeration,
            target_sample_rate=args.sample_rate,
        )
        print(json.dumps(res))
        sys.exit(0)
    except Exception as exc:
        err = {"status": "error", "error": str(exc)}
        print(json.dumps(err), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
