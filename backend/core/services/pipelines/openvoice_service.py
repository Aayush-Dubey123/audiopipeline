"""
OpenVoice V2 and MeloTTS tone color conversion inference pipeline.

Provides structured text-to-speech synthesis and tone color transfer using
MeloTTS English base speaker models and OpenVoice V2 tone conversion neural networks.
Also exposes a sentence-level text splitter used by the long-form chunk pipeline.
"""

import argparse
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
    language: str = "en"


# ---------------------------------------------------------------------------
# Text splitter — used by both the CLI harness and the long-form chunk pipeline
# ---------------------------------------------------------------------------

# Abbreviations that must NOT be treated as sentence-ending periods.
_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc",
    "ie", "eg", "approx", "dept", "est", "govt", "inc", "ltd", "no",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "st", "ave", "blvd",
}

# Sentence-boundary pattern:
#   [.!?।॥]     — ASCII terminators plus Devanagari danda / double-danda
#   (\s|$)       — followed by whitespace or end-of-string
_SENT_END = re.compile(r"[.!?।॥]+(?=\s|$)")

# Decimal / ordinal protection: digits around a period (e.g. 1.5, 3rd, Rs.500)
_DECIMAL_GUARD = re.compile(r"(\d+)\.(?=\d)")
# Abbreviation guard: word followed by period then whitespace+capital or end
_ABBREV_GUARD = re.compile(r"\b([A-Za-z]{1,6})\.(?=\s)")


def _is_abbreviation(word: str) -> bool:
    """
    Return True if the word is a known abbreviation that should not end a sentence.

    Args:
        word (str): Candidate word (stripped of trailing period).

    Returns:
        bool: True if the word is in the protected abbreviation set.
    """
    return word.lower().rstrip(".") in _ABBREVIATIONS


def split_text_with_boundaries(text: str, max_chars: int = 1000) -> Tuple[List[str], set]:
    """
    Split a long text into chunks of at most max_chars characters and record paragraph starts.

    Splitting rules (in priority order):
      1. Split at sentence boundaries (.  !  ?  ।  ॥) followed by whitespace,
         but NOT after known abbreviations (Dr., Mr., etc.) or decimal numbers.
      2. If a sentence segment still exceeds max_chars, fall back to the last
         comma or semicolon before the limit.
      3. If no comma/semicolon is found, fall back to the last whitespace.
      4. Hard-cut at max_chars only if no whitespace exists (should not happen
         for normal prose).

    Each chunk is stripped of leading/trailing whitespace. Empty chunks are
    discarded. Paragraphs (double-newline) are always split points, and the index
    of chunks starting a new paragraph is returned in a set.

    Args:
        text (str): Input script text (any language, Devanagari or Latin).
        max_chars (int): Maximum characters per chunk (default 1000).

    Returns:
        Tuple[List[str], set]: Ordered list of text chunks and set of paragraph-start indices.
    """
    if not text or not text.strip():
        return [], set()

    chunks: List[str] = []
    paragraph_start_indices = set()
    paragraphs = re.split(r"\n{2,}", text)

    def flush(buf: str) -> List[str]:
        """Break a buffer that still exceeds max_chars at best sub-word boundary."""
        results: List[str] = []
        while len(buf) > max_chars:
            cut = max_chars
            for pat in (r"[,;]\s", r"\s"):
                m = None
                for match in re.finditer(pat, buf[:max_chars]):
                    m = match
                if m:
                    cut = m.end()
                    break
            results.append(buf[:cut].strip())
            buf = buf[cut:].lstrip()
        if buf.strip():
            results.append(buf.strip())
        return results

    for para_idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue

        start_chunk_idx = len(chunks)
        if para_idx > 0 and start_chunk_idx > 0:
            paragraph_start_indices.add(start_chunk_idx)

        positions: List[int] = []
        for m in _SENT_END.finditer(para):
            end = m.end()
            before = para[:m.start()].rstrip()
            last_word = re.split(r"\s", before)[-1] if before else ""
            after_char = para[end:end + 1] if end < len(para) else ""
            if after_char.isdigit():
                continue
            if _is_abbreviation(last_word):
                continue
            positions.append(end)

        segments: List[str] = []
        prev = 0
        for pos in positions:
            seg = para[prev:pos].strip()
            if seg:
                segments.append(seg)
            prev = pos
        tail = para[prev:].strip()
        if tail:
            segments.append(tail)

        if not segments:
            segments = [para]

        current = ""
        for seg in segments:
            if len(seg) > max_chars:
                if current.strip():
                    chunks.extend(flush(current.strip()))
                    current = ""
                chunks.extend(flush(seg))
                continue

            if current and len(current) + 1 + len(seg) > max_chars:
                chunks.extend(flush(current.strip()))
                current = seg
            else:
                current = (current + " " + seg).lstrip() if current else seg

        if current.strip():
            chunks.extend(flush(current.strip()))

    final_chunks = [c for c in chunks if c.strip()]
    return final_chunks, paragraph_start_indices


def split_text_into_chunks(text: str, max_chars: int = 1000) -> List[str]:
    """
    Split a long text into chunks of at most max_chars characters.

    Args:
        text (str): Input script text (any language, Devanagari or Latin).
        max_chars (int): Maximum characters per chunk (default 1000).

    Returns:
        List[str]: Ordered list of non-empty text chunks.
    """
    chunks, _ = split_text_with_boundaries(text, max_chars)
    return chunks


# ---------------------------------------------------------------------------
# Hindi / Marathi input preparation
# ---------------------------------------------------------------------------


def _has_devanagari(text: str) -> bool:
    """
    Return True if the text contains at least one Devanagari Unicode character.

    Args:
        text (str): Input text to check.

    Returns:
        bool: True when Devanagari characters are present.
    """
    return bool(re.search(r"[\u0900-\u097F]", text))


def prepare_hindi_text(text: str) -> str:
    """
    Prepare Hindi/Marathi text for MMS-TTS input.

    If the text is already in Devanagari script it is returned as-is.
    Romanized / Hinglish text is NOT accepted for the Hindi pipeline — the
    MMS-TTS VITS phoneme model maps Devanagari characters to phonemes; any
    Latin characters that survive tokenisation are silently skipped, which
    causes the 83-char/s rate seen in benchmarks.  Callers must supply
    native Devanagari for the 'hi'/'mr' language path.

    Args:
        text (str): Raw input text.

    Returns:
        str: Validated and stripped text ready for MMS-TTS tokenisation.

    Raises:
        ValueError: If the text contains no Devanagari characters.
    """
    cleaned = text.strip()
    if not _has_devanagari(cleaned):
        raise ValueError(
            "Hindi/Marathi TTS requires native Devanagari script input. "
            "Romanized (Hinglish) text will be silently dropped by the MMS-TTS phoneme model. "
            "Please convert your text to Devanagari before submitting."
        )
    return cleaned


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
        self._mms_models: Dict[str, Tuple[Any, Any]] = {}
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
        self._mms_models.clear()
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

    def _load_mms_tts(self, language: str) -> Tuple[Any, Any]:
        """
        Lazily initialize and load the Meta MMS-TTS model and tokenizer for Hindi or Marathi.

        Args:
            language (str): Target language ISO code ('hi' or 'mr').

        Returns:
            Tuple[Any, Any]: Instantiated (VitsTokenizer, VitsModel).

        Raises:
            OpenVoiceInferenceError: If model loading fails.
        """
        lang = language.lower().strip()
        if lang in self._mms_models:
            return self._mms_models[lang]

        if "cuda" in self.device:
            try:
                from core.services.pipelines.rvc_service import RVCService
                RVCService.unload_all()
            except Exception:
                pass

        model_name = "facebook/mms-tts-hin" if lang == "hi" else "facebook/mms-tts-mar"
        try:
            from transformers import VitsModel, VitsTokenizer

            log.info("Loading MMS-TTS model '%s' on %s...", model_name, self.device)
            tokenizer = VitsTokenizer.from_pretrained(model_name)
            model = VitsModel.from_pretrained(model_name).to(self.device)
            model.eval()
            self._mms_models[lang] = (tokenizer, model)
            log.info("MMS-TTS model '%s' successfully loaded.", model_name)
            return self._mms_models[lang]
        except Exception as exc:
            log.error("Failed to load MMS-TTS model '%s': %s", model_name, exc)
            raise OpenVoiceInferenceError(f"MMS-TTS ({language}) initialization failed: {exc}") from exc

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
        force_recompute: bool = False,
    ) -> torch.Tensor:
        """
        Extract or retrieve cached speaker embedding (SE) from reference audio.

        Args:
            reference_wav_path (str): Filepath to the clean 24kHz reference WAV.
            cache_on_disk (bool): Whether to persist extracted tensor to disk.
            force_recompute (bool): If True, ignore caches and recompute embedding. Defaults to False.

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

        disk_cache_path = ref_path.with_suffix(".se.pth")

        if not force_recompute:
            if cache_key in self._target_se_cache:
                cached_mtime, tensor = self._target_se_cache[cache_key]
                if cached_mtime == current_mtime:
                    log.debug("Using in-memory target speaker embedding for: %s", ref_path.name)
                    return tensor

            if cache_on_disk and disk_cache_path.exists():
                if disk_cache_path.stat().st_mtime >= current_mtime:
                    log.info("Loading cached target speaker embedding from: %s", disk_cache_path.name)
                    tensor = torch.load(str(disk_cache_path), map_location=self.device, weights_only=True)
                    self._target_se_cache[cache_key] = (current_mtime, tensor)
                    return tensor

        converter = self._load_converter()
        try:
            log.info("Extracting target speaker embedding from: %s (force_recompute=%s)", ref_path.name, force_recompute)
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
        language: str = "en",
        force_recompute_se: bool = False,
    ) -> GenerationResult:
        """
        Generate synthesized speech transferred to the tone of reference audio.

        Supports multilingual engines: English (MeloTTS), Hindi (MMS-TTS), and Marathi (MMS-TTS).

        Args:
            text (str): Script text to synthesize in English, Hindi, or Marathi.
            reference_audio_path (str): Filepath to reference audio WAV clip.
            output_path (str): Destination path for final converted WAV.
            speaker_id (str): Base MeloTTS speaker profile name ('EN-Default', 'EN-US', etc.).
            speed (float): Playback rate multiplier. Defaults to 1.0.
            tau (float): OpenVoice conversion temperature. Defaults to 0.3.
            language (str): Target language ISO code ('en', 'hi', or 'mr'). Defaults to 'en'.
            force_recompute_se (bool): If True, recompute target speaker embedding from audio. Defaults to False.

        Returns:
            GenerationResult: Comprehensive metrics and output file metadata.

        Raises:
            ValueError: If text is empty or invalid.
            ReferenceAudioNotFoundError: If reference audio file does not exist.
            OpenVoiceInferenceError: If synthesis, language engine, or conversion fails.
        """
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Input text cannot be empty or whitespace.")

        lang = language.lower().strip()
        if lang not in ("en", "hi", "mr"):
            raise OpenVoiceInferenceError(
                f"Unsupported language '{language}'. Supported languages: English ('en'), Hindi ('hi'), Marathi ('mr')."
            )

        # Auto-route chunks with Devanagari script to Hindi MMS-TTS engine
        if lang == "en" and any("\u0900" <= c <= "\u097F" for c in clean_text):
            lang = "hi"

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
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_src_file = tmp.name

            tts_start = time.time()

            if lang in ("hi", "mr"):
                # 1. Validate and prepare Devanagari text
                try:
                    effective_text = prepare_hindi_text(clean_text)
                except ValueError as ve:
                    raise OpenVoiceInferenceError(str(ve)) from ve

                log.info(
                    "Hindi/Marathi input: raw_chars=%d, devanagari_chars=%d",
                    len(clean_text),
                    len(effective_text),
                )

                # 2. Base TTS Synthesis with Meta MMS-TTS (Hindi / Marathi)
                mms_tok, mms_model = self._load_mms_tts(lang)
                inputs = mms_tok(effective_text, return_tensors="pt")
                token_count = inputs["input_ids"].shape[-1]
                log.info(
                    "MMS-TTS tokenised input: %d tokens for %d chars (%.1f chars/token)",
                    token_count,
                    len(effective_text),
                    len(effective_text) / max(token_count, 1),
                )
                if token_count == 0:
                    raise OpenVoiceInferenceError(
                        f"MMS-TTS tokeniser produced 0 tokens for {lang.upper()} input. "
                        "Ensure the text contains valid Devanagari script."
                    )
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    mms_out = mms_model(**inputs)
                raw_wav = mms_out.waveform.squeeze().cpu().numpy()
                audio_duration_produced = len(raw_wav) / 16000
                chars_per_second = len(effective_text) / max(audio_duration_produced, 0.001)
                log.info(
                    "MMS-TTS produced %.2fs of audio for %d chars (%.1f chars/s) — expected 10-15 chars/s",
                    audio_duration_produced,
                    len(effective_text),
                    chars_per_second,
                )
                sf.write(temp_src_file, raw_wav, 16000)
                base_tts_duration = time.time() - tts_start
                log.info("MMS-TTS base audio (%s) generated in %.2fs", lang, base_tts_duration)

                converter = self._load_converter()
                source_se = converter.extract_se(temp_src_file)
            else:
                # 1. Base TTS Synthesis with MeloTTS (English)
                tts_model = self._load_tts()
                spk_map = getattr(tts_model.hps.data, "spk2id", {})
                try:
                    numeric_spk_id = spk_map[speaker_id]
                except (KeyError, TypeError):
                    numeric_spk_id = getattr(spk_map, speaker_id, 0)

                tts_model.tts_to_file(
                    clean_text,
                    numeric_spk_id,
                    temp_src_file,
                    speed=speed,
                    quiet=True,
                )
                base_tts_duration = time.time() - tts_start
                log.info("MeloTTS base audio generated in %.2fs", base_tts_duration)

                converter = self._load_converter()
                source_se = self.get_source_speaker_embedding("en-default.pth")

            # 2. Extract / Retrieve Speaker Embeddings
            target_se = self.extract_target_speaker_embedding(
                str(ref_file),
                force_recompute=force_recompute_se,
            )

            # 3. Tone Color Conversion
            conv_start = time.time()
            conv_target_path = str(out_file)
            is_part_file = conv_target_path.endswith(".part")
            if is_part_file:
                # Soundfile cannot infer audio format from .part; synthesize to temporary .wav then move
                conv_target_path = conv_target_path[:-5] if conv_target_path.endswith(".part") else f"{conv_target_path}.wav"

            converter.convert(
                audio_src_path=temp_src_file,
                src_se=source_se,
                tgt_se=target_se,
                output_path=conv_target_path,
                tau=tau,
                message="@MyShell",
            )
            if is_part_file and os.path.exists(conv_target_path):
                if os.path.exists(str(out_file)):
                    os.remove(str(out_file))
                os.replace(conv_target_path, str(out_file))

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
                language=lang,
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
    CLI execution entry point for standalone verification, benchmarking, and Hindi diagnostics.

    Modes
    -----
    --mode generate   : Single TTS generation pass (default).
    --mode hindi_diag : Hindi diagnostic — tests native Devanagari and validates chars/s.
    --mode splitter   : Splitter self-check against edge-case texts.
    --mode benchmark  : EN-1k, EN-5k, HI-native, EN-1k repeat benchmarks + 30-min estimate.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="AI Voice Studio: OpenVoice V2 + Multilingual TTS CLI Harness"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "hindi_diag", "splitter", "benchmark"],
        help="Execution mode",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="This is an automated verification test for OpenVoice V2 voice cloning.",
        help="Text script to synthesize",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code ('en', 'hi', or 'mr')",
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

    # ------------------------------------------------------------------ #
    # MODE: splitter — no GPU needed
    # ------------------------------------------------------------------ #
    if args.mode == "splitter":
        print("=" * 60)
        print("  Splitter self-check")
        print("=" * 60)

        cases = [
            # (description, text, max_chars, expected_chunks_ge)
            ("Short English",
             "Hello world. This is a test.", 200, 1),
            ("Abbreviation protection",
             "Dr. Smith works at approx. 9 a.m. every day. He is great.", 200, 1),
            ("Decimal protection",
             "The price is Rs. 1.5 per unit. Buy 10.5 kg.", 200, 1),
            ("Devanagari danda split",
             "नमस्ते। आज मौसम अच्छा है। कल बारिश होगी।", 50, 2),
            ("Long unbroken text falls back to whitespace",
             "word " * 300, 100, 5),
            ("Paragraph boundary",
             "Para one sentence.\n\nPara two sentence.", 200, 2),
            ("English then Devanagari",
             "Hello! यह एक परीक्षण है। Great.", 100, 2),
        ]

        all_pass = True
        for desc, text, max_c, min_chunks in cases:
            chunks = split_text_into_chunks(text, max_chars=max_c)
            ok = (
                len(chunks) >= min_chunks
                and all(len(c) <= max_c for c in chunks)
                and all(not c[0].isspace() and not c[-1].isspace() for c in chunks)
            )
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print("  [{status}] {desc}: {n} chunks, max_len={ml}".format(
                status=status, desc=desc, n=len(chunks),
                ml=max(len(c) for c in chunks) if chunks else 0))
            for i, c in enumerate(chunks):
                preview = c[:60].replace("\n", " ")
                print("         chunk[{}]: {} chars | {!r}...".format(i, len(c), preview))

        print("=" * 60)
        print("  Splitter: " + ("ALL PASS" if all_pass else "SOME FAILURES"))
        print("=" * 60)
        sys.exit(0 if all_pass else 1)

    # ------------------------------------------------------------------ #
    # Remaining modes need a service + reference wav
    # ------------------------------------------------------------------ #
    service = OpenVoiceService(device=args.device)

    ref_path = Path(args.reference)
    if not ref_path.is_absolute():
        ref_path = service.backend_dir / ref_path

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = service.backend_dir / out_path

    # ------------------------------------------------------------------ #
    # MODE: hindi_diag
    # ------------------------------------------------------------------ #
    if args.mode == "hindi_diag":
        print("=" * 60)
        print("  Hindi Diagnostic")
        print("=" * 60)

        native_hi = (
            "नमस्ते! आज हम आर्टिफिशियल इंटेलिजेंस के क्षेत्र में एक नई तकनीक के "
            "बारे में बात करेंगे जिसे टेक्स्ट-टू-स्पीच कहा जाता है। "
            "यह तकनीक लिखे हुए शब्दों को प्राकृतिक आवाज़ में बदलती है।"
        )

        romanized_hi = (
            "namaste! aaj hum artificial intelligence ke kshetra mein "
            "ek nayi takneek ke baare mein baat karenge."
        )

        print("  Case 1: Native Devanagari ({} chars)".format(len(native_hi)))
        hi_out = service.backend_dir / "storage" / "benchmark" / "diag_hi_native.wav"
        hi_out.parent.mkdir(parents=True, exist_ok=True)
        try:
            t0 = time.perf_counter()
            r = service.generate(
                text=native_hi,
                reference_audio_path=str(ref_path),
                output_path=str(hi_out),
                language="hi",
            )
            elapsed = time.perf_counter() - t0
            cps = len(native_hi) / max(r.duration_seconds, 0.001)
            print("    Audio: {:.2f}s | Wall: {:.2f}s | chars/s: {:.1f} (target 10-15)".format(
                r.duration_seconds, elapsed, cps))
            print("    PASS" if 8 <= cps <= 20 else "    WARNING: chars/s out of expected range")
        except Exception as exc:
            print("    FAIL: {}".format(exc))

        print("  Case 2: Romanized text (should raise ValueError)")
        try:
            prepare_hindi_text(romanized_hi)
            print("    FAIL: no error raised for romanized text")
        except ValueError as ve:
            print("    PASS: raised ValueError as expected")
            print("    Message: {}".format(str(ve)[:80]))

        print("=" * 60)
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # MODE: benchmark
    # ------------------------------------------------------------------ #
    if args.mode == "benchmark":
        import json
        out_dir = service.backend_dir / "storage" / "benchmark"
        out_dir.mkdir(parents=True, exist_ok=True)

        en_1k = (
            "Welcome to the AI Voice Studio benchmark session. Today we are testing the "
            "long-form text-to-speech pipeline that will power client scripts of up to "
            "thirty minutes in length. The system uses MeloTTS as the base synthesis "
            "engine and the OpenVoice V2 tone color converter to clone the target speaker "
            "from a short reference clip. Our goal for this benchmark is to measure wall "
            "clock time per thousand characters, the real-time factor relative to the "
            "generated audio length, and the peak VRAM footprint on the RTX 3050 GPU. "
            "We will also observe whether speed degrades across repeated calls in the same "
            "warm process. Let us begin the first pass now."
        )
        en_5k = (en_1k + "\n\n") * 5
        hi_native = (
            "नमस्ते! आज हम आर्टिफिशियल इंटेलिजेंस के क्षेत्र में एक नई और रोमांचक "
            "तकनीक के बारे में बात करेंगे जिसे टेक्स्ट-टू-स्पीच कहा जाता है। "
            "यह तकनीक लिखे हुए शब्दों को प्राकृतिक आवाज़ में बदलती है। "
            "हमारा उद्देश्य यह जानना है कि एक छोटी संदर्भ क्लिप से आवाज़ की नकल "
            "कितनी अच्छी तरह की जा सकती है।"
        )

        runs = [
            ("EN-1k-cold", en_1k, "en", "bench2_en1k_cold.wav"),
            ("EN-5k-warm", en_5k, "en", "bench2_en5k.wav"),
            ("HI-native", hi_native, "hi", "bench2_hi_native.wav"),
            ("EN-1k-repeat1", en_1k, "en", "bench2_en1k_r1.wav"),
            ("EN-1k-repeat2", en_1k, "en", "bench2_en1k_r2.wav"),
        ]

        results = []
        for label, text, lang, fname in runs:
            print("Running {} ({} chars)...".format(label, len(text)))
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            try:
                r = service.generate(
                    text=text,
                    reference_audio_path=str(ref_path),
                    output_path=str(out_dir / fname),
                    language=lang,
                )
            except Exception as exc:
                print("  FAILED: {}".format(exc))
                continue
            elapsed = time.perf_counter() - t0
            alloc = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            res_mb = torch.cuda.max_memory_reserved() / 1024**2 if torch.cuda.is_available() else 0
            cps = len(text) / max(r.duration_seconds, 0.001)
            row = {
                "label": label, "chars": len(text),
                "audio_s": round(r.duration_seconds, 2),
                "wall_s": round(elapsed, 2),
                "s_per_1k": round(elapsed / len(text) * 1000, 2),
                "rtf": round(elapsed / r.duration_seconds, 3),
                "vram_alloc_mb": round(alloc, 1),
                "vram_res_mb": round(res_mb, 1),
                "chars_per_s": round(cps, 1),
            }
            results.append(row)
            print("  audio={audio_s}s wall={wall_s}s s/1k={s_per_1k} RTF={rtf} VRAM={vram_alloc_mb}MB chars/s={chars_per_s}".format(**row))

        target_chars = int(30 * 150 * 5.5)
        warm_rate = next((r["s_per_1k"] for r in results if "repeat" in r["label"]), None)
        if warm_rate:
            est_s = target_chars / 1000 * warm_rate
            print("30-min estimate: ~{} chars -> {:.0f}s ({:.1f} min) at {:.2f}s/1k".format(
                target_chars, est_s, est_s / 60, warm_rate))

        rp = out_dir / "benchmark_results_v2.json"
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("Results saved to {}".format(rp))
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # MODE: generate (default)
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("AI Voice Studio - OpenVoice V2 Multilingual Inference Test")
    print("=" * 60)
    print("Device:           {}".format(service.device))
    print("Language:         {}".format(args.language))
    print("Input text:       {}".format(args.text))
    print("Reference audio:  {}".format(ref_path))
    print("Output path:      {}".format(out_path))
    print("Speech speed:     {}".format(args.speed))
    print("Tau:              {}".format(args.tau))
    print("-" * 60)

    try:
        result = service.generate(
            text=args.text,
            reference_audio_path=str(ref_path),
            output_path=str(out_path),
            speed=args.speed,
            tau=args.tau,
            language=args.language,
        )
        print("RESULT: SUCCESS")
        print("Generated Audio Duration: {:.2f}s".format(result.duration_seconds))
        print("Audio Sample Rate:        {} Hz".format(result.sample_rate))
        print("Output File Size:         {:,} bytes".format(result.file_size_bytes))
        print("Base TTS Time:            {:.2f}s".format(result.base_tts_duration_seconds))
        print("Tone Conversion Time:     {:.2f}s".format(result.tone_conversion_duration_seconds))
        print("Total Pipeline Time:      {:.2f}s".format(result.total_duration_seconds))
        print("Peak VRAM Allocated:      {:.2f} MB".format(result.peak_vram_allocated_mb))
        print("Peak VRAM Reserved:       {:.2f} MB".format(result.peak_vram_reserved_mb))
        print("Watermark Applied:        {}".format(result.watermark_applied))
        print("=" * 60)
    except Exception as exc:
        print("RESULT: FAILED - {}".format(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
