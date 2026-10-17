"""
Optional text enhancement and normalization service for text-to-speech synthesis.

Provides robust rule-based text cleaning and optional external LLM fallback chains
(Groq, Gemini, OpenRouter) to clean punctuation, numbers, and formatting for audio synthesis.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import httpx

from core import logger
from core.config import get_settings

logging = logger(__name__)
settings = get_settings()

COMMON_ENGLISH_ABBREVIATIONS = {
    r"\bmr\.": "mister",
    r"\bmrs\.": "missus",
    r"\bms\.": "miss",
    r"\bdr\.": "doctor",
    r"\bprof\.": "professor",
    r"\bst\.": "saint",
    r"\betc\.": "etcetera",
    r"\be\.g\.": "for example",
    r"\bi\.e\.": "that is",
    r"\bvs\.": "versus",
    r"\bapprox\.": "approximately",
    r"\bmin\.": "minutes",
    r"\bsec\.": "seconds",
    r"\bdept\.": "department",
    r"\bno\.": "number",
}

CURRENCY_MAP = {
    r"\$([0-9]+(?:\.[0-9]+)?)": r"\1 dollars",
    r"₹([0-9]+(?:\.[0-9]+)?)": r"\1 rupees",
    r"€([0-9]+(?:\.[0-9]+)?)": r"\1 euros",
    r"£([0-9]+(?:\.[0-9]+)?)": r"\1 pounds",
}


@dataclass(frozen=True)
class CleanupResult:
    """
    Result container for cleaned script text.
    """
    original: str
    cleaned: str
    provider_used: str
    service_available: bool


class LLMService:
    """
    Text preprocessing service with graceful offline fallback and provider chains.
    """

    @staticmethod
    def rule_based_cleanup(text: str, language: str = "en") -> str:
        """
        Deterministically normalize punctuation, abbreviations, and symbols for TTS.

        Args:
            text (str): Raw input script text.
            language (str): Target language ISO code ('en', 'hi', or 'mr').

        Returns:
            str: Cleaned and normalized script string.
        """
        logging.info("Executing LLMService.rule_based_cleanup")
        if not text:
            return ""

        cleaned = text.strip()

        # 1. Normalize smart quotes and em-dashes
        cleaned = cleaned.replace("“", '"').replace("”", '"')
        cleaned = cleaned.replace("‘", "'").replace("’", "'")
        cleaned = cleaned.replace("—", ", ").replace("–", ", ")

        # 2. Strip Markdown formatting artifacts (links, bold, headers, code)
        cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)  # [text](url) -> text
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)          # **bold** -> bold
        cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)              # *italic* -> italic
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)                # `code` -> code
        cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)   # # Header -> Header

        # 3. Collapse repeated punctuation
        cleaned = re.sub(r"\?+", "?", cleaned)
        cleaned = re.sub(r"!+", "!", cleaned)
        cleaned = re.sub(r"\.{4,}", "...", cleaned)

        # 4. English-specific abbreviation and symbol expansions
        if language.lower() == "en":
            for pattern, replacement in COMMON_ENGLISH_ABBREVIATIONS.items():
                cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

            for pattern, replacement in CURRENCY_MAP.items():
                cleaned = re.sub(pattern, replacement, cleaned)

            cleaned = cleaned.replace("&", " and ")
            cleaned = cleaned.replace("%", " percent ")

        # 5. Hindi & Marathi-specific normalizations
        if language.lower() in ("hi", "mr"):
            # Normalize Devanagari numerals to standard digits for consistent TTS pronunciation
            devanagari_digits = "०१२३४५६७८९"
            for d_idx, d_char in enumerate(devanagari_digits):
                cleaned = cleaned.replace(d_char, str(d_idx))

        # 6. Normalize extraneous whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def clean_text(cls, text: str, language: str = "en") -> CleanupResult:
        """
        Enhance and normalize script text using external LLMs if available, or local rules.

        Never throws an exception on network, key, or quota failures. Always returns
        a safe, clean result so core audio inference is never blocked.

        Args:
            text (str): Input script to clean.
            language (str): Target language code ('en', 'hi', or 'mr').

        Returns:
            CleanupResult: Normalized text and status metadata.
        """
        logging.info("Executing LLMService.clean_text")
        clean_input = text.strip()
        if not clean_input:
            return CleanupResult(
                original="",
                cleaned="",
                provider_used="none",
                service_available=True,
            )

        # Rule-based cleaning is always guaranteed and fast
        deterministic_clean = cls.rule_based_cleanup(clean_input, language=language)

        # 1. Attempt Groq API if key is set
        groq_key = getattr(settings, "GROQ_API_KEY", None)
        if groq_key and groq_key.strip():
            try:
                logging.info("Attempting Groq text enhancement...")
                prompt = (
                    "You are a speech synthesis preprocessor. Clean and format the following text for natural text-to-speech reading. "
                    "Fix punctuation, expand ambiguous abbreviations, and keep the exact original meaning and language. "
                    "Do NOT add introductory comments. Return ONLY the cleaned text.\n\n"
                    f"Text: {clean_input}"
                )
                headers = {
                    "Authorization": f"Bearer {groq_key.strip()}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 512,
                }
                with httpx.Client(timeout=8.0) as client:
                    resp = client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
                    if resp.status_code == 200:
                        llm_out = resp.json()["choices"][0]["message"]["content"].strip()
                        if llm_out:
                            logging.info("Groq text enhancement succeeded")
                            return CleanupResult(
                                original=clean_input,
                                cleaned=llm_out,
                                provider_used="groq",
                                service_available=True,
                            )
            except Exception as groq_err:
                logging.warning(f"Groq enhancement call failed: {groq_err}. Falling back...")

        # 2. Attempt Gemini API if key is set
        gemini_key = getattr(settings, "GEMINI_API_KEY", None)
        if gemini_key and gemini_key.strip():
            try:
                logging.info("Attempting Gemini text enhancement...")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key.strip()}"
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": (
                                "Format the following text for natural text-to-speech reading. "
                                "Fix punctuation and numbers, preserve exact language and meaning. "
                                f"Return ONLY the cleaned text:\n\n{clean_input}"
                            )
                        }]
                    }]
                }
                with httpx.Client(timeout=8.0) as client:
                    resp = client.post(url, json=payload)
                    if resp.status_code == 200:
                        candidates = resp.json().get("candidates", [])
                        if candidates:
                            part_text = candidates[0]["content"]["parts"][0]["text"].strip()
                            if part_text:
                                logging.info("Gemini text enhancement succeeded")
                                return CleanupResult(
                                    original=clean_input,
                                    cleaned=part_text,
                                    provider_used="gemini",
                                    service_available=True,
                                )
            except Exception as gemini_err:
                logging.warning(f"Gemini enhancement call failed: {gemini_err}. Falling back...")

        # 3. Default to deterministic rule-based cleaner
        logging.info("Using deterministic rule-based text cleanup (no external provider)")
        has_keys = bool(groq_key or gemini_key or getattr(settings, "OPENROUTER_API_KEY", None))
        return CleanupResult(
            original=clean_input,
            cleaned=deterministic_clean,
            provider_used="rule_based",
            service_available=has_keys,
        )
