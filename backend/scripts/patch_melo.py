"""
Reproducible patching utility for MeloTTS site-packages installation.

Resolves eager multilingual import coupling in MeloTTS so that English-only
synthesis does not require Japanese (mecab-python3/unidic-lite) or Korean
(num2words) dictionary packages.
"""

import argparse
import importlib
import importlib.util
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("patch_melo")


DISTRIBUTE_PHONE_CODE = '''def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word'''

LAZY_LANGUAGE_MAP_CODE = '''from . import cleaned_text_to_sequence
import copy
import importlib

class _LazyLanguageModuleMap(dict):
    _mod_names = {
        "ZH": "chinese",
        "JP": "japanese",
        "EN": "english",
        "ZH_MIX_EN": "chinese_mix",
        "KR": "korean",
        "FR": "french",
        "SP": "spanish",
        "ES": "spanish",
    }

    def __contains__(self, item):
        return item in self._mod_names

    def __getitem__(self, item):
        if dict.__contains__(self, item):
            return super().__getitem__(item)
        if item in self._mod_names:
            mod = importlib.import_module(f".{self._mod_names[item]}", package=__package__)
            self[item] = mod
            return mod
        raise KeyError(item)

language_module_map = _LazyLanguageModuleMap()'''

GET_BERT_CODE = '''def get_bert(norm_text, word2ph, language, device):
    if language == "EN":
        from .english_bert import get_bert_feature
        return get_bert_feature(norm_text, word2ph, device)
    elif language == "ZH":
        from .chinese_bert import get_bert_feature
        return get_bert_feature(norm_text, word2ph, device)
    elif language == "JP":
        from .japanese_bert import get_bert_feature
        return get_bert_feature(norm_text, word2ph, device)
    elif language == "ZH_MIX_EN":
        from .chinese_mix import get_bert_feature
        return get_bert_feature(norm_text, word2ph, device)
    elif language in ("SP", "ES"):
        from .spanish_bert import get_bert_feature
        return get_bert_feature(norm_text, word2ph, device)
    elif language == "FR":
        from .french_bert import get_bert_feature
        return get_bert_feature(norm_text, word2ph, device)
    elif language == "KR":
        from .korean import get_bert_feature
        return get_bert_feature(norm_text, word2ph, device)
    else:
        raise ValueError(f"Language {language} not supported for BERT features")'''


def find_melo_dir() -> Path:
    """
    Locate the root directory of the installed melo package.

    Returns:
        Path: Filesystem path to the installed melo directory.

    Raises:
        RuntimeError: If melo is not installed in the active environment.
    """
    try:
        spec = importlib.util.find_spec("melo")
        if spec is None or spec.origin is None:
            raise ImportError("melo package spec not found")
        return Path(spec.origin).parent
    except Exception as exc:
        raise RuntimeError(
            "Could not locate installed 'melo' package. Ensure it is installed via: "
            "pip install --no-deps git+https://github.com/myshell-ai/MeloTTS.git"
        ) from exc


def patch_english(melo_dir: Path) -> bool:
    """
    Patch melo/text/english.py to inline distribute_phone.

    Args:
        melo_dir (Path): Melo root package path.

    Returns:
        bool: True if already patched or patch succeeded.
    """
    english_py = melo_dir / "text" / "english.py"
    if not english_py.exists():
        log.error("File not found: %s", english_py)
        return False

    content = english_py.read_text(encoding="utf-8")
    if "from .japanese import distribute_phone" not in content:
        log.info("[OK] melo/text/english.py is already patched.")
        return True

    bak = english_py.with_suffix(".py.bak")
    if not bak.exists():
        shutil.copyfile(english_py, bak)

    new_content = content.replace("from .japanese import distribute_phone", DISTRIBUTE_PHONE_CODE)
    english_py.write_text(new_content, encoding="utf-8")
    log.info("[PATCHED] inlined distribute_phone in %s", english_py)
    return True


def patch_cleaner(melo_dir: Path) -> bool:
    """
    Patch melo/text/cleaner.py to make language module imports lazy.

    Args:
        melo_dir (Path): Melo root package path.

    Returns:
        bool: True if already patched or patch succeeded.
    """
    cleaner_py = melo_dir / "text" / "cleaner.py"
    if not cleaner_py.exists():
        log.error("File not found: %s", cleaner_py)
        return False

    content = cleaner_py.read_text(encoding="utf-8")
    if "_LazyLanguageModuleMap" in content:
        log.info("[OK] melo/text/cleaner.py is already patched.")
        return True

    bak = cleaner_py.with_suffix(".py.bak")
    if not bak.exists():
        shutil.copyfile(cleaner_py, bak)

    # Replace the top imports up to language_module_map
    target = (
        "from . import chinese, japanese, english, chinese_mix, korean, french, spanish\n"
        "from . import cleaned_text_to_sequence\n"
        "import copy\n\n"
        'language_module_map = {"ZH": chinese, "JP": japanese, "EN": english, \'ZH_MIX_EN\': chinese_mix, \'KR\': korean,\n'
        "                    'FR': french, 'SP': spanish, 'ES': spanish}"
    )

    if target in content:
        new_content = content.replace(target, LAZY_LANGUAGE_MAP_CODE)
    else:
        # Fallback line-by-line replacement
        lines = content.splitlines()
        clean_lines = []
        skip = False
        for line in lines:
            if line.startswith("from . import chinese, japanese, english"):
                skip = True
                continue
            if skip and line.strip().endswith("'ES': spanish}"):
                skip = False
                continue
            if not skip:
                clean_lines.append(line)
        new_content = LAZY_LANGUAGE_MAP_CODE + "\n" + "\n".join(clean_lines).lstrip()

    cleaner_py.write_text(new_content, encoding="utf-8")
    log.info("[PATCHED] lazy language_module_map in %s", cleaner_py)
    return True


def patch_init(melo_dir: Path) -> bool:
    """
    Patch melo/text/__init__.py to make BERT loading language-conditional.

    Args:
        melo_dir (Path): Melo root package path.

    Returns:
        bool: True if already patched or patch succeeded.
    """
    init_py = melo_dir / "text" / "__init__.py"
    if not init_py.exists():
        log.error("File not found: %s", init_py)
        return False

    content = init_py.read_text(encoding="utf-8")
    if 'if language == "EN":' in content and "from .english_bert import get_bert_feature" in content:
        log.info("[OK] melo/text/__init__.py is already patched.")
        return True

    bak = init_py.with_suffix(".py.bak")
    if not bak.exists():
        shutil.copyfile(init_py, bak)

    # Find get_bert definition and replace body
    if "def get_bert(norm_text, word2ph, language, device):" in content:
        parts = content.split("def get_bert(norm_text, word2ph, language, device):")
        new_content = parts[0] + GET_BERT_CODE + "\n"
        init_py.write_text(new_content, encoding="utf-8")
        log.info("[PATCHED] conditional get_bert in %s", init_py)
        return True

    log.error("Could not find get_bert in %s", init_py)
    return False


def verify_melo() -> bool:
    """
    Verify that melo.api TTS and English clean_text can be imported and executed.

    Returns:
        bool: True if verification passes cleanly.
    """
    try:
        from melo.api import TTS
        from melo.text.cleaner import clean_text

        norm, phones, tones, word2ph = clean_text("Test sentence for verification.", "EN")
        assert len(phones) > 0
        assert len(tones) > 0
        log.info("[VERIFIED] melo.api.TTS and English cleaner executed successfully.")
        return True
    except Exception as exc:
        log.error("[VERIFICATION FAILED] %s", exc)
        return False


def revert_patches(melo_dir: Path) -> None:
    """
    Revert all patches using stored .bak backup files.

    Args:
        melo_dir (Path): Melo root package path.

    Returns:
        None
    """
    targets = [
        melo_dir / "text" / "english.py",
        melo_dir / "text" / "cleaner.py",
        melo_dir / "text" / "__init__.py",
    ]
    for target in targets:
        bak = target.with_suffix(".py.bak")
        if bak.exists():
            shutil.copyfile(bak, target)
            bak.unlink()
            log.info("[REVERTED] restored %s from backup", target.name)
        else:
            log.warning("[SKIP] no backup found for %s", target.name)


def main() -> None:
    """
    CLI execution entry point.
    """
    parser = argparse.ArgumentParser(description="MeloTTS English-Only Reproducible Patcher")
    parser.add_argument("--verify-only", action="store_true", help="Check verification only without modifying files")
    parser.add_argument("--revert", action="store_true", help="Revert files from .bak backups")
    args = parser.parse_args()

    try:
        melo_dir = find_melo_dir()
        log.info("Found MeloTTS installation at: %s", melo_dir)
    except Exception as exc:
        log.error(str(exc))
        sys.exit(1)

    if args.revert:
        revert_patches(melo_dir)
        sys.exit(0)

    if args.verify_only:
        ok = verify_melo()
        sys.exit(0 if ok else 1)

    # Apply patches
    p1 = patch_english(melo_dir)
    p2 = patch_cleaner(melo_dir)
    p3 = patch_init(melo_dir)

    if not (p1 and p2 and p3):
        log.error("One or more patches failed to apply.")
        sys.exit(1)

    log.info("Running verification test...")
    if verify_melo():
        log.info("MeloTTS English-only patch is successfully active and verified.")
        sys.exit(0)
    else:
        log.error("Patch verification failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
