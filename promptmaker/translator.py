"""Pluggable translation step (non-English input -> English).

Any object with ``translate(text: str) -> str`` can be passed to
:class:`promptmaker.PromptMaker` as ``translator``.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Dict, Optional, Protocol


class Translator(Protocol):
    def translate(self, text: str) -> str: ...


def is_english(text: str) -> bool:
    """True when the text is (almost) entirely ASCII letters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    ascii_letters = sum(1 for c in letters if c.isascii())
    return ascii_letters / len(letters) >= 0.9


class NoTranslator:
    """Returns the text unchanged."""

    def translate(self, text: str) -> str:
        return text


class DictionaryTranslator:
    """Offline word-by-word fallback using lexicon/ko_en.json.

    It is not a real translator: it only swaps known words so that the
    keyword search still finds something without network access.
    """

    def __init__(self, dictionary: Dict[str, str]):
        # Longest keys first so that "비 오는" wins over "비".
        self.pairs = sorted(((k, v) for k, v in dictionary.items() if not k.startswith("_")),
                            key=lambda kv: -len(kv[0]))

    def translate(self, text: str) -> str:
        out = text
        for ko, en in self.pairs:
            if ko in out:
                out = out.replace(ko, f" {en} ")
        # Remove remaining non-ASCII (untranslated particles etc.)
        out = re.sub(r"[^\x00-\x7F]+", " ", out)
        return re.sub(r"\s+", " ", out).strip()


class GoogleTranslator:
    """Google Translate.

    Uses the ``deep-translator`` package when installed, otherwise the public
    ``translate.googleapis.com`` endpoint through urllib (no extra packages).
    On any failure it falls back to ``fallback`` (if given) or the original text.
    """

    def __init__(self, target: str = "en", timeout: float = 8.0,
                 fallback: Optional[Translator] = None):
        self.target = target
        self.timeout = timeout
        self.fallback = fallback
        self.last_error: Optional[str] = None
        self.last_backend: str = ""

    def _deep_translator(self, text: str) -> Optional[str]:
        try:
            from deep_translator import GoogleTranslator as _DT  # type: ignore
        except ImportError:
            return None
        return _DT(source="auto", target=self.target).translate(text)

    def _http(self, text: str) -> str:
        params = urllib.parse.urlencode(
            {"client": "gtx", "sl": "auto", "tl": self.target, "dt": "t", "q": text}
        )
        url = f"https://translate.googleapis.com/translate_a/single?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return "".join(part[0] for part in data[0] if part and part[0])

    def translate(self, text: str) -> str:
        self.last_error = None
        self.last_backend = "google"
        try:
            result = self._deep_translator(text)
            if result is None:
                result = self._http(text)
            if result:
                return result
        except Exception as exc:  # network errors, quota, parsing...
            self.last_error = f"{type(exc).__name__}: {exc}"
        if self.fallback is not None:
            self.last_backend = "fallback"
            return self.fallback.translate(text)
        self.last_backend = "none"
        return text
