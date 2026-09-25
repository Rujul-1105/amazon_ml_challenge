"""Devanagari → Latin transliteration (offline, MIT-licensed).

Uses the `indic-transliteration` library (Sanskrit-related; supports ITRANS,
IAST, ISO schemes). Falls back to `unidecode` for non-Devanagari or for
unsupported input. Result is normalized ASCII so it can be matched against
Latin-script addresses by string similarity.

Why this is innovative:
    Many Indian business names appear in mixed scripts — Devanagari in the
    business_name, Latin in the business_address (and vice-versa). Without
    cross-script matching, two records for the same real business look
    completely different to a string similarity model. The `indic-transliteration`
    library is MIT-licensed, runs offline (no API calls — fair-play compliant),
    and gives us a deterministic, reproducible romanization.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache

import regex
from indic_transliteration import sanscript
from unidecode import unidecode

log = logging.getLogger(__name__)

DEVANAGARI_RE = regex.compile(r"[ऀ-ॿ]+")
NON_PRINTABLE_RE = regex.compile(r"[\x00-\x1f\x7f]+")


# ---------------------------------------------------------------------------
# Devanagari → Latin transliteration
# ---------------------------------------------------------------------------

@lru_cache(maxsize=200_000)
def _devanagari_to_iso(text: str) -> str:
    """Convert Devanagari text to ISO 15919 (with diacritics)."""
    return sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ISO)


def to_latin(text: str) -> str:
    """Return a Latin-script, ASCII-friendly version of `text`.

    - Devanagari characters are romanized via IAST (with diacritics), then
      diacritics are stripped using `unidecode`.
    - Other non-ASCII characters are passed through `unidecode` directly.
    - Cached on (str) — single record can be processed many times.
    """
    if text is None:
        return ""
    s = str(text)
    if not s:
        return ""
    # Quick path: ASCII-only, no Devanagari
    if DEVANAGARI_RE.search(s) is None:
        return _ascii_safe(s)
    # Romanize Devanagari, then ASCII-normalize
    try:
        romanized = _devanagari_to_iso(s)
    except Exception as e:  # noqa: BLE001
        log.debug("Transliteration failed for %r: %s; falling back to unidecode", s[:40], e)
        romanized = s
    return _ascii_safe(romanized)


def _ascii_safe(s: str) -> str:
    """Strip non-printables and convert to ASCII-friendly form."""
    if not s:
        return ""
    s = NON_PRINTABLE_RE.sub(" ", s)
    s = unicodedata.normalize("NFKC", s)
    try:
        s = unidecode(s)
    except Exception:  # noqa: BLE001
        pass
    return s


# ---------------------------------------------------------------------------
# Per-record fields
# ---------------------------------------------------------------------------

def has_devanagari(text: str) -> bool:
    if not text:
        return False
    return DEVANAGARI_RE.search(text) is not None


def devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    dev = sum(len(m.group()) for m in DEVANAGARI_RE.finditer(text))
    return dev / total if total else 0.0
