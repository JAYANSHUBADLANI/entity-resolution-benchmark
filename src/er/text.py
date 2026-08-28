"""Text normalisation shared by blocking and by the similarity features.

Deliberately conservative. Aggressive normalisation (stemming, aggressive stopword lists) can
manufacture agreement that is not there, which flatters the matcher and does not survive contact
with a different dataset.
"""
from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^a-z0-9\s]+")
_SPACE = re.compile(r"\s+")


def normalise(value) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"nan", "none"}:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def tokens(value) -> list[str]:
    normalised = normalise(value)
    return normalised.split() if normalised else []
