"""Normalize and clean raw document text."""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


def clean_text(raw: str) -> str:
    if not raw:
        return ""
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
