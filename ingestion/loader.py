"""Load text from PDF, plain text, and Markdown bytes."""

from __future__ import annotations

import logging
from io import BytesIO

logger = logging.getLogger(__name__)


def load_bytes(filename: str, data: bytes) -> str:
    name = (filename or "document").lower()
    if name.endswith(".pdf"):
        return _load_pdf(data)
    if name.endswith((".md", ".markdown", ".txt", ".text")):
        return data.decode("utf-8", errors="replace")
    logger.warning("Unknown extension for %s; decoding as utf-8", filename)
    return data.decode("utf-8", errors="replace")


def _load_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception as e:  # noqa: BLE001
            logger.debug("Page extract failed: %s", e)
            t = ""
        parts.append(t)
    return "\n\n".join(parts)
