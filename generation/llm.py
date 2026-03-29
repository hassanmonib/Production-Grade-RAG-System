"""OpenAI chat completion client."""

from __future__ import annotations

import logging

from openai import OpenAI

from config.settings import get_settings

logger = logging.getLogger(__name__)


class OpenAIClient:
    def __init__(self) -> None:
        s = get_settings()
        self.model = s.llm_model
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            s = get_settings()
            if not s.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._client = OpenAI(api_key=s.openai_api_key)
        return self._client

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        msg = r.choices[0].message.content or ""
        return msg.strip()
