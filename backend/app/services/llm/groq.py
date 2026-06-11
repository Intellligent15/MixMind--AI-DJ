"""Groq `LLMProvider` on the shared `CachedJSONProvider` base.

JSON mode (`response_format={"type": "json_object"}`) forces the model
to emit a valid JSON object; lenient parsing in the base tolerates the
occasional wrapper drift.
"""

from __future__ import annotations

import logging

from groq import Groq

from app.services.llm.base import CachedJSONProvider

logger = logging.getLogger(__name__)

# llama-4-scout — 30K TPM on Groq's free tier, strong at structured JSON.
DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Free-tier inference is fast; this only fires on network trouble.
GROQ_TIMEOUT_SECONDS = 30.0


class GroqProvider(CachedJSONProvider):
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 temperature: float | None = None):
        self.api_key = api_key
        self.model = model
        if temperature is not None:
            self.temperature = temperature

    def _chat_json(self, system: str, user: str) -> str:
        client = Groq(api_key=self.api_key, timeout=GROQ_TIMEOUT_SECONDS)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""
