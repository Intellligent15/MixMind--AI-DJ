"""Gemini `LLMProvider` on the shared `CachedJSONProvider` base.

Uses `response_mime_type="application/json"` so Gemini's constrained
decoder guarantees syntactically valid JSON; semantic validation
(candidate ids, style enum) happens in planner v2's Pydantic schema.
"""

from __future__ import annotations

import logging

import google.genai as genai

from app.services.llm.base import CachedJSONProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"

# Healthy ceiling for Flash; on timeout the worker falls back rather
# than hanging the Celery task.
GEMINI_TIMEOUT_MS = 30_000


class GeminiProvider(CachedJSONProvider):
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 temperature: float | None = None):
        self.api_key = api_key
        self.model = model
        if temperature is not None:
            self.temperature = temperature

    def _chat_json(self, system: str, user: str) -> str:
        client = genai.Client(
            api_key=self.api_key,
            http_options=genai.types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
        )
        response = client.models.generate_content(
            model=self.model,
            contents=user,
            config=genai.types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                temperature=self.temperature,
            ),
        )
        return response.text or ""
