"""DigitalOcean Serverless Inference `LLMProvider` (OpenAI-compatible)
on the shared `CachedJSONProvider` base. gpt-oss occasionally wraps or
prefixes its JSON; the base's lenient parser handles it.
"""

from __future__ import annotations

import logging

from openai import OpenAI

from app.services.llm.base import CachedJSONProvider

logger = logging.getLogger(__name__)

BASE_URL = "https://inference.do-ai.run/v1"

# gpt-oss-120b is the strongest model on the tier-1 DO subscription and
# fast enough (~10s) to stay under the 30s timeout.
DEFAULT_MODEL = "openai-gpt-oss-120b"

DO_TIMEOUT_SECONDS = 30.0


class DigitalOceanProvider(CachedJSONProvider):
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 temperature: float | None = None):
        self.api_key = api_key
        self.model = model
        if temperature is not None:
            self.temperature = temperature

    def _chat_json(self, system: str, user: str) -> str:
        client = OpenAI(
            base_url=BASE_URL, api_key=self.api_key, timeout=DO_TIMEOUT_SECONDS
        )
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
