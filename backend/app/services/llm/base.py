"""Shared LLM-provider machinery.

v1 duplicated the hash-cache + JSON-parse logic across three providers.
This base class owns it once:

* content-addressed response cache in storage (`mix_plan_logs/<sha>.json`)
  whose key covers system prompt, user prompt, model, temperature, AND a
  caller-supplied `nonce` — bumping the nonce is how "re-roll this
  transition" gets a genuinely fresh plan instead of a cache hit;
* explicit temperature (v1 silently used provider defaults);
* lenient JSON extraction that tolerates markdown fences, stray
  prefixes, and trailing prose (promoted from the DigitalOcean provider,
  where gpt-oss needed it).

Concrete providers implement one method: `_chat_json(system, user) ->
str`, a synchronous call that asks the API for a JSON response.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from app.services.storage import get_storage

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE = 0.7


def parse_json_lenient(raw_text: str) -> Any:
    """Parse model JSON, tolerating junk prefixes/suffixes and fences."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw_text):
        if ch in "[{":
            try:
                obj, _ = decoder.raw_decode(raw_text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("Invalid JSON from LLM: no decodable JSON value found")


class CachedJSONProvider(ABC):
    """Base for providers that return JSON completions with storage caching."""

    model: str = ""
    temperature: float = DEFAULT_TEMPERATURE

    @abstractmethod
    def _chat_json(self, system: str, user: str) -> str:
        """Synchronous provider call. Must request JSON output."""

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        nonce: int = 0,
        cache_namespace: str = "mix_plan_logs",
    ) -> Any:
        storage = get_storage()
        payload = {
            "system": system,
            "user": user,
            "model": self.model,
            "temperature": self.temperature,
            "nonce": nonce,
        }
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        h = hashlib.sha256(payload_bytes).hexdigest()
        cache_key = f"{cache_namespace}/{h}.json"

        if await storage.exists(cache_key):
            logger.info("%s: cache hit for %s", type(self).__name__, cache_key)
            data = await storage.read(cache_key)
            return json.loads(data.decode("utf-8"))["response"]

        logger.info(
            "%s: cache miss for %s, calling LLM (model=%s, nonce=%d)",
            type(self).__name__, cache_key, self.model, nonce,
        )
        raw_text = self._chat_json(system, user)
        obj = parse_json_lenient(raw_text)

        log_data = {
            "system_instruction": system,
            "prompt": user,
            "model": self.model,
            "temperature": self.temperature,
            "nonce": nonce,
            "response": obj,
        }
        await storage.write(
            cache_key, json.dumps(log_data, indent=2).encode("utf-8")
        )
        return obj

    # ------------------------------------------------------------------
    # Legacy v1 contract, kept so existing tests / the legacy planner path
    # keep working. Free-form tool-call planning; planner v2 doesn't use it.
    # ------------------------------------------------------------------
    async def plan_transition(
        self,
        from_song: dict[str, Any],
        to_song: dict[str, Any],
        tools_schema: str,
    ) -> list[dict[str, Any]]:
        from app.services.llm.legacy_prompt import LEGACY_SYSTEM_PROMPT

        system = LEGACY_SYSTEM_PROMPT.replace("{tools_schema}", tools_schema)
        user = json.dumps({"A": from_song, "B": to_song}, indent=2)
        obj = await self.complete_json(system=system, user=user)
        if isinstance(obj, dict) and isinstance(obj.get("plan"), list):
            return obj["plan"]
        if isinstance(obj, list):
            return obj
        raise ValueError(
            f"Expected JSON object with 'plan' list, got {type(obj).__name__}"
        )
