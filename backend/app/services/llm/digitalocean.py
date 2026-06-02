"""DigitalOcean Serverless Inference `LLMProvider`. Same contract as
`GroqProvider`: hash the inputs, look up `mix_plan_logs/<sha>.json` in
storage, call the LLM only on cache miss, persist prompt+response.

DO's endpoint is OpenAI-compatible, so we use the official `openai`
client pointed at `https://inference.do-ai.run/v1`. JSON mode
(`response_format={"type": "json_object"}`) forces a JSON object, so we
ask for `{"plan": [...]}` and unwrap — exactly like Groq. To keep the
prompt body in lock-step with Groq (the only intentional difference vs
Gemini is the wrapper) we import `SYSTEM_PROMPT` from `groq` rather than
maintaining a third copy.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from openai import OpenAI

from app.services.llm.groq import SYSTEM_PROMPT
from app.services.storage import get_storage

logger = logging.getLogger(__name__)

BASE_URL = "https://inference.do-ai.run/v1"

# gpt-oss-120b is the strongest model on the tier-1 DO subscription
# (commercial GPT-5.x / Claude are 403-gated to higher tiers) and is the
# only top performer fast enough for the render pipeline. Swap via
# settings.do_inference_model once a tier upgrade unlocks GPT-5.5/Claude.
DEFAULT_MODEL = "openai-gpt-oss-120b"

# Match the other providers' 30s ceiling so a hung call falls back to the
# deterministic planner rather than stalling the Celery task.
DO_TIMEOUT_SECONDS = 30.0


def _parse_plan_json(raw_text: str) -> Any:
    """Parse the model's JSON, tolerating a junk prefix or extra wrapper.

    gpt-oss occasionally emits a stray leading `{` (or double-wraps the
    object) before the real `{"plan": [...]}`, which makes a strict
    json.loads fail. On failure we scan for the first position that
    decodes to a complete JSON value — raw_decode ignores trailing junk,
    so this also handles markdown fences and trailing prose.
    """
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


class DigitalOceanProvider:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model

    async def plan_transition(
        self,
        from_song: dict[str, Any],
        to_song: dict[str, Any],
        tools_schema: str,
    ) -> list[dict[str, Any]]:
        storage = get_storage()

        # Cache key includes the model and formatted system prompt so
        # swapping either invalidates stale plans automatically.
        system_instruction = SYSTEM_PROMPT.replace("{tools_schema}", tools_schema)
        payload = {
            "from_song": from_song,
            "to_song": to_song,
            "tools_schema": tools_schema,
            "model": self.model,
            "system_instruction": system_instruction,
        }
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        h = hashlib.sha256(payload_bytes).hexdigest()
        cache_key = f"mix_plan_logs/{h}.json"

        if await storage.exists(cache_key):
            logger.info("DigitalOceanProvider: cache hit for %s", cache_key)
            data = await storage.read(cache_key)
            return json.loads(data.decode("utf-8"))["response"]

        logger.info("DigitalOceanProvider: cache miss for %s, calling LLM", cache_key)

        client = OpenAI(
            base_url=BASE_URL, api_key=self.api_key, timeout=DO_TIMEOUT_SECONDS
        )
        # Songs are labeled "A"/"B" everywhere — including user-message
        # keys — to reinforce the tool-call field requirement.
        prompt = json.dumps({"A": from_song, "B": to_song}, indent=2)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        raw_text = response.choices[0].message.content or ""
        try:
            obj = _parse_plan_json(raw_text)
        except ValueError:
            logger.error("DigitalOceanProvider: invalid JSON: %s", raw_text[:500])
            raise

        # Prefer {"plan": [...]} (what we asked for); fall back to a raw
        # list in case the model ignored the wrap instruction.
        if isinstance(obj, dict) and isinstance(obj.get("plan"), list):
            plan = obj["plan"]
        elif isinstance(obj, list):
            plan = obj
        else:
            raise ValueError(
                f"Expected JSON object with 'plan' list, got {type(obj).__name__}"
            )

        log_data = {
            "prompt": prompt,
            "system_instruction": system_instruction,
            "model": self.model,
            "response": plan,
        }
        await storage.write(cache_key, json.dumps(log_data, indent=2).encode("utf-8"))
        return plan
