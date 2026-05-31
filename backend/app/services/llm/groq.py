"""Groq concrete `LLMProvider`. Mirrors `GeminiProvider`'s contract:
hash the inputs, look up `mix_plan_logs/<sha>.json` in storage, call
the LLM only on cache miss, persist prompt+response on success.

The `groq` SDK is OpenAI-compatible. JSON mode (`response_format=
{"type": "json_object"}`) forces the model to emit a valid JSON
object — so the system prompt asks for `{"plan": [...]}` and we
unwrap. We also tolerate a raw list (some models ignore the wrap
instruction) for robustness.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from groq import Groq

from app.services.storage import get_storage

logger = logging.getLogger(__name__)

# Default to llama-4-scout — 30K TPM on Groq's free tier (vs 8K for
# gpt-oss-120b), 17B/16-expert MoE, strong at structured JSON. Swap
# via settings.groq_model if you want to A/B with another model.
DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Same 30s ceiling as Gemini. Free tier inference is fast enough that
# this only fires on network trouble.
GROQ_TIMEOUT_SECONDS = 30.0

SYSTEM_PROMPT = """You are an expert DJ transitioning between two tracks.
You will be provided with the musical analysis (BPM, key, sections, downbeats, energy curve) and vocal-safe regions for two songs: Song A (outgoing) and Song B (incoming). The `vocal_safe_regions` are time intervals where no vocals are present — that is where cuts, stem swaps, and drop swaps should land.

Your goal is to plan a seamless, professional transition from A to B. Return a JSON object with a single key "plan" whose value is the list of tool call objects. You have the following tools available:
{tools_schema}

Rules:
1. Always start with exactly one `set_transition_window` call to define the alignment and crossfade duration.
2. Hard cuts, stem swaps, and drop swaps should land inside the provided `vocal_safe_regions`. Outside them, prefer crossfades.
3. You must emit exactly 4 `crossfade_stem` calls, one for each stem: "vocals", "drums", "bass", "other". All 4 MUST share the exact same `start_bar`, `duration_bars`, and `curve` ("equal_power" or "linear").
4. If keys clash, you may use `pitch_shift` (permanent) or `temporary_pitch_shift` (returns to native key) on Song B. Permanent `pitch_shift` is capped at ±2 semitones — beyond that, pyrubberband artifacts outweigh the harmonic benefit, and the executor will clamp. Prefer `temporary_pitch_shift` for larger excursions.
5. If tempos clash significantly, you may use `set_tempo_ramp` on Song B to ramp from A's BPM to B's BPM over a specified window.
6. The output must be a valid JSON object of the form {{"plan": [...]}}.
"""


class GroqProvider:
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

        # Cache key includes the model — swapping models invalidates cache.
        payload = {
            "from_song": from_song,
            "to_song": to_song,
            "tools_schema": tools_schema,
            "model": self.model,
        }
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        h = hashlib.sha256(payload_bytes).hexdigest()
        cache_key = f"mix_plan_logs/{h}.json"

        if await storage.exists(cache_key):
            logger.info("GroqProvider: cache hit for %s", cache_key)
            data = await storage.read(cache_key)
            return json.loads(data.decode("utf-8"))["response"]

        logger.info("GroqProvider: cache miss for %s, calling LLM", cache_key)

        client = Groq(api_key=self.api_key, timeout=GROQ_TIMEOUT_SECONDS)
        system_instruction = SYSTEM_PROMPT.format(tools_schema=tools_schema)
        prompt = json.dumps({"Song A": from_song, "Song B": to_song}, indent=2)

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
            obj = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.error("GroqProvider: invalid JSON: %s", raw_text[:500])
            raise ValueError(f"Invalid JSON from LLM: {e}")

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
