import hashlib
import json
import logging
from typing import Any

import google.genai as genai

from app.services.storage import get_storage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert DJ transitioning between two tracks.
You will be provided with the musical analysis (BPM, keys, sections, beat grid, downbeats, energy curve, vocal segments), lyrics with timestamps (aligned when available, otherwise raw transcription), and vocal-safe regions for two songs: Song A (outgoing) and Song B (incoming).

Your goal is to plan a seamless, professional transition from A to B. You must return a JSON list of tool calls. You have the following tools available:
{tools_schema}

Rules:
1. Always start with exactly one `set_transition_window` call to define the alignment and crossfade duration.
2. Hard cuts, stem swaps, and drop swaps should land inside the provided `vocal_safe_regions`. Outside them, prefer crossfades.
3. You must emit exactly 4 `crossfade_stem` calls, one for each stem: "vocals", "drums", "bass", "other". All 4 MUST share the exact same `start_bar`, `duration_bars`, and `curve` ("equal_power" or "linear").
4. If keys clash, you may use `pitch_shift` (permanent) or `temporary_pitch_shift` (returns to native key) on Song B. Permanent `pitch_shift` is capped at ±2 semitones — beyond that, pyrubberband artifacts outweigh the harmonic benefit, and the executor will clamp. Prefer `temporary_pitch_shift` for larger excursions.
5. If tempos clash significantly, you may use `set_tempo_ramp` on Song B to ramp from A's BPM to B's BPM over a specified window.
6. The output must be a valid JSON list containing only tool call objects.
"""

# Timeout for a single Gemini generate_content call. 30s is a healthy
# ceiling for Flash; if we hit it, the worker falls back to the
# deterministic planner rather than hanging the Celery task.
GEMINI_TIMEOUT_MS = 30_000

class GeminiProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # We instantiate the client lazily or per-call if needed, but for now we do it per provider init.
        # However, to allow testing via patching genai.Client, we will instantiate it in the method.
    
    async def plan_transition(
        self,
        from_song: dict[str, Any],
        to_song: dict[str, Any],
        tools_schema: str,
    ) -> list[dict[str, Any]]:
        # 1. Compute Cache Key
        storage = get_storage()
        
        payload = {
            "from_song": from_song,
            "to_song": to_song,
            "tools_schema": tools_schema,
        }
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        h = hashlib.sha256(payload_bytes).hexdigest()
        cache_key = f"mix_plan_logs/{h}.json"
        
        # 2. Check Cache
        if await storage.exists(cache_key):
            logger.info("GeminiProvider: cache hit for %s", cache_key)
            data = await storage.read(cache_key)
            parsed = json.loads(data.decode("utf-8"))
            return parsed["response"]
            
        logger.info("GeminiProvider: cache miss for %s, calling LLM", cache_key)
        
        # 3. Call LLM
        client = genai.Client(
            api_key=self.api_key,
            http_options=genai.types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
        )
        system_instruction = SYSTEM_PROMPT.format(tools_schema=tools_schema)
        
        prompt = json.dumps({"Song A": from_song, "Song B": to_song}, indent=2)
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
            ),
        )
        
        try:
            plan = json.loads(response.text)
        except json.JSONDecodeError as e:
            logger.error("GeminiProvider: Failed to parse JSON response: %s", response.text)
            raise ValueError(f"Invalid JSON from LLM: {e}")
            
        if not isinstance(plan, list):
            raise ValueError(f"Expected JSON list, got {type(plan).__name__}")
            
        # 4. Save Cache
        log_data = {
            "prompt": prompt,
            "system_instruction": system_instruction,
            "response": plan,
        }
        await storage.write(cache_key, json.dumps(log_data, indent=2).encode("utf-8"))
        
        return plan
