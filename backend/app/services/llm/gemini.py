import hashlib
import json
import logging
from typing import Any

import google.genai as genai

from app.services.storage import get_storage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert DJ transitioning between two tracks.
You will be provided with the musical analysis (BPM, key, sections, downbeats, energy curve) and vocal-safe regions for two songs labeled "A" (outgoing) and "B" (incoming). The `vocal_safe_regions` are time intervals where no vocals are present — that is where cuts, stem swaps, and drop swaps should land.

CRITICAL: In every tool call, the `song`, `from_song`, and `to_song` fields MUST be exactly the single-character strings "A" or "B". Never "Song A", "song_a", "1", or any other variation — the executor will reject anything else.

Your goal is to plan a seamless, professional transition from A to B. You must return a JSON list of tool calls. You have the following tools available:
{tools_schema}

Rules:
1. Always start with exactly one `set_transition_window` call to define the alignment and crossfade duration.
2. SEAM HEADROOM (critical): `from_song_time_start` MUST be <= A's `analysis.max_seam_time`, and `to_song_time_start` MUST be <= B's `analysis.max_seam_time`. Equivalently, `from_song_time_start + duration_bars * 4 * 60 / bpm_a` must be <= `A.duration - 2s`, so the crossfade finishes at least 2s before A runs out. These values are pre-computed to leave room for the full crossfade plus a safety buffer. If you violate this, the executor will shrink your crossfade to whatever audio remains — typically producing an abrupt cut rather than a smooth blend. For the most musical result in A, place `from_song_time_start` at the start of A's last section (the outro) as long as that start is <= A's `max_seam_time`.
3. Hard cuts, stem swaps, and drop swaps should land inside the provided `vocal_safe_regions`. Outside them, prefer crossfades.
4. You must emit exactly 4 `crossfade_stem` calls, one for each stem: "vocals", "drums", "bass", "other". They MAY use different `start_bar`, `duration_bars`, and `curve` ("equal_power" or "linear") values to build stem-swap transitions where one stem fades out before another. Every call uses `"from_song": "A"` and `"to_song": "B"`.
5. If keys clash, you may use `pitch_shift` (permanent) or `temporary_pitch_shift` (returns to native key) on song B (i.e. `"song": "B"`). Permanent `pitch_shift` is capped at ±2 semitones — beyond that, pyrubberband artifacts outweigh the harmonic benefit, and the executor will clamp. Prefer `temporary_pitch_shift` for larger excursions.
6. If tempos clash significantly, you may use `set_tempo_ramp` on song B (`"song": "B"`) to ramp from A's BPM to B's BPM over a specified window.
7. The output must be a valid JSON list containing only tool call objects.

Transition styles (pick ONE based on the two songs' energy_curve, BPM, key, and vocal_safe_regions; emit the matching tool calls — do NOT label which style you picked):

- Classic Blend — use when A and B have similar energy and compatible keys. All 4 `crossfade_stem` calls share `start_bar=0`, `duration_bars=16`, `curve="equal_power"`. Smooth long blend; the default when nothing else fits.

- Vocal-First Out — use when A and B are different songs but have similar tempo, especially when A has a strong vocal hook you want to clear before B's instrumental enters. Vocals call uses an earlier and shorter envelope (e.g. `start_bar=0`, `duration_bars=8`); drums / bass / other call a later one (e.g. `start_bar=8`, `duration_bars=8`). A's instrumental hands off after the vocal is already gone.

- Drop Swap — use when B starts with a drop and A has a clean bar at its end inside a `vocal_safe_region`. All 4 stems share `start_bar` aligned to that safe downbeat and a short `duration_bars` (2 to 4); `curve="equal_power"`. Snaps from A to B at the drop.

- Drum-Bridge — use when both songs are drum-driven. Vocals / bass / other fade first (e.g. `start_bar=0`, `duration_bars=8`), drums fade later (e.g. `start_bar=4`, `duration_bars=12`) so A's drums hold while B's drums come in early and bridge the two beat grids.

- Filter Sweep Out — use when A is bright/high-energy and B is darker, or when you want to "deconstruct" A before B enters. Emit one `filter_sweep` on `"song": "A"` with `type="lowpass"`, `start_time` aligned to A's last 8 bars before the seam, `end_time` at the seam, `start_cutoff_hz=20000`, `end_cutoff_hz=200`. Combine with a standard 4-stem crossfade (e.g. Classic Blend params) — the sweep removes A's brightness so B's entry feels like opening a curtain. Do NOT also use `pitch_shift` on B here; the sweep is the focal effect.

- Loop & Echo Trail — use when A has a clean instrumental tail (no vocals in the last 8 bars per `vocal_safe_regions`) and B intros softly. Emit one `loop_section` on A around its last 4 bars before the seam (`beats=16`, `repeats=2`, `bpm=A.bpm`) to extend the bridge, then one `echo_out` on A at the loop's end (`beats=4`, `feedback=0.5`, `bpm=A.bpm`) so A dissolves into echoes while the 4 `crossfade_stem` calls bring B in over the top. The crossfade can be short (`duration_bars=4`) because A is already vanishing.

Advanced moves (use sparingly, when the style above calls for them):
- `swap_stem` is a hard cut on a SINGLE stem at a downbeat (`time` is in OUTPUT-timeline seconds, not original-song). Use to instant-swap drums while letting the other stems crossfade smoothly. Pair with 3 normal `crossfade_stem` calls for the other stems.
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
        # 1. Compute Cache Key — includes the formatted system prompt so a
        #    prompt change invalidates cached plans automatically.
        storage = get_storage()

        system_instruction = SYSTEM_PROMPT.format(tools_schema=tools_schema)
        payload = {
            "from_song": from_song,
            "to_song": to_song,
            "tools_schema": tools_schema,
            "system_instruction": system_instruction,
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
        # Songs are labeled "A"/"B" in the prompt to match what the
        # system prompt requires in tool-call fields. Models parrot the
        # labels from the user message — keeping them short reinforces
        # the SongRef constraint.
        prompt = json.dumps({"A": from_song, "B": to_song}, indent=2)
        
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
