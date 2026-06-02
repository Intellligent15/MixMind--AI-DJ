import hashlib
import json
import logging
from typing import Any

import google.genai as genai

from app.services.storage import get_storage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert DJ planning a seamless transition from track A (outgoing) into track B (incoming). A great transition is varied and musical — NOT the same long crossfade every time.

INPUT — for each of A and B you get:
- bpm, key, camelot_key, time_signature, seconds_per_bar, duration
- max_seam_time: the latest time you may leave A / enter B. A hard ceiling (see Hard rules).
- sections: [{"start","end","energy"}] — your structural map. energy is 0..1 normalized to the song's hottest section: ~1.0 = a drop or chorus, low values = intro, breakdown, or outro.
- vocal_safe_regions: [{"start","end"}] — spans with NO vocals. The ONLY places a hard cut or stem swap may land.

PLAN IN THIS ORDER:
1. Choose the seam. from_song_time_start = where you leave A; to_song_time_start = where you enter B. Land both on a section boundary near a downbeat (use seconds_per_bar to reason in bars). Use energy to make it musical: blend a high-energy A tail into a high-energy B section, or drop B in where A has gone quiet. Keep both <= max_seam_time.
2. Choose ONE style below that actually fits these two songs. Reach past Classic Blend whenever the songs give you a reason to — variety across transitions is the goal.
3. Emit that style's tool calls.

TRANSITION STYLES (pick one; emit its calls — do not name it):
- Classic Blend — similar energy, compatible keys, nothing special to exploit. All 4 crossfade_stem: start_bar=0, duration_bars=16, equal_power. The fallback, not the default.
- Vocal-First Out — A has a strong vocal hook to clear before B enters. vocals: start_bar=0, duration_bars=8; drums/bass/other: start_bar=8, duration_bars=8. A's instrumental hands off after its vocal is gone.
- Drum-Bridge — both tracks drum-driven. vocals/bass/other: start_bar=0, duration_bars=8; drums: start_bar=4, duration_bars=12, so A's drums hold while B's come in early and bridge the two grids.
- Drop Swap — B opens on a drop (B's first section energy near 1.0) and A has a clean bar inside a vocal_safe_region at the seam. All 4 stems: same start_bar on that safe downbeat, duration_bars=2..4, equal_power. Snaps A→B at the drop.
- Filter Sweep Out — A is bright/high-energy, B darker, or you want to deconstruct A. One filter_sweep on A: lowpass, start_time = seam minus 8 bars, end_time = seam, start_cutoff_hz=20000, end_cutoff_hz=200, plus a normal 16-bar 4-stem crossfade. Do NOT pitch_shift B here.
- Loop & Echo Trail — A has a clean instrumental tail (no vocals in its last 8 bars) and B intros softly. One loop_section on A's last 4 bars (beats=16, repeats=2, bpm=A.bpm), then one echo_out at the loop end (beats=4, feedback=0.5, bpm=A.bpm); the 4 crossfade_stem calls can be short (duration_bars=4).

DON'T LET A LINGER: by default A only goes silent when B reaches full, so a long crossfade keeps A audible across the whole overlap and muddies the mix. Set `a_fade_out_bars` shorter than `duration_bars` so A clears out early while B keeps swelling in on its own — e.g. duration_bars=16, a_fade_out_bars=8 means A is gone halfway through while B keeps rising. Use this on most transitions; reserve a_fade_out_bars == duration_bars for when you genuinely want A and B locked together to the end. Keep crossfades punchy (8–16 bars), don't reflexively max the duration.

Available tools:
{tools_schema}

WORKED EXAMPLES (copy these shapes; change the numbers to fit the songs):

Example — Drum-Bridge, A and B both 128 BPM (seconds_per_bar 1.875), seam at A 180.0s / B 32.0s:
[
  {"tool": "set_transition_window", "from_song_time_start": 180.0, "to_song_time_start": 32.0, "duration_bars": 16},
  {"tool": "crossfade_stem", "stem": "vocals", "from_song": "A", "to_song": "B", "start_bar": 0, "duration_bars": 8, "curve": "equal_power"},
  {"tool": "crossfade_stem", "stem": "bass", "from_song": "A", "to_song": "B", "start_bar": 0, "duration_bars": 8, "curve": "equal_power"},
  {"tool": "crossfade_stem", "stem": "other", "from_song": "A", "to_song": "B", "start_bar": 0, "duration_bars": 8, "curve": "equal_power"},
  {"tool": "crossfade_stem", "stem": "drums", "from_song": "A", "to_song": "B", "start_bar": 4, "duration_bars": 12, "curve": "equal_power"}
]

Example — Filter Sweep Out, A 128 BPM bright, B 124 BPM darker, seam at A 200.0s / B 16.0s:
[
  {"tool": "set_transition_window", "from_song_time_start": 200.0, "to_song_time_start": 16.0, "duration_bars": 16},
  {"tool": "set_tempo_ramp", "song": "B", "start_time": 16.0, "end_time": 47.0, "start_bpm": 128.0, "end_bpm": 124.0},
  {"tool": "filter_sweep", "song": "A", "type": "lowpass", "start_time": 185.0, "end_time": 200.0, "start_cutoff_hz": 20000, "end_cutoff_hz": 200},
  {"tool": "crossfade_stem", "stem": "vocals", "from_song": "A", "to_song": "B", "start_bar": 0, "duration_bars": 16, "a_fade_out_bars": 8, "curve": "equal_power"},
  {"tool": "crossfade_stem", "stem": "drums", "from_song": "A", "to_song": "B", "start_bar": 0, "duration_bars": 16, "a_fade_out_bars": 8, "curve": "equal_power"},
  {"tool": "crossfade_stem", "stem": "bass", "from_song": "A", "to_song": "B", "start_bar": 0, "duration_bars": 16, "a_fade_out_bars": 8, "curve": "equal_power"},
  {"tool": "crossfade_stem", "stem": "other", "from_song": "A", "to_song": "B", "start_bar": 0, "duration_bars": 16, "a_fade_out_bars": 8, "curve": "equal_power"}
]

Also available: swap_stem — a hard cut of ONE stem at a downbeat (time is in OUTPUT-timeline seconds). Use it to instant-swap drums while the other 3 stems crossfade.

HARD RULES (break one and the whole plan is discarded for a worse fallback):
1. song / from_song / to_song are EXACTLY "A" or "B" — never "Song A", "song_a", "1".
2. Exactly one set_transition_window, and it comes first.
3. Exactly 4 crossfade_stem calls — vocals, drums, bass, other — each from_song "A", to_song "B". Their start_bar / duration_bars / curve may differ; that is how stem-swap styles are built.
4. SEAM HEADROOM: from_song_time_start <= A.max_seam_time AND to_song_time_start <= B.max_seam_time. These are pre-computed with the full safety buffer baked in — use them literally, never derive your own headroom from duration/bpm.
5. Hard cuts, swap_stem, and Drop Swap must land inside a vocal_safe_region.
6. KEYS: pitch-shift B only if A and B camelot_key are neither equal nor adjacent (adjacent = same number with the other letter, or number ±1 with the same letter). When they clash, prefer temporary_pitch_shift on B; permanent pitch_shift is capped at ±2 semitones. Compatible keys → no pitch shift at all.
7. TEMPO: if bpm differs enough to matter, set_tempo_ramp on B from A's bpm to B's bpm.
8. Output ONLY a JSON list of tool-call objects — no prose, no wrapper key.
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

        system_instruction = SYSTEM_PROMPT.replace("{tools_schema}", tools_schema)
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
