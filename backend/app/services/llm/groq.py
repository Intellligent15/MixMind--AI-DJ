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

SYSTEM_PROMPT = """You are an expert DJ planning a seamless transition from track A (outgoing) into track B (incoming). A great transition is varied and musical — NOT the same long crossfade every time.

INPUT — for each of A and B you get:
- bpm, key, camelot_key, time_signature, seconds_per_bar, duration
- max_seam_time: the latest time you may leave A / enter B. A hard ceiling (see Hard rules).
- sections: [{"start","end","energy"}] — your structural map. energy is 0..1 normalized to the song's hottest section: ~1.0 = a drop or chorus, low values = intro, breakdown, or outro.
- vocal_safe_regions: [{"start","end"}] — spans with NO vocals. The ONLY places a hard cut or stem swap may land.

PLAN IN THIS ORDER:
1. Choose A's OUT point (from_song_time_start): a downbeat LATE in A — inside A's final or second-to-last section — at or before A.max_seam_time. This is where A bows out.
2. Choose B's IN point (to_song_time_start): a downbeat EARLY in B — its intro or its FIRST energy rise (first drop/chorus), normally within B's first 1–3 sections. The listener should hear almost all of B, so NEVER enter B in its back half. B.max_seam_time is only a hard ceiling, NOT a target — entering anywhere near it means B is already ending. Match energy by blending A's tail into B's FIRST high-energy section, or dropping B's intro in where A has gone quiet.
3. Pick the style whose TRIGGER fits these two songs (next section). Different pairs must hit different styles — variety across the set is the goal.
4. Emit that style's tool calls.

TRANSITION STYLES — reason from the section energies AT YOUR SEAM: A_out = energy of A's out-section, B_in = energy of B's in-section. Pick the FIRST style whose trigger fits; emit its calls (don't name it). Do NOT default to one style across the set.
- Drop Swap — TRIGGER: B_in >= 0.95 (B enters on a drop) AND A's OUT downbeat sits inside a vocal_safe_region. All 4 crossfade_stem: same start_bar on that safe downbeat, duration_bars=2..4, equal_power. Snaps A→B at the drop.
- Drum-Bridge — TRIGGER: both drum-driven at the seam (A_out >= 0.7 AND B_in >= 0.7). vocals/bass/other: start_bar=0, duration_bars=8; drums: start_bar=4, duration_bars=12, so A's drums hold while B's come in early and bridge the two grids.
- Filter Sweep Out — TRIGGER: A's tail is hot but B comes in calmer (A_out >= 0.8 AND B_in <= 0.6). One filter_sweep on A: lowpass, start_time = seam minus 8 bars, end_time = seam, start_cutoff_hz=20000, end_cutoff_hz=200, plus a normal 16-bar 4-stem crossfade. Do NOT pitch_shift B here.
- Loop & Echo Trail — TRIGGER: A's last 8 bars are inside a vocal_safe_region AND B intros soft (B_in <= 0.5). One loop_section on A's last 4 bars (beats=16, repeats=2, bpm=A.bpm), then one echo_out at the loop end (beats=4, feedback=0.5, bpm=A.bpm); the 4 crossfade_stem calls can be short (duration_bars=4).
- Vocal-First Out — TRIGGER: A's OUT point still has vocals (no vocal_safe_region there) and a strong hook to clear before B enters. vocals: start_bar=0, duration_bars=8; drums/bass/other: start_bar=8, duration_bars=8. A's instrumental hands off after its vocal is gone.
- Classic Blend — FALLBACK only, when no trigger above fits. All 4 crossfade_stem: start_bar=0, duration_bars=16, equal_power. Don't pick this just because it's safe.

DON'T LET A LINGER: by default A only goes silent when B reaches full, so a long crossfade keeps A audible across the whole overlap and muddies the mix. Set `a_fade_out_bars` shorter than `duration_bars` so A clears out early while B keeps swelling in on its own — e.g. duration_bars=16, a_fade_out_bars=8 means A is gone halfway through while B keeps rising. Use this on most transitions; reserve a_fade_out_bars == duration_bars for when you genuinely want A and B locked together to the end. Keep crossfades punchy (8–16 bars), don't reflexively max the duration.

Available tools:
{tools_schema}

WORKED EXAMPLES (copy these shapes; change the numbers to fit the songs):

Example — Drum-Bridge, A and B both 128 BPM (seconds_per_bar 1.875), seam at A 180.0s / B 32.0s:
{"plan": [
  {"tool": "set_transition_window", "from_song_time_start": 180.0, "to_song_time_start": 32.0, "duration_bars": 16},
  {"tool": "crossfade_stem", "stem": "vocals", "from_song": "A", "to_song": "B", "start_bar": 0, "duration_bars": 8, "curve": "equal_power"},
  {"tool": "crossfade_stem", "stem": "bass", "from_song": "A", "to_song": "B", "start_bar": 0, "duration_bars": 8, "curve": "equal_power"},
  {"tool": "crossfade_stem", "stem": "other", "from_song": "A", "to_song": "B", "start_bar": 0, "duration_bars": 8, "curve": "equal_power"},
  {"tool": "crossfade_stem", "stem": "drums", "from_song": "A", "to_song": "B", "start_bar": 4, "duration_bars": 12, "curve": "equal_power"}
]}

Example — Filter Sweep Out, A 128 BPM bright, B 124 BPM darker, seam at A 200.0s / B 16.0s:
{"plan": [
  {"tool": "set_transition_window", "from_song_time_start": 200.0, "to_song_time_start": 16.0, "duration_bars": 16},
  {"tool": "set_tempo_ramp", "song": "B", "start_time": 47.0, "end_time": 78.0, "start_bpm": 128.0, "end_bpm": 124.0},
  {"tool": "filter_sweep", "song": "A", "type": "lowpass", "start_time": 185.0, "end_time": 200.0, "start_cutoff_hz": 20000, "end_cutoff_hz": 200},
  {"tool": "crossfade_stem", "stem": "vocals", "from_song": "A", "to_song": "B", "start_bar": 0, "duration_bars": 16, "a_fade_out_bars": 8, "curve": "equal_power"},
  {"tool": "crossfade_stem", "stem": "drums", "from_song": "A", "to_song": "B", "start_bar": 0, "duration_bars": 16, "a_fade_out_bars": 8, "curve": "equal_power"},
  {"tool": "crossfade_stem", "stem": "bass", "from_song": "A", "to_song": "B", "start_bar": 0, "duration_bars": 16, "a_fade_out_bars": 8, "curve": "equal_power"},
  {"tool": "crossfade_stem", "stem": "other", "from_song": "A", "to_song": "B", "start_bar": 0, "duration_bars": 16, "a_fade_out_bars": 8, "curve": "equal_power"}
]}

Also available: swap_stem — a hard cut of ONE stem at a downbeat (time is in OUTPUT-timeline seconds). Use it to instant-swap drums while the other 3 stems crossfade.

HARD RULES (break one and the whole plan is discarded for a worse fallback):
1. song / from_song / to_song are EXACTLY "A" or "B" — never "Song A", "song_a", "1".
2. Exactly one set_transition_window, and it comes first.
3. Exactly 4 crossfade_stem calls — vocals, drums, bass, other — each from_song "A", to_song "B". Their start_bar / duration_bars / curve may differ; that is how stem-swap styles are built.
4. SEAM HEADROOM: from_song_time_start <= A.max_seam_time AND to_song_time_start <= B.max_seam_time. These are pre-computed with the full safety buffer baked in — use them literally, never derive your own headroom from duration/bpm.
5. Hard cuts, swap_stem, and Drop Swap must land inside a vocal_safe_region.
6. KEYS: pitch-shift B only if A and B camelot_key are neither equal nor adjacent (adjacent = same number with the other letter, or number ±1 with the same letter). When they clash, use temporary_pitch_shift on B (<= 2 semitones) — it bends B during the overlap and returns it to its real key before the transition ends. NEVER use permanent pitch_shift: it detunes B for the rest of the song, so when B is later mixed OUT into the next track it snaps back to its real pitch at the stitch and you hear a glitch. Compatible keys → no pitch shift at all.
7. TEMPO (beatmatch): if A.bpm and B.bpm differ by more than ~2%, add ONE set_tempo_ramp on B. B is auto-stretched to A's bpm at the seam, so it is beat-locked while A plays — KEEP it locked for the whole crossfade, then ramp it up to its own tempo only AFTER A is gone. Set start_time = to_song_time_start + N × B.seconds_per_bar, where N = the largest (start_bar + duration_bars) across your crossfade_stem calls (the bar where the last stem finishes); end_time = start_time + 16 × B.seconds_per_bar; start_bpm = A.bpm; end_bpm = B.bpm. NEVER start the ramp at the seam — that speeds B up while A is still playing and the beats drift apart.
8. Output ONLY a JSON object of the form {"plan": [ ...tool-call objects... ]} — no prose.
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

        # Cache key includes the model and the formatted system prompt so
        # swapping either invalidates stale plans automatically. Without
        # this, a prompt fix would never reach pairs that already have a
        # cached (wrong) plan.
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
            logger.info("GroqProvider: cache hit for %s", cache_key)
            data = await storage.read(cache_key)
            return json.loads(data.decode("utf-8"))["response"]

        logger.info("GroqProvider: cache miss for %s, calling LLM", cache_key)

        client = Groq(api_key=self.api_key, timeout=GROQ_TIMEOUT_SECONDS)
        # Per the system prompt, the songs are labeled "A" and "B"
        # everywhere — including the user message keys. The model parrots
        # whatever labels the prompt uses, so keep them as "A"/"B" to
        # reinforce the tool-call field requirement.
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
