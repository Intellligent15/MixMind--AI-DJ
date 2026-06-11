"""Planner v2 orchestration.

build_plan_v2() is the single entry point the render worker calls:

  1. build the seam-candidate menus (pure math, always valid);
  2. compute the pair facts the model should NOT have to derive
     (tempo gap, key-compatibility verdict);
  3. ask the LLM for a TransitionDecision (style hint / user override /
     previously-used styles ride along as context; a reroll nonce busts
     the response cache);
  4. expand the decision deterministically into tool calls.

Every step that can fail degrades gracefully: a bad decision triggers
one repair attempt (default knobs, keep the model's style if legal),
and total LLM failure falls back to either a pinned-style default
expansion or the v1 deterministic planner. The caller learns which path
produced the plan via PlanOutcome.source — no more silent fallbacks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import ValidationError

from app.services.mixer.archetypes import (
    ArchetypeError,
    camelot_compatible,
    default_decision,
    expand,
)
from app.services.mixer.candidates import (
    PairCandidates,
    build_pair_candidates,
    enrich_sections,
)
from app.services.mixer.decision import TransitionDecision, TransitionStyle
from app.services.mixer.plan import build_pair_plan, compute_pitch_shift
from app.services.mixer.types import AnalysisBundle, MixPlanJSON

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SongMeta:
    """Identity + planning inputs the worker snapshots from the DB."""

    title: str
    artist: str | None
    bundle: AnalysisBundle
    energy_curve: list[float]
    safe_regions: list[dict]


@dataclass(frozen=True)
class PlanOutcome:
    plan: MixPlanJSON
    source: str            # "llm_v2" | "llm_v2_repaired" | "style_default" | "deterministic_fallback"
    style: str | None
    rationale: str | None


def _song_llm_input(meta: SongMeta, candidates: list) -> dict:
    b = meta.bundle
    return {
        "title": meta.title,
        "artist": meta.artist,
        "bpm": b.bpm,
        "key": b.key,
        "camelot_key": b.camelot_key,
        "duration": round(b.duration, 1),
        "sections": enrich_sections(b.sections, meta.energy_curve)[:12],
        "candidates": [c.to_llm_dict() for c in candidates],
    }


def _pair_facts(a: AnalysisBundle, b: AnalysisBundle) -> dict:
    tempo_gap_pct = (
        round(abs(a.bpm - b.bpm) / a.bpm * 100.0, 1) if a.bpm and b.bpm else None
    )
    compatible = camelot_compatible(a.camelot_key, b.camelot_key)
    if compatible:
        key_verdict = "compatible — no pitch handling needed"
    else:
        try:
            delta = compute_pitch_shift(a.key, b.key)
        except ValueError:
            delta = 0
        key_verdict = (
            f"clash — B will be held {delta:+d} semitones in A's key during "
            f"the blend, automatically"
        )
    return {
        "tempo_gap_percent": tempo_gap_pct,
        "tempo_note": "B is beatmatched to A automatically; ignore tempo math",
        "key_verdict": key_verdict,
    }


def _parse_decision(obj) -> TransitionDecision:
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        raise ValueError(f"decision is not a JSON object: {type(obj).__name__}")
    # Some models wrap: {"decision": {...}} / {"plan": {...}}.
    for key in ("decision", "plan", "transition"):
        if key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    return TransitionDecision.model_validate(obj)


def _repair_decision(
    obj, candidates: PairCandidates
) -> TransitionDecision:
    """Second chance for a near-miss decision: keep whatever fields are
    legal, default the rest."""
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        raise ValueError("irreparable decision payload")
    style = None
    raw_style = obj.get("style")
    if isinstance(raw_style, str):
        try:
            style = TransitionStyle(raw_style.strip().lower())
        except ValueError:
            style = None
    base = default_decision(candidates, style=style)
    out = obj.get("out")
    in_ = obj.get("in")
    data = base.model_dump(by_alias=True)
    if isinstance(out, str) and candidates.find(out) and out.startswith("A"):
        data["out"] = out
    if isinstance(in_, str) and candidates.find(in_) and in_.startswith("B"):
        data["in"] = in_
    if isinstance(obj.get("rationale"), str):
        data["rationale"] = obj["rationale"][:600]
    return TransitionDecision.model_validate(data)


async def build_plan_v2(
    provider,
    a: SongMeta,
    b: SongMeta,
    *,
    style_hint: str | None = None,
    style_override: str | None = None,
    previous_styles: list[str] | None = None,
    nonce: int = 0,
) -> PlanOutcome:
    # Lazy import keeps the mixer package importable without the llm
    # package's heavy provider dependencies (pure unit tests, tooling).
    from app.services.llm.prompts import (
        DECISION_SYSTEM_PROMPT,
        decision_user_prompt,
    )

    candidates = build_pair_candidates(
        a.bundle, b.bundle,
        a.energy_curve, b.energy_curve,
        a.safe_regions, b.safe_regions,
    )

    pinned_style: TransitionStyle | None = None
    if style_override:
        try:
            pinned_style = TransitionStyle(style_override)
        except ValueError:
            logger.warning("planner_v2: unknown style_override %r", style_override)

    if not candidates.out_candidates or not candidates.in_candidates:
        logger.warning(
            "planner_v2: no usable seam candidates; deterministic fallback"
        )
        return PlanOutcome(
            plan=build_pair_plan(a.bundle, b.bundle),
            source="deterministic_fallback",
            style=None,
            rationale="songs too short for candidate generation",
        )

    context: dict = {}
    if pinned_style is not None:
        context["forced_style"] = (
            f"The user pinned this transition's style to '{pinned_style.value}'. "
            f"You MUST use it; choose only the seams and knobs."
        )
    elif style_hint:
        context["suggested_style"] = style_hint
    if previous_styles:
        context["styles_used_so_far"] = previous_styles

    user = decision_user_prompt(
        _song_llm_input(a, candidates.out_candidates),
        _song_llm_input(b, candidates.in_candidates),
        _pair_facts(a.bundle, b.bundle),
        context or None,
    )

    decision: TransitionDecision | None = None
    source = "llm_v2"
    try:
        obj = await provider.complete_json(
            system=DECISION_SYSTEM_PROMPT, user=user, nonce=nonce
        )
        try:
            decision = _parse_decision(obj)
        except (ValidationError, ValueError) as exc:
            logger.warning("planner_v2: repairing decision (%s)", exc)
            decision = _repair_decision(obj, candidates)
            source = "llm_v2_repaired"
    except Exception as exc:
        logger.error("planner_v2: LLM decision failed: %s", exc)

    if decision is not None and pinned_style is not None:
        if decision.style != pinned_style:
            decision = decision.model_copy(update={"style": pinned_style})
            decision = decision.model_copy(
                update={"duration_bars": decision.normalized_duration()}
            )

    if decision is not None:
        try:
            plan = expand(decision, a.bundle, b.bundle, candidates)
            return PlanOutcome(
                plan=plan, source=source,
                style=decision.style.value,
                rationale=decision.rationale or None,
            )
        except ArchetypeError as exc:
            logger.error("planner_v2: expansion failed (%s); using defaults", exc)

    # LLM unavailable or decision unusable: pinned style still expands
    # deterministically; otherwise fall back to the v1 planner.
    try:
        fallback = default_decision(candidates, style=pinned_style)
        plan = expand(fallback, a.bundle, b.bundle, candidates)
        return PlanOutcome(
            plan=plan, source="style_default",
            style=fallback.style.value, rationale=fallback.rationale,
        )
    except ArchetypeError as exc:
        logger.error("planner_v2: default expansion failed (%s)", exc)
        return PlanOutcome(
            plan=build_pair_plan(a.bundle, b.bundle),
            source="deterministic_fallback",
            style=None, rationale=None,
        )
