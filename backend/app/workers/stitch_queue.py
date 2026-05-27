import asyncio
import logging
import uuid
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

from sqlalchemy import select, update

from app.core.db import SessionLocal
from app.models import (
    Analysis,
    MixPlan,
    MixPlanStatus,
    Queue,
    QueueItem,
    QueueRender,
    QueueRenderStatus,
    Song,
)
from app.services.storage import get_storage
from app.workers import celery_app

logger = logging.getLogger(__name__)

CLAIMABLE_STATUSES = (
    QueueRenderStatus.pending,
    QueueRenderStatus.failed,
)


def _get_mix0_sample(plan_json: list[dict], rate_A: float, T_orig: float, sr: int = 44100) -> int:
    window = next(c for c in plan_json if c["tool"] == "set_transition_window")
    tempo_ramp = next((c for c in plan_json if c["tool"] == "set_tempo_ramp"), None)

    a_seam_orig = window["from_song_time_start"]
    b_seam_orig = window["to_song_time_start"]

    a_seam_samp = int(a_seam_orig * sr)
    b_seam_samp_post = int(b_seam_orig * sr / rate_A)

    if tempo_ramp:
        ramp_start_orig = tempo_ramp["start_time"]
        ramp_end_orig = tempo_ramp["end_time"]
        ramp_start_samp = int(ramp_start_orig * sr)
        ramp_end_samp = int(ramp_end_orig * sr)
        ramp_len = ramp_end_samp - ramp_start_samp

        num_points = 10
        t_source = np.linspace(0, ramp_len, num_points)
        rates = np.linspace(rate_A, 1.0, num_points)
        t_target = 0.0
        for i in range(1, num_points):
            dt = t_source[i] - t_source[i-1]
            avg_rate = (rates[i] + rates[i-1]) / 2.0
            t_target += dt / avg_rate

        if T_orig >= ramp_end_orig:
            ramp_end_target = int(ramp_start_samp / rate_A) + int(t_target)
            stretched_B_sample = ramp_end_target + int((T_orig - ramp_end_orig) * sr)
        elif T_orig >= ramp_start_orig:
            fraction = (T_orig - ramp_start_orig) / (ramp_end_orig - ramp_start_orig)
            stretched_B_sample = int(ramp_start_samp / rate_A) + int(t_target * fraction)
        else:
            stretched_B_sample = int(T_orig * sr / rate_A)
    else:
        stretched_B_sample = int(T_orig * sr / rate_A)

    return stretched_B_sample - b_seam_samp_post + a_seam_samp


def _get_mix1_sample(T_orig: float, sr: int = 44100) -> int:
    return int(T_orig * sr)


@celery_app.task(name="app.workers.stitch_queue.stitch_queue")
def stitch_queue(queue_id: str) -> str | None:
    queue_uuid = uuid.UUID(queue_id)
    storage = get_storage()

    with SessionLocal() as db:
        render_row = db.scalar(select(QueueRender).where(QueueRender.queue_id == queue_uuid))
        if render_row is None:
            logger.warning("stitch_queue: no QueueRender found for %s", queue_id)
            return None

        claim = db.execute(
            update(QueueRender)
            .where(QueueRender.id == render_row.id)
            .where(QueueRender.status.in_(CLAIMABLE_STATUSES))
            .values(status=QueueRenderStatus.rendering, error_text=None)
        )
        db.commit()
        if claim.rowcount == 0:
            db.refresh(render_row)
            logger.info("stitch_queue: %s already %s, skipping", queue_id, render_row.status.value)
            return None

        render_row_id = render_row.id

        # Fetch all queue items
        items = db.scalars(
            select(QueueItem)
            .where(QueueItem.queue_id == queue_uuid)
            .order_by(QueueItem.position)
        ).all()
        
        if len(items) < 2:
            _mark_failed(render_row_id, "Queue must have at least 2 songs")
            return None

        # Fetch MixPlans
        mix_plans = []
        for i in range(len(items) - 1):
            mp = db.scalar(
                select(MixPlan)
                .where(MixPlan.queue_id == queue_uuid)
                .where(MixPlan.from_song_id == items[i].song_id)
                .where(MixPlan.to_song_id == items[i+1].song_id)
            )
            if not mp or mp.status != MixPlanStatus.ready or not mp.rendered_audio_path:
                _mark_failed(render_row_id, f"MixPlan for pair {i} not ready")
                return None
            mix_plans.append(mp)

        # Fetch Analyses for BPMs
        analyses = {}
        for item in items:
            an = db.scalar(select(Analysis).where(Analysis.song_id == item.song_id))
            if not an:
                _mark_failed(render_row_id, f"Analysis missing for song {item.song_id}")
                return None
            analyses[item.song_id] = an

    # Now stitch!
    sr = 44100
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        
        # Download all WAVs
        wav_paths = []
        for i, mp in enumerate(mix_plans):
            dest = tmp / f"mix_{i}.wav"
            asyncio.run(storage.download_file(mp.rendered_audio_path, dest))
            wav_paths.append(dest)

        # Load all audio into memory (each is ~40MB, total ~800MB for 20 songs, very safe)
        audios = []
        for p in wav_paths:
            y, _ = sf.read(str(p), dtype=np.float32)
            if y.ndim == 1:
                y = np.column_stack((y, y))
            audios.append(y)

        stitched = [audios[0]]
        
        for i in range(len(mix_plans) - 1):
            mix0 = stitched[-1] # The accumulated mix so far (or we just accumulate chunks)
            mix1 = audios[i+1]
            
            mp0 = mix_plans[i]
            mp1 = mix_plans[i+1]
            
            # The song in the middle is items[i+1].song_id
            mid_song_id = items[i+1].song_id
            an_a = analyses[items[i].song_id]
            an_b = analyses[mid_song_id]
            rate_A = an_a.bpm / an_b.bpm
            
            plan0 = mp0.plan_json
            plan1 = mp1.plan_json
            
            tempo_ramp = next((c for c in plan0 if c["tool"] == "set_tempo_ramp"), None)
            window0 = next(c for c in plan0 if c["tool"] == "set_transition_window")
            window1 = next(c for c in plan1 if c["tool"] == "set_transition_window")
            
            a_seam1 = window1["from_song_time_start"]
            
            if tempo_ramp:
                safe_start_T = tempo_ramp["end_time"]
            else:
                # Estimate crossfade end
                b_seam0 = window0["to_song_time_start"]
                dur_bars = window0["duration_bars"]
                sec_per_bar_b = (60.0 / an_b.bpm) * an_b.time_signature
                safe_start_T = b_seam0 + dur_bars * sec_per_bar_b

            safe_end_T = a_seam1
            T_orig = (safe_start_T + safe_end_T) / 2.0
            
            # Prevent overlap failure
            if T_orig > safe_end_T:
                T_orig = safe_end_T - 1.0
                
            S0 = _get_mix0_sample(plan0, rate_A, T_orig, sr)
            S1 = _get_mix1_sample(T_orig, sr)
            
            # To accumulate cleanly, we replace stitched[-1] with its sliced version
            mix0_sliced = mix0[:S0]
            mix1_sliced = mix1[S1:]
            
            # 50ms crossfade
            xfade_samples = int(0.050 * sr)
            if mix0_sliced.shape[0] < xfade_samples or mix1_sliced.shape[0] < xfade_samples:
                xfade_samples = min(mix0_sliced.shape[0], mix1_sliced.shape[0])
                
            if xfade_samples > 0:
                t = np.linspace(0.0, 1.0, xfade_samples, endpoint=False, dtype=np.float32)
                gain0 = np.cos(t * (np.pi / 2.0))
                gain1 = np.sin(t * (np.pi / 2.0))
                
                xfade_region = (
                    gain0[:, None] * mix0_sliced[-xfade_samples:] + 
                    gain1[:, None] * mix1_sliced[:xfade_samples]
                )
                mix0_sliced = mix0_sliced[:-xfade_samples]
                mix1_sliced = mix1_sliced[xfade_samples:]
                stitched[-1] = mix0_sliced
                stitched.append(xfade_region)
                stitched.append(mix1_sliced)
            else:
                stitched[-1] = mix0_sliced
                stitched.append(mix1_sliced)

        final_audio = np.concatenate(stitched)
        
        out_dest = tmp / "final.flac"
        sf.write(str(out_dest), final_audio, sr, format="FLAC", subtype="PCM_16")
        
        with open(out_dest, "rb") as f:
            flac_bytes = f.read()

    key = f"queue_mixes/{queue_id}.flac"
    asyncio.run(storage.write(key, flac_bytes))

    with SessionLocal() as db:
        row = db.get(QueueRender, render_row_id)
        if row:
            row.rendered_audio_path = key
            row.status = QueueRenderStatus.ready
            row.error_text = None
            db.commit()
            
    return queue_id


def _mark_failed(render_id: uuid.UUID, message: str) -> None:
    with SessionLocal() as db:
        row = db.get(QueueRender, render_id)
        if row is None:
            return
        row.status = QueueRenderStatus.failed
        row.error_text = message[:1000]
        db.commit()
