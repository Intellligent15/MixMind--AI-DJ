from typing import Any


def _get_envelope_value(envelope: dict[str, Any], time_sec: float) -> float:
    """Get the RMS value of the envelope at a specific time."""
    hop_seconds = envelope.get("hop_seconds", 0.1)
    rms_list = envelope.get("rms", [])
    if not rms_list:
        return 0.0
    
    idx = int(time_sec / hop_seconds)
    if idx < 0:
        return 0.0
    if idx >= len(rms_list):
        return rms_list[-1]
    return rms_list[idx]


def vocal_safe_regions(
    transcription_segments: list[dict[str, Any]],
    envelope: dict[str, Any],
    aligned_words: list[dict[str, Any]] | None = None,
    word_prob_min: float = 0.35,
    segment_logprob_min: float = -1.2,
    stem_rms_presence: float = 0.02,
    stem_rms_quiet: float = 0.01,
    min_safe_region_seconds: float = 1.5,
    duration_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Compute vocal-safe regions for a song.
    If aligned_words is provided, it uses the alignment output to filter out Whisper hallucinations.
    """
    hop_seconds = envelope.get("hop_seconds", 0.1)
    rms_list = envelope.get("rms", [])
    total_frames = len(rms_list)
    if not total_frames:
        # Default to entire duration safe if no envelope
        return [{"start": 0.0, "end": duration_seconds, "safe": True, "reason": "no_envelope"}]
        
    duration = duration_seconds or (total_frames * hop_seconds)
    
    usable_intervals = []
    
    # 1. Use aligned_words if we have them (authoritative)
    if aligned_words is not None:
        for w in aligned_words:
            # We trust words that were matched or substituted from Genius
            if w.get("source") in ("whisper_match", "whisper_substitution", "interpolated"):
                start = w.get("start")
                end = w.get("end")
                if start is not None and end is not None:
                    # We trust words that were matched or substituted from Genius.
                    # Do not do a secondary RMS check here because a whisper or highly
                    # compressed vocal might dip below the RMS threshold. The sequence
                    # alignment acts as our definitive ground truth.
                    usable_intervals.append((start, end))
                        
    else:
        # 2. Fall back to raw Whisper + RMS heuristic
        for seg in transcription_segments:
            if seg.get("avg_logprob", 0.0) < segment_logprob_min:
                continue
                
            for w in seg.get("words", []):
                if w.get("probability", 0.0) < word_prob_min:
                    continue
                    
                start = w.get("start", 0.0)
                end = w.get("end", 0.0)
                mid = (start + end) / 2
                rms_val = _get_envelope_value(envelope, mid)
                if rms_val >= stem_rms_presence:
                    usable_intervals.append((start, end))

    # Sort and merge usable intervals
    usable_intervals.sort()
    merged = []
    for interval in usable_intervals:
        if not merged:
            merged.append(interval)
        else:
            last = merged[-1]
            # merge if overlapping or extremely close (e.g., < 0.2s)
            if interval[0] <= last[1] + 0.2:
                merged[-1] = (last[0], max(last[1], interval[1]))
            else:
                merged.append(interval)
                
    # Now invert merged intervals to find safe gaps
    safe_regions = []
    current_time = 0.0
    
    for start, end in merged:
        gap_start = current_time
        gap_end = start
        
        if gap_end - gap_start >= min_safe_region_seconds:
            # Check RMS quietness in this gap. Allow up to 15% of frames to be slightly noisy
            # to account for Demucs stem bleed (e.g. loud synths/drums leaking into vocals).
            noisy_frames = 0
            total_frames_in_gap = 0
            for i in range(int(gap_start / hop_seconds), int(gap_end / hop_seconds)):
                if i < len(rms_list):
                    total_frames_in_gap += 1
                    if rms_list[i] >= stem_rms_quiet:
                        noisy_frames += 1
            
            is_quiet = total_frames_in_gap == 0 or (noisy_frames / total_frames_in_gap) < 0.15
            
            if is_quiet:
                safe_regions.append({
                    "start": gap_start,
                    "end": gap_end,
                    "safe": True,
                    "reason": "quiet_gap"
                })
                
        current_time = end
        
    # Check final gap
    gap_start = current_time
    gap_end = duration
    if gap_end - gap_start >= min_safe_region_seconds:
        noisy_frames = 0
        total_frames_in_gap = 0
        for i in range(int(gap_start / hop_seconds), int(gap_end / hop_seconds)):
            if i < len(rms_list):
                total_frames_in_gap += 1
                if rms_list[i] >= stem_rms_quiet:
                    noisy_frames += 1
                    
        is_quiet = total_frames_in_gap == 0 or (noisy_frames / total_frames_in_gap) < 0.15
        
        if is_quiet:
            safe_regions.append({
                "start": gap_start,
                "end": gap_end,
                "safe": True,
                "reason": "quiet_gap"
            })
            
    return safe_regions
