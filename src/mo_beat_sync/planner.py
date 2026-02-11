from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .db import MatrixOneClient
from .models import BeatSlot, PlannedClip


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _interp_curve_value(
    times: Sequence[float],
    values: Sequence[float],
    target_s: float,
    default_value: float,
) -> float:
    if not times or not values:
        return float(default_value)
    count = min(len(times), len(values))
    if count <= 0:
        return float(default_value)
    x = np.asarray(times[:count], dtype=np.float32)
    y = np.asarray(values[:count], dtype=np.float32)
    if x.size == 0 or y.size == 0:
        return float(default_value)
    return float(np.interp(target_s, x, y))


def _compute_boundary_strength(
    structure_times: Sequence[float],
    structure_strengths: Sequence[float],
    slot_start_s: float,
    window_s: float = 0.45,
) -> float:
    if not structure_times or not structure_strengths:
        return 0.0
    count = min(len(structure_times), len(structure_strengths))
    if count <= 0:
        return 0.0

    best = 0.0
    for idx in range(count):
        ts_s = float(structure_times[idx])
        dist_s = abs(slot_start_s - ts_s)
        if dist_s > window_s:
            continue
        weight = 1.0 - (dist_s / max(window_s, 1e-6))
        strength = _clamp01(float(structure_strengths[idx]))
        best = max(best, strength * weight)
    return _clamp01(best)


def build_beat_slots(
    beat_times: Sequence[float],
    beat_strengths: Sequence[float],
    tension_times: Sequence[float],
    tension_values: Sequence[float],
    structure_times: Sequence[float],
    structure_strengths: Sequence[float],
    duration_limit_s: float,
    beats_per_clip: int,
    min_slot_s: float,
) -> List[BeatSlot]:
    if duration_limit_s <= 0:
        return []

    filtered = [(t, beat_strengths[idx]) for idx, t in enumerate(beat_times) if t <= duration_limit_s]
    if not filtered:
        slot_len = max(min_slot_s, 0.8)
        slots = []
        start = 0.0
        slot_idx = 0
        while start < duration_limit_s - 1e-6:
            end = min(duration_limit_s, start + slot_len)
            slots.append(
                BeatSlot(
                    slot_idx=slot_idx,
                    start_s=start,
                    end_s=end,
                    duration_s=end - start,
                    target_energy=0.5,
                    target_tension=0.5,
                    boundary_strength=0.0,
                )
            )
            start = end
            slot_idx += 1
        return slots

    times = [x[0] for x in filtered]
    strengths = [x[1] for x in filtered]

    boundaries = [0.0]
    step = max(1, beats_per_clip)
    for idx in range(step - 1, len(times), step):
        ts = times[idx]
        if ts - boundaries[-1] >= min_slot_s:
            boundaries.append(ts)

    for idx, ts in enumerate(structure_times):
        if ts <= 0.0 or ts >= duration_limit_s:
            continue
        strength = float(structure_strengths[idx]) if idx < len(structure_strengths) else 0.0
        if strength >= 0.30:
            boundaries.append(float(ts))

    if duration_limit_s - boundaries[-1] >= 0.25:
        boundaries.append(duration_limit_s)

    if len(boundaries) < 2:
        boundaries = [0.0, duration_limit_s]

    boundaries = sorted(set(float(x) for x in boundaries))

    merged: List[float] = [boundaries[0]]
    for x in boundaries[1:]:
        if x - merged[-1] < min_slot_s and x < duration_limit_s - 1e-6:
            continue
        merged.append(x)

    if merged[-1] < duration_limit_s:
        merged[-1] = duration_limit_s

    beat_intervals = [b - a for a, b in zip(times, times[1:]) if b - a > 1e-6]
    median_beat_interval = float(np.median(beat_intervals)) if beat_intervals else max(0.3, min_slot_s)
    soft_max_slot_s = max(min_slot_s * 2.0, median_beat_interval * max(3, 2 * beats_per_clip))

    expanded: List[float] = [merged[0]]
    for idx in range(len(merged) - 1):
        start_s = merged[idx]
        end_s = merged[idx + 1]
        span_s = end_s - start_s
        if span_s <= soft_max_slot_s:
            if end_s - expanded[-1] >= min_slot_s * 0.90 or end_s >= duration_limit_s - 1e-6:
                expanded.append(end_s)
            continue

        pieces = max(1, int(math.ceil(span_s / soft_max_slot_s)))
        for piece_idx in range(1, pieces + 1):
            cut_s = start_s + span_s * (piece_idx / pieces)
            if cut_s - expanded[-1] >= min_slot_s * 0.90 or piece_idx == pieces:
                expanded.append(min(duration_limit_s, cut_s))
    merged = expanded

    slots: List[BeatSlot] = []
    global_energy = sum(strengths) / max(len(strengths), 1)
    global_tension = (
        float(sum(tension_values) / max(len(tension_values), 1)) if tension_values else float(global_energy)
    )
    for slot_idx in range(len(merged) - 1):
        start_s = merged[slot_idx]
        end_s = merged[slot_idx + 1]
        if end_s - start_s < 0.25:
            continue

        vals = [s for t, s in filtered if start_s <= t < end_s]
        target_energy = sum(vals) / len(vals) if vals else global_energy
        target_tension = _interp_curve_value(
            times=tension_times,
            values=tension_values,
            target_s=(start_s + end_s) / 2.0,
            default_value=global_tension,
        )
        boundary_strength = 0.0
        if slot_idx > 0:
            boundary_strength = _compute_boundary_strength(
                structure_times=structure_times,
                structure_strengths=structure_strengths,
                slot_start_s=start_s,
            )

        slots.append(
            BeatSlot(
                slot_idx=slot_idx,
                start_s=float(start_s),
                end_s=float(end_s),
                duration_s=float(end_s - start_s),
                target_energy=_clamp01(float(target_energy)),
                target_tension=_clamp01(float(target_tension)),
                boundary_strength=_clamp01(float(boundary_strength)),
            )
        )
    return slots


def _to_embedding_literal(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    text = str(value)
    if not text:
        return None
    return text


@dataclass
class _BeamState:
    total_score: float
    planned: List[PlannedClip]
    used_clip_ids: frozenset[int]
    prev_embedding: str | None
    prev_movie: str | None
    prev_motion: float | None


def _collect_candidates(
    db: MatrixOneClient,
    slot: BeatSlot,
    movie_checksums: Sequence[str],
    feature_sig: str,
    prev_embedding: str | None,
    candidate_limit: int,
    max_candidates: int,
) -> List[dict]:
    seen: set[int] = set()
    picked: List[dict] = []
    boundary_strength = _clamp01(slot.boundary_strength)
    for tol in (0.20, 0.35, 0.55):
        rows = db.query_clip_candidates(
            movie_checksums=movie_checksums,
            feature_sig=feature_sig,
            target_duration_s=slot.duration_s,
            target_energy=slot.target_energy,
            target_tension=slot.target_tension,
            boundary_strength=boundary_strength,
            prev_embedding_literal=prev_embedding,
            limit=candidate_limit,
            duration_tol=tol,
        )
        if not rows:
            continue
        for row in rows:
            clip_id = int(row["clip_id"])
            if clip_id in seen:
                continue
            seen.add(clip_id)
            picked.append(row)
            if len(picked) >= max_candidates:
                return picked
    return picked


def plan_clips(
    db: MatrixOneClient,
    slots: Sequence[BeatSlot],
    movie_checksums: Sequence[str],
    feature_sig: str,
    candidate_limit: int = 150,
    beam_width: int = 10,
    per_state_candidates: int = 18,
    slot_skip_penalty: float = 0.45,
    min_reuse_gap: int = 4,
    reuse_penalty: float = 0.14,
) -> List[PlannedClip]:
    if not slots:
        return []
    beam_width = max(1, int(beam_width))
    per_state_candidates = max(4, int(per_state_candidates))

    beam: List[_BeamState] = [
        _BeamState(
            total_score=0.0,
            planned=[],
            used_clip_ids=frozenset(),
            prev_embedding=None,
            prev_movie=None,
            prev_motion=None,
        )
    ]
    query_cache: dict[tuple[int, str], List[dict]] = {}

    for slot in slots:
        boundary_strength = _clamp01(slot.boundary_strength)
        transition_target = 0.10 + 0.80 * boundary_strength
        motion_jump_target = 0.08 + 0.50 * boundary_strength
        prefer_alt_movie = boundary_strength >= 0.60
        next_states: List[_BeamState] = []

        for state in beam:
            cache_key = (slot.slot_idx, state.prev_embedding or "")
            rows = query_cache.get(cache_key)
            if rows is None:
                rows = _collect_candidates(
                    db=db,
                    slot=slot,
                    movie_checksums=movie_checksums,
                    feature_sig=feature_sig,
                    prev_embedding=state.prev_embedding,
                    candidate_limit=candidate_limit,
                    max_candidates=per_state_candidates,
                )
                query_cache[cache_key] = rows
            if not rows:
                continue

            for row in rows[:per_state_candidates]:
                clip_id = int(row["clip_id"])
                repeat_pen = 0.0
                last_use_slot = None
                for old in reversed(state.planned):
                    if old.clip_id == clip_id:
                        last_use_slot = old.slot_idx
                        break
                if last_use_slot is not None:
                    gap = slot.slot_idx - last_use_slot
                    if gap < max(1, min_reuse_gap):
                        continue
                    repeat_pen = reuse_penalty * (1.0 + 1.0 / max(1.0, float(gap)))

                movie_path = str(row["movie_path"])
                row_score = float(row["score"])
                vec_distance = float(row.get("vec_distance", 0.0))
                transition_distance = float(row.get("transition_distance", abs(vec_distance - transition_target)))
                motion = float(row.get("motion_energy", 0.0))
                motion_jump = abs(motion - state.prev_motion) if state.prev_motion is not None else 0.0
                motion_jump_distance = abs(motion_jump - motion_jump_target)
                movie_penalty = 0.0
                if state.prev_movie is not None and movie_path == state.prev_movie:
                    movie_penalty = 0.12 if prefer_alt_movie else 0.02
                recent_same_movie = sum(1 for x in state.planned[-3:] if x.movie_path == movie_path)
                if recent_same_movie > 0:
                    movie_penalty += 0.03 * recent_same_movie

                local_score = row_score + 0.18 * motion_jump_distance + movie_penalty + repeat_pen

                source_duration = float(row["duration_s"])
                target_duration = float(slot.duration_s)
                stretch_rate = source_duration / max(target_duration, 1e-6)
                stretch_rate = max(0.90, min(1.10, stretch_rate))
                planned_clip = PlannedClip(
                    slot_idx=slot.slot_idx,
                    clip_id=clip_id,
                    movie_path=movie_path,
                    start_s=float(row["start_s"]),
                    end_s=float(row["end_s"]),
                    source_duration_s=source_duration,
                    target_duration_s=target_duration,
                    stretch_rate=float(stretch_rate),
                    score=float(local_score),
                    vec_distance=float(row.get("vec_distance", 0.0)),
                    energy_distance=float(row.get("energy_distance", 0.0)),
                    tension_distance=float(row.get("tension_distance", 0.0)),
                    transition_distance=float(transition_distance),
                    boundary_strength=float(boundary_strength),
                )
                next_states.append(
                    _BeamState(
                        total_score=state.total_score + local_score,
                        planned=state.planned + [planned_clip],
                        used_clip_ids=state.used_clip_ids.union({clip_id}),
                        prev_embedding=_to_embedding_literal(row.get("embedding")),
                        prev_movie=movie_path,
                        prev_motion=motion,
                    )
                )

        if not next_states:
            # Keep search alive when a slot has no viable candidate.
            for state in beam:
                next_states.append(
                    _BeamState(
                        total_score=state.total_score + slot_skip_penalty,
                        planned=state.planned,
                        used_clip_ids=state.used_clip_ids,
                        prev_embedding=state.prev_embedding,
                        prev_movie=state.prev_movie,
                        prev_motion=state.prev_motion,
                    )
                )

        next_states.sort(key=lambda s: (-len(s.planned), s.total_score))
        beam = next_states[:beam_width]

    if not beam:
        return []
    best = sorted(beam, key=lambda s: (-len(s.planned), s.total_score))[0]
    return best.planned
