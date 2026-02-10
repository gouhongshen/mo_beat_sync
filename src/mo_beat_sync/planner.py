from __future__ import annotations

import math
from typing import List, Sequence

from .db import MatrixOneClient
from .models import BeatSlot, PlannedClip


def build_beat_slots(
    beat_times: Sequence[float],
    beat_strengths: Sequence[float],
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
            slots.append(BeatSlot(slot_idx=slot_idx, start_s=start, end_s=end, duration_s=end - start, target_energy=0.5))
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

    if duration_limit_s - boundaries[-1] >= 0.25:
        boundaries.append(duration_limit_s)

    if len(boundaries) < 2:
        boundaries = [0.0, duration_limit_s]

    merged: List[float] = [boundaries[0]]
    for x in boundaries[1:]:
        if x - merged[-1] < min_slot_s and x < duration_limit_s - 1e-6:
            continue
        merged.append(x)

    if merged[-1] < duration_limit_s:
        merged[-1] = duration_limit_s

    slots: List[BeatSlot] = []
    global_energy = sum(strengths) / max(len(strengths), 1)
    for slot_idx in range(len(merged) - 1):
        start_s = merged[slot_idx]
        end_s = merged[slot_idx + 1]
        if end_s - start_s < 0.25:
            continue

        vals = [s for t, s in filtered if start_s <= t < end_s]
        target_energy = sum(vals) / len(vals) if vals else global_energy

        slots.append(
            BeatSlot(
                slot_idx=slot_idx,
                start_s=float(start_s),
                end_s=float(end_s),
                duration_s=float(end_s - start_s),
                target_energy=float(target_energy),
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


def plan_clips(
    db: MatrixOneClient,
    run_id: int,
    slots: Sequence[BeatSlot],
    candidate_limit: int = 150,
) -> List[PlannedClip]:
    selected: List[PlannedClip] = []
    used_clip_ids: set[int] = set()
    prev_embedding: str | None = None
    prev_movie: str | None = None

    for slot in slots:
        best_row = None
        for tol in (0.20, 0.35, 0.55):
            rows = db.query_clip_candidates(
                run_id=run_id,
                target_duration_s=slot.duration_s,
                target_energy=slot.target_energy,
                prev_embedding_literal=prev_embedding,
                limit=candidate_limit,
                duration_tol=tol,
            )
            if not rows:
                continue

            first_different_movie = None
            first_any = None
            for row in rows:
                clip_id = int(row["clip_id"])
                if clip_id in used_clip_ids:
                    continue
                if first_any is None:
                    first_any = row
                if prev_movie is None or row["movie_path"] != prev_movie:
                    first_different_movie = row
                    break

            best_row = first_different_movie or first_any
            if best_row is not None:
                break

        if best_row is None:
            continue

        clip_id = int(best_row["clip_id"])
        source_duration = float(best_row["duration_s"])
        target_duration = float(slot.duration_s)
        stretch_rate = source_duration / max(target_duration, 1e-6)
        stretch_rate = max(0.90, min(1.10, stretch_rate))

        selected.append(
            PlannedClip(
                slot_idx=slot.slot_idx,
                clip_id=clip_id,
                movie_path=str(best_row["movie_path"]),
                start_s=float(best_row["start_s"]),
                end_s=float(best_row["end_s"]),
                source_duration_s=source_duration,
                target_duration_s=target_duration,
                stretch_rate=float(stretch_rate),
                score=float(best_row["score"]),
                vec_distance=float(best_row["vec_distance"]),
                energy_distance=float(best_row["energy_distance"]),
            )
        )

        used_clip_ids.add(clip_id)
        prev_movie = str(best_row["movie_path"])
        prev_embedding = _to_embedding_literal(best_row.get("embedding"))

    return selected
