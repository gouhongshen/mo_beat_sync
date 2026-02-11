from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class VideoMeta:
    path: str
    duration_s: float
    fps: float
    width: int
    height: int


@dataclass
class SongAnalysis:
    path: str
    wav_path: str
    duration_s: float
    bpm: float
    beat_times: List[float]
    beat_strengths: List[float]
    tension_times: List[float]
    tension_values: List[float]
    structure_times: List[float]
    structure_strengths: List[float]
    checksum: str = ""
    analysis_sig: str = ""


@dataclass
class ClipFeature:
    movie_path: str
    start_s: float
    end_s: float
    duration_s: float
    motion_energy: float
    brightness: float
    saturation: float
    embedding: List[float]
    movie_checksum: str = ""
    feature_sig: str = ""


@dataclass
class BeatSlot:
    slot_idx: int
    start_s: float
    end_s: float
    duration_s: float
    target_energy: float
    target_tension: float
    boundary_strength: float


@dataclass
class PlannedClip:
    slot_idx: int
    clip_id: int
    movie_path: str
    start_s: float
    end_s: float
    source_duration_s: float
    target_duration_s: float
    stretch_rate: float
    score: float
    vec_distance: float
    energy_distance: float
    tension_distance: float = 0.0
    transition_distance: float = 0.0
    boundary_strength: float = 0.0


def vector_to_sql_literal(values: List[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"
