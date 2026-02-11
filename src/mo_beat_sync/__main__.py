from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto-generate beat-synced video from source movies and songs, backed by MatrixOne",
    )
    parser.add_argument("--movies-dir", default="source_movies")
    parser.add_argument("--songs-dir", default="source_songs")
    parser.add_argument("--output-path", default="outputs/final/beat_sync.mp4")
    parser.add_argument("--workdir", default="workdir")

    parser.add_argument("--db-host", default=os.getenv("MO_HOST", "127.0.0.1"))
    parser.add_argument("--db-port", type=int, default=int(os.getenv("MO_PORT", "6001")))
    parser.add_argument("--db-user", default=os.getenv("MO_USER", "root"))
    parser.add_argument("--db-password", default=os.getenv("MO_PASSWORD", "111"))
    parser.add_argument("--db-name", default=os.getenv("MO_DATABASE", "mo_beat_sync"))

    parser.add_argument("--song-keyword", default="", help="Pick a song whose filename contains this keyword")
    parser.add_argument("--target-duration-s", type=float, default=50.0)
    parser.add_argument("--max-movies", type=int, default=12)
    parser.add_argument("--max-clips-per-movie", type=int, default=120)
    parser.add_argument("--beats-per-clip", type=int, default=2)
    parser.add_argument("--chunk-min-minutes", type=float, default=2.0)
    parser.add_argument("--chunk-max-minutes", type=float, default=5.0)
    parser.add_argument(
        "--chunk-overlap-seconds",
        type=float,
        default=float(os.getenv("MO_CHUNK_OVERLAP_SECONDS", "1.5")),
    )
    parser.add_argument("--sample-width", type=int, default=320)
    parser.add_argument(
        "--gpu-mode",
        choices=["auto", "on", "off"],
        default=os.getenv("MO_GPU_MODE", "auto"),
        help="GPU usage mode for ffmpeg decode/encode.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("MO_WORKERS", "0") or "0"),
        help="Chunk processing workers. 0 means use pipeline default.",
    )
    parser.add_argument(
        "--beat-model",
        choices=["auto", "librosa", "madmom"],
        default=os.getenv("MO_BEAT_MODEL", "auto"),
        help="Beat tracking backend.",
    )
    parser.add_argument(
        "--embedding-model",
        choices=["auto", "hist96", "clip"],
        default=os.getenv("MO_EMBEDDING_MODEL", "auto"),
        help="Clip embedding backend.",
    )
    parser.add_argument(
        "--planner-beam-width",
        type=int,
        default=int(os.getenv("MO_PLANNER_BEAM_WIDTH", "10")),
        help="Beam width for global clip planning.",
    )
    parser.add_argument(
        "--planner-per-state-candidates",
        type=int,
        default=int(os.getenv("MO_PLANNER_PER_STATE_CANDIDATES", "18")),
        help="Candidate count expanded per beam state and slot.",
    )
    parser.add_argument(
        "--planner-slot-skip-penalty",
        type=float,
        default=float(os.getenv("MO_PLANNER_SLOT_SKIP_PENALTY", "0.45")),
        help="Penalty when a slot cannot be matched.",
    )
    parser.add_argument(
        "--planner-min-reuse-gap",
        type=int,
        default=int(os.getenv("MO_PLANNER_MIN_REUSE_GAP", "4")),
        help="Minimum slot gap before reusing the same source clip.",
    )
    parser.add_argument(
        "--planner-reuse-penalty",
        type=float,
        default=float(os.getenv("MO_PLANNER_REUSE_PENALTY", "0.14")),
        help="Penalty factor for reused clips.",
    )

    parser.add_argument("--scene-threshold", type=float, default=0.42)
    parser.add_argument("--min-scene-len-s", type=float, default=1.0)
    parser.add_argument("--scene-sample-fps", type=float, default=3.0)
    parser.add_argument("--min-clip-s", type=float, default=0.8)
    parser.add_argument("--max-clip-s", type=float, default=3.6)
    parser.add_argument("--min-slot-s", type=float, default=0.6)

    parser.add_argument("--render-width", type=int, default=1080)
    parser.add_argument("--render-height", type=int, default=1920)
    parser.add_argument("--render-fps", type=int, default=30)

    parser.add_argument("--drop-existing", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = PipelineConfig(
        movies_dir=str(Path(args.movies_dir).resolve()),
        songs_dir=str(Path(args.songs_dir).resolve()),
        output_path=str(Path(args.output_path).resolve()),
        workdir=str(Path(args.workdir).resolve()),
        db_host=args.db_host,
        db_port=args.db_port,
        db_user=args.db_user,
        db_password=args.db_password,
        db_name=args.db_name,
        drop_existing=args.drop_existing,
        max_movies=args.max_movies,
        max_clips_per_movie=args.max_clips_per_movie,
        scene_threshold=args.scene_threshold,
        min_scene_len_s=args.min_scene_len_s,
        scene_sample_fps=args.scene_sample_fps,
        min_clip_s=args.min_clip_s,
        max_clip_s=args.max_clip_s,
        beats_per_clip=args.beats_per_clip,
        min_slot_s=args.min_slot_s,
        target_duration_s=args.target_duration_s,
        render_width=args.render_width,
        render_height=args.render_height,
        render_fps=args.render_fps,
        chunk_min_minutes=args.chunk_min_minutes,
        chunk_max_minutes=args.chunk_max_minutes,
        chunk_overlap_seconds=args.chunk_overlap_seconds,
        sample_width=args.sample_width,
        gpu_mode=args.gpu_mode,
        workers=args.workers if args.workers > 0 else PipelineConfig.workers,
        song_keyword=args.song_keyword,
        beat_model=args.beat_model,
        embedding_model=args.embedding_model,
        planner_beam_width=args.planner_beam_width,
        planner_per_state_candidates=args.planner_per_state_candidates,
        planner_slot_skip_penalty=args.planner_slot_skip_penalty,
        planner_min_reuse_gap=args.planner_min_reuse_gap,
        planner_reuse_penalty=args.planner_reuse_penalty,
    )

    try:
        report = run_pipeline(cfg)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
