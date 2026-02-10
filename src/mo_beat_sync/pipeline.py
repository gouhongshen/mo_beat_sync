from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .db import MatrixOneClient
from .media import (
    AUDIO_EXTS,
    VIDEO_EXTS,
    analyze_song,
    check_tools,
    detect_ffmpeg_capabilities,
    extract_movie_clip_features_chunked,
    list_media_files,
    read_video_meta,
    write_report_json,
)
from .planner import build_beat_slots, plan_clips
from .renderer import render_final_video

DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 4) // 2))


@dataclass
class PipelineConfig:
    movies_dir: str
    songs_dir: str
    output_path: str
    workdir: str
    db_host: str = "127.0.0.1"
    db_port: int = 6001
    db_user: str = "root"
    db_password: str = "111"
    db_name: str = "mo_beat_sync"
    drop_existing: bool = False
    max_movies: int = 12
    max_clips_per_movie: int = 120
    scene_threshold: float = 0.42
    min_scene_len_s: float = 1.0
    scene_sample_fps: float = 3.0
    min_clip_s: float = 0.8
    max_clip_s: float = 3.6
    beats_per_clip: int = 2
    min_slot_s: float = 0.6
    target_duration_s: float = 50.0
    render_width: int = 1080
    render_height: int = 1920
    render_fps: int = 30
    chunk_min_minutes: float = 2.0
    chunk_max_minutes: float = 5.0
    chunk_overlap_seconds: float = 1.5
    sample_width: int = 320
    gpu_mode: str = "auto"
    workers: int = DEFAULT_WORKERS
    song_keyword: str = ""


class PipelineError(RuntimeError):
    pass


def _pick_song(song_files: list[str], keyword: str) -> str:
    if not song_files:
        raise PipelineError("No songs found in songs_dir")
    if keyword:
        key = keyword.lower().strip()
        for p in song_files:
            if key in Path(p).name.lower():
                return p
        raise PipelineError(f"No song matched keyword: {keyword}")
    return song_files[0]


def _safe_config(cfg: PipelineConfig) -> dict:
    payload = asdict(cfg)
    payload["db_password"] = "***"
    return payload


def _process_one_movie(
    movie_path: str,
    scene_threshold: float,
    min_scene_len_s: float,
    scene_sample_fps: float,
    min_clip_s: float,
    max_clip_s: float,
    max_clips_per_movie: int,
    chunk_min_minutes: float,
    chunk_max_minutes: float,
    chunk_overlap_seconds: float,
    sample_width: int,
    use_gpu_decode: bool,
) -> tuple[list, dict]:
    video_meta = read_video_meta(movie_path)
    clips, report = extract_movie_clip_features_chunked(
        video_meta=video_meta,
        scene_threshold=scene_threshold,
        min_scene_len_s=min_scene_len_s,
        sample_fps=scene_sample_fps,
        min_clip_s=min_clip_s,
        max_clip_s=max_clip_s,
        max_clips_per_movie=max_clips_per_movie,
        chunk_min_s=chunk_min_minutes * 60.0,
        chunk_max_s=chunk_max_minutes * 60.0,
        chunk_overlap_s=chunk_overlap_seconds,
        sample_width=sample_width,
        use_gpu_decode=use_gpu_decode,
    )
    return clips, report


def run_pipeline(config: PipelineConfig) -> dict:
    load_dotenv()
    check_tools()
    gpu_mode = config.gpu_mode.lower().strip()
    if gpu_mode not in {"auto", "on", "off"}:
        raise PipelineError("gpu_mode must be one of: auto, on, off")
    caps = detect_ffmpeg_capabilities()
    if gpu_mode == "off":
        use_gpu_decode = False
        use_gpu_encode = False
    elif gpu_mode == "on":
        use_gpu_decode = True
        use_gpu_encode = True
    else:
        use_gpu_decode = caps["cuda_hwaccel"]
        use_gpu_encode = caps["nvenc"]

    movies = list_media_files(config.movies_dir, VIDEO_EXTS)
    songs = list_media_files(config.songs_dir, AUDIO_EXTS)
    if not movies:
        raise PipelineError(f"No movies found in: {config.movies_dir}")
    if not songs:
        raise PipelineError(f"No songs found in: {config.songs_dir}")

    movies = movies[: max(1, config.max_movies)]
    song_path = _pick_song(songs, config.song_keyword)

    embedding_dim = 96
    db = MatrixOneClient(
        host=config.db_host,
        port=config.db_port,
        user=config.db_user,
        password=config.db_password,
        database=config.db_name,
        embedding_dim=embedding_dim,
    )

    run_id: Optional[int] = None
    report: dict = {}
    try:
        db.ensure_schema(drop_existing=config.drop_existing)
        run_id = db.start_run(_safe_config(config))

        wav_dir = str((Path(config.workdir) / "wav_cache").resolve())
        song = analyze_song(song_path=song_path, wav_dir=wav_dir)
        song_id = db.insert_song(run_id, song)

        all_clips = []
        movie_reports = []
        worker_count = max(1, min(config.workers, len(movies)))
        if worker_count == 1:
            for movie_path in movies:
                clips, report = _process_one_movie(
                    movie_path=movie_path,
                    scene_threshold=config.scene_threshold,
                    min_scene_len_s=config.min_scene_len_s,
                    scene_sample_fps=config.scene_sample_fps,
                    min_clip_s=config.min_clip_s,
                    max_clip_s=config.max_clip_s,
                    max_clips_per_movie=config.max_clips_per_movie,
                    chunk_min_minutes=config.chunk_min_minutes,
                    chunk_max_minutes=config.chunk_max_minutes,
                    chunk_overlap_seconds=config.chunk_overlap_seconds,
                    sample_width=config.sample_width,
                    use_gpu_decode=use_gpu_decode,
                )
                all_clips.extend(clips)
                movie_reports.append(report)
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(
                        _process_one_movie,
                        movie_path,
                        config.scene_threshold,
                        config.min_scene_len_s,
                        config.scene_sample_fps,
                        config.min_clip_s,
                        config.max_clip_s,
                        config.max_clips_per_movie,
                        config.chunk_min_minutes,
                        config.chunk_max_minutes,
                        config.chunk_overlap_seconds,
                        config.sample_width,
                        use_gpu_decode,
                    ): movie_path
                    for movie_path in movies
                }
                for future in as_completed(futures):
                    movie_path = futures[future]
                    try:
                        clips, report = future.result()
                    except Exception as exc:
                        raise PipelineError(f"Movie processing failed: {movie_path}, error={exc}") from exc
                    all_clips.extend(clips)
                    movie_reports.append(report)

        movie_reports.sort(key=lambda x: x["movie_path"])

        inserted = db.insert_clips(run_id, all_clips)
        if inserted <= 0:
            raise PipelineError("No clip features were inserted into MatrixOne")

        duration_limit = min(song.duration_s, config.target_duration_s)
        slots = build_beat_slots(
            beat_times=song.beat_times,
            beat_strengths=song.beat_strengths,
            duration_limit_s=duration_limit,
            beats_per_clip=config.beats_per_clip,
            min_slot_s=config.min_slot_s,
        )
        if not slots:
            raise PipelineError("No beat slots generated")

        planned = plan_clips(db=db, run_id=run_id, slots=slots)
        if not planned:
            raise PipelineError("No clips selected for edit plan")

        plan_id = db.create_plan(run_id=run_id, song_id=song_id, output_path=config.output_path, slot_count=len(slots))
        db.insert_plan_items(plan_id, planned)

        temp_dir = str((Path(config.workdir) / f"run_{run_id}").resolve())
        final_video = render_final_video(
            plan=planned,
            song_path=song.path,
            output_path=config.output_path,
            temp_dir=temp_dir,
            width=config.render_width,
            height=config.render_height,
            fps=config.render_fps,
            use_gpu_encode=use_gpu_encode,
        )

        summary = db.get_run_summary(run_id)
        report = {
            "run_id": run_id,
            "song": {
                "path": song.path,
                "duration_s": song.duration_s,
                "bpm": song.bpm,
                "beats": len(song.beat_times),
            },
            "movies": movie_reports,
            "clip_inserted": inserted,
            "slot_count": len(slots),
            "planned_count": len(planned),
            "plan_id": plan_id,
            "final_video": final_video,
            "matrixone_summary": summary,
            "gpu": {
                "mode": gpu_mode,
                "ffmpeg_caps": caps,
                "decode_enabled": use_gpu_decode,
                "encode_enabled": use_gpu_encode,
            },
        }

        report_path = str((Path(config.workdir) / f"run_{run_id}" / "report.json").resolve())
        write_report_json(report_path, report)
        db.finish_run(run_id, "SUCCESS", f"output={final_video}")
        report["report_path"] = report_path
        return report
    except Exception as exc:
        if run_id is not None:
            db.finish_run(run_id, "FAILED", str(exc))
        raise
    finally:
        db.close()
