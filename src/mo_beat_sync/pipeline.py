from __future__ import annotations

import os
import hashlib
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
    build_time_chunks,
    check_tools,
    detect_ffmpeg_capabilities,
    extract_movie_chunk_features,
    file_sha256,
    list_media_files,
    read_video_meta,
    write_report_json,
)
from .planner import build_beat_slots, plan_clips
from .renderer import render_final_video

DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 4) // 2))
SONG_ANALYSIS_VERSION = "song_v2_sr22050_hop512_tension_structure"
VIDEO_FEATURE_VERSION = "video_v2_chunk_scene_embed96"


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
    beat_model: str = "auto"
    embedding_model: str = "auto"
    planner_beam_width: int = 10
    planner_per_state_candidates: int = 18
    planner_slot_skip_penalty: float = 0.45
    planner_min_reuse_gap: int = 4
    planner_reuse_penalty: float = 0.14


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


def _build_song_analysis_sig(mode: str) -> str:
    return f"{SONG_ANALYSIS_VERSION}_beat_{mode}"


def _build_video_feature_sig(config: PipelineConfig, embedding_mode: str) -> str:
    payload = (
        f"scene={config.scene_threshold:.4f}|"
        f"minscene={config.min_scene_len_s:.3f}|"
        f"sfps={config.scene_sample_fps:.3f}|"
        f"minclip={config.min_clip_s:.3f}|"
        f"maxclip={config.max_clip_s:.3f}|"
        f"samplew={config.sample_width}|"
        f"chunkmin={config.chunk_min_minutes:.3f}|"
        f"chunkmax={config.chunk_max_minutes:.3f}|"
        f"overlap={config.chunk_overlap_seconds:.3f}|"
        f"maxclips={config.max_clips_per_movie}|"
        f"embedding={embedding_mode}"
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]
    return f"{VIDEO_FEATURE_VERSION}_{digest}"


def _resolve_beat_mode(requested_mode: str) -> str:
    mode = (requested_mode or "auto").lower().strip()
    if mode == "librosa":
        return mode
    if mode == "madmom":
        try:
            import madmom.features.beats  # type: ignore

            return "madmom"
        except Exception:
            return "librosa"
    try:
        import madmom.features.beats  # type: ignore

        return "madmom"
    except Exception:
        return "librosa"


def _resolve_embedding_mode(requested_mode: str) -> str:
    mode = (requested_mode or "auto").lower().strip()
    if mode == "hist96":
        return mode
    if mode == "clip":
        try:
            import torch  # type: ignore
            import transformers  # type: ignore
            from PIL import Image  # noqa: F401

            _ = torch
            _ = transformers
            return "clip"
        except Exception:
            return "hist96"
    try:
        import torch  # type: ignore
        import transformers  # type: ignore
        from PIL import Image  # noqa: F401

        _ = torch
        _ = transformers
        return "clip"
    except Exception:
        return "hist96"


def _process_one_chunk(
    movie_path: str,
    movie_checksum: str,
    movie_duration_s: float,
    chunk_start_s: float,
    chunk_end_s: float,
    max_clips_for_chunk: int,
    scene_threshold: float,
    min_scene_len_s: float,
    scene_sample_fps: float,
    min_clip_s: float,
    max_clip_s: float,
    chunk_overlap_seconds: float,
    sample_width: int,
    use_gpu_decode: bool,
    embedding_model: str,
) -> tuple[str, str, float, float, list, int]:
    clips, scene_count = extract_movie_chunk_features(
        movie_path=movie_path,
        movie_duration_s=movie_duration_s,
        chunk_start_s=chunk_start_s,
        chunk_end_s=chunk_end_s,
        scene_threshold=scene_threshold,
        min_scene_len_s=min_scene_len_s,
        sample_fps=scene_sample_fps,
        min_clip_s=min_clip_s,
        max_clip_s=max_clip_s,
        max_clips_for_chunk=max_clips_for_chunk,
        chunk_overlap_s=chunk_overlap_seconds,
        sample_width=sample_width,
        use_gpu_decode=use_gpu_decode,
        embedding_model=embedding_model,
    )
    return movie_path, movie_checksum, chunk_start_s, chunk_end_s, clips, scene_count


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
    beat_mode = _resolve_beat_mode(config.beat_model)
    embedding_mode = _resolve_embedding_mode(config.embedding_model)

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
        song_checksum = file_sha256(song_path)
        song_sig = _build_song_analysis_sig(beat_mode)
        cached_song = db.find_song_cache(song_checksum, song_sig)
        if cached_song is not None:
            cached_song_id = int(cached_song["song_id"])
            song = db.load_song_analysis(cached_song_id)
            song.path = song_path
            song.checksum = song_checksum
            song.analysis_sig = song_sig
            song_id = db.clone_song_to_run(
                run_id=run_id,
                cached_song_id=cached_song_id,
                song_path=song_path,
                wav_path=song.wav_path,
            )
            song_cached = True
        else:
            song = analyze_song(song_path=song_path, wav_dir=wav_dir, beat_model=beat_mode)
            song.checksum = song_checksum
            song.analysis_sig = song_sig
            song_id = db.insert_song(run_id, song)
            song_cached = False

        video_feature_sig = _build_video_feature_sig(config, embedding_mode=embedding_mode)
        movie_checksums: list[str] = []
        tasks: list[tuple[str, str, float, float, float, int]] = []
        movie_reports_map: dict[str, dict] = {}
        for movie_path in movies:
            video_meta = read_video_meta(movie_path)
            movie_checksum = file_sha256(movie_path)
            movie_checksums.append(movie_checksum)
            cached_clip_count = db.count_cached_clips(movie_checksum=movie_checksum, feature_sig=video_feature_sig)
            chunks = build_time_chunks(
                duration_s=video_meta.duration_s,
                min_chunk_s=config.chunk_min_minutes * 60.0,
                max_chunk_s=config.chunk_max_minutes * 60.0,
            )
            if not chunks:
                continue
            movie_reports_map[movie_path] = {
                "movie_path": movie_path,
                "movie_checksum": movie_checksum,
                "duration_s": video_meta.duration_s,
                "chunk_count": len(chunks),
                "scene_count": 0,
                "clip_count": cached_clip_count,
                "cache_hit": cached_clip_count > 0,
            }
            if cached_clip_count > 0:
                continue
            chunk_cap = max(8, int((config.max_clips_per_movie + len(chunks) - 1) / len(chunks)))
            for chunk_start_s, chunk_end_s in chunks:
                tasks.append(
                    (
                        movie_path,
                        movie_checksum,
                        video_meta.duration_s,
                        chunk_start_s,
                        chunk_end_s,
                        chunk_cap,
                    )
                )

        all_clips = []
        if tasks:
            worker_count = max(1, min(config.workers, len(tasks)))
            if worker_count == 1:
                for movie_path, movie_checksum, movie_duration_s, chunk_start_s, chunk_end_s, chunk_cap in tasks:
                    _, _, _, _, clips, scene_count = _process_one_chunk(
                        movie_path,
                        movie_checksum,
                        movie_duration_s,
                        chunk_start_s,
                        chunk_end_s,
                        chunk_cap,
                        config.scene_threshold,
                        config.min_scene_len_s,
                        config.scene_sample_fps,
                        config.min_clip_s,
                        config.max_clip_s,
                        config.chunk_overlap_seconds,
                        config.sample_width,
                        use_gpu_decode,
                        embedding_mode,
                    )
                    for clip in clips:
                        clip.movie_checksum = movie_checksum
                        clip.feature_sig = video_feature_sig
                    all_clips.extend(clips)
                    movie_reports_map[movie_path]["scene_count"] += scene_count
                    movie_reports_map[movie_path]["clip_count"] += len(clips)
            else:
                with ProcessPoolExecutor(max_workers=worker_count) as pool:
                    futures = {
                        pool.submit(
                            _process_one_chunk,
                            movie_path,
                            movie_checksum,
                            movie_duration_s,
                            chunk_start_s,
                            chunk_end_s,
                            chunk_cap,
                            config.scene_threshold,
                            config.min_scene_len_s,
                            config.scene_sample_fps,
                            config.min_clip_s,
                            config.max_clip_s,
                            config.chunk_overlap_seconds,
                            config.sample_width,
                            use_gpu_decode,
                            embedding_mode,
                        ): (movie_path, movie_checksum, chunk_start_s, chunk_end_s)
                        for movie_path, movie_checksum, movie_duration_s, chunk_start_s, chunk_end_s, chunk_cap in tasks
                    }
                    for future in as_completed(futures):
                        chunk_info = futures[future]
                        try:
                            movie_path, movie_checksum, _, _, clips, scene_count = future.result()
                        except Exception as exc:
                            raise PipelineError(
                                f"Chunk processing failed: movie={chunk_info[0]}, chunk=({chunk_info[2]:.2f},{chunk_info[3]:.2f}), error={exc}"
                            ) from exc
                        for clip in clips:
                            clip.movie_checksum = movie_checksum
                            clip.feature_sig = video_feature_sig
                        all_clips.extend(clips)
                        movie_reports_map[movie_path]["scene_count"] += scene_count
                        movie_reports_map[movie_path]["clip_count"] += len(clips)

        dedup: dict[tuple[str, int, int], object] = {}
        for clip in all_clips:
            key = (clip.movie_checksum or clip.movie_path, int(round(clip.start_s * 1000.0)), int(round(clip.end_s * 1000.0)))
            if key not in dedup:
                dedup[key] = clip
        deduped_clips = list(dedup.values())

        clips_by_movie: dict[str, list] = {}
        for clip in deduped_clips:
            clips_by_movie.setdefault(clip.movie_path, []).append(clip)

        all_clips = []
        for movie_path, clips in clips_by_movie.items():
            clips.sort(key=lambda x: (x.start_s, x.end_s))
            if len(clips) > config.max_clips_per_movie:
                step = (len(clips) - 1) / max(1, config.max_clips_per_movie - 1)
                pick = [clips[int(round(i * step))] for i in range(config.max_clips_per_movie)]
                clips = pick
            movie_reports_map[movie_path]["clip_count"] = len(clips)
            all_clips.extend(clips)

        movie_reports = sorted(movie_reports_map.values(), key=lambda x: x["movie_path"])

        inserted = db.insert_clips(run_id, all_clips)
        available_clips = db.count_clips_for_assets(movie_checksums=movie_checksums, feature_sig=video_feature_sig)
        if available_clips <= 0:
            raise PipelineError("No clip features available in MatrixOne for selected movies")

        duration_limit = min(song.duration_s, config.target_duration_s)
        slots = build_beat_slots(
            beat_times=song.beat_times,
            beat_strengths=song.beat_strengths,
            tension_times=song.tension_times,
            tension_values=song.tension_values,
            structure_times=song.structure_times,
            structure_strengths=song.structure_strengths,
            duration_limit_s=duration_limit,
            beats_per_clip=config.beats_per_clip,
            min_slot_s=config.min_slot_s,
        )
        if not slots:
            raise PipelineError("No beat slots generated")

        planned = plan_clips(
            db=db,
            slots=slots,
            movie_checksums=movie_checksums,
            feature_sig=video_feature_sig,
            beam_width=config.planner_beam_width,
            per_state_candidates=config.planner_per_state_candidates,
            slot_skip_penalty=config.planner_slot_skip_penalty,
            min_reuse_gap=config.planner_min_reuse_gap,
            reuse_penalty=config.planner_reuse_penalty,
        )
        if not planned:
            raise PipelineError("No clips selected for edit plan")
        boundary_slot_count = sum(1 for s in slots if s.boundary_strength >= 0.55)
        boundary_plan_count = sum(1 for p in planned if p.boundary_strength >= 0.55)
        avg_tension_distance = (
            sum(float(p.tension_distance) for p in planned) / max(1, len(planned))
        )
        avg_transition_distance = (
            sum(float(p.transition_distance) for p in planned) / max(1, len(planned))
        )

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
                "structure_boundaries": len(song.structure_times),
                "tension_samples": len(song.tension_values),
                "cache_hit": song_cached,
                "beat_model": beat_mode,
            },
            "movies": movie_reports,
            "clip_inserted": inserted,
            "clip_available": available_clips,
            "slot_count": len(slots),
            "planned_count": len(planned),
            "alignment": {
                "boundary_slot_count": boundary_slot_count,
                "boundary_planned_count": boundary_plan_count,
                "avg_tension_distance": avg_tension_distance,
                "avg_transition_distance": avg_transition_distance,
            },
            "plan_id": plan_id,
            "final_video": final_video,
            "matrixone_summary": summary,
            "gpu": {
                "mode": gpu_mode,
                "ffmpeg_caps": caps,
                "decode_enabled": use_gpu_decode,
                "encode_enabled": use_gpu_encode,
            },
            "models": {
                "beat_model_requested": config.beat_model,
                "beat_model_resolved": beat_mode,
                "embedding_model_requested": config.embedding_model,
                "embedding_model_resolved": embedding_mode,
                "planner_beam_width": config.planner_beam_width,
                "planner_per_state_candidates": config.planner_per_state_candidates,
                "planner_slot_skip_penalty": config.planner_slot_skip_penalty,
                "planner_min_reuse_gap": config.planner_min_reuse_gap,
                "planner_reuse_penalty": config.planner_reuse_penalty,
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
