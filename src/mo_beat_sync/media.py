from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import librosa
import numpy as np

from .models import ClipFeature, SongAnalysis, VideoMeta

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"}


class MediaError(RuntimeError):
    pass


def detect_ffmpeg_capabilities() -> dict:
    caps = {"cuda_hwaccel": False, "nvenc": False}
    try:
        hw = run_cmd(["ffmpeg", "-hide_banner", "-hwaccels"])
        caps["cuda_hwaccel"] = "cuda" in hw.lower()
    except Exception:
        caps["cuda_hwaccel"] = False

    try:
        enc = run_cmd(["ffmpeg", "-hide_banner", "-encoders"])
        caps["nvenc"] = "h264_nvenc" in enc.lower()
    except Exception:
        caps["nvenc"] = False
    return caps


def list_media_files(folder: str, exts: set[str]) -> List[str]:
    path = Path(folder)
    if not path.exists():
        return []
    files = [str(p.resolve()) for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def check_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise MediaError(f"{tool} not found in PATH")


def run_cmd(cmd: Sequence[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MediaError(f"Command failed: {' '.join(cmd)}\n{proc.stderr.strip()}")
    return proc.stdout


def ffprobe_json(path: str) -> dict:
    out = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            path,
        ]
    )
    return json.loads(out)


def _parse_fps(raw: str) -> float:
    if not raw or raw == "0/0":
        return 0.0
    if "/" in raw:
        num, den = raw.split("/", 1)
        den_val = float(den)
        if den_val == 0:
            return 0.0
        return float(num) / den_val
    return float(raw)


def read_video_meta(path: str) -> VideoMeta:
    meta = ffprobe_json(path)
    stream = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), None)
    if not stream:
        raise MediaError(f"No video stream found: {path}")
    duration_s = float(meta.get("format", {}).get("duration", stream.get("duration", 0.0)) or 0.0)
    fps = _parse_fps(stream.get("avg_frame_rate", "0/0"))
    width = int(stream.get("width", 0) or 0)
    height = int(stream.get("height", 0) or 0)
    return VideoMeta(path=path, duration_s=duration_s, fps=fps, width=width, height=height)


def convert_audio_to_wav(song_path: str, wav_path: str, sample_rate: int = 22050) -> None:
    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            song_path,
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            wav_path,
        ]
    )


def analyze_song(song_path: str, wav_dir: str, sample_rate: int = 22050) -> SongAnalysis:
    wav_path = str((Path(wav_dir) / (Path(song_path).stem + ".wav")).resolve())
    convert_audio_to_wav(song_path, wav_path, sample_rate=sample_rate)

    y, sr = librosa.load(wav_path, sr=sample_rate, mono=True)
    duration_s = float(librosa.get_duration(y=y, sr=sr))
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    onset = librosa.onset.onset_strength(y=y, sr=sr)
    if len(beat_frames) > 0:
        idx = np.clip(beat_frames, 0, len(onset) - 1)
        strengths = onset[idx].astype(np.float32)
        if strengths.max() > strengths.min():
            strengths = (strengths - strengths.min()) / (strengths.max() - strengths.min())
        else:
            strengths = np.zeros_like(strengths)
        beat_strengths = strengths.tolist()
    else:
        bpm = float(tempo) if np.size(tempo) == 1 else float(np.mean(tempo))
        interval = 60.0 / max(bpm, 120.0)
        beat_times = list(np.arange(0.0, duration_s, interval))
        beat_strengths = [0.5] * len(beat_times)

    bpm = float(tempo) if np.size(tempo) == 1 else float(np.mean(tempo))
    return SongAnalysis(
        path=str(Path(song_path).resolve()),
        wav_path=wav_path,
        duration_s=duration_s,
        bpm=bpm,
        beat_times=beat_times,
        beat_strengths=beat_strengths,
    )


def _frame_hist_hs(frame: np.ndarray, h_bins: int = 32, s_bins: int = 32) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def build_time_chunks(duration_s: float, min_chunk_s: float, max_chunk_s: float) -> List[Tuple[float, float]]:
    if duration_s <= 0:
        return []
    if max_chunk_s <= 0:
        return [(0.0, duration_s)]
    if min_chunk_s <= 0:
        min_chunk_s = max_chunk_s
    if min_chunk_s > max_chunk_s:
        min_chunk_s, max_chunk_s = max_chunk_s, min_chunk_s
    if duration_s <= max_chunk_s:
        return [(0.0, duration_s)]

    target = (min_chunk_s + max_chunk_s) / 2.0
    chunk_count = max(1, int(round(duration_s / target)))
    chunk_count = max(chunk_count, int(math.ceil(duration_s / max_chunk_s)))
    while chunk_count > 1 and duration_s / chunk_count < min_chunk_s:
        chunk_count -= 1

    chunk_len = duration_s / chunk_count
    chunks = []
    for i in range(chunk_count):
        start_s = i * chunk_len
        end_s = duration_s if i == chunk_count - 1 else (i + 1) * chunk_len
        chunks.append((start_s, end_s))
    return chunks


def _extract_chunk_sampled_frames(
    video_path: str,
    chunk_start_s: float,
    chunk_end_s: float,
    sample_fps: float,
    sample_width: int,
    use_gpu_decode: bool,
) -> Tuple[List[float], List[np.ndarray]]:
    duration_s = max(0.01, chunk_end_s - chunk_start_s)
    out_dir = Path(tempfile.mkdtemp(prefix="mo_chunk_frames_"))
    out_pattern = str((out_dir / "frame_%06d.jpg").resolve())
    vf = f"fps={sample_fps},scale={sample_width}:-2:force_original_aspect_ratio=decrease"

    base_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{chunk_start_s:.6f}",
        "-t",
        f"{duration_s:.6f}",
    ]
    gpu_cmd = base_cmd + ["-hwaccel", "cuda", "-i", video_path, "-vf", vf, "-q:v", "5", out_pattern]
    cpu_cmd = base_cmd + ["-i", video_path, "-vf", vf, "-q:v", "5", out_pattern]

    try:
        if use_gpu_decode:
            run_cmd(gpu_cmd)
        else:
            run_cmd(cpu_cmd)
    except Exception:
        run_cmd(cpu_cmd)

    frame_paths = sorted(out_dir.glob("frame_*.jpg"))
    timestamps: List[float] = []
    frames: List[np.ndarray] = []
    step_s = 1.0 / max(sample_fps, 1e-6)

    for idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        ts = min(chunk_end_s, chunk_start_s + idx * step_s)
        timestamps.append(ts)
        frames.append(frame)

    shutil.rmtree(out_dir, ignore_errors=True)
    return timestamps, frames


def _detect_scene_segments_from_sampled_frames(
    timestamps: Sequence[float],
    frames: Sequence[np.ndarray],
    chunk_start_s: float,
    chunk_end_s: float,
    scene_threshold: float,
    min_scene_len_s: float,
) -> List[Tuple[float, float]]:
    if not timestamps or len(frames) < 2:
        return [(chunk_start_s, chunk_end_s)]

    scenes: List[Tuple[float, float]] = []
    last_cut = chunk_start_s
    prev_hist = _frame_hist_hs(frames[0], h_bins=16, s_bins=16)

    for idx in range(1, len(frames)):
        ts = float(timestamps[idx])
        hist = _frame_hist_hs(frames[idx], h_bins=16, s_bins=16)
        dist = cv2.compareHist(prev_hist.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
        if dist >= scene_threshold and (ts - last_cut) >= min_scene_len_s:
            scenes.append((last_cut, ts))
            last_cut = ts
        prev_hist = hist

    if chunk_end_s - last_cut >= min_scene_len_s:
        scenes.append((last_cut, chunk_end_s))
    elif not scenes:
        scenes.append((chunk_start_s, chunk_end_s))
    return scenes


def _extract_clip_features_from_sampled_frames(
    movie_path: str,
    timestamps: Sequence[float],
    frames: Sequence[np.ndarray],
    segments: Sequence[Tuple[float, float]],
) -> List[ClipFeature]:
    results: List[ClipFeature] = []
    if len(timestamps) != len(frames) or not timestamps:
        return results

    for start_s, end_s in segments:
        if end_s - start_s <= 0:
            continue

        indices = [i for i, ts in enumerate(timestamps) if start_s <= ts < end_s]
        if len(indices) < 2:
            continue

        selected_frames = [frames[i] for i in indices]
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in selected_frames]
        motions = []
        for i in range(1, len(grays)):
            motions.append(float(np.mean(np.abs(grays[i] - grays[i - 1])) / 255.0))

        hsvs = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in selected_frames]
        brightness = float(np.mean([np.mean(hsv[:, :, 2]) / 255.0 for hsv in hsvs]))
        saturation = float(np.mean([np.mean(hsv[:, :, 1]) / 255.0 for hsv in hsvs]))
        motion_energy = float(np.mean(motions)) if motions else 0.0

        emb_pick_count = min(12, len(selected_frames))
        emb_idx = np.linspace(0, len(selected_frames) - 1, emb_pick_count, dtype=int)
        emb_frames = [selected_frames[i] for i in emb_idx]
        embedding = _embedding_from_frames(emb_frames, bins=32)

        results.append(
            ClipFeature(
                movie_path=movie_path,
                start_s=float(start_s),
                end_s=float(end_s),
                duration_s=float(end_s - start_s),
                motion_energy=motion_energy,
                brightness=brightness,
                saturation=saturation,
                embedding=embedding,
            )
        )
    return results


def extract_movie_clip_features_chunked(
    video_meta: VideoMeta,
    scene_threshold: float,
    min_scene_len_s: float,
    sample_fps: float,
    min_clip_s: float,
    max_clip_s: float,
    max_clips_per_movie: int,
    chunk_min_s: float,
    chunk_max_s: float,
    chunk_overlap_s: float,
    sample_width: int,
    use_gpu_decode: bool,
) -> Tuple[List[ClipFeature], dict]:
    chunks = build_time_chunks(video_meta.duration_s, min_chunk_s=chunk_min_s, max_chunk_s=chunk_max_s)
    all_clips: List[ClipFeature] = []
    total_scenes = 0

    def _restrict_segments_to_window(
        segments: Sequence[Tuple[float, float]],
        win_start_s: float,
        win_end_s: float,
        min_len_s: float,
    ) -> List[Tuple[float, float]]:
        restricted: List[Tuple[float, float]] = []
        for seg_start_s, seg_end_s in segments:
            start_s = max(seg_start_s, win_start_s)
            end_s = min(seg_end_s, win_end_s)
            if end_s - start_s >= min_len_s:
                restricted.append((start_s, end_s))
        return restricted

    for chunk_start_s, chunk_end_s in chunks:
        extended_start_s = max(0.0, chunk_start_s - max(0.0, chunk_overlap_s))
        extended_end_s = min(video_meta.duration_s, chunk_end_s + max(0.0, chunk_overlap_s))
        timestamps, frames = _extract_chunk_sampled_frames(
            video_path=video_meta.path,
            chunk_start_s=extended_start_s,
            chunk_end_s=extended_end_s,
            sample_fps=sample_fps,
            sample_width=sample_width,
            use_gpu_decode=use_gpu_decode,
        )
        if len(frames) < 2:
            continue

        scenes = _detect_scene_segments_from_sampled_frames(
            timestamps=timestamps,
            frames=frames,
            chunk_start_s=extended_start_s,
            chunk_end_s=extended_end_s,
            scene_threshold=scene_threshold,
            min_scene_len_s=min_scene_len_s,
        )
        total_scenes += len(scenes)

        chunk_cap = max(8, int(math.ceil(max_clips_per_movie / max(1, len(chunks)))))
        segments = build_clip_segments(
            scenes=scenes,
            min_clip_s=min_clip_s,
            max_clip_s=max_clip_s,
            max_clips=chunk_cap,
        )
        segments = _restrict_segments_to_window(
            segments=segments,
            win_start_s=chunk_start_s,
            win_end_s=chunk_end_s,
            min_len_s=min_clip_s,
        )
        chunk_clips = _extract_clip_features_from_sampled_frames(
            movie_path=video_meta.path,
            timestamps=timestamps,
            frames=frames,
            segments=segments,
        )
        all_clips.extend(chunk_clips)

    dedup: dict[tuple[str, int, int], ClipFeature] = {}
    for clip in all_clips:
        key = (clip.movie_path, int(round(clip.start_s * 1000.0)), int(round(clip.end_s * 1000.0)))
        if key not in dedup:
            dedup[key] = clip
    all_clips = list(dedup.values())

    if len(all_clips) > max_clips_per_movie:
        chosen_idx = np.linspace(0, len(all_clips) - 1, max_clips_per_movie, dtype=int)
        all_clips = [all_clips[i] for i in chosen_idx]

    report = {
        "movie_path": video_meta.path,
        "duration_s": video_meta.duration_s,
        "chunk_count": len(chunks),
        "scene_count": total_scenes,
        "clip_count": len(all_clips),
    }
    return all_clips, report


def detect_scene_segments(
    video_path: str,
    duration_s: float,
    scene_threshold: float,
    min_scene_len_s: float,
    sample_fps: float,
) -> List[Tuple[float, float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise MediaError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / max(sample_fps, 1e-6))))

    scenes: List[Tuple[float, float]] = []
    last_cut = 0.0
    prev_hist = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        ts = frame_idx / fps
        hist = _frame_hist_hs(frame, h_bins=16, s_bins=16)
        if prev_hist is not None:
            dist = cv2.compareHist(prev_hist.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
            if dist >= scene_threshold and (ts - last_cut) >= min_scene_len_s:
                scenes.append((last_cut, ts))
                last_cut = ts
        prev_hist = hist
        frame_idx += 1

    cap.release()
    if duration_s - last_cut >= min_scene_len_s:
        scenes.append((last_cut, duration_s))
    elif not scenes:
        scenes.append((0.0, duration_s))
    return scenes


def build_clip_segments(
    scenes: Sequence[Tuple[float, float]],
    min_clip_s: float,
    max_clip_s: float,
    max_clips: int,
) -> List[Tuple[float, float]]:
    clips: List[Tuple[float, float]] = []
    for start_s, end_s in scenes:
        duration = end_s - start_s
        if duration < min_clip_s:
            continue
        if duration <= max_clip_s:
            clips.append((start_s, end_s))
            continue

        chunks = max(1, int(math.ceil(duration / max_clip_s)))
        chunk_len = duration / chunks
        for idx in range(chunks):
            c_start = start_s + idx * chunk_len
            c_end = min(end_s, c_start + chunk_len)
            if c_end - c_start >= min_clip_s:
                clips.append((c_start, c_end))

    if len(clips) <= max_clips:
        return clips

    chosen_idx = np.linspace(0, len(clips) - 1, max_clips, dtype=int)
    return [clips[i] for i in chosen_idx]


def _read_frame_at(cap: cv2.VideoCapture, ts_s: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, ts_s * 1000.0)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def _embedding_from_frames(frames: Iterable[np.ndarray], bins: int = 32) -> List[float]:
    vectors = []
    for frame in frames:
        ch_features = []
        for ch in range(3):
            hist = cv2.calcHist([frame], [ch], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
            ch_features.append(hist)
        vectors.append(np.concatenate(ch_features, axis=0))

    if not vectors:
        return [0.0] * (bins * 3)

    emb = np.mean(np.stack(vectors, axis=0), axis=0)
    norm = float(np.linalg.norm(emb))
    if norm > 1e-9:
        emb = emb / norm
    return emb.astype(np.float32).tolist()


def extract_clip_features(video_meta: VideoMeta, segments: Sequence[Tuple[float, float]]) -> List[ClipFeature]:
    cap = cv2.VideoCapture(video_meta.path)
    if not cap.isOpened():
        raise MediaError(f"Cannot open video: {video_meta.path}")

    results: List[ClipFeature] = []
    for start_s, end_s in segments:
        duration_s = max(0.0, end_s - start_s)
        if duration_s <= 0:
            continue

        sample_cnt = max(4, min(12, int(round(duration_s * 2))))
        times = np.linspace(start_s, max(start_s + 1e-3, end_s - 1e-3), sample_cnt).tolist()
        frames: List[np.ndarray] = []
        for ts_s in times:
            frame = _read_frame_at(cap, ts_s)
            if frame is not None:
                frames.append(frame)

        if len(frames) < 2:
            continue

        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
        motions = []
        for i in range(1, len(grays)):
            motions.append(float(np.mean(np.abs(grays[i] - grays[i - 1])) / 255.0))

        hsvs = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames]
        brightness = float(np.mean([np.mean(hsv[:, :, 2]) / 255.0 for hsv in hsvs]))
        saturation = float(np.mean([np.mean(hsv[:, :, 1]) / 255.0 for hsv in hsvs]))
        motion_energy = float(np.mean(motions)) if motions else 0.0
        embedding = _embedding_from_frames(frames, bins=32)

        results.append(
            ClipFeature(
                movie_path=video_meta.path,
                start_s=float(start_s),
                end_s=float(end_s),
                duration_s=float(duration_s),
                motion_energy=motion_energy,
                brightness=brightness,
                saturation=saturation,
                embedding=embedding,
            )
        )

    cap.release()
    return results


def write_report_json(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
