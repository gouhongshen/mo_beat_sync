from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import hashlib
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


_CLIP_RUNTIME: dict | None = None
_CLIP_PROJECTOR: np.ndarray | None = None


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


def file_sha256(path: str, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


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


def _try_madmom_beats(wav_path: str) -> List[float]:
    try:
        from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
    except Exception:
        return []
    try:
        activations = RNNBeatProcessor()(wav_path)
        beats = DBNBeatTrackingProcessor(fps=100)(activations)
        return [float(x) for x in beats]
    except Exception:
        return []


def _resolve_beat_times(
    y: np.ndarray,
    sr: int,
    wav_path: str,
    hop_length: int,
    beat_model: str,
) -> Tuple[List[float], float]:
    mode = (beat_model or "auto").lower().strip()
    if mode not in {"auto", "librosa", "madmom"}:
        mode = "auto"

    if mode in {"auto", "madmom"}:
        madmom_beats = _try_madmom_beats(wav_path)
        if madmom_beats:
            if len(madmom_beats) >= 2:
                intervals = np.diff(np.asarray(madmom_beats, dtype=np.float32))
                valid = intervals[intervals > 1e-6]
                median_interval = float(np.median(valid)) if valid.size > 0 else 0.0
                bpm = 60.0 / median_interval if median_interval > 1e-6 else 120.0
            else:
                bpm = 120.0
            return madmom_beats, bpm
        if mode == "madmom":
            raise MediaError("beat_model=madmom requested but madmom is unavailable or failed")

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length, units="frames")
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length).tolist()
    bpm = float(tempo) if np.size(tempo) == 1 else float(np.mean(tempo))
    return beats, bpm


def _ensure_clip_runtime() -> dict | None:
    global _CLIP_RUNTIME
    if _CLIP_RUNTIME is not None:
        return _CLIP_RUNTIME
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except Exception:
        _CLIP_RUNTIME = {}
        return _CLIP_RUNTIME

    try:
        model_id = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id)
        model.eval()
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            model = model.to(device)
        _CLIP_RUNTIME = {
            "torch": torch,
            "Image": Image,
            "processor": processor,
            "model": model,
            "device": device,
            "model_id": model_id,
        }
    except Exception:
        _CLIP_RUNTIME = {}
    return _CLIP_RUNTIME


def _get_clip_projector(in_dim: int, out_dim: int = 96) -> np.ndarray:
    global _CLIP_PROJECTOR
    if _CLIP_PROJECTOR is not None and _CLIP_PROJECTOR.shape == (in_dim, out_dim):
        return _CLIP_PROJECTOR
    rng = np.random.default_rng(20260211)
    mat = rng.normal(0.0, 1.0 / math.sqrt(max(1, in_dim)), size=(in_dim, out_dim)).astype(np.float32)
    _CLIP_PROJECTOR = mat
    return _CLIP_PROJECTOR


def _embedding_from_frames_clip(frames: Sequence[np.ndarray]) -> List[float] | None:
    runtime = _ensure_clip_runtime()
    if not runtime or "model" not in runtime:
        return None
    torch = runtime["torch"]
    processor = runtime["processor"]
    model = runtime["model"]
    Image = runtime["Image"]
    device = runtime["device"]

    if not frames:
        return None
    pick = min(8, len(frames))
    idx = np.linspace(0, len(frames) - 1, pick, dtype=int)
    images = [Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)) for i in idx]

    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        features = model.get_image_features(**inputs)
        features = features / torch.clamp(features.norm(dim=-1, keepdim=True), min=1e-6)
        pooled = features.mean(dim=0).detach().cpu().numpy().astype(np.float32)

    proj = _get_clip_projector(in_dim=int(pooled.shape[0]), out_dim=96)
    emb = pooled @ proj
    norm = float(np.linalg.norm(emb))
    if norm > 1e-9:
        emb = emb / norm
    return emb.astype(np.float32).tolist()


def _normalize_01(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    arr = values.astype(np.float32)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _smooth_signal(values: np.ndarray, win: int) -> np.ndarray:
    if values.size == 0 or win <= 1:
        return values.astype(np.float32)
    width = max(1, int(win))
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def _detect_structure_peaks(
    times: np.ndarray,
    novelty: np.ndarray,
    duration_s: float,
    min_gap_s: float = 2.5,
    quantile: float = 0.88,
) -> Tuple[List[float], List[float]]:
    if times.size < 3 or novelty.size < 3:
        return [], []

    threshold = float(np.quantile(novelty, quantile))
    step_s = float(np.median(np.diff(times))) if times.size > 1 else 0.023
    min_gap_frames = max(1, int(round(min_gap_s / max(step_s, 1e-6))))

    peak_indices: List[int] = []
    peak_values: List[float] = []
    for idx in range(1, len(novelty) - 1):
        center = float(novelty[idx])
        if center < threshold:
            continue
        if center < float(novelty[idx - 1]) or center < float(novelty[idx + 1]):
            continue

        ts_s = float(times[idx])
        if ts_s < 1.0 or ts_s > duration_s - 1.0:
            continue

        if peak_indices and idx - peak_indices[-1] < min_gap_frames:
            if center > peak_values[-1]:
                peak_indices[-1] = idx
                peak_values[-1] = center
            continue

        peak_indices.append(idx)
        peak_values.append(center)

    if not peak_indices:
        return [], []

    strength_values = _normalize_01(np.asarray(peak_values, dtype=np.float32))
    return [float(times[i]) for i in peak_indices], strength_values.astype(np.float32).tolist()


def analyze_song(song_path: str, wav_dir: str, sample_rate: int = 22050, beat_model: str = "auto") -> SongAnalysis:
    wav_path = str((Path(wav_dir) / (Path(song_path).stem + ".wav")).resolve())
    convert_audio_to_wav(song_path, wav_path, sample_rate=sample_rate)

    y, sr = librosa.load(wav_path, sr=sample_rate, mono=True)
    duration_s = float(librosa.get_duration(y=y, sr=sr))
    hop_length = 512
    beat_times, bpm = _resolve_beat_times(y=y, sr=sr, wav_path=wav_path, hop_length=hop_length, beat_model=beat_model)
    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length).astype(int)

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length).astype(np.float32)
    if not np.isfinite(bpm) or bpm <= 0:
        bpm = 120.0

    if len(beat_frames) > 0 and len(onset) > 0:
        idx = np.clip(beat_frames, 0, len(onset) - 1)
        strengths = _normalize_01(onset[idx].astype(np.float32))
        beat_strengths = strengths.tolist()
    else:
        interval = 60.0 / max(bpm, 120.0)
        beat_times = list(np.arange(0.0, duration_s, interval))
        beat_strengths = [0.5] * len(beat_times)

    rms = librosa.feature.rms(y=y, hop_length=hop_length).squeeze().astype(np.float32)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length).squeeze().astype(np.float32)

    frame_count = int(min(len(onset), len(rms), len(centroid)))
    onset = onset[:frame_count]
    rms = rms[:frame_count]
    centroid = centroid[:frame_count]
    tension_times = librosa.frames_to_time(np.arange(frame_count), sr=sr, hop_length=hop_length).astype(np.float32)

    if frame_count > 0:
        onset_n = _normalize_01(onset)
        rms_n = _normalize_01(rms)
        centroid_n = _normalize_01(centroid)
        tension_raw = 0.50 * onset_n + 0.35 * rms_n + 0.15 * centroid_n
        tension_values = _smooth_signal(tension_raw, win=9)
    else:
        tension_values = np.zeros((0,), dtype=np.float32)

    if frame_count > 1:
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length).astype(np.float32)
            chroma = chroma[:, :frame_count]
            chroma_jump = np.linalg.norm(np.diff(chroma, axis=1, prepend=chroma[:, :1]), axis=0).astype(np.float32)
        except Exception:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length).astype(np.float32)
            mfcc = mfcc[:, :frame_count]
            chroma_jump = np.linalg.norm(np.diff(mfcc, axis=1, prepend=mfcc[:, :1]), axis=0).astype(np.float32)

        rms_jump = np.abs(np.diff(_normalize_01(rms), prepend=_normalize_01(rms[:1]))).astype(np.float32)
        tension_jump = np.abs(
            np.diff(_normalize_01(tension_values), prepend=_normalize_01(tension_values[:1]))
        ).astype(np.float32)
        common_len = int(min(len(chroma_jump), len(rms_jump), len(tension_jump), len(tension_times), len(tension_values)))
        chroma_jump = chroma_jump[:common_len]
        rms_jump = rms_jump[:common_len]
        tension_jump = tension_jump[:common_len]
        tension_times = tension_times[:common_len]
        tension_values = tension_values[:common_len]
        novelty = (
            0.55 * _normalize_01(chroma_jump)
            + 0.25 * _normalize_01(tension_jump)
            + 0.20 * _normalize_01(rms_jump)
        ).astype(np.float32)
        novelty = _smooth_signal(novelty, win=11)
        structure_times, structure_strengths = _detect_structure_peaks(
            times=tension_times,
            novelty=novelty,
            duration_s=duration_s,
            min_gap_s=2.5,
            quantile=0.88,
        )
    else:
        structure_times, structure_strengths = [], []

    return SongAnalysis(
        path=str(Path(song_path).resolve()),
        wav_path=wav_path,
        duration_s=duration_s,
        bpm=bpm,
        beat_times=beat_times,
        beat_strengths=beat_strengths,
        tension_times=tension_times.tolist(),
        tension_values=tension_values.astype(np.float32).tolist(),
        structure_times=structure_times,
        structure_strengths=structure_strengths,
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
    embedding_model: str,
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
        embedding = _embedding_from_frames(emb_frames, bins=32, embedding_model=embedding_model)

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


def extract_movie_chunk_features(
    movie_path: str,
    movie_duration_s: float,
    chunk_start_s: float,
    chunk_end_s: float,
    scene_threshold: float,
    min_scene_len_s: float,
    sample_fps: float,
    min_clip_s: float,
    max_clip_s: float,
    max_clips_for_chunk: int,
    chunk_overlap_s: float,
    sample_width: int,
    use_gpu_decode: bool,
    embedding_model: str = "hist96",
) -> Tuple[List[ClipFeature], int]:
    extended_start_s = max(0.0, chunk_start_s - max(0.0, chunk_overlap_s))
    extended_end_s = min(movie_duration_s, chunk_end_s + max(0.0, chunk_overlap_s))

    timestamps, frames = _extract_chunk_sampled_frames(
        video_path=movie_path,
        chunk_start_s=extended_start_s,
        chunk_end_s=extended_end_s,
        sample_fps=sample_fps,
        sample_width=sample_width,
        use_gpu_decode=use_gpu_decode,
    )
    if len(frames) < 2:
        return [], 0

    scenes = _detect_scene_segments_from_sampled_frames(
        timestamps=timestamps,
        frames=frames,
        chunk_start_s=extended_start_s,
        chunk_end_s=extended_end_s,
        scene_threshold=scene_threshold,
        min_scene_len_s=min_scene_len_s,
    )
    segments = build_clip_segments(
        scenes=scenes,
        min_clip_s=min_clip_s,
        max_clip_s=max_clip_s,
        max_clips=max_clips_for_chunk,
    )
    segments = _restrict_segments_to_window(
        segments=segments,
        win_start_s=chunk_start_s,
        win_end_s=chunk_end_s,
        min_len_s=min_clip_s,
    )
    chunk_clips = _extract_clip_features_from_sampled_frames(
        movie_path=movie_path,
        timestamps=timestamps,
        frames=frames,
        segments=segments,
        embedding_model=embedding_model,
    )
    return chunk_clips, len(scenes)


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
    embedding_model: str = "hist96",
) -> Tuple[List[ClipFeature], dict]:
    chunks = build_time_chunks(video_meta.duration_s, min_chunk_s=chunk_min_s, max_chunk_s=chunk_max_s)
    all_clips: List[ClipFeature] = []
    total_scenes = 0

    for chunk_start_s, chunk_end_s in chunks:
        chunk_cap = max(8, int(math.ceil(max_clips_per_movie / max(1, len(chunks)))))
        chunk_clips, scene_count = extract_movie_chunk_features(
            movie_path=video_meta.path,
            movie_duration_s=video_meta.duration_s,
            chunk_start_s=chunk_start_s,
            chunk_end_s=chunk_end_s,
            scene_threshold=scene_threshold,
            min_scene_len_s=min_scene_len_s,
            sample_fps=sample_fps,
            min_clip_s=min_clip_s,
            max_clip_s=max_clip_s,
            max_clips_for_chunk=chunk_cap,
            chunk_overlap_s=chunk_overlap_s,
            sample_width=sample_width,
            use_gpu_decode=use_gpu_decode,
            embedding_model=embedding_model,
        )
        total_scenes += scene_count
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


def _embedding_from_frames(frames: Iterable[np.ndarray], bins: int = 32, embedding_model: str = "hist96") -> List[float]:
    frame_list = list(frames)
    mode = (embedding_model or "hist96").lower().strip()
    if mode in {"auto", "clip"}:
        clip_embedding = _embedding_from_frames_clip(frame_list)
        if clip_embedding is not None:
            return clip_embedding

    vectors = []
    for frame in frame_list:
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


def extract_clip_features(
    video_meta: VideoMeta,
    segments: Sequence[Tuple[float, float]],
    embedding_model: str = "hist96",
) -> List[ClipFeature]:
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
        embedding = _embedding_from_frames(frames, bins=32, embedding_model=embedding_model)

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
