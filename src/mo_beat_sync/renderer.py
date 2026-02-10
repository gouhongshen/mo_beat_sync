from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

from .models import PlannedClip


class RenderError(RuntimeError):
    pass


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RenderError(f"Command failed: {' '.join(cmd)}\n{proc.stderr.strip()}")


def _render_single_clip(
    clip: PlannedClip,
    out_path: str,
    width: int,
    height: int,
    fps: int,
    use_gpu_encode: bool,
) -> None:
    duration_s = max(0.01, clip.end_s - clip.start_s)
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        f"setsar=1,"
        f"fps={fps},"
        f"setpts=PTS/{clip.stretch_rate:.8f},"
        f"trim=duration={clip.target_duration_s:.6f},"
        "setpts=PTS-STARTPTS"
    )
    cmd_gpu = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{clip.start_s:.6f}",
        "-t",
        f"{duration_s:.6f}",
        "-i",
        clip.movie_path,
        "-vf",
        vf,
        "-an",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-cq",
        "23",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]
    cmd_cpu = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{clip.start_s:.6f}",
        "-t",
        f"{duration_s:.6f}",
        "-i",
        clip.movie_path,
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]
    if use_gpu_encode:
        try:
            _run(cmd_gpu)
            return
        except RenderError:
            _run(cmd_cpu)
            return
    _run(cmd_cpu)


def render_final_video(
    plan: Sequence[PlannedClip],
    song_path: str,
    output_path: str,
    temp_dir: str,
    width: int,
    height: int,
    fps: int,
    use_gpu_encode: bool,
) -> str:
    if not plan:
        raise RenderError("No plan items available for rendering")

    temp_path = Path(temp_dir).resolve()
    temp_path.mkdir(parents=True, exist_ok=True)
    part_dir = temp_path / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    part_files: list[str] = []
    for idx, clip in enumerate(plan):
        part_file = str((part_dir / f"part_{idx:04d}.mp4").resolve())
        _render_single_clip(clip, part_file, width=width, height=height, fps=fps, use_gpu_encode=use_gpu_encode)
        part_files.append(part_file)

    concat_file = temp_path / "concat.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for p in part_files:
            escaped = p.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    assembled = str((temp_path / "assembled.mp4").resolve())
    assemble_gpu = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-cq",
        "21",
        "-pix_fmt",
        "yuv420p",
        "-an",
        assembled,
    ]
    assemble_cpu = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-an",
        assembled,
    ]
    if use_gpu_encode:
        try:
            _run(assemble_gpu)
        except RenderError:
            _run(assemble_cpu)
    else:
        _run(assemble_cpu)

    output = str(Path(output_path).resolve())
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            assembled,
            "-i",
            song_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            output,
        ]
    )
    return output
