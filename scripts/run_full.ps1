$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

if (!(Test-Path ".venv")) {
  python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
python -m pip install -U pip | Out-Null
python -m pip install -r requirements.txt

New-Item -ItemType Directory -Force -Path source_movies | Out-Null
New-Item -ItemType Directory -Force -Path source_songs | Out-Null
New-Item -ItemType Directory -Force -Path outputs\final | Out-Null
New-Item -ItemType Directory -Force -Path workdir | Out-Null

$ArgsList = @(
  "-m", "mo_beat_sync",
  "--movies-dir", "source_movies",
  "--songs-dir", "source_songs",
  "--output-path", "outputs/final/beat_sync.mp4",
  "--workdir", "workdir"
)

if ($env:MO_WORKERS) {
  $ArgsList += @("--workers", $env:MO_WORKERS)
}
if ($env:MO_GPU_MODE) {
  $ArgsList += @("--gpu-mode", $env:MO_GPU_MODE)
}
if ($env:MO_CHUNK_MIN_MINUTES -and $env:MO_CHUNK_MAX_MINUTES) {
  $ArgsList += @("--chunk-min-minutes", $env:MO_CHUNK_MIN_MINUTES)
  $ArgsList += @("--chunk-max-minutes", $env:MO_CHUNK_MAX_MINUTES)
}
if ($env:MO_CHUNK_OVERLAP_SECONDS) {
  $ArgsList += @("--chunk-overlap-seconds", $env:MO_CHUNK_OVERLAP_SECONDS)
}

$env:PYTHONPATH = "src"
python @ArgsList
