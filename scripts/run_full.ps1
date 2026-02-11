$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

if (!(Test-Path ".venv")) {
  python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
python -m pip install -U pip | Out-Null
python -m pip install -r requirements.txt
if ($env:MO_INSTALL_ML -eq "1") {
  python -m pip install -r requirements-ml.txt
}
if ($env:MO_INSTALL_MADMOM -eq "1") {
  try {
    python -m pip install -r requirements-ml-beat.txt
  } catch {
    Write-Host "[WARN] madmom install failed, fallback to librosa beat backend."
  }
}

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
if ($env:MO_BEAT_MODEL) {
  $ArgsList += @("--beat-model", $env:MO_BEAT_MODEL)
}
if ($env:MO_EMBEDDING_MODEL) {
  $ArgsList += @("--embedding-model", $env:MO_EMBEDDING_MODEL)
}
if ($env:MO_PLANNER_BEAM_WIDTH) {
  $ArgsList += @("--planner-beam-width", $env:MO_PLANNER_BEAM_WIDTH)
}
if ($env:MO_PLANNER_PER_STATE_CANDIDATES) {
  $ArgsList += @("--planner-per-state-candidates", $env:MO_PLANNER_PER_STATE_CANDIDATES)
}
if ($env:MO_PLANNER_SLOT_SKIP_PENALTY) {
  $ArgsList += @("--planner-slot-skip-penalty", $env:MO_PLANNER_SLOT_SKIP_PENALTY)
}
if ($env:MO_PLANNER_MIN_REUSE_GAP) {
  $ArgsList += @("--planner-min-reuse-gap", $env:MO_PLANNER_MIN_REUSE_GAP)
}
if ($env:MO_PLANNER_REUSE_PENALTY) {
  $ArgsList += @("--planner-reuse-penalty", $env:MO_PLANNER_REUSE_PENALTY)
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
