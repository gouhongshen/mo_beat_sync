# MatrixOne Beat Sync Pipeline

This project automatically generates a beat-synced short video from:
- movie sources in `source_movies/`
- music sources in `source_songs/`

No model training is required.

Pipeline overview:
1. Scan media folders.
2. Extract beat timing, structural boundaries, and a tension curve from music (`madmom/librosa`).
3. Split each movie into time chunks (default: 2-5 minutes) and process chunks in parallel.
4. Sample frames with `ffmpeg` (CUDA decode optional), then run scene detection and feature extraction (supports chunk overlap, default: 1.5s).
5. Store clip features and embeddings in MatrixOne (`vecf32(96)`).
6. Reuse cached song/video analysis by checksum when inputs are unchanged.
7. Use MatrixOne vector retrieval + SQL scoring (duration + energy + tension + transition) to build an edit plan.
8. Render the final video with `ffmpeg` (NVENC optional) and mix with music.

## 1. Folder Layout

Under project root:
- `source_movies/`: movie assets (`mp4/mkv/mov/...`)
- `source_songs/`: music assets (`mp3/wav/m4a/...`)

Outputs:
- `outputs/final/beat_sync.mp4`
- `workdir/run_<run_id>/report.json`

## 2. Start MatrixOne (Optional Reference)

If MatrixOne is not running, from your MatrixOne repo:

```bash
source $HOME/.zshrc && moenv
./mo-service -debug-http=:11235 -launch etc/launch/launch.toml > log.log 2>&1 &
```

Connection check:

```bash
mysql -h 127.0.0.1 -P 6001 -u root -p111 -e "select 1"
```

## 3. Install and Run

### One-command Run

Linux/macOS:

```bash
cd /Users/ghs-mo/MOWorkSpace/experiments/mo_beat_sync
export MO_GPU_MODE=on
export MO_WORKERS=10
export MO_CHUNK_MIN_MINUTES=2
export MO_CHUNK_MAX_MINUTES=5
export MO_CHUNK_OVERLAP_SECONDS=1.5
./scripts/run_full.sh
```

Windows PowerShell:

```powershell
cd /Users/ghs-mo/MOWorkSpace/experiments/mo_beat_sync
$env:MO_GPU_MODE="on"
$env:MO_WORKERS="10"
$env:MO_CHUNK_MIN_MINUTES="2"
$env:MO_CHUNK_MAX_MINUTES="5"
$env:MO_CHUNK_OVERLAP_SECONDS="1.5"
./scripts/run_full.ps1
```

### Manual Run

```bash
cd /Users/ghs-mo/MOWorkSpace/experiments/mo_beat_sync
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python -m mo_beat_sync \
  --movies-dir source_movies \
  --songs-dir source_songs \
  --output-path outputs/final/beat_sync.mp4 \
  --workdir workdir
```

## 4. Common Parameters

```bash
PYTHONPATH=src python -m mo_beat_sync \
  --song-keyword "edm" \
  --target-duration-s 60 \
  --beats-per-clip 2 \
  --beat-model auto \
  --embedding-model auto \
  --planner-beam-width 10 \
  --planner-per-state-candidates 18 \
  --planner-slot-skip-penalty 0.45 \
  --planner-min-reuse-gap 4 \
  --planner-reuse-penalty 0.14 \
  --gpu-mode on \
  --workers 6 \
  --chunk-min-minutes 2 \
  --chunk-max-minutes 5 \
  --chunk-overlap-seconds 1.5 \
  --sample-width 320 \
  --max-movies 12 \
  --max-clips-per-movie 120 \
  --drop-existing
```

Environment variables:
- `MO_HOST` `MO_PORT` `MO_USER` `MO_PASSWORD` `MO_DATABASE`
- `MO_WORKERS` (parallel chunk workers)
- `MO_GPU_MODE` (`auto/on/off`)
- `MO_BEAT_MODEL` (`auto/librosa/madmom`)
- `MO_EMBEDDING_MODEL` (`auto/hist96/clip`)
- `MO_PLANNER_BEAM_WIDTH` (default `10`)
- `MO_PLANNER_PER_STATE_CANDIDATES` (default `18`)
- `MO_PLANNER_SLOT_SKIP_PENALTY` (default `0.45`)
- `MO_PLANNER_MIN_REUSE_GAP` (default `4`)
- `MO_PLANNER_REUSE_PENALTY` (default `0.14`)
- `MO_INSTALL_ML` (`1` means install `requirements-ml.txt` in run scripts)
- `MO_INSTALL_MADMOM` (`1` means try to install `requirements-ml-beat.txt`; failure falls back to librosa)
- `MO_CHUNK_MIN_MINUTES` `MO_CHUNK_MAX_MINUTES`
- `MO_CHUNK_OVERLAP_SECONDS`

## 5. MatrixOne Capabilities Used

- Vector column: `clips.embedding vecf32(96)`
- Vector distance: `l2_distance(embedding, '[...]')`
- Vector index: `IVF Flat` (created automatically if supported)
- SQL scoring analytics: duration distance, energy distance, tension distance, transition distance
- Post-run analytics in `report.json`: boundary slot counts, average tension/transition distances
- Input cache keys: song/video SHA-256 checksum + processing signature

## 6. Optional Pretrained Models (No Training)

You can improve beat and visual semantic quality with pretrained small models:
- Beat tracking: `madmom` (`--beat-model madmom` or `auto`)
- Visual embedding: `CLIP ViT-B/32` via `transformers` (`--embedding-model clip` or `auto`)

Install optional dependencies:

```bash
pip install -r requirements-ml.txt
```

Install optional madmom beat backend (may fail on some platforms/Python toolchains):

```bash
pip install -r requirements-ml-beat.txt
```

If these packages are not installed, `auto` mode falls back to baseline (`librosa` + `hist96`) automatically.

## 7. Windows + GPU Notes

For Windows + RTX 4080:
1. Ensure `ffmpeg -hwaccels` contains `cuda`.
2. Ensure `ffmpeg -encoders` contains `h264_nvenc`.
3. Recommended runtime args: `--gpu-mode on --workers 10 --chunk-min-minutes 2 --chunk-max-minutes 5 --chunk-overlap-seconds 1.5`.
   `workers` is chunk-level now, so it can speed up even with a single movie.
4. If driver/codec compatibility issues happen, use `--gpu-mode auto` to allow CPU fallback.

## 8. Tables

The pipeline creates:
- `pipeline_runs`
- `songs`
- `song_beats`
- `clips`
- `edit_plans`
- `plan_items`

## 9. Example SQL

```sql
use mo_beat_sync;

-- Clip contribution by source movie
select movie_path, count(*) as clips
from clips
group by movie_path
order by clips desc
limit 20;

-- Plan item quality for a specific plan
select slot_idx, clip_id, score, vec_distance, energy_distance, target_duration_s
from plan_items
where plan_id = 1
order by slot_idx;
```

## 10. Notes

- First run may take longer because dependencies are installed.
- Large movie assets can make feature extraction slow, depending on CPU/GPU and disk throughput.
- If vector index is unavailable in MatrixOne, the pipeline still works but retrieval may be slower.
