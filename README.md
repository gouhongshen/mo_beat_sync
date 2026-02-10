# MatrixOne Beat Sync Pipeline

This project automatically generates a beat-synced short video from:
- movie sources in `source_movies/`
- music sources in `source_songs/`

No model training is required.

Pipeline overview:
1. Scan media folders.
2. Extract beat timing and beat strength from music (`librosa`).
3. Split each movie into time chunks (default: 2-5 minutes) and process chunks in parallel.
4. Sample frames with `ffmpeg` (CUDA decode optional), then run scene detection and feature extraction (supports chunk overlap, default: 1.5s).
5. Store clip features and embeddings in MatrixOne (`vecf32(96)`).
6. Use MatrixOne vector retrieval + SQL scoring to build an edit plan.
7. Render the final video with `ffmpeg` (NVENC optional) and mix with music.

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
- `MO_WORKERS` (parallel movie workers)
- `MO_GPU_MODE` (`auto/on/off`)
- `MO_CHUNK_MIN_MINUTES` `MO_CHUNK_MAX_MINUTES`
- `MO_CHUNK_OVERLAP_SECONDS`

## 5. MatrixOne Capabilities Used

- Vector column: `clips.embedding vecf32(96)`
- Vector distance: `l2_distance(embedding, '[...]')`
- Vector index: `HNSW` (created automatically if supported)
- SQL analytics: clip distribution, motion strength, source contribution

## 6. Windows + GPU Notes

For Windows + RTX 4080:
1. Ensure `ffmpeg -hwaccels` contains `cuda`.
2. Ensure `ffmpeg -encoders` contains `h264_nvenc`.
3. Recommended runtime args: `--gpu-mode on --workers 10 --chunk-min-minutes 2 --chunk-max-minutes 5 --chunk-overlap-seconds 1.5`.
4. If driver/codec compatibility issues happen, use `--gpu-mode auto` to allow CPU fallback.

## 7. Tables

The pipeline creates:
- `pipeline_runs`
- `songs`
- `song_beats`
- `clips`
- `edit_plans`
- `plan_items`

## 8. Example SQL

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

## 9. Notes

- First run may take longer because dependencies are installed.
- Large movie assets can make feature extraction slow, depending on CPU/GPU and disk throughput.
- If vector index is unavailable in MatrixOne, the pipeline still works but retrieval may be slower.
