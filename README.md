# MatrixOne Beat Sync Pipeline

这个项目会把 `source_movies/` 的电影素材和 `source_songs/` 的音乐自动处理成一条踩点视频，且全程不训练模型。

流程：
1. 扫描素材目录。
2. 从音乐中提取节拍和强弱（`librosa`）。
3. 把每部电影按时长切成多个块（默认 2~5 分钟），并发处理每个块。
4. 用 `ffmpeg`（可启用 CUDA 解码）按低帧率抽样，再做镜头切分和片段特征提取（支持块间 overlap，默认 1.5 秒）。
5. 把片段特征与向量写入 MatrixOne（`vecf32(96)`）。
6. 用 MatrixOne 向量检索 + SQL 打分生成剪辑计划。
7. 用 `ffmpeg`（可启用 NVENC 编码）自动渲染并叠加音乐输出最终视频。

## 1. 目录约定

在项目根目录下放置：
- `source_movies/`：十几部电影素材（mp4/mkv/mov...）
- `source_songs/`：十几首音乐（mp3/wav/m4a...）

输出：
- `outputs/final/beat_sync.mp4`
- `workdir/run_<run_id>/report.json`

## 2. 启动 MatrixOne（可选参考）

如果你本地还没起服务，可在 MatrixOne 仓库下执行：

```bash
source $HOME/.zshrc && moenv
./mo-service -debug-http=:11235 -launch etc/launch/launch.toml > log.log 2>&1 &
```

验证连接：

```bash
mysql -h 127.0.0.1 -P 6001 -u root -p111 -e "select 1"
```

## 3. 安装依赖并运行

### 一键运行

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

### 手动运行

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

## 4. 常用参数

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

数据库参数可通过命令行或环境变量：
- `MO_HOST` `MO_PORT` `MO_USER` `MO_PASSWORD` `MO_DATABASE`
- `MO_WORKERS`（电影并发处理进程数）
- `MO_GPU_MODE`（`auto/on/off`）
- `MO_CHUNK_MIN_MINUTES` `MO_CHUNK_MAX_MINUTES`
- `MO_CHUNK_OVERLAP_SECONDS`

## 5. MatrixOne 用到的能力

- 向量列：`clips.embedding vecf32(96)`
- 向量函数：`l2_distance(embedding, '[...]')`
- 向量索引：`HNSW`（若集群支持会自动创建）
- 分析 SQL：统计片段分布、运动强度、各素材贡献等

## 6. Windows + GPU 建议

如果你是 Windows + RTX 4080，建议：
1. 确保 `ffmpeg -hwaccels` 能看到 `cuda`，且 `ffmpeg -encoders` 有 `h264_nvenc`。
2. 运行时设置：`--gpu-mode on --workers 10 --chunk-min-minutes 2 --chunk-max-minutes 5`。
   建议加上 `--chunk-overlap-seconds 1.5`，减少块边界丢片段。
3. 如果出现驱动/编解码兼容问题，先切到 `--gpu-mode auto`，程序会回退到 CPU 路径。

## 7. 数据表

自动创建以下表：
- `pipeline_runs`
- `songs`
- `song_beats`
- `clips`
- `edit_plans`
- `plan_items`

## 8. 示例分析 SQL

```sql
use mo_beat_sync;

-- 每个素材贡献了多少候选片段
select movie_path, count(*) as clips
from clips
group by movie_path
order by clips desc
limit 20;

-- 查看某次计划的片段质量
select slot_idx, clip_id, score, vec_distance, energy_distance, target_duration_s
from plan_items
where plan_id = 1
order by slot_idx;
```

## 9. 注意事项

- 首次跑依赖下载会比较慢。
- 电影素材较大时，特征提取会花较久（取决于机器性能）。
- 如果 MatrixOne 没有启用向量索引，本项目仍可运行，只是召回性能会下降。
