from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Iterable, List, Sequence

import pymysql
from pymysql.cursors import DictCursor

from .models import ClipFeature, PlannedClip, SongAnalysis, vector_to_sql_literal


class MatrixOneClient:
    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        embedding_dim: int,
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.embedding_dim = embedding_dim
        self.conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=None,
            cursorclass=DictCursor,
            autocommit=True,
            charset="utf8mb4",
        )

    def close(self) -> None:
        self.conn.close()

    def _exec(self, sql: str, params: Sequence[Any] | None = None) -> int:
        with self.conn.cursor() as cur:
            affected = cur.execute(sql, params)
        return affected

    def _query(self, sql: str, params: Sequence[Any] | None = None) -> List[dict]:
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())

    def _executemany(self, sql: str, params: Iterable[Sequence[Any]]) -> int:
        with self.conn.cursor() as cur:
            affected = cur.executemany(sql, list(params))
        return affected

    def _exec_ignore_error(self, sql: str) -> None:
        try:
            self._exec(sql)
        except Exception:
            pass

    def ensure_schema(self, drop_existing: bool = False) -> None:
        self._exec(f"create database if not exists `{self.database}`")
        self._exec(f"use `{self.database}`")

        if drop_existing:
            for table in ("plan_items", "edit_plans", "song_beats", "clips", "songs", "pipeline_runs"):
                self._exec(f"drop table if exists {table}")

        self._exec(
            """
            create table if not exists pipeline_runs (
                run_id bigint auto_increment primary key,
                created_at timestamp default current_timestamp,
                status varchar(32) not null,
                config_json text,
                message varchar(1024)
            )
            """
        )

        self._exec(
            """
            create table if not exists songs (
                song_id bigint auto_increment primary key,
                run_id bigint not null,
                path varchar(2048) not null,
                wav_path varchar(2048) not null,
                duration_s double not null,
                bpm double not null,
                beat_count int not null,
                checksum varchar(64) not null default '',
                analysis_sig varchar(128) not null default '',
                analysis_json longtext,
                created_at timestamp default current_timestamp,
                key idx_song_run (run_id),
                key idx_song_cache (checksum, analysis_sig)
            )
            """
        )

        self._exec(
            """
            create table if not exists song_beats (
                song_id bigint not null,
                beat_idx int not null,
                ts_s double not null,
                strength double not null,
                primary key (song_id, beat_idx)
            )
            """
        )

        self._exec(
            f"""
            create table if not exists clips (
                clip_id bigint auto_increment primary key,
                run_id bigint not null,
                movie_path varchar(2048) not null,
                movie_checksum varchar(64) not null default '',
                feature_sig varchar(128) not null default '',
                start_s double not null,
                end_s double not null,
                duration_s double not null,
                motion_energy double not null,
                brightness double not null,
                saturation double not null,
                embedding vecf32({self.embedding_dim}),
                created_at timestamp default current_timestamp,
                key idx_clips_run (run_id),
                key idx_clips_asset (movie_checksum, feature_sig),
                key idx_clips_duration (duration_s),
                key idx_clips_motion (motion_energy)
            )
            """
        )

        self._exec(
            """
            create table if not exists edit_plans (
                plan_id bigint auto_increment primary key,
                run_id bigint not null,
                song_id bigint not null,
                output_path varchar(2048) not null,
                slot_count int not null,
                created_at timestamp default current_timestamp,
                key idx_plan_run (run_id)
            )
            """
        )

        self._exec(
            """
            create table if not exists plan_items (
                plan_id bigint not null,
                slot_idx int not null,
                clip_id bigint not null,
                start_s double not null,
                end_s double not null,
                source_duration_s double not null,
                target_duration_s double not null,
                stretch_rate double not null,
                score double not null,
                vec_distance double not null,
                energy_distance double not null,
                primary key (plan_id, slot_idx),
                key idx_plan_item_clip (clip_id)
            )
            """
        )

        self._exec_ignore_error("alter table songs add column checksum varchar(64) not null default ''")
        self._exec_ignore_error("alter table songs add column analysis_sig varchar(128) not null default ''")
        self._exec_ignore_error("alter table songs add column analysis_json longtext")
        self._exec_ignore_error("create index idx_song_cache on songs(checksum, analysis_sig)")
        self._exec_ignore_error("set experimental_hnsw_index=1")
        self._exec_ignore_error("alter table clips add column movie_checksum varchar(64) not null default ''")
        self._exec_ignore_error("alter table clips add column feature_sig varchar(128) not null default ''")
        self._exec_ignore_error("create index idx_clips_asset on clips(movie_checksum, feature_sig)")
        self._exec_ignore_error("drop index idx_clips_embedding_hnsw on clips")

        try:
            self._exec("set experimental_ivf_index=1")
        except Exception:
            pass

        ivf_sql_candidates = (
            """
            create index idx_clips_embedding_ivf
            using ivfflat on clips(embedding)
            op_type "vector_l2_ops" lists 256
            """,
            """
            create index idx_clips_embedding_ivf
            using ivfflat on clips(embedding)
            lists 256 op_type "vector_l2_ops"
            """,
            """
            create index idx_clips_embedding_ivf
            using ivfflat on clips(embedding)
            lists 256
            """,
        )
        for sql in ivf_sql_candidates:
            try:
                self._exec(sql)
                break
            except Exception:
                continue

    def start_run(self, config: dict) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "insert into pipeline_runs(status, config_json, message) values(%s,%s,%s)",
                ("RUNNING", json.dumps(config, ensure_ascii=False), ""),
            )
            run_id = int(cur.lastrowid)
        return run_id

    def finish_run(self, run_id: int, status: str, message: str) -> None:
        self._exec(
            "update pipeline_runs set status=%s, message=%s where run_id=%s",
            (status, message[:1024], run_id),
        )

    def insert_song(self, run_id: int, song: SongAnalysis) -> int:
        analysis_json = json.dumps(
            {
                "tension_times": song.tension_times,
                "tension_values": song.tension_values,
                "structure_times": song.structure_times,
                "structure_strengths": song.structure_strengths,
            },
            ensure_ascii=False,
        )
        with self.conn.cursor() as cur:
            cur.execute(
                """
                insert into songs(
                    run_id, path, wav_path, duration_s, bpm, beat_count,
                    checksum, analysis_sig, analysis_json
                )
                values(%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    run_id,
                    song.path,
                    song.wav_path,
                    song.duration_s,
                    song.bpm,
                    len(song.beat_times),
                    song.checksum,
                    song.analysis_sig,
                    analysis_json,
                ),
            )
            song_id = int(cur.lastrowid)

        rows = [(song_id, i, float(t), float(song.beat_strengths[i])) for i, t in enumerate(song.beat_times)]
        if rows:
            self._executemany(
                "insert into song_beats(song_id, beat_idx, ts_s, strength) values(%s,%s,%s,%s)",
                rows,
            )
        return song_id

    def insert_clips(self, run_id: int, clips: Sequence[ClipFeature]) -> int:
        if not clips:
            return 0

        rows = [
            (
                run_id,
                clip.movie_path,
                clip.movie_checksum,
                clip.feature_sig,
                clip.start_s,
                clip.end_s,
                clip.duration_s,
                clip.motion_energy,
                clip.brightness,
                clip.saturation,
                vector_to_sql_literal(clip.embedding),
            )
            for clip in clips
        ]
        return self._executemany(
            """
            insert into clips(
                run_id, movie_path, movie_checksum, feature_sig, start_s, end_s, duration_s,
                motion_energy, brightness, saturation, embedding
            ) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            rows,
        )

    def find_song_cache(self, checksum: str, analysis_sig: str) -> dict | None:
        rows = self._query(
            """
            select song_id
            from songs
            where checksum=%s and analysis_sig=%s
            order by song_id desc
            limit 1
            """,
            (checksum, analysis_sig),
        )
        return rows[0] if rows else None

    def load_song_analysis(self, song_id: int) -> SongAnalysis:
        rows = self._query(
            """
            select
                song_id, path, wav_path, duration_s, bpm, checksum, analysis_sig, analysis_json
            from songs
            where song_id=%s
            limit 1
            """,
            (song_id,),
        )
        if not rows:
            raise RuntimeError(f"Song not found: song_id={song_id}")
        row = rows[0]
        beat_rows = self._query(
            """
            select ts_s, strength
            from song_beats
            where song_id=%s
            order by beat_idx asc
            """,
            (song_id,),
        )
        beat_times = [float(x["ts_s"]) for x in beat_rows]
        beat_strengths = [float(x["strength"]) for x in beat_rows]

        payload = {}
        raw_json = row.get("analysis_json")
        if raw_json:
            if isinstance(raw_json, bytes):
                raw_json = raw_json.decode("utf-8")
            try:
                payload = json.loads(str(raw_json))
            except Exception:
                payload = {}

        return SongAnalysis(
            path=str(row["path"]),
            wav_path=str(row["wav_path"]),
            duration_s=float(row["duration_s"]),
            bpm=float(row["bpm"]),
            beat_times=beat_times,
            beat_strengths=beat_strengths,
            tension_times=[float(x) for x in payload.get("tension_times", [])],
            tension_values=[float(x) for x in payload.get("tension_values", [])],
            structure_times=[float(x) for x in payload.get("structure_times", [])],
            structure_strengths=[float(x) for x in payload.get("structure_strengths", [])],
            checksum=str(row.get("checksum", "") or ""),
            analysis_sig=str(row.get("analysis_sig", "") or ""),
        )

    def clone_song_to_run(
        self,
        run_id: int,
        cached_song_id: int,
        song_path: str,
        wav_path: str,
    ) -> int:
        cached = self._query(
            """
            select duration_s, bpm, beat_count, checksum, analysis_sig, analysis_json
            from songs
            where song_id=%s
            limit 1
            """,
            (cached_song_id,),
        )
        if not cached:
            raise RuntimeError(f"Cached song not found: song_id={cached_song_id}")
        c = cached[0]
        with self.conn.cursor() as cur:
            cur.execute(
                """
                insert into songs(
                    run_id, path, wav_path, duration_s, bpm, beat_count,
                    checksum, analysis_sig, analysis_json
                ) values(%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    run_id,
                    song_path,
                    wav_path,
                    float(c["duration_s"]),
                    float(c["bpm"]),
                    int(c["beat_count"]),
                    str(c.get("checksum", "") or ""),
                    str(c.get("analysis_sig", "") or ""),
                    c.get("analysis_json"),
                ),
            )
            new_song_id = int(cur.lastrowid)
        self._exec(
            """
            insert into song_beats(song_id, beat_idx, ts_s, strength)
            select %s, beat_idx, ts_s, strength
            from song_beats
            where song_id=%s
            """,
            (new_song_id, cached_song_id),
        )
        return new_song_id

    def count_cached_clips(self, movie_checksum: str, feature_sig: str) -> int:
        rows = self._query(
            """
            select count(*) as clip_count
            from clips
            where movie_checksum=%s and feature_sig=%s
            """,
            (movie_checksum, feature_sig),
        )
        if not rows:
            return 0
        return int(rows[0]["clip_count"] or 0)

    def count_clips_for_assets(self, movie_checksums: Sequence[str], feature_sig: str) -> int:
        checksums = [x for x in movie_checksums if x]
        if not checksums:
            return 0
        placeholders = ",".join(["%s"] * len(checksums))
        rows = self._query(
            f"""
            select count(*) as clip_count
            from clips
            where feature_sig=%s and movie_checksum in ({placeholders})
            """,
            (feature_sig, *checksums),
        )
        if not rows:
            return 0
        return int(rows[0]["clip_count"] or 0)

    def query_clip_candidates(
        self,
        movie_checksums: Sequence[str],
        feature_sig: str,
        target_duration_s: float,
        target_energy: float,
        target_tension: float,
        boundary_strength: float,
        prev_embedding_literal: str | None,
        limit: int,
        duration_tol: float,
    ) -> List[dict]:
        checksums = [x for x in movie_checksums if x]
        if not checksums:
            return []
        min_d = max(0.2, target_duration_s * (1.0 - duration_tol))
        max_d = target_duration_s * (1.0 + duration_tol)
        target_tension = max(0.0, min(1.0, float(target_tension)))
        boundary_strength = max(0.0, min(1.0, float(boundary_strength)))
        target_transition = 0.10 + 0.80 * boundary_strength
        raw_tension_expr = "(0.62 * motion_energy + 0.23 * saturation + 0.15 * brightness)"
        clip_tension_expr = (
            f"(case "
            f"when {raw_tension_expr} < cast(0 as double) then cast(0 as double) "
            f"when {raw_tension_expr} > cast(1 as double) then cast(1 as double) "
            f"else cast({raw_tension_expr} as double) end)"
        )

        placeholders = ",".join(["%s"] * len(checksums))

        if prev_embedding_literal is not None:
            sql = """
                select
                    clip_id,
                    movie_path,
                    start_s,
                    end_s,
                    duration_s,
                    motion_energy,
                    brightness,
                    saturation,
                    embedding,
                    """
            sql += f"""
                    {clip_tension_expr} as clip_tension,
                    l2_distance(embedding, %s) as vec_distance,
                    abs(l2_distance(embedding, %s) - %s) as transition_distance,
                    abs(motion_energy - %s) as energy_distance,
                    abs(({clip_tension_expr}) - %s) as tension_distance,
                    abs(duration_s - %s) as duration_distance,
                    (
                        0.26 * abs(duration_s - %s)
                        + 0.20 * abs(motion_energy - %s)
                        + 0.24 * abs(({clip_tension_expr}) - %s)
                        + 0.30 * abs(l2_distance(embedding, %s) - %s)
                    ) as score
                from clips
                where feature_sig = %s and movie_checksum in ("""
            sql += placeholders
            sql += """) and duration_s between %s and %s
                order by score asc
                limit %s
            """
            params = (
                prev_embedding_literal,
                prev_embedding_literal,
                target_transition,
                target_energy,
                target_tension,
                target_duration_s,
                target_duration_s,
                target_energy,
                target_tension,
                prev_embedding_literal,
                target_transition,
                feature_sig,
                *checksums,
                min_d,
                max_d,
                limit,
            )
        else:
            sql = """
                select
                    clip_id,
                    movie_path,
                    start_s,
                    end_s,
                    duration_s,
                    motion_energy,
                    brightness,
                    saturation,
                    embedding,
                    """
            sql += f"""
                    {clip_tension_expr} as clip_tension,
                    0.0 as vec_distance,
                    0.0 as transition_distance,
                    abs(motion_energy - %s) as energy_distance,
                    abs(({clip_tension_expr}) - %s) as tension_distance,
                    abs(duration_s - %s) as duration_distance,
                    (
                        0.34 * abs(duration_s - %s)
                        + 0.30 * abs(motion_energy - %s)
                        + 0.36 * abs(({clip_tension_expr}) - %s)
                    ) as score
                from clips
                where feature_sig = %s and movie_checksum in ("""
            sql += placeholders
            sql += """) and duration_s between %s and %s
                order by score asc
                limit %s
            """
            params = (
                target_energy,
                target_tension,
                target_duration_s,
                target_duration_s,
                target_energy,
                target_tension,
                feature_sig,
                *checksums,
                min_d,
                max_d,
                limit,
            )
        return self._query(sql, params)

    def create_plan(self, run_id: int, song_id: int, output_path: str, slot_count: int) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "insert into edit_plans(run_id, song_id, output_path, slot_count) values(%s,%s,%s,%s)",
                (run_id, song_id, output_path, slot_count),
            )
            plan_id = int(cur.lastrowid)
        return plan_id

    def insert_plan_items(self, plan_id: int, items: Sequence[PlannedClip]) -> int:
        if not items:
            return 0
        rows = [
            (
                plan_id,
                it.slot_idx,
                it.clip_id,
                it.start_s,
                it.end_s,
                it.source_duration_s,
                it.target_duration_s,
                it.stretch_rate,
                it.score,
                it.vec_distance,
                it.energy_distance,
            )
            for it in items
        ]
        return self._executemany(
            """
            insert into plan_items(
                plan_id, slot_idx, clip_id, start_s, end_s,
                source_duration_s, target_duration_s, stretch_rate,
                score, vec_distance, energy_distance
            ) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            rows,
        )

    def get_run_summary(self, run_id: int) -> dict:
        clip_stats = self._query(
            """
            select
                count(*) as clip_count,
                round(avg(duration_s), 4) as avg_duration,
                round(avg(motion_energy), 4) as avg_motion,
                round(min(duration_s), 4) as min_duration,
                round(max(duration_s), 4) as max_duration
            from clips
            where run_id=%s
            """,
            (run_id,),
        )[0]

        top_movies = self._query(
            """
            select
                movie_path,
                count(*) as clip_count,
                round(avg(motion_energy), 4) as avg_motion,
                round(avg(duration_s), 4) as avg_duration
            from clips
            where run_id=%s
            group by movie_path
            order by clip_count desc
            limit 10
            """,
            (run_id,),
        )

        return {
            "clip_stats": clip_stats,
            "top_movies": top_movies,
        }
