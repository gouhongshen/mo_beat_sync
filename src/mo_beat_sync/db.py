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
                created_at timestamp default current_timestamp,
                key idx_song_run (run_id)
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
                start_s double not null,
                end_s double not null,
                duration_s double not null,
                motion_energy double not null,
                brightness double not null,
                saturation double not null,
                embedding vecf32({self.embedding_dim}),
                created_at timestamp default current_timestamp,
                key idx_clips_run (run_id),
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

        try:
            self._exec("set experimental_hnsw_index=1")
            self._exec(
                """
                create index idx_clips_embedding_hnsw
                using hnsw on clips(embedding)
                op_type "vector_l2_ops" m 32 ef_construction 64 ef_search 64
                """
            )
        except Exception:
            # Index already exists or vector index is disabled.
            pass

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
        with self.conn.cursor() as cur:
            cur.execute(
                """
                insert into songs(run_id, path, wav_path, duration_s, bpm, beat_count)
                values(%s,%s,%s,%s,%s,%s)
                """,
                (run_id, song.path, song.wav_path, song.duration_s, song.bpm, len(song.beat_times)),
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
                run_id, movie_path, start_s, end_s, duration_s,
                motion_energy, brightness, saturation, embedding
            ) values(%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            rows,
        )

    def query_clip_candidates(
        self,
        run_id: int,
        target_duration_s: float,
        target_energy: float,
        prev_embedding_literal: str | None,
        limit: int,
        duration_tol: float,
    ) -> List[dict]:
        min_d = max(0.2, target_duration_s * (1.0 - duration_tol))
        max_d = target_duration_s * (1.0 + duration_tol)

        if prev_embedding_literal is not None:
            sql = """
                select
                    clip_id,
                    movie_path,
                    start_s,
                    end_s,
                    duration_s,
                    motion_energy,
                    embedding,
                    l2_distance(embedding, %s) as vec_distance,
                    abs(motion_energy - %s) as energy_distance,
                    abs(duration_s - %s) as duration_distance,
                    (
                        0.55 * l2_distance(embedding, %s)
                        + 0.30 * abs(motion_energy - %s)
                        + 0.15 * abs(duration_s - %s)
                    ) as score
                from clips
                where run_id = %s and duration_s between %s and %s
                order by score asc
                limit %s
            """
            params = (
                prev_embedding_literal,
                target_energy,
                target_duration_s,
                prev_embedding_literal,
                target_energy,
                target_duration_s,
                run_id,
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
                    embedding,
                    0.0 as vec_distance,
                    abs(motion_energy - %s) as energy_distance,
                    abs(duration_s - %s) as duration_distance,
                    (
                        0.70 * abs(motion_energy - %s)
                        + 0.30 * abs(duration_s - %s)
                    ) as score
                from clips
                where run_id = %s and duration_s between %s and %s
                order by score asc
                limit %s
            """
            params = (
                target_energy,
                target_duration_s,
                target_energy,
                target_duration_s,
                run_id,
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
