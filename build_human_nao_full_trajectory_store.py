from __future__ import annotations

import argparse
import csv
import gzip
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from human_nao_event_cache import parse_member_events
from human_nao_source import HumanNAODataSource
from plot_human_nao_random_plus_best import build_game_index
from plot_human_nao_trajectories import load_achievements


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs"
DEFAULT_DB_PATH = OUTPUT_DIR / "human_nao_full_trajectories.sqlite3"
DEFAULT_MANIFEST_CSV = OUTPUT_DIR / "human_nao_full_trajectory_manifest.csv"
DEFAULT_EVENTS_CSV_GZ = OUTPUT_DIR / "human_nao_full_trajectory_events.csv.gz"


GameDict = dict[str, object]
EventTuple = tuple[int, int, float, float, int, int, int, int, int, str]
ParsedGameTuple = tuple[GameDict, list[EventTuple], float, float, int, int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse all stitched human NetHack games once and store sparse "
            "trajectories in SQLite plus optional CSV exports."
        )
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--events-csv-gz", type=Path, default=DEFAULT_EVENTS_CSV_GZ)
    parser.add_argument("--limit", type=int, default=0, help="Only parse the first N stitched games.")
    parser.add_argument(
        "--commit-every",
        type=int,
        default=250,
        help="Commit DB writes every N parsed games.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing outputs before rebuilding.",
    )
    parser.add_argument(
        "--skip-csv-export",
        action="store_true",
        help="Build the SQLite store only.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parser worker processes. 0 means all logical cores.",
    )
    parser.add_argument(
        "--chunk-games",
        type=int,
        default=128,
        help="Stitched games parsed per worker task.",
    )
    return parser.parse_args()


def create_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS games (
            game_key TEXT PRIMARY KEY,
            player_name TEXT NOT NULL,
            local_gameid INTEGER NOT NULL,
            death TEXT NOT NULL,
            version TEXT NOT NULL,
            turns INTEGER NOT NULL,
            points INTEGER NOT NULL,
            maxlvl INTEGER NOT NULL,
            member_count INTEGER NOT NULL,
            final_score REAL NOT NULL,
            final_progression_pct REAL NOT NULL,
            final_dlvl INTEGER NOT NULL,
            final_xp INTEGER NOT NULL,
            final_home_level INTEGER NOT NULL,
            reached_astral INTEGER NOT NULL,
            ascended INTEGER NOT NULL,
            event_count INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            game_key TEXT NOT NULL,
            event_idx INTEGER NOT NULL,
            turn INTEGER NOT NULL,
            score REAL NOT NULL,
            progression_pct REAL NOT NULL,
            dlvl INTEGER NOT NULL,
            xp INTEGER NOT NULL,
            home_level INTEGER NOT NULL,
            reached_astral INTEGER NOT NULL,
            ascended INTEGER NOT NULL,
            member_name TEXT NOT NULL,
            PRIMARY KEY (game_key, event_idx)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_events_game_turn
        ON events (game_key, turn)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_games_rank
        ON games (final_progression_pct DESC, final_score DESC)
        """
    )
    conn.commit()


def maybe_remove_outputs(args: argparse.Namespace) -> None:
    if not args.overwrite:
        return

    for path in (args.db_path, args.manifest_csv, args.events_csv_gz):
        if path.exists():
            path.unlink()

    for suffix in ("-wal", "-shm"):
        extra = Path(str(args.db_path) + suffix)
        if extra.exists():
            extra.unlink()


def load_done_game_keys(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT game_key FROM games").fetchall()
    return {row[0] for row in rows}


def game_key(game: GameDict) -> str:
    return f"{game['player_name']}#{int(game['local_gameid'])}"


def build_sparse_game_events(
    source_base_dir: Path,
    game: GameDict,
    achievements: dict[str, float],
) -> tuple[list[EventTuple], float, float, int, int, int, int, int]:
    expected_turns = int(game["turns"])
    current_score = 0.0
    current_progression = 0.0
    current_dlvl = 0
    current_xp = 0
    current_home = 0
    current_astral = 0
    current_ascended = 0
    last_turn = -1
    events: list[EventTuple] = []

    for member_name in list(game["members"]):
        payload = parse_member_events(
            member_path=source_base_dir / Path(member_name),
            achievements=achievements,
            max_turn=expected_turns,
            expected_turns=expected_turns,
        )

        for turn, score, progression_pct, dlvl, xp, home_level, reached_astral, ascended in zip(
            payload["turns"],
            payload["scores"],
            payload["progressions"],
            payload["dlvls"],
            payload["xps"],
            payload["homes"],
            payload["astrals"],
            payload["ascends"],
        ):
            turn = int(turn)
            score = float(score)
            progression_pct = float(progression_pct)
            dlvl = int(dlvl)
            xp = int(xp)
            home_level = int(home_level)
            reached_astral = int(reached_astral)
            ascended = int(ascended)

            if turn < last_turn:
                continue

            current_score = max(current_score, score)
            current_progression = max(current_progression, progression_pct)
            current_dlvl = max(current_dlvl, dlvl)
            current_xp = max(current_xp, xp)
            current_home = max(current_home, home_level)
            current_astral = max(current_astral, reached_astral)
            current_ascended = max(current_ascended, ascended)

            if events and turn == events[-1][1]:
                event_idx, _, prev_score, prev_progression, prev_dlvl, prev_xp, prev_home, prev_astral, prev_ascended, prev_member = events[-1]
                events[-1] = (
                    event_idx,
                    turn,
                    max(prev_score, current_score),
                    max(prev_progression, current_progression),
                    max(prev_dlvl, current_dlvl),
                    max(prev_xp, current_xp),
                    max(prev_home, current_home),
                    max(prev_astral, current_astral),
                    max(prev_ascended, current_ascended),
                    prev_member if prev_member == member_name else member_name,
                )
            elif (
                not events
                or current_score != events[-1][2]
                or current_progression != events[-1][3]
                or current_dlvl != events[-1][4]
                or current_xp != events[-1][5]
                or current_home != events[-1][6]
                or current_astral != events[-1][7]
                or current_ascended != events[-1][8]
            ):
                events.append(
                    (
                        len(events),
                        turn,
                        current_score,
                        current_progression,
                        current_dlvl,
                        current_xp,
                        current_home,
                        current_astral,
                        current_ascended,
                        member_name,
                    )
                )

            last_turn = turn

    if not events:
        return [], 0.0, 0.0, 0, 0, 0, 0, 0

    return (
        events,
        float(events[-1][2]),
        float(events[-1][3]),
        int(events[-1][4]),
        int(events[-1][5]),
        int(events[-1][6]),
        int(events[-1][7]),
        int(events[-1][8]),
    )


def chunk_games(games: list[GameDict], chunk_size: int) -> list[list[GameDict]]:
    return [games[i : i + chunk_size] for i in range(0, len(games), chunk_size)]


def parse_game_chunk(
    source_base_dir_str: str,
    games: list[GameDict],
    achievements: dict[str, float],
) -> list[ParsedGameTuple]:
    source_base_dir = Path(source_base_dir_str)
    parsed: list[ParsedGameTuple] = []
    for game in games:
        (
            events,
            final_score,
            final_progression_pct,
            final_dlvl,
            final_xp,
            final_home_level,
            reached_astral,
            ascended,
        ) = build_sparse_game_events(
            source_base_dir=source_base_dir,
            game=game,
            achievements=achievements,
        )
        parsed.append(
            (
                game,
                events,
                final_score,
                final_progression_pct,
                final_dlvl,
                final_xp,
                final_home_level,
                reached_astral,
                ascended,
            )
        )
    return parsed


def insert_game(
    conn: sqlite3.Connection,
    game: GameDict,
    events: list[EventTuple],
    final_score: float,
    final_progression_pct: float,
    final_dlvl: int,
    final_xp: int,
    final_home_level: int,
    reached_astral: int,
    ascended: int,
) -> None:
    key = game_key(game)
    conn.execute("DELETE FROM events WHERE game_key = ?", (key,))
    conn.execute("DELETE FROM games WHERE game_key = ?", (key,))
    conn.execute(
        """
        INSERT INTO games (
            game_key,
            player_name,
            local_gameid,
            death,
            version,
            turns,
            points,
            maxlvl,
            member_count,
            final_score,
            final_progression_pct,
            final_dlvl,
            final_xp,
            final_home_level,
            reached_astral,
            ascended,
            event_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            key,
            str(game["player_name"]),
            int(game["local_gameid"]),
            str(game["death"]),
            str(game["version"]),
            int(game["turns"]),
            int(game["points"]),
            int(game["maxlvl"]),
            len(list(game["members"])),
            final_score,
            final_progression_pct,
            final_dlvl,
            final_xp,
            final_home_level,
            reached_astral,
            ascended,
            len(events),
        ),
    )
    conn.executemany(
        """
        INSERT INTO events (
            game_key,
            event_idx,
            turn,
            score,
            progression_pct,
            dlvl,
            xp,
            home_level,
            reached_astral,
            ascended,
            member_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                key,
                event_idx,
                turn,
                score,
                progression_pct,
                dlvl,
                xp,
                home_level,
                reached_astral,
                ascended,
                member_name,
            )
            for event_idx, turn, score, progression_pct, dlvl, xp, home_level, reached_astral, ascended, member_name in events
        ],
    )


def process_parsed_batch(conn: sqlite3.Connection, parsed_batch: list[ParsedGameTuple]) -> tuple[int, int]:
    game_count = 0
    event_count = 0
    for (
        game,
        events,
        final_score,
        final_progression_pct,
        final_dlvl,
        final_xp,
        final_home_level,
        reached_astral,
        ascended,
    ) in parsed_batch:
        insert_game(
            conn=conn,
            game=game,
            events=events,
            final_score=final_score,
            final_progression_pct=final_progression_pct,
            final_dlvl=final_dlvl,
            final_xp=final_xp,
            final_home_level=final_home_level,
            reached_astral=reached_astral,
            ascended=ascended,
        )
        game_count += 1
        event_count += len(events)
    return game_count, event_count


def export_manifest_csv(conn: sqlite3.Connection, manifest_csv: Path) -> None:
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "game_key",
                "player_name",
                "local_gameid",
                "death",
                "version",
                "turns",
                "points",
                "maxlvl",
                "member_count",
                "final_score",
                "final_progression_pct",
                "final_dlvl",
                "final_xp",
                "final_home_level",
                "reached_astral",
                "ascended",
                "event_count",
            ]
        )
        cursor = conn.execute(
            """
            SELECT
                game_key,
                player_name,
                local_gameid,
                death,
                version,
                turns,
                points,
                maxlvl,
                member_count,
                final_score,
                final_progression_pct,
                final_dlvl,
                final_xp,
                final_home_level,
                reached_astral,
                ascended,
                event_count
            FROM games
            ORDER BY player_name, local_gameid
            """
        )
        writer.writerows(cursor)


def export_events_csv_gz(conn: sqlite3.Connection, events_csv_gz: Path) -> None:
    events_csv_gz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(events_csv_gz, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "game_key",
                "event_idx",
                "turn",
                "score",
                "progression_pct",
                "dlvl",
                "xp",
                "home_level",
                "reached_astral",
                "ascended",
                "member_name",
            ]
        )
        cursor = conn.execute(
            """
            SELECT
                game_key,
                event_idx,
                turn,
                score,
                progression_pct,
                dlvl,
                xp,
                home_level,
                reached_astral,
                ascended,
                member_name
            FROM events
            ORDER BY game_key, event_idx
            """
        )
        while True:
            rows = cursor.fetchmany(50000)
            if not rows:
                break
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    args.db_path = args.db_path.resolve()
    args.manifest_csv = args.manifest_csv.resolve()
    args.events_csv_gz = args.events_csv_gz.resolve()

    maybe_remove_outputs(args)

    achievements = load_achievements()

    with HumanNAODataSource() as source:
        if source.mode != "dir":
            raise SystemExit("Expected extracted human data directory, but source is not in directory mode.")
        stitched_games = build_game_index(source)
        source_base_dir = source.base_dir

    if args.limit > 0:
        stitched_games = stitched_games[: args.limit]

    workers = args.workers if args.workers > 0 else max(1, os.cpu_count() or 1)
    chunk_size = max(1, args.chunk_games)

    conn = sqlite3.connect(str(args.db_path))
    try:
        create_schema(conn)
        done_keys = load_done_game_keys(conn)

        remaining_games = [game for game in stitched_games if game_key(game) not in done_keys]
        total_games = len(stitched_games)
        start_time = time.perf_counter()
        parsed_count = total_games - len(remaining_games)
        event_rows_written = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

        progress = tqdm(
            total=total_games,
            initial=parsed_count,
            desc="Parsing stitched human games",
            unit="game",
        )

        since_commit = 0
        remaining_chunks = chunk_games(remaining_games, chunk_size)

        if workers == 1:
            for games_chunk in remaining_chunks:
                parsed_batch = parse_game_chunk(
                    source_base_dir_str=str(source_base_dir),
                    games=games_chunk,
                    achievements=achievements,
                )
                games_written, events_written = process_parsed_batch(conn, parsed_batch)
                parsed_count += games_written
                since_commit += games_written
                event_rows_written += events_written
                progress.update(games_written)

                elapsed = time.perf_counter() - start_time
                gps = parsed_count / elapsed if elapsed > 0 else 0.0
                remaining = total_games - parsed_count
                eta_minutes = (remaining / gps / 60.0) if gps > 0 else 0.0
                progress.set_postfix(
                    mode="serial",
                    workers=1,
                    games_per_s=f"{gps:.2f}",
                    events=event_rows_written,
                    eta_min=f"{eta_minutes:.1f}",
                )

                if since_commit >= args.commit_every:
                    conn.commit()
                    since_commit = 0
        else:
            if remaining_chunks:
                tasks = [(str(source_base_dir), games_chunk, achievements) for games_chunk in remaining_chunks]
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    for parsed_batch in executor.map(parse_game_chunk, *zip(*tasks), chunksize=1):
                        games_written, events_written = process_parsed_batch(conn, parsed_batch)
                        parsed_count += games_written
                        since_commit += games_written
                        event_rows_written += events_written
                        progress.update(games_written)

                        elapsed = time.perf_counter() - start_time
                        gps = parsed_count / elapsed if elapsed > 0 else 0.0
                        remaining = total_games - parsed_count
                        eta_minutes = (remaining / gps / 60.0) if gps > 0 else 0.0
                        progress.set_postfix(
                            mode="proc",
                            workers=workers,
                            games_per_s=f"{gps:.2f}",
                            events=event_rows_written,
                            eta_min=f"{eta_minutes:.1f}",
                        )

                        if since_commit >= args.commit_every:
                            conn.commit()
                            since_commit = 0

        conn.commit()
        progress.close()

        manifest_start = time.perf_counter()
        export_manifest_csv(conn, args.manifest_csv)
        manifest_seconds = time.perf_counter() - manifest_start

        events_export_seconds = 0.0
        if not args.skip_csv_export:
            events_start = time.perf_counter()
            export_events_csv_gz(conn, args.events_csv_gz)
            events_export_seconds = time.perf_counter() - events_start

        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        total_seconds = time.perf_counter() - start_time
        game_rows = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        event_rows = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

        print(
            {
                "games": game_rows,
                "events": event_rows,
                "db_path": str(args.db_path),
                "db_size_bytes": args.db_path.stat().st_size,
                "manifest_csv": str(args.manifest_csv),
                "manifest_size_bytes": args.manifest_csv.stat().st_size if args.manifest_csv.exists() else 0,
                "events_csv_gz": str(args.events_csv_gz),
                "events_csv_gz_size_bytes": args.events_csv_gz.stat().st_size if args.events_csv_gz.exists() else 0,
                "total_seconds": total_seconds,
                "games_per_second": (game_rows / total_seconds) if total_seconds > 0 else 0.0,
                "workers": workers,
                "chunk_games": chunk_size,
                "manifest_export_seconds": manifest_seconds,
                "events_export_seconds": events_export_seconds,
            }
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
