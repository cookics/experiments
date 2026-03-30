from __future__ import annotations

import bz2
import pickle
import sqlite3
import struct
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_PATH = ROOT / "nld-nao" / "human_member_events.sqlite3"
PARSER_VERSION = 3

TURN_TOKEN = b"T:"
SCORE_TOKEN = b"S:"
DLVL_TOKEN = b"Dlvl:"
XP_TOKEN = b"Xp:"
EXP_TOKEN = b"Exp:"
HOME_TOKEN = b"Home "
ASTRAL_TOKEN = b"Astral Plane"
ASCEND_TOKEN = b"You ascend"
ASCEND_TOKEN_LOWER = b"you ascend"


def _last_numeric_value(data: bytes, token: bytes) -> int | None:
    idx = data.rfind(token)
    if idx == -1:
        return None

    start = idx + len(token)
    end = start
    while end < len(data) and 48 <= data[end] <= 57:
        end += 1

    if end == start:
        return None
    return int(data[start:end])


def _finalize_curve(values_by_turn: dict[int, float], max_turn: int) -> pd.Series | None:
    if not values_by_turn:
        return None
    series = pd.Series(values_by_turn).sort_index()
    return series.reindex(range(max_turn + 1)).ffill().fillna(0.0)


def parse_member_events(
    member_path: Path,
    achievements: dict[str, float],
    max_turn: int,
    expected_turns: int | None,
) -> dict[str, list[float] | int]:
    progression = 0.0
    score = 0.0
    last_turn = -1
    best_dlvl = 0
    best_xp = 0
    best_home = 0
    reached_astral = False
    ascended = False

    turns: list[int] = []
    scores: list[float] = []
    progressions: list[float] = []
    dlvls: list[int] = []
    xps: list[int] = []
    homes: list[int] = []
    astrals: list[int] = []
    ascends: list[int] = []

    with member_path.open("rb") as raw_member:
        tty = bz2.BZ2File(raw_member)

        while True:
            header = tty.read(12)
            if not header:
                break

            sec, usec, length = struct.unpack("<iii", header)
            if sec < 0 or usec < 0 or length < 1:
                break

            data = tty.read(length)
            if not data:
                break

            if (
                TURN_TOKEN not in data
                and SCORE_TOKEN not in data
                and DLVL_TOKEN not in data
                and XP_TOKEN not in data
                and EXP_TOKEN not in data
                and HOME_TOKEN not in data
                and ASTRAL_TOKEN not in data
                and ASCEND_TOKEN not in data
                and ASCEND_TOKEN_LOWER not in data
            ):
                continue

            candidates = [progression]
            dlvl = _last_numeric_value(data, DLVL_TOKEN)
            if dlvl is not None:
                best_dlvl = max(best_dlvl, dlvl)
                candidates.append(achievements.get(f"Dlvl:{dlvl}", progression))

            xp = _last_numeric_value(data, XP_TOKEN)
            if xp is None:
                xp = _last_numeric_value(data, EXP_TOKEN)
            if xp is not None:
                best_xp = max(best_xp, xp)
                candidates.append(achievements.get(f"Xp:{xp}", progression))

            home = _last_numeric_value(data, HOME_TOKEN)
            if home is not None:
                best_home = max(best_home, home)
                candidates.append(achievements.get(f"Home {home}", progression))
            if ASTRAL_TOKEN in data:
                reached_astral = True
                candidates.append(achievements.get("Astral Plane", progression))
            if ASCEND_TOKEN in data or ASCEND_TOKEN_LOWER in data:
                ascended = True
                candidates.append(achievements.get("You ascend t", progression))

            progression = max(candidates)
            progression_percent = progression * 100.0

            parsed_score = _last_numeric_value(data, SCORE_TOKEN)
            if parsed_score is not None:
                score = max(score, float(parsed_score))

            turn = _last_numeric_value(data, TURN_TOKEN)
            if turn is None:
                continue
            if expected_turns is not None and turn > expected_turns:
                continue
            if turn < last_turn:
                continue

            if turns and turn == turns[-1]:
                scores[-1] = max(scores[-1], score)
                progressions[-1] = max(progressions[-1], progression_percent)
                dlvls[-1] = max(dlvls[-1], best_dlvl)
                xps[-1] = max(xps[-1], best_xp)
                homes[-1] = max(homes[-1], best_home)
                astrals[-1] = max(astrals[-1], int(reached_astral))
                ascends[-1] = max(ascends[-1], int(ascended))
            elif not turns or score != scores[-1] or progression_percent != progressions[-1]:
                turns.append(turn)
                scores.append(score)
                progressions.append(progression_percent)
                dlvls.append(best_dlvl)
                xps.append(best_xp)
                homes.append(best_home)
                astrals.append(int(reached_astral))
                ascends.append(int(ascended))

            last_turn = turn
            if turn >= max_turn:
                break

    return {
        "turns": turns,
        "scores": scores,
        "progressions": progressions,
        "dlvls": dlvls,
        "xps": xps,
        "homes": homes,
        "astrals": astrals,
        "ascends": ascends,
        "last_turn": last_turn,
    }


class HumanMemberEventCache:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_CACHE_PATH
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS member_cache (
                member_name TEXT PRIMARY KEY,
                file_size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                parser_version INTEGER NOT NULL,
                parsed_max_turn INTEGER NOT NULL,
                payload BLOB NOT NULL
            )
            """
        )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> HumanMemberEventCache:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def clear(self) -> None:
        self.conn.execute("DELETE FROM member_cache")
        self.conn.commit()

    def get_or_build(
        self,
        member_name: str,
        member_path: Path,
        achievements: dict[str, float],
        max_turn: int,
        expected_turns: int | None,
    ) -> dict[str, list[float] | int]:
        stat = member_path.stat()
        row = self.conn.execute(
            """
            SELECT file_size, mtime_ns, parser_version, parsed_max_turn, payload
            FROM member_cache
            WHERE member_name = ?
            """,
            (member_name,),
        ).fetchone()

        if row is not None:
            file_size, mtime_ns, parser_version, parsed_max_turn, payload = row
            if (
                int(file_size) == stat.st_size
                and int(mtime_ns) == stat.st_mtime_ns
                and int(parser_version) == PARSER_VERSION
                and int(parsed_max_turn) >= max_turn
            ):
                return pickle.loads(payload)

        payload = parse_member_events(
            member_path=member_path,
            achievements=achievements,
            max_turn=max_turn,
            expected_turns=expected_turns,
        )
        self.conn.execute(
            """
            INSERT INTO member_cache (
                member_name, file_size, mtime_ns, parser_version, parsed_max_turn, payload
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(member_name) DO UPDATE SET
                file_size = excluded.file_size,
                mtime_ns = excluded.mtime_ns,
                parser_version = excluded.parser_version,
                parsed_max_turn = excluded.parsed_max_turn,
                payload = excluded.payload
            """,
            (
                member_name,
                stat.st_size,
                stat.st_mtime_ns,
                PARSER_VERSION,
                max_turn,
                sqlite3.Binary(pickle.dumps(payload, protocol=5)),
            ),
        )
        self.conn.commit()
        return payload


def build_curves_from_cached_members(
    cache: HumanMemberEventCache,
    base_dir: Path,
    member_names: list[str],
    achievements: dict[str, float],
    max_turn: int,
    expected_turns: int | None,
) -> tuple[pd.Series | None, pd.Series | None]:
    last_turn = -1
    score_by_turn: dict[int, float] = {}
    progression_by_turn: dict[int, float] = {}
    progression = 0.0
    score = 0.0

    for member_name in member_names:
        payload = cache.get_or_build(
            member_name=member_name,
            member_path=base_dir / Path(member_name),
            achievements=achievements,
            max_turn=max_turn,
            expected_turns=expected_turns,
        )

        turns = payload["turns"]
        scores = payload["scores"]
        progressions = payload["progressions"]

        for turn, cached_score, cached_progression in zip(turns, scores, progressions):
            turn = int(turn)
            cached_score = float(cached_score)
            cached_progression = float(cached_progression)

            if expected_turns is not None and turn > expected_turns:
                continue
            if turn < last_turn:
                continue

            score = max(score, cached_score)
            progression = max(progression, cached_progression)
            score_by_turn[turn] = score
            progression_by_turn[turn] = progression
            last_turn = turn

            if turn >= max_turn:
                break

        if last_turn >= max_turn:
            break

    return _finalize_curve(score_by_turn, max_turn), _finalize_curve(progression_by_turn, max_turn)

