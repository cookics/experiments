from __future__ import annotations

import argparse
import bz2
import datetime
import json
import random
import re
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from human_nao_source import HumanNAODataSource, NAO_ZIP_PATH


ROOT = Path(__file__).resolve().parent
ACHIEVEMENTS_PATH = ROOT / "_balrog_src" / "balrog" / "environments" / "nle" / "achievements.json"
OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_PNG = OUTPUT_DIR / "human_nld_nao_progression_turns_3000.png"
OUTPUT_CSV = OUTPUT_DIR / "human_nld_nao_progression_turns_3000_summary.csv"

DLVL_RE = re.compile(r"Dlvl:(\d+)")
XP_RE = re.compile(r"(?:Xp|Exp):(\d+)")
TURN_RE = re.compile(r"\bT:(\d+)")
HOME_RE = re.compile(r"\bHome ([1-5])\b")
ASTRAL_RE = re.compile(r"\bAstral Plane\b")
ASCEND_RE = re.compile(r"\bYou ascend\b", re.IGNORECASE)
ALT_TIMEFMT = re.compile(r"(.*\.\d\d)_(\d\d)_(\d\d.*)")
FIVE_MINS = 5 * 60
TURN_TOKEN = b"T:"
DLVL_TOKEN = b"Dlvl:"
XP_TOKEN = b"Xp:"
EXP_TOKEN = b"Exp:"
HOME_TOKEN = b"Home "
ASTRAL_TOKEN = b"Astral Plane"
ASCEND_TOKEN = b"You ascend"
ASCEND_TOKEN_LOWER = b"you ascend"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--max-turn", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_achievements() -> dict[str, float]:
    with ACHIEVEMENTS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_ttyrecs(source, sample_size: int, seed: int) -> list[str]:
    ttyrecs = [name for name in source.namelist() if name.endswith(".ttyrec.bz2")]
    rng = random.Random(seed)
    return rng.sample(ttyrecs, min(sample_size, len(ttyrecs)))


def altorg_filename_to_timestamp(member_name: str) -> float:
    ts = member_name.split("/")[-1][:-11]
    ts = ALT_TIMEFMT.sub(r"\1:\2:\3", ts)
    dt = datetime.datetime.fromisoformat(ts)
    return dt.replace(tzinfo=datetime.timezone.utc).timestamp()


def parse_xlog_line(line: str) -> dict[str, str]:
    record: dict[str, str] = {}
    for field in line.strip().split(":"):
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        record[key] = value
    return record


def load_player_games(xlog_path: Path, player_name: str) -> list[tuple[int, int, int]]:
    games: list[tuple[int, int, int]] = []
    gameid = 0
    with xlog_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record = parse_xlog_line(line)
            if record.get("name", "").lower() != player_name.lower():
                continue
            if "starttime" not in record or "endtime" not in record:
                continue
            gameid += 1
            games.append((gameid, int(record["starttime"]), int(record["endtime"])))
    return games


def assign_ttyrecs_to_games(
    ttyrecs: list[str], games: list[tuple[int, int, int]]
) -> list[tuple[str, int]]:
    assigned: list[list[float | int | str]] = []
    for member_name in ttyrecs:
        assigned.append([member_name, altorg_filename_to_timestamp(member_name), -1, -1, -1])

    assigned.sort(key=lambda row: row[1])
    games = sorted(games, key=lambda row: row[1])

    gg = 0
    tt = 0
    while gg < len(games) and tt < len(assigned):
        if assigned[tt][1] > games[gg][2]:
            gg += 1
        elif assigned[tt][1] < games[gg][1] - FIVE_MINS:
            assigned[tt][2] = -games[gg][0]
            assigned[tt][3] = games[gg][1]
            assigned[tt][4] = games[gg][2]
            tt += 1
        else:
            assigned[tt][2] = games[gg][0]
            assigned[tt][3] = games[gg][1]
            assigned[tt][4] = games[gg][2]
            tt += 1

    return [(str(member_name), int(gameid)) for member_name, _, gameid, _, _ in assigned if int(gameid) != -1]


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


def parse_human_progression_members(
    source,
    member_names: list[str],
    achievements: dict[str, float],
    max_turn: int | None,
    expected_turns: int | None = None,
) -> pd.Series | None:
    progression = 0.0
    last_turn = -1
    progression_by_turn: dict[int, float] = {}

    for member_name in member_names:
        with source.open(member_name) as raw_member:
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
                    candidates.append(achievements.get(f"Dlvl:{dlvl}", progression))

                xp = _last_numeric_value(data, XP_TOKEN)
                if xp is None:
                    xp = _last_numeric_value(data, EXP_TOKEN)
                if xp is not None:
                    candidates.append(achievements.get(f"Xp:{xp}", progression))

                home = _last_numeric_value(data, HOME_TOKEN)
                if home is not None:
                    candidates.append(achievements.get(f"Home {home}", progression))
                if ASTRAL_TOKEN in data:
                    candidates.append(achievements.get("Astral Plane", progression))
                if ASCEND_TOKEN in data or ASCEND_TOKEN_LOWER in data:
                    candidates.append(achievements.get("You ascend t", progression))

                progression = max(candidates)
                turn = _last_numeric_value(data, TURN_TOKEN)
                if turn is None:
                    continue

                # Old NAO ttyrecs sometimes expose partially-redrawn status lines at
                # segment boundaries, which can yield bogus tiny or huge `T` values.
                # Enforce monotone in-game turns and optionally clip to xlog metadata.
                if expected_turns is not None and turn > expected_turns:
                    continue
                if turn <= last_turn:
                    continue
                last_turn = turn
                progression_by_turn[turn] = progression * 100.0

                if max_turn is not None and turn >= max_turn:
                    break

            if max_turn is not None and progression_by_turn and max(progression_by_turn) >= max_turn:
                break

    if not progression_by_turn:
        return None

    return pd.Series(progression_by_turn, name=member_names[-1]).sort_index()


def parse_human_progression_curve(
    source, member_name: str, achievements: dict[str, float], max_turn: int
) -> pd.Series | None:
    return parse_human_progression_members(source, [member_name], achievements, max_turn)


def aggregate_curves(curves: list[pd.Series], max_turn: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    turns = pd.Index(range(max_turn + 1), name="turn")
    aligned = [curve.reindex(turns).ffill().fillna(0.0) for curve in curves]
    frame = pd.concat(aligned, axis=1)
    return frame.mean(axis=1), frame.quantile(0.25, axis=1), frame.quantile(0.75, axis=1)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    achievements = load_achievements()
    with HumanNAODataSource() as source:
        selected = sample_ttyrecs(source, args.sample_size, args.seed)

        curves: list[pd.Series] = []
        summary_rows = []

        for idx, member_name in enumerate(selected, start=1):
            curve = parse_human_progression_curve(source, member_name, achievements, args.max_turn)
            if curve is None:
                continue

            curves.append(curve)
            summary_rows.append(
                {
                    "member_name": member_name,
                    "final_turn_observed": int(curve.index.max()),
                    "final_progression_percent": float(curve.iloc[-1]),
                }
            )

            if idx % 100 == 0:
                print(f"Parsed {idx} / {len(selected)} human ttyrecs...")

    if not curves:
        raise SystemExit("No human progression curves were parsed.")

    mean_curve, q25_curve, q75_curve = aggregate_curves(curves, args.max_turn)

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#f6f3ee")
    ax.set_facecolor("#fffdfa")

    ax.fill_between(
        mean_curve.index,
        q25_curve.values,
        q75_curve.values,
        color="#b6c9d6",
        alpha=0.35,
        label="25-75% band",
    )
    ax.plot(
        mean_curve.index,
        mean_curve.values,
        color="#1f5c7a",
        linewidth=3,
        label=f"Human mean ({len(curves)} games)",
    )

    ax.set_title("Human NetHack Progression From NLD-NAO (First 3,000 Turns)", fontsize=18, pad=12)
    ax.set_xlabel("NetHack Turn `T`", fontsize=13)
    ax.set_ylabel("BALROG NLE Progression (%)", fontsize=13)
    ax.set_xlim(0, args.max_turn)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.22, linewidth=0.9, color="#8f7f6d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6c6258")
    ax.spines["bottom"].set_color("#6c6258")
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    pd.DataFrame(summary_rows).sort_values(
        ["final_progression_percent", "final_turn_observed"], ascending=[False, False]
    ).to_csv(OUTPUT_CSV, index=False)

    print(f"Saved human progression plot to {OUTPUT_PNG}")
    print(f"Saved human summary to {OUTPUT_CSV}")
    print(f"Parsed {len(curves)} human games from {NAO_ZIP_PATH.name}")


if __name__ == "__main__":
    main()
