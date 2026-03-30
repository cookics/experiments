from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.human_nao_source import HumanNAODataSource, NAO_EXTRACTED_ROOT, NAO_ZIP_PATH
from analysis.plot_human_nao_trajectories import (
    altorg_filename_to_timestamp,
    assign_ttyrecs_to_games,
    load_achievements,
    parse_human_progression_members,
    parse_xlog_line,
)


ROOT = Path(__file__).resolve().parent.parent
XLOG_PATH = ROOT / "nld-nao" / "xlogfile.full.txt"
OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_PNG = OUTPUT_DIR / "human_random_50_plus_best_full.png"
OUTPUT_CSV = OUTPUT_DIR / "human_random_50_plus_best_manifest.csv"
INDEX_CACHE_PATH = ROOT / "nld-nao" / "nld-nao-dir-aa.stitched_games.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-random", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turn", type=int, default=80000)
    return parser.parse_args()


def build_ttyrecs_by_player(source) -> dict[str, list[str]]:
    ttyrecs_by_player: dict[str, list[str]] = defaultdict(list)
    for member_name in source.namelist():
        if not member_name.endswith(".ttyrec.bz2"):
            continue
        player_name = Path(member_name).parts[-2].lower()
        ttyrecs_by_player[player_name].append(member_name)

    for ttyrecs in ttyrecs_by_player.values():
        ttyrecs.sort(key=altorg_filename_to_timestamp)

    return dict(ttyrecs_by_player)


def load_games_for_players(xlog_path: Path, player_names: set[str]) -> dict[str, list[dict[str, object]]]:
    game_counts: dict[str, int] = defaultdict(int)
    games_by_player: dict[str, list[dict[str, object]]] = defaultdict(list)

    with xlog_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record = parse_xlog_line(line)
            player_name = record.get("name", "").lower()
            if player_name not in player_names:
                continue
            if "starttime" not in record or "endtime" not in record:
                continue

            game_counts[player_name] += 1
            games_by_player[player_name].append(
                {
                    "local_gameid": game_counts[player_name],
                    "starttime": int(record["starttime"]),
                    "endtime": int(record["endtime"]),
                    "death": record.get("death", ""),
                    "points": int(record.get("points", 0) or 0),
                    "turns": int(record.get("turns", 0) or 0),
                    "maxlvl": int(record.get("maxlvl", 0) or 0),
                    "version": record.get("version", ""),
                }
            )

    return dict(games_by_player)


def _source_signature() -> dict[str, int]:
    xlog_stat = XLOG_PATH.stat()
    signature = {
        "xlog_size": xlog_stat.st_size,
        "xlog_mtime_ns": xlog_stat.st_mtime_ns,
        "has_extracted_root": int(NAO_EXTRACTED_ROOT.exists()),
    }
    if NAO_ZIP_PATH.exists():
        zip_stat = NAO_ZIP_PATH.stat()
        signature["zip_size"] = zip_stat.st_size
        signature["zip_mtime_ns"] = zip_stat.st_mtime_ns
    return signature


def _load_cached_game_index() -> list[dict[str, object]] | None:
    if not INDEX_CACHE_PATH.exists():
        return None

    with INDEX_CACHE_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("signature") != _source_signature():
        return None
    return list(payload.get("games", []))


def _write_cached_game_index(stitched_games: list[dict[str, object]]) -> None:
    payload = {
        "signature": _source_signature(),
        "games": stitched_games,
    }
    with INDEX_CACHE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _build_game_index_uncached(source) -> list[dict[str, object]]:
    ttyrecs_by_player = build_ttyrecs_by_player(source)
    games_by_player = load_games_for_players(XLOG_PATH, set(ttyrecs_by_player))

    stitched_games: list[dict[str, object]] = []
    for player_name, ttyrecs in ttyrecs_by_player.items():
        games = games_by_player.get(player_name)
        if not games:
            continue

        assignments = assign_ttyrecs_to_games(
            ttyrecs,
            [(int(game["local_gameid"]), int(game["starttime"]), int(game["endtime"])) for game in games],
        )
        members_by_gameid: dict[int, list[str]] = defaultdict(list)
        for member_name, local_gameid in assignments:
            members_by_gameid[local_gameid].append(member_name)

        for game in games:
            local_gameid = int(game["local_gameid"])
            members = members_by_gameid.get(local_gameid)
            if not members:
                continue

            stitched_games.append(
                {
                    **game,
                    "player_name": player_name,
                    "members": sorted(members, key=altorg_filename_to_timestamp),
                }
            )

    return stitched_games


def build_game_index(source=None) -> list[dict[str, object]]:
    cached_games = _load_cached_game_index()
    if cached_games is not None:
        return cached_games

    if source is None:
        with HumanNAODataSource() as local_source:
            stitched_games = _build_game_index_uncached(local_source)
    else:
        stitched_games = _build_game_index_uncached(source)

    _write_cached_game_index(stitched_games)
    return stitched_games


def choose_best_game(stitched_games: list[dict[str, object]]) -> dict[str, object]:
    def sort_key(game: dict[str, object]) -> tuple[int, int, int, int]:
        death = str(game["death"]).lower()
        return (
            1 if death == "ascended" else 0,
            int(game["maxlvl"]),
            int(game["points"]),
            int(game["turns"]),
        )

    return max(stitched_games, key=sort_key)


def harmonize_curve_with_metadata(
    curve: pd.Series, game: dict[str, object], achievements: dict[str, float]
) -> pd.Series:
    expected_turns = int(game["turns"])
    if expected_turns > 0:
        curve = curve[curve.index <= expected_turns]

    if curve.empty:
        return curve

    death = str(game["death"]).lower()
    target_progression = float(curve.iloc[-1])
    if death == "ascended":
        target_progression = max(target_progression, achievements["You ascend t"] * 100.0)

    if expected_turns > 0:
        curve.loc[expected_turns] = max(float(curve.iloc[-1]), target_progression)

    return curve.sort_index()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    achievements = load_achievements()
    rng = random.Random(args.seed)

    with HumanNAODataSource() as source:
        stitched_games = build_game_index(source)
        best_game = choose_best_game(stitched_games)

        remaining_games = [
            game
            for game in stitched_games
            if not (
                game["player_name"] == best_game["player_name"]
                and int(game["local_gameid"]) == int(best_game["local_gameid"])
            )
        ]
        rng.shuffle(remaining_games)

        curves: list[tuple[dict[str, object], pd.Series]] = []

        best_curve = parse_human_progression_members(
            source,
            member_names=list(best_game["members"]),
            achievements=achievements,
            max_turn=args.max_turn,
            expected_turns=int(best_game["turns"]),
        )
        if best_curve is None:
            raise SystemExit("Could not parse the selected best human run.")
        best_curve = harmonize_curve_with_metadata(best_curve, best_game, achievements)
        curves.append((best_game, best_curve))

        parsed_random = 0
        attempted_random = 0
        for game in remaining_games:
            if parsed_random >= args.num_random:
                break
            attempted_random += 1
            curve = parse_human_progression_members(
                source,
                member_names=list(game["members"]),
                achievements=achievements,
                max_turn=args.max_turn,
                expected_turns=int(game["turns"]),
            )
            if curve is None:
                continue
            curve = harmonize_curve_with_metadata(curve, game, achievements)
            if curve.empty:
                continue
            curves.append((game, curve))
            parsed_random += 1
            if parsed_random % 10 == 0 or parsed_random == args.num_random:
                print(f"Parsed {parsed_random} / {args.num_random} random stitched games...")

    if parsed_random < args.num_random:
        print(f"Only parsed {parsed_random} random stitched games after {attempted_random} attempts.")

    if not curves:
        raise SystemExit("No stitched human curves were parsed.")

    manifest_rows = []
    best_key = (best_game["player_name"], int(best_game["local_gameid"]))
    max_observed_turn = 0

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor("#f6f3ee")
    ax.set_facecolor("#fffdfa")

    for game, curve in curves:
        game_key = (game["player_name"], int(game["local_gameid"]))
        is_best = game_key == best_key
        final_progression = float(curve.iloc[-1])
        final_turn = int(curve.index.max())
        max_observed_turn = max(max_observed_turn, final_turn)

        manifest_rows.append(
            {
                "player_name": game["player_name"],
                "local_gameid": int(game["local_gameid"]),
                "death": game["death"],
                "points": int(game["points"]),
                "turns": int(game["turns"]),
                "maxlvl": int(game["maxlvl"]),
                "ttyrec_parts": len(game["members"]),
                "final_turn_observed": final_turn,
                "final_progression_percent": final_progression,
                "is_best": is_best,
            }
        )

        if is_best:
            ax.plot(
                curve.index,
                curve.values,
                color="#7a1f2b",
                linewidth=3.2,
                drawstyle="steps-post",
                zorder=5,
                label=f"Best run: {game['player_name']} ({final_progression:.2f}%)",
            )
            ax.scatter([final_turn], [final_progression], color="#d1495b", s=85, zorder=6)
            ax.annotate(
                f"{game['death']} at turn {final_turn:,}\n{final_progression:.2f}%",
                xy=(final_turn, final_progression),
                xytext=(-140, -4),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.28", fc="#fff2ef", ec="#d1495b", alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#d1495b", lw=1.4),
                fontsize=10.5,
            )
        else:
            ax.plot(
                curve.index,
                curve.values,
                color="#3c6e71",
                linewidth=1.0,
                alpha=0.25,
                drawstyle="steps-post",
                zorder=2,
            )

    ax.set_title("50 Random Human NetHack Runs Plus One Best Run (NLD-NAO, Stitched Games)", fontsize=18, pad=12)
    ax.set_xlabel("NetHack Turn `T`", fontsize=13)
    ax.set_ylabel("BALROG NLE Progression (%)", fontsize=13)
    ax.set_xlim(0, max_observed_turn)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.22, linewidth=0.9, color="#8f7f6d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6c6258")
    ax.spines["bottom"].set_color("#6c6258")
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    pd.DataFrame(manifest_rows).sort_values(
        ["is_best", "final_progression_percent", "final_turn_observed"],
        ascending=[False, False, False],
    ).to_csv(OUTPUT_CSV, index=False)

    print(f"Total stitched games available in shard: {len(stitched_games)}")
    print(f"Best run player: {best_game['player_name']}")
    print(f"Best run death: {best_game['death']}")
    print(f"Best run maxlvl: {best_game['maxlvl']}")
    print(f"Best run points: {best_game['points']}")
    print(f"Parsed random runs: {parsed_random}")
    print(f"Saved plot to {OUTPUT_PNG}")
    print(f"Saved manifest to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

