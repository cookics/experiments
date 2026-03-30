from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from analysis.human_nao_source import HumanNAODataSource
from analysis.plot_human_nao_random_plus_best import build_game_index, choose_best_game, harmonize_curve_with_metadata
from analysis.plot_human_nao_trajectories import load_achievements, parse_human_progression_members
from analysis.plot_nle_trajectories import find_nle_csvs, load_data


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_PNG = OUTPUT_DIR / "nle_trajectories_progression_zoom_3000_10_llm_plus_25_humans.png"
OUTPUT_CSV = OUTPUT_DIR / "nle_trajectories_progression_zoom_3000_10_llm_plus_25_humans_manifest.csv"

TURN_RE = re.compile(r"\bT:(\d+)")
DLVL_RE = re.compile(r"Dlvl:(\d+)")
XP_RE = re.compile(r"(?:Xp|Exp):(\d+)")
HOME_RE = re.compile(r"\bHome ([1-5])\b")
ASTRAL_RE = re.compile(r"\bAstral Plane\b")
ASCEND_RE = re.compile(r"\bYou ascend\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-turn", type=int, default=3000)
    parser.add_argument("--human-random", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_llm_progression_curve_by_turn(
    csv_path: Path, achievements: dict[str, float], max_turn: int
) -> pd.Series | None:
    df = pd.read_csv(csv_path, usecols=["Observation"])

    progression_by_turn: dict[int, float] = {}
    progression = 0.0
    last_turn = -1

    for observation in df["Observation"].fillna(""):
        turn_match = TURN_RE.search(observation)
        if not turn_match:
            continue

        turn = int(turn_match.group(1))
        if turn <= last_turn:
            continue
        if turn > max_turn:
            break
        last_turn = turn

        candidates = [progression]

        dlvl_match = DLVL_RE.search(observation)
        xp_match = XP_RE.search(observation)
        home_match = HOME_RE.search(observation)

        if dlvl_match:
            candidates.append(achievements.get(f"Dlvl:{int(dlvl_match.group(1))}", progression))
        if xp_match:
            candidates.append(achievements.get(f"Xp:{int(xp_match.group(1))}", progression))
        if home_match:
            candidates.append(achievements.get(f"Home {int(home_match.group(1))}", progression))
        if ASTRAL_RE.search(observation):
            candidates.append(achievements.get("Astral Plane", progression))
        if ASCEND_RE.search(observation):
            candidates.append(achievements.get("You ascend t", progression))

        progression = max(candidates)
        progression_by_turn[turn] = progression * 100.0

    if not progression_by_turn:
        return None

    return pd.Series(progression_by_turn, name=csv_path.name).sort_index()


def average_curve(curves: list[pd.Series], turns: pd.Index) -> pd.Series:
    aligned = [curve.reindex(turns).ffill().fillna(0.0) for curve in curves]
    return pd.concat(aligned, axis=1).mean(axis=1)


def build_llm_records(max_turn: int, achievements: dict[str, float]) -> list[dict]:
    data = load_data()
    records: list[dict] = []

    for leaderboard in data.get("leaderboards", []):
        if leaderboard.get("name") != "LLM":
            continue

        for result in leaderboard.get("results", []):
            if result.get("trajs") is not True:
                continue

            folder = ROOT / Path(result["folder"])
            csv_paths = find_nle_csvs(folder)
            if not csv_paths:
                continue

            curves = []
            for csv_path in csv_paths:
                curve = load_llm_progression_curve_by_turn(csv_path, achievements, max_turn=max_turn)
                if curve is not None:
                    curves.append(curve)

            if not curves:
                continue

            turns = pd.Index(range(max_turn + 1), name="turn")
            mean_curve = average_curve(curves, turns)

            records.append(
                {
                    "model_name": result["name"],
                    "date": result.get("date", ""),
                    "official_nle_progress_percent": float(result.get("nle", [0.0])[0] or 0.0),
                    "run_count": len(curves),
                    "mean_curve": mean_curve,
                    "final_progression_percent_at_max_turn": float(mean_curve.iloc[-1]),
                }
            )

    records.sort(
        key=lambda record: (record["final_progression_percent_at_max_turn"], record["model_name"].lower()),
        reverse=True,
    )
    return records


def build_human_curves(max_turn: int, human_random: int, seed: int, achievements: dict[str, float]) -> tuple[list[dict], dict]:
    rng = random.Random(seed)

    with HumanNAODataSource() as source:
        stitched_games = build_game_index(source)
        best_game = choose_best_game(stitched_games)

        best_curve = parse_human_progression_members(
            source,
            member_names=list(best_game["members"]),
            achievements=achievements,
            max_turn=max_turn,
            expected_turns=int(best_game["turns"]),
        )
        if best_curve is None:
            raise SystemExit("Could not parse the best stitched human run.")
        best_curve = harmonize_curve_with_metadata(best_curve, best_game, achievements)
        best_curve = best_curve[best_curve.index <= max_turn]
        best_curve = best_curve.reindex(range(max_turn + 1)).ffill().fillna(0.0)

        remaining_games = [
            game
            for game in stitched_games
            if not (
                game["player_name"] == best_game["player_name"]
                and int(game["local_gameid"]) == int(best_game["local_gameid"])
            )
        ]
        rng.shuffle(remaining_games)

        random_records: list[dict] = []
        for game in remaining_games:
            if len(random_records) >= human_random:
                break

            curve = parse_human_progression_members(
                source,
                member_names=list(game["members"]),
                achievements=achievements,
                max_turn=max_turn,
                expected_turns=int(game["turns"]),
            )
            if curve is None:
                continue
            curve = harmonize_curve_with_metadata(curve, game, achievements)
            curve = curve[curve.index <= max_turn]
            if curve.empty:
                continue
            curve = curve.reindex(range(max_turn + 1)).ffill().fillna(0.0)

            random_records.append(
                {
                    "player_name": game["player_name"],
                    "local_gameid": int(game["local_gameid"]),
                    "death": game["death"],
                    "curve": curve,
                    "final_progression_percent_at_max_turn": float(curve.iloc[-1]),
                }
            )

    best_record = {
        "player_name": best_game["player_name"],
        "local_gameid": int(best_game["local_gameid"]),
        "death": best_game["death"],
        "curve": best_curve,
        "final_progression_percent_at_max_turn": float(best_curve.iloc[-1]),
    }
    return random_records, best_record


def plot_overlay(
    llm_records: list[dict], human_random_records: list[dict], best_human_record: dict, max_turn: int
) -> list[dict]:
    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor("#f7f1e8")
    ax.set_facecolor("#fffaf3")

    manifest_rows: list[dict] = []

    human_records = human_random_records + [best_human_record]
    highlighted_human = max(
        human_records,
        key=lambda record: (record["final_progression_percent_at_max_turn"], record["label"] if "label" in record else record["player_name"]),
    )

    for human_record in human_records:
        curve = human_record["curve"]
        is_highlighted = (
            human_record["player_name"] == highlighted_human["player_name"]
            and int(human_record["local_gameid"]) == int(highlighted_human["local_gameid"])
        )
        if is_highlighted:
            continue
        ax.plot(
            curve.index,
            curve.values,
            color="#7aa6a1",
            linewidth=1.0,
            alpha=0.24,
            drawstyle="steps-post",
            zorder=1,
        )
        manifest_rows.append(
            {
                "group": "human_other",
                "label": human_record["player_name"],
                "final_progression_percent_at_3000_turns": human_record["final_progression_percent_at_max_turn"],
                "details": human_record["death"],
            }
        )

    best_curve = highlighted_human["curve"]
    ax.plot(
        best_curve.index,
        best_curve.values,
        color="#7a1f2b",
        linewidth=3.0,
        alpha=0.98,
        drawstyle="steps-post",
        zorder=4,
    )
    manifest_rows.append(
        {
            "group": "human_highlighted",
            "label": highlighted_human["player_name"],
            "final_progression_percent_at_3000_turns": highlighted_human["final_progression_percent_at_max_turn"],
            "details": highlighted_human["death"],
        }
    )

    cmap = plt.get_cmap("tab20")
    model_handles: list[Line2D] = []
    model_labels: list[str] = []
    for idx, record in enumerate(llm_records):
        color = cmap(idx % 20)
        mean_curve = record["mean_curve"]
        label = f"{record['model_name']} ({record['final_progression_percent_at_max_turn']:.2f}%)"
        ax.plot(
            mean_curve.index,
            mean_curve.values,
            color=color,
            linewidth=2.3,
            alpha=0.97,
            zorder=3,
        )
        model_handles.append(Line2D([0], [0], color=color, linewidth=2.8))
        model_labels.append(label)
        manifest_rows.append(
            {
                "group": "llm_model",
                "label": record["model_name"],
                "final_progression_percent_at_3000_turns": record["final_progression_percent_at_max_turn"],
                "details": f"official_nle={record['official_nle_progress_percent']:.4f}; runs={record['run_count']}",
            }
        )

    annotation = (
        "x-axis: NetHack turn T for both humans and LLMs\n"
        "y-axis: BALROG NLE progression reconstructed from Dlvl/Xp/endgame milestones"
    )
    ax.text(
        0.012,
        0.988,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        color="#5d544a",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff6ea", ec="#d8c8b6", alpha=0.96),
    )

    best_turn = int(best_curve.index.max())
    best_progress = float(best_curve.iloc[-1])
    ax.scatter([best_turn], [best_progress], color="#c8465a", s=55, zorder=5)
    ax.annotate(
        f"Top human at T={max_turn:,}\n{highlighted_human['player_name']} ({best_progress:.2f}%)",
        xy=(best_turn, best_progress),
        xytext=(16, 10),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.28", fc="#fff2ef", ec="#c8465a", alpha=0.95),
    )

    ax.set_title("LLM NLE Progression vs Human Runs (First 3,000 NetHack Turns)", fontsize=18, pad=12)
    ax.set_xlabel("NetHack Turn `T`", fontsize=13)
    ax.set_ylabel("BALROG NLE Progression (%)", fontsize=13)
    ax.set_xlim(0, max_turn)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.22, linewidth=0.9, color="#8f7f6d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6c6258")
    ax.spines["bottom"].set_color("#6c6258")

    legend_handles = [
        Line2D([0], [0], color="#7aa6a1", linewidth=2.0, alpha=0.55),
        Line2D([0], [0], color="#7a1f2b", linewidth=3.0),
    ] + model_handles
    legend_labels = [
        f"{len(human_records) - 1} other humans",
        f"Top human at T={max_turn:,}: {highlighted_human['player_name']} ({highlighted_human['final_progression_percent_at_max_turn']:.2f}%)",
    ] + model_labels

    ax.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        ncol=1,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return manifest_rows


def main() -> None:
    args = parse_args()
    achievements = load_achievements()
    llm_records = build_llm_records(max_turn=args.max_turn, achievements=achievements)
    if not llm_records:
        raise SystemExit("No LLM trajectories with parsable turn counters were found.")

    human_random_records, best_human_record = build_human_curves(
        max_turn=args.max_turn,
        human_random=args.human_random,
        seed=args.seed,
        achievements=achievements,
    )
    manifest_rows = plot_overlay(llm_records, human_random_records, best_human_record, max_turn=args.max_turn)

    pd.DataFrame(manifest_rows).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved combined plot to {OUTPUT_PNG}")
    print(f"Saved manifest to {OUTPUT_CSV}")
    print(f"Plotted {len(llm_records)} LLM curves, {len(human_random_records)} random human runs, and 1 best human run.")


if __name__ == "__main__":
    main()

