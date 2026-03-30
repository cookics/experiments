from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.human_nao_source import HumanNAODataSource
from analysis.plot_human_nao_trajectories import (
    assign_ttyrecs_to_games,
    load_achievements,
    load_player_games,
    parse_human_progression_members,
)


ROOT = Path(__file__).resolve().parent.parent
XLOG_PATH = ROOT / "nld-nao" / "xlogfile.full.txt"
SUMMARY_PATH = ROOT / "analysis_outputs" / "human_nld_nao_progression_turns_3000_summary.csv"
OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_PNG = OUTPUT_DIR / "human_best_nld_nao_progression_full.png"
OUTPUT_CSV = OUTPUT_DIR / "human_best_nld_nao_progression_full.csv"


def collect_game_members(source, member_name: str) -> tuple[list[str], int, int | None]:
    player_name = Path(member_name).parts[-2]
    ttyrecs = [name for name in source.namelist() if name.endswith(".ttyrec.bz2") and f"/{player_name}/" in name]
    games = load_player_games(XLOG_PATH, player_name)
    assignments = assign_ttyrecs_to_games(ttyrecs, games)
    member_to_gameid = {ttyrec: gameid for ttyrec, gameid in assignments}

    if member_name not in member_to_gameid:
        raise SystemExit(f"Could not match {member_name} to a human game in {XLOG_PATH}")

    target_gameid = member_to_gameid[member_name]
    game_members = sorted(
        [ttyrec for ttyrec, gameid in assignments if gameid == target_gameid]
    )
    return game_members, target_gameid, None


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    summary = pd.read_csv(SUMMARY_PATH)
    best_row = summary.sort_values("final_progression_percent", ascending=False).iloc[0]
    member_name = best_row["member_name"]

    achievements = load_achievements()
    with HumanNAODataSource() as source:
        game_members, gameid, expected_turns = collect_game_members(source, member_name)
        curve = parse_human_progression_members(
            source,
            member_names=game_members,
            achievements=achievements,
            max_turn=1_000_000,
            expected_turns=expected_turns,
        )

    if curve is None:
        raise SystemExit(f"Could not parse progression curve for stitched game containing {member_name}")

    # Recover the highest milestone label from the achievement table.
    achievement_lookup = {round(value * 100.0, 12): key for key, value in achievements.items()}
    final_progress = float(curve.iloc[-1])
    best_label = achievement_lookup.get(round(final_progress, 12), "unknown")
    final_turn = int(curve.index.max())

    plot_df = pd.DataFrame({"turn": curve.index, "progression_percent": curve.values})
    plot_df.to_csv(OUTPUT_CSV, index=False)

    milestone_mask = plot_df["progression_percent"].diff().fillna(plot_df["progression_percent"]) > 0
    milestone_df = plot_df[milestone_mask]

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#f6f3ee")
    ax.set_facecolor("#fffdfa")

    ax.plot(
        curve.index,
        curve.values,
        color="#7a1f2b",
        linewidth=3,
        drawstyle="steps-post",
        label="Best sampled human run",
    )

    if not milestone_df.empty:
        ax.scatter(
            milestone_df["turn"],
            milestone_df["progression_percent"],
            color="#b23a48",
            s=26,
            alpha=0.95,
            zorder=4,
            label="Progress jumps",
        )

    ax.scatter([final_turn], [final_progress], color="#d1495b", s=80, zorder=5)
    ax.annotate(
        f"{best_label} at turn {final_turn:,}\n{final_progress:.2f}%",
        xy=(final_turn, final_progress),
        xytext=(-140, -10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff3ef", ec="#d1495b", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#d1495b", lw=1.5),
        fontsize=11,
    )

    ax.set_title("Best Human NetHack Run in Sample (Stitched Full Game)", fontsize=18, pad=12)
    ax.set_xlabel("NetHack Turn `T`", fontsize=13)
    ax.set_ylabel("BALROG NLE Progression (%)", fontsize=13)
    ax.set_xlim(left=0)
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

    print(f"Best run: {member_name}")
    print(f"Stitched parts: {len(game_members)}")
    print(f"Game id: {gameid}")
    print(f"Final turn: {final_turn}")
    print(f"Final progression: {final_progress:.6f}%")
    print(f"Best milestone: {best_label}")
    print(f"Saved plot to {OUTPUT_PNG}")
    print(f"Saved trajectory data to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

