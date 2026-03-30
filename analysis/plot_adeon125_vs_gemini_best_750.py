from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.human_nao_source import HumanNAODataSource
from analysis.plot_gemini3_vs_humans import load_gemini_records, parse_human_metric_curves
from analysis.plot_human_nao_trajectories import load_achievements


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_PNG = OUTPUT_DIR / "adeon125_vs_gemini_best_750.png"
OUTPUT_CSV = OUTPUT_DIR / "adeon125_vs_gemini_best_750_manifest.csv"

MAX_TURN = 750
TARGET_LABEL = "adeon#125"
TARGET_DEATH = "ascended"
TARGET_MEMBERS = [
    "nld-nao-unzipped/Adeon/2009-02-28.10:31:15.ttyrec.bz2",
    "nld-nao-unzipped/Adeon/2009-02-28.13:40:59.ttyrec.bz2",
    "nld-nao-unzipped/Adeon/2009-02-28.15:40:30.ttyrec.bz2",
    "nld-nao-unzipped/Adeon/2009-02-28.16:58:02.ttyrec.bz2",
    "nld-nao-unzipped/Adeon/2009-02-28.18:19:43.ttyrec.bz2",
]


def load_best_gemini(achievements: dict[str, float]) -> dict:
    gemini_records = load_gemini_records(max_turn=MAX_TURN, achievements=achievements)
    return max(
        gemini_records,
        key=lambda record: (
            float(record["progression_curve"].loc[MAX_TURN]),
            float(record["score_curve"].loc[MAX_TURN]),
            record["run_name"],
        ),
    )


def load_target_human(achievements: dict[str, float]) -> dict:
    with HumanNAODataSource() as source:
        score_curve, progression_curve = parse_human_metric_curves(
            source,
            member_names=TARGET_MEMBERS,
            achievements=achievements,
            max_turn=MAX_TURN,
            expected_turns=None,
        )
        if score_curve is None or progression_curve is None:
            raise SystemExit(f"Could not parse human run {TARGET_LABEL}.")

    return {
        "label": TARGET_LABEL,
        "death": TARGET_DEATH,
        "score_curve": score_curve,
        "progression_curve": progression_curve,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    achievements = load_achievements()
    gemini = load_best_gemini(achievements)
    human = load_target_human(achievements)

    fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    fig.patch.set_facecolor("#f7f1e8")

    for ax in axes:
        ax.set_facecolor("#fffaf3")
        ax.grid(True, alpha=0.22, linewidth=0.9, color="#8f7f6d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#6c6258")
        ax.spines["bottom"].set_color("#6c6258")

    gemini_color = "#6a1b9a"
    human_color = "#b23a48"

    axes[0].plot(
        gemini["progression_curve"].index,
        gemini["progression_curve"].values,
        color=gemini_color,
        linewidth=3.0,
        drawstyle="steps-post",
        label=f"Gemini best: {gemini['run_name']}",
    )
    axes[0].plot(
        human["progression_curve"].index,
        human["progression_curve"].values,
        color=human_color,
        linewidth=3.0,
        drawstyle="steps-post",
        label=f"Human: {human['label']}",
    )
    axes[0].set_ylabel("BALROG NLE Progression (%)", fontsize=13)
    axes[0].set_title("Best Saved Human vs Gemini-3-Pro Best Run (First 750 Turns)", fontsize=18, pad=12)
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].plot(
        gemini["score_curve"].index,
        gemini["score_curve"].values,
        color=gemini_color,
        linewidth=3.0,
        drawstyle="steps-post",
        label=f"Gemini best: {gemini['run_name']}",
    )
    axes[1].plot(
        human["score_curve"].index,
        human["score_curve"].values,
        color=human_color,
        linewidth=3.0,
        drawstyle="steps-post",
        label=f"Human: {human['label']}",
    )
    axes[1].set_ylabel("Status-Line Score `S`", fontsize=13)
    axes[1].set_xlabel("NetHack Turn `T`", fontsize=13)
    axes[1].legend(frameon=False, loc="upper left")

    for ax, metric_key in [
        (axes[0], "progression_curve"),
        (axes[1], "score_curve"),
    ]:
        gemini_final = float(gemini[metric_key].loc[MAX_TURN])
        human_final = float(human[metric_key].loc[MAX_TURN])
        ax.scatter([MAX_TURN], [gemini_final], color=gemini_color, s=55, zorder=5)
        ax.scatter([MAX_TURN], [human_final], color=human_color, s=55, zorder=5)

    axes[0].annotate(
        f"{float(gemini['progression_curve'].loc[MAX_TURN]):.2f}%",
        xy=(MAX_TURN, float(gemini["progression_curve"].loc[MAX_TURN])),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        color=gemini_color,
    )
    axes[0].annotate(
        f"{float(human['progression_curve'].loc[MAX_TURN]):.2f}%",
        xy=(MAX_TURN, float(human["progression_curve"].loc[MAX_TURN])),
        xytext=(10, -18),
        textcoords="offset points",
        fontsize=10,
        color=human_color,
    )
    axes[1].annotate(
        f"{float(gemini['score_curve'].loc[MAX_TURN]):.0f}",
        xy=(MAX_TURN, float(gemini["score_curve"].loc[MAX_TURN])),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        color=gemini_color,
    )
    axes[1].annotate(
        f"{float(human['score_curve'].loc[MAX_TURN]):.0f}",
        xy=(MAX_TURN, float(human["score_curve"].loc[MAX_TURN])),
        xytext=(10, -18),
        textcoords="offset points",
        fontsize=10,
        color=human_color,
    )

    axes[0].set_xlim(0, MAX_TURN)
    axes[1].set_xlim(0, MAX_TURN)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    pd.DataFrame(
        [
            {
                "label": gemini["run_name"],
                "group": "gemini_best",
                "progression_at_750": float(gemini["progression_curve"].loc[MAX_TURN]),
                "score_at_750": float(gemini["score_curve"].loc[MAX_TURN]),
            },
            {
                "label": human["label"],
                "group": "human_saved_best",
                "progression_at_750": float(human["progression_curve"].loc[MAX_TURN]),
                "score_at_750": float(human["score_curve"].loc[MAX_TURN]),
                "death": human["death"],
            },
        ]
    ).to_csv(OUTPUT_CSV, index=False)

    print(f"Saved comparison plot to {OUTPUT_PNG}")
    print(f"Saved manifest to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

