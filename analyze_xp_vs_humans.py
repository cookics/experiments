from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_human_dataset_effort import (
    CLUSTER_COLORS,
    GEMINI_COLORS,
    fit_effort_gmm,
    load_games,
    percentile_of_value,
)
from plot_nle_trajectories import find_nle_csvs, load_data
from train_gemini_continuation_value_model import initial_state, update_state_from_bytes


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_PNG = OUTPUT_DIR / "llm_vs_human_experience_panels.png"
RUN_CSV = OUTPUT_DIR / "llm_run_xp_vs_trying_humans.csv"
MODEL_CSV = OUTPUT_DIR / "llm_model_xp_vs_trying_humans.csv"
SUMMARY_JSON = OUTPUT_DIR / "llm_xp_vs_humans_summary.json"


def parse_llm_run_final(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path, usecols=["Observation"])
    state = initial_state()
    last_turn = 0
    for observation in df["Observation"].fillna(""):
        turn = update_state_from_bytes(str(observation).encode("utf-8", errors="ignore"), state)
        if turn is not None and turn >= last_turn:
            last_turn = turn
    return {
        "turns": float(last_turn),
        "final_xp": float(state["best_xl"]),
        "final_dlvl": float(state["best_dlvl"]),
        "final_score": float(state["best_score"]),
        "final_gold": float(state["best_gold"]),
    }


def build_llm_run_table() -> pd.DataFrame:
    data = load_data()
    rows: list[dict[str, object]] = []
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
            for csv_path in sorted(csv_paths, key=lambda p: p.stem):
                parsed = parse_llm_run_final(csv_path)
                rows.append(
                    {
                        "model_name": result["name"],
                        "date": result.get("date", ""),
                        "folder": str(folder.relative_to(ROOT)),
                        "run_name": csv_path.stem,
                        **parsed,
                    }
                )
    return pd.DataFrame(rows)


def human_cluster_hist(
    ax,
    games: pd.DataFrame,
    *,
    selected_models: pd.DataFrame | None,
    title: str,
    xlabel: str,
) -> None:
    order = ["instant_quit", "instant_die", "substantive", "deep"]
    data = [
        games.loc[games["cluster_name"] == name, "final_xp"].astype(float).to_numpy(dtype=np.float64)
        for name in order
    ]
    max_xp = int(max(games["final_xp"].max(), 30))
    bins = np.arange(-0.5, max_xp + 1.5, 1.0)
    ax.hist(
        data,
        bins=bins,
        stacked=True,
        color=[CLUSTER_COLORS[name] for name in order],
        edgecolor="white",
        linewidth=0.35,
    )
    if selected_models is not None:
        for idx, row in enumerate(selected_models.itertuples(index=False)):
            color = GEMINI_COLORS[idx % len(GEMINI_COLORS)]
            ax.axvline(float(row.mean_final_xp), color=color, linewidth=2.0, alpha=0.95)
    ax.set_title(title, fontsize=12.5)
    ax.set_xlabel(xlabel)
    ax.set_xlim(-0.5, max_xp + 0.5)
    ax.grid(True, axis="y", alpha=0.14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def overlay_hist(
    ax,
    all_values: np.ndarray,
    trying_values: np.ndarray,
    *,
    selected_models: pd.DataFrame | None,
    title: str,
    xlabel: str,
) -> None:
    max_xp = int(max(np.max(all_values), np.max(trying_values), 30))
    bins = np.arange(-0.5, max_xp + 1.5, 1.0)
    ax.hist(all_values, bins=bins, color="#d8dee3", alpha=0.75, edgecolor="white", linewidth=0.35, label="All humans")
    ax.hist(trying_values, bins=bins, histtype="step", color="#1f4f78", linewidth=2.2, label="Trying humans")
    if selected_models is not None:
        for idx, row in enumerate(selected_models.itertuples(index=False)):
            color = GEMINI_COLORS[idx % len(GEMINI_COLORS)]
            ax.axvline(float(row.mean_final_xp), color=color, linewidth=2.0, alpha=0.95)
    ax.set_title(title, fontsize=12.5)
    ax.set_xlabel(xlabel)
    ax.set_xlim(-0.5, max_xp + 0.5)
    ax.grid(True, axis="y", alpha=0.14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    humans = load_games()
    humans, _, _, _ = fit_effort_gmm(humans)
    trying = humans[humans["trying_strict"]].copy()

    llm_runs = build_llm_run_table()
    llm_runs["xp_pct_all_humans"] = llm_runs["final_xp"].map(lambda x: percentile_of_value(humans["final_xp"], x))
    llm_runs["xp_pct_trying_humans"] = llm_runs["final_xp"].map(lambda x: percentile_of_value(trying["final_xp"], x))
    llm_runs["dlvl_pct_trying_humans"] = llm_runs["final_dlvl"].map(lambda x: percentile_of_value(trying["final_dlvl"], x))
    llm_runs["score_pct_trying_humans"] = llm_runs["final_score"].map(lambda x: percentile_of_value(trying["final_score"], x))
    llm_runs.to_csv(RUN_CSV, index=False)

    llm_models = (
        llm_runs.groupby(["model_name", "date", "folder"], as_index=False)
        .agg(
            run_count=("run_name", "size"),
            mean_final_xp=("final_xp", "mean"),
            median_final_xp=("final_xp", "median"),
            max_run_xp=("final_xp", "max"),
            mean_final_dlvl=("final_dlvl", "mean"),
            mean_final_score=("final_score", "mean"),
            mean_turns=("turns", "mean"),
            mean_xp_pct_all_humans=("xp_pct_all_humans", "mean"),
            mean_xp_pct_trying_humans=("xp_pct_trying_humans", "mean"),
            best_run_xp_pct_trying_humans=("xp_pct_trying_humans", "max"),
        )
        .sort_values(["mean_final_xp", "mean_final_dlvl", "model_name"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    llm_models["mean_xp_value_pct_trying_humans"] = llm_models["mean_final_xp"].map(
        lambda x: percentile_of_value(trying["final_xp"], x)
    )
    llm_models["median_xp_value_pct_trying_humans"] = llm_models["median_final_xp"].map(
        lambda x: percentile_of_value(trying["final_xp"], x)
    )
    llm_models.to_csv(MODEL_CSV, index=False)

    selected = llm_models.head(8).copy()
    if "Gemini-3-Pro" in set(llm_models["model_name"]) and "Gemini-3-Pro" not in set(selected["model_name"]):
        selected = pd.concat([selected, llm_models[llm_models["model_name"] == "Gemini-3-Pro"]], ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor("#f4efe8")
    for ax in axes.flat:
        ax.set_facecolor("#fffaf3")

    human_cluster_hist(
        axes[0, 0],
        humans,
        selected_models=selected,
        title="All humans: final XP level by effort cluster",
        xlabel="Final experience level",
    )
    overlay_hist(
        axes[0, 1],
        humans["final_xp"].to_numpy(dtype=np.float64),
        trying["final_xp"].to_numpy(dtype=np.float64),
        selected_models=selected,
        title="Trying humans vs top LLM model means",
        xlabel="Final experience level",
    )

    model_bins = np.arange(-0.5, max(int(llm_models["mean_final_xp"].max()), 12) + 1.5, 1.0)
    axes[1, 0].hist(
        llm_models["mean_final_xp"].to_numpy(dtype=np.float64),
        bins=model_bins,
        color="#6c8aa6",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.45,
    )
    axes[1, 0].set_title("LLM model means: final XP")
    axes[1, 0].set_xlabel("Mean final experience level across runs")
    axes[1, 0].grid(True, axis="y", alpha=0.14)
    axes[1, 0].spines["top"].set_visible(False)
    axes[1, 0].spines["right"].set_visible(False)

    pct_bins = np.linspace(0, 100, 21)
    axes[1, 1].hist(
        llm_models["mean_xp_value_pct_trying_humans"].to_numpy(dtype=np.float64),
        bins=pct_bins,
        color="#457b9d",
        alpha=0.84,
        edgecolor="white",
        linewidth=0.45,
    )
    axes[1, 1].set_title("LLM model means: percentile among trying humans")
    axes[1, 1].set_xlabel("Percentile in trying-human final XP distribution")
    axes[1, 1].grid(True, axis="y", alpha=0.14)
    axes[1, 1].spines["top"].set_visible(False)
    axes[1, 1].spines["right"].set_visible(False)

    cluster_handles = [
        plt.Line2D([0], [0], color=CLUSTER_COLORS[name], linewidth=8)
        for name in ["instant_quit", "instant_die", "substantive", "deep"]
    ]
    cluster_labels = ["instant_quit", "instant_die", "substantive", "deep"]
    subset_handles = [
        plt.Line2D([0], [0], color="#d8dee3", linewidth=8),
        plt.Line2D([0], [0], color="#1f4f78", linewidth=2.2),
    ]
    subset_labels = ["All humans", "Trying humans"]
    model_handles = [
        plt.Line2D([0], [0], color=GEMINI_COLORS[idx % len(GEMINI_COLORS)], linewidth=2.0)
        for idx in range(len(selected))
    ]
    model_labels = [str(name) for name in selected["model_name"]]
    fig.legend(
        cluster_handles + subset_handles + model_handles,
        cluster_labels + subset_labels + model_labels,
        loc="upper center",
        ncol=5,
        frameon=True,
        framealpha=0.95,
    )
    fig.suptitle("LLMs vs humans in final experience level", fontsize=22, y=0.992)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUTPUT_PNG, dpi=220)
    plt.close(fig)

    summary = {
        "human_counts": {
            "all": int(len(humans)),
            "trying": int(len(trying)),
            "trying_share": float(len(trying) / len(humans)),
        },
        "all_humans_final_xp_quantiles": humans["final_xp"].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_dict(),
        "trying_humans_final_xp_quantiles": trying["final_xp"].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_dict(),
        "llm_model_count": int(len(llm_models)),
        "top_models_by_mean_final_xp": llm_models.head(10).to_dict(orient="records"),
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        json.dumps(
            {
                "output_png": str(OUTPUT_PNG),
                "run_csv": str(RUN_CSV),
                "model_csv": str(MODEL_CSV),
                "summary_json": str(SUMMARY_JSON),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
