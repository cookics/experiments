from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_human_best_median_vs_gemini import load_gemini_sparse_curves
from plot_human_nao_trajectories import load_achievements
from train_gemini_latent_skill_model import (
    DB_PATH,
    GEMINI_CSV,
    HORIZON,
    PREDICTIONS_CSV,
    build_gemini_feature_table,
    build_human_feature_table,
    find_bad_games,
    load_human_data,
    make_skill_target,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs"
DIST_PNG = OUTPUT_DIR / "human_vs_gemini_metric_distributions_1000.png"
ECDF_PNG = OUTPUT_DIR / "human_vs_gemini_progress_ecdf.png"
SUMMARY_JSON = OUTPUT_DIR / "human_vs_gemini_metric_distribution_summary.json"


def progression_from_features(df: pd.DataFrame) -> pd.Series:
    ach = load_achievements()
    dlvl_map = {i: ach[f"Dlvl:{i}"] * 100.0 for i in range(1, 100) if f"Dlvl:{i}" in ach}
    xp_map = {i: ach[f"Xp:{i}"] * 100.0 for i in range(1, 100) if f"Xp:{i}" in ach}
    home_map = {i: ach[f"Home {i}"] * 100.0 for i in range(1, 6) if f"Home {i}" in ach}
    return pd.concat(
        [
            df["dlvl_final"].map(dlvl_map).fillna(0.0),
            df["xp_final"].map(xp_map).fillna(0.0),
            df["home_final"].map(home_map).fillna(0.0),
        ],
        axis=1,
    ).max(axis=1)


def stats_dict(series: pd.Series) -> dict[str, float]:
    q = series.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p10": float(q[0.10]),
        "p25": float(q[0.25]),
        "p75": float(q[0.75]),
        "p90": float(q[0.90]),
        "p95": float(q[0.95]),
        "p99": float(q[0.99]),
        "max": float(series.max()),
    }


def ecdf_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values.astype(float))
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


def plot_hist_with_markers(
    ax,
    human_values: np.ndarray,
    gemini_values: np.ndarray,
    gemini_labels: list[str],
    *,
    title: str,
    x_label: str,
    bins: int | np.ndarray,
    human_color: str = "#6c8aa6",
    gemini_colors: list[str] | None = None,
) -> None:
    ax.hist(
        human_values,
        bins=bins,
        density=True,
        alpha=0.68,
        color=human_color,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(float(np.mean(human_values)), color="#1d3557", linestyle="--", linewidth=2.0, label="Human mean")
    ax.axvline(float(np.median(human_values)), color="#264653", linestyle=":", linewidth=2.0, label="Human median")
    colors = gemini_colors or ["#355070", "#6d597a", "#b56576", "#457b9d", "#2a9d8f"]
    for color, value, label in zip(colors, gemini_values, gemini_labels):
        ax.axvline(float(value), color=color, linewidth=2.0, alpha=0.95, label=label)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(x_label)
    ax.grid(True, axis="y", alpha=0.14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    games, events = load_human_data(DB_PATH)
    bad_games = find_bad_games(games, events)
    games = games[~games["game_key"].astype(str).isin(bad_games)].copy()
    events = events[~events["game_key"].astype(str).isin(bad_games)].copy()
    games = make_skill_target(games)

    human_features = build_human_feature_table(games, events, HORIZON)
    human = games.merge(human_features, on="game_key", how="inner")
    human["progression_1000"] = progression_from_features(human)

    gemini_features = build_gemini_feature_table(HORIZON)
    gemini_features["progression_1000"] = progression_from_features(gemini_features)

    human_pred = pd.read_csv(PREDICTIONS_CSV)
    gemini_pred = pd.read_csv(GEMINI_CSV)
    human = human.merge(human_pred[["game_key", "pred_skill_pct"]], on="game_key", how="left")
    gemini_features = gemini_features.merge(
        gemini_pred[["run_name", "pred_skill_pct", "pred_pct_among_human_preds"]],
        on="run_name",
        how="left",
    )

    gemini_score_curves, gemini_progress_curves = load_gemini_sparse_curves()
    gemini_final_progress = {
        curve.run_name: float(curve.values[-1]) if len(curve.values) else 0.0
        for curve in gemini_progress_curves
    }
    gemini_final_score = {
        curve.run_name: float(curve.values[-1]) if len(curve.values) else 0.0
        for curve in gemini_score_curves
    }
    gemini_features["final_progression"] = gemini_features["run_name"].map(gemini_final_progress).fillna(0.0)
    gemini_features["final_score"] = gemini_features["run_name"].map(gemini_final_score).fillna(0.0)

    human_final_progress = games["final_progression_pct"].astype(float)
    human_progress_1000 = human["progression_1000"].astype(float)
    human_depth_1000 = human["dlvl_final"].astype(float)
    human_xp_1000 = human["xp_final"].astype(float)
    human_score_1000_log = np.log1p(human["score_final"].astype(float))
    human_latent_pred = human["pred_skill_pct"].astype(float)

    gem_labels = gemini_features["run_name"].astype(str).tolist()
    gem_final_progress = gemini_features["final_progression"].to_numpy(dtype=np.float64)
    gem_progress_1000 = gemini_features["progression_1000"].to_numpy(dtype=np.float64)
    gem_depth_1000 = gemini_features["dlvl_final"].to_numpy(dtype=np.float64)
    gem_xp_1000 = gemini_features["xp_final"].to_numpy(dtype=np.float64)
    gem_score_1000_log = np.log1p(gemini_features["score_final"].astype(float).to_numpy(dtype=np.float64))
    gem_latent_pred = gemini_features["pred_skill_pct"].to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.patch.set_facecolor("#f4efe8")
    for ax in axes.flat:
        ax.set_facecolor("#fffaf3")

    plot_hist_with_markers(
        axes[0, 0],
        human_final_progress.to_numpy(dtype=np.float64),
        gem_final_progress,
        gem_labels,
        title="Final BALROG progression",
        x_label="Progression (%)",
        bins=np.linspace(0, 100, 41),
    )
    plot_hist_with_markers(
        axes[0, 1],
        human_progress_1000.to_numpy(dtype=np.float64),
        gem_progress_1000,
        gem_labels,
        title="Progression by turn 1000",
        x_label="Progression at T=1000 (%)",
        bins=np.linspace(0, max(25.0, human_progress_1000.max(), gem_progress_1000.max()) + 1, 30),
    )
    plot_hist_with_markers(
        axes[0, 2],
        human_depth_1000.to_numpy(dtype=np.float64),
        gem_depth_1000,
        gem_labels,
        title="Depth by turn 1000",
        x_label="Best dungeon level by T=1000",
        bins=np.arange(-0.5, max(13.5, human_depth_1000.max(), gem_depth_1000.max()) + 1.5, 1.0),
    )
    plot_hist_with_markers(
        axes[1, 0],
        human_score_1000_log.to_numpy(dtype=np.float64),
        gem_score_1000_log,
        gem_labels,
        title="Score by turn 1000",
        x_label="log1p(score at T=1000)",
        bins=35,
    )
    plot_hist_with_markers(
        axes[1, 1],
        human_xp_1000.to_numpy(dtype=np.float64),
        gem_xp_1000,
        gem_labels,
        title="XP by turn 1000",
        x_label="Best XP level by T=1000",
        bins=np.arange(-0.5, max(10.5, human_xp_1000.max(), gem_xp_1000.max()) + 1.5, 1.0),
    )
    plot_hist_with_markers(
        axes[1, 2],
        human_latent_pred.to_numpy(dtype=np.float64),
        gem_latent_pred,
        gem_labels,
        title="Predicted latent skill from first 1000 turns",
        x_label="Predicted latent skill percentile",
        bins=np.linspace(0, 100, 31),
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, framealpha=0.95)
    fig.suptitle("Humans vs Gemini-3-Pro: metric distributions", fontsize=22, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(DIST_PNG, dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor("#f4efe8")
    for ax in axes:
        ax.set_facecolor("#fffaf3")

    panels = [
        (axes[0], human_final_progress.to_numpy(dtype=np.float64), gem_final_progress, "Final progression", "Progression (%)"),
        (axes[1], human_progress_1000.to_numpy(dtype=np.float64), gem_progress_1000, "Progression at T=1000", "Progression (%)"),
        (axes[2], human_latent_pred.to_numpy(dtype=np.float64), gem_latent_pred, "Predicted latent skill", "Predicted percentile"),
    ]
    colors = ["#355070", "#6d597a", "#b56576", "#457b9d", "#2a9d8f"]
    for ax, human_vals, gem_vals, title, xlabel in panels:
        x, y = ecdf_values(human_vals)
        ax.plot(x, y, color="#355070", linewidth=2.4, label="Humans")
        for color, value, label in zip(colors, gem_vals, gem_labels):
            ax.axvline(float(value), color=color, linewidth=1.9, alpha=0.95, label=label)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Human CDF")
        ax.grid(True, alpha=0.14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, framealpha=0.95)
    fig.suptitle("Where Gemini sits inside the human distributions", fontsize=19, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(ECDF_PNG, dpi=220)
    plt.close(fig)

    summary = {
        "final_progression_humans": stats_dict(human_final_progress),
        "progression_1000_humans": stats_dict(human_progress_1000),
        "depth_1000_humans": stats_dict(human_depth_1000),
        "xp_1000_humans": stats_dict(human_xp_1000),
        "score_1000_humans": stats_dict(human["score_final"].astype(float)),
        "latent_skill_pred_humans": stats_dict(human_latent_pred),
        "gemini_runs": gemini_features[
            [
                "run_name",
                "final_progression",
                "progression_1000",
                "dlvl_final",
                "xp_final",
                "score_final",
                "pred_skill_pct",
                "pred_pct_among_human_preds",
            ]
        ].to_dict(orient="records"),
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        json.dumps(
            {
                "dist_png": str(DIST_PNG),
                "ecdf_png": str(ECDF_PNG),
                "summary_json": str(SUMMARY_JSON),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
