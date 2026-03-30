from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_human_dataset_effort import GEMINI_COLORS, fit_effort_gmm, load_games, percentile_of_value
from plot_nle_trajectories import find_nle_csvs, load_data
from train_gemini_continuation_value_model import initial_state, update_state_from_bytes


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs"
DB_PATH = OUTPUT_DIR / "human_nao_full_trajectories.sqlite3"
HORIZONS = [1000, 2000, 3000, 4000, 5000, 6000]

HIST_PNG = OUTPUT_DIR / "llm_vs_trying_humans_xp_horizon_histograms.png"
LINES_PNG = OUTPUT_DIR / "llm_xp_horizon_lines.png"
RUN_CSV = OUTPUT_DIR / "llm_run_xp_horizons_vs_trying_humans.csv"
MODEL_CSV = OUTPUT_DIR / "llm_model_xp_horizons_vs_trying_humans.csv"
HUMAN_CSV = OUTPUT_DIR / "human_trying_xp_horizons_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "llm_xp_horizons_summary.json"


def load_events() -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    try:
        events = pd.read_sql_query(
            """
            SELECT game_key, event_idx, turn, xp
            FROM events
            ORDER BY game_key, turn, event_idx
            """,
            conn,
        )
    finally:
        conn.close()
    events["game_key"] = events["game_key"].astype(str)
    return events


def build_human_horizon_table(games: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = games[
        [
            "game_key",
            "player_name",
            "turns",
            "final_xp",
            "trying_strict",
            "cluster_name",
        ]
    ].copy()

    for horizon in HORIZONS:
        early = events[events["turn"] <= horizon]
        last_rows = early.groupby("game_key", sort=False).tail(1)[["game_key", "xp"]].rename(columns={"xp": f"xp_t{horizon}"})
        out = out.merge(last_rows, on="game_key", how="left")
        ended_mask = out["turns"].astype(int) <= horizon
        out.loc[ended_mask & out[f"xp_t{horizon}"].isna(), f"xp_t{horizon}"] = out.loc[
            ended_mask & out[f"xp_t{horizon}"].isna(), "final_xp"
        ]
        out[f"xp_t{horizon}"] = out[f"xp_t{horizon}"].fillna(0.0).astype(float)

    return out


def parse_llm_run_horizons(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path, usecols=["Observation"])
    state = initial_state()
    last_turn = 0
    checkpoints = {h: None for h in HORIZONS}

    for observation in df["Observation"].fillna(""):
        turn = update_state_from_bytes(str(observation).encode("utf-8", errors="ignore"), state)
        if turn is None or turn < last_turn:
            continue
        last_turn = turn
        for horizon in HORIZONS:
            if checkpoints[horizon] is None and turn >= horizon:
                checkpoints[horizon] = float(state["best_xl"])
        if turn >= HORIZONS[-1]:
            break

    for horizon in HORIZONS:
        if checkpoints[horizon] is None:
            checkpoints[horizon] = float(state["best_xl"])

    row = {"turns": float(last_turn), "final_xp": float(state["best_xl"])}
    for horizon in HORIZONS:
        row[f"xp_t{horizon}"] = float(checkpoints[horizon])
    return row


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
                parsed = parse_llm_run_horizons(csv_path)
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


def add_percentiles(llm_runs: pd.DataFrame, humans: pd.DataFrame, trying: pd.DataFrame) -> pd.DataFrame:
    out = llm_runs.copy()
    for horizon in HORIZONS:
        col = f"xp_t{horizon}"
        out[f"{col}_pct_all_humans"] = out[col].map(lambda x: percentile_of_value(humans[col], x))
        out[f"{col}_pct_trying_humans"] = out[col].map(lambda x: percentile_of_value(trying[col], x))
    return out


def build_model_summary(llm_runs: pd.DataFrame, trying: pd.DataFrame) -> pd.DataFrame:
    agg_spec: dict[str, tuple[str, str]] = {
        "run_count": ("run_name", "size"),
        "mean_turns": ("turns", "mean"),
        "mean_final_xp": ("final_xp", "mean"),
        "median_final_xp": ("final_xp", "median"),
        "max_final_xp": ("final_xp", "max"),
    }
    for horizon in HORIZONS:
        col = f"xp_t{horizon}"
        pct_col = f"{col}_pct_trying_humans"
        agg_spec[f"mean_{col}"] = (col, "mean")
        agg_spec[f"median_{col}"] = (col, "median")
        agg_spec[f"max_{col}"] = (col, "max")
        agg_spec[f"mean_{pct_col}"] = (pct_col, "mean")
        agg_spec[f"best_{pct_col}"] = (pct_col, "max")

    out = (
        llm_runs.groupby(["model_name", "date", "folder"], as_index=False)
        .agg(**agg_spec)
        .sort_values(["mean_xp_t6000", "mean_final_xp", "model_name"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    for horizon in HORIZONS:
        col = f"mean_xp_t{horizon}"
        med_col = f"median_xp_t{horizon}"
        out[f"{col}_value_pct_trying_humans"] = out[col].map(lambda x: percentile_of_value(trying[f"xp_t{horizon}"], x))
        out[f"{med_col}_value_pct_trying_humans"] = out[med_col].map(lambda x: percentile_of_value(trying[f"xp_t{horizon}"], x))
    return out


def plot_histograms(humans: pd.DataFrame, trying: pd.DataFrame, selected_models: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor("#f4efe8")
    for ax in axes.flat:
        ax.set_facecolor("#fffaf3")

    max_xp = int(max(humans["final_xp"].max(), 30))
    bins = np.arange(-0.5, max_xp + 1.5, 1.0)

    for ax, horizon in zip(axes.flat, HORIZONS):
        human_vals = humans[f"xp_t{horizon}"].to_numpy(dtype=np.float64)
        trying_vals = trying[f"xp_t{horizon}"].to_numpy(dtype=np.float64)
        ax.hist(human_vals, bins=bins, color="#d8dee3", alpha=0.78, edgecolor="white", linewidth=0.35, label="All humans")
        ax.hist(trying_vals, bins=bins, histtype="step", color="#1f4f78", linewidth=2.1, label="Trying humans")
        for idx, row in enumerate(selected_models.itertuples(index=False)):
            color = GEMINI_COLORS[idx % len(GEMINI_COLORS)]
            ax.axvline(float(getattr(row, f"mean_xp_t{horizon}")), color=color, linewidth=2.0, alpha=0.95)
        ax.set_title(f"T = {horizon}")
        ax.set_xlabel("Best experience level by horizon")
        ax.set_xlim(-0.5, max_xp + 0.5)
        ax.grid(True, axis="y", alpha=0.14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color="#d8dee3", linewidth=8),
        plt.Line2D([0], [0], color="#1f4f78", linewidth=2.1),
    ] + [plt.Line2D([0], [0], color=GEMINI_COLORS[idx % len(GEMINI_COLORS)], linewidth=2.0) for idx in range(len(selected_models))]
    labels = ["All humans", "Trying humans"] + [str(name) for name in selected_models["model_name"]]
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True, framealpha=0.95)
    fig.suptitle("Experience level by NetHack turn horizon", fontsize=22, y=0.992)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(HIST_PNG, dpi=220)
    plt.close(fig)


def plot_model_lines(model_summary: pd.DataFrame, selected_models: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    fig.patch.set_facecolor("#f4efe8")
    for ax in axes:
        ax.set_facecolor("#fffaf3")

    horizons = np.array(HORIZONS, dtype=np.int32)
    for idx, row in enumerate(selected_models.itertuples(index=False)):
        color = GEMINI_COLORS[idx % len(GEMINI_COLORS)]
        mean_xps = np.array([getattr(row, f"mean_xp_t{h}") for h in HORIZONS], dtype=np.float64)
        pct_vals = np.array([getattr(row, f"mean_xp_t{h}_value_pct_trying_humans") for h in HORIZONS], dtype=np.float64)
        axes[0].plot(horizons, mean_xps, color=color, linewidth=2.4, marker="o", label=str(row.model_name))
        axes[1].plot(horizons, pct_vals, color=color, linewidth=2.4, marker="o", label=str(row.model_name))

    axes[0].set_title("LLM model means in experience level")
    axes[0].set_ylabel("Mean best XP by horizon")
    axes[0].grid(True, alpha=0.14)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].set_title("LLM model mean XP as percentile of trying humans")
    axes[1].set_xlabel("NetHack turn horizon")
    axes[1].set_ylabel("Percentile among trying humans")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.14)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].set_xticks(horizons)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, framealpha=0.95)
    fig.suptitle("LLM experience-level trajectories across horizons", fontsize=21, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(LINES_PNG, dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    humans = load_games()
    humans, _, _, _ = fit_effort_gmm(humans)
    events = load_events()
    human_h = build_human_horizon_table(humans, events)
    trying = human_h[human_h["trying_strict"]].copy()

    llm_runs = add_percentiles(build_llm_run_table(), human_h, trying)
    llm_runs.to_csv(RUN_CSV, index=False)

    model_summary = build_model_summary(llm_runs, trying)
    model_summary.to_csv(MODEL_CSV, index=False)

    human_summary_rows = []
    for horizon in HORIZONS:
        col = f"xp_t{horizon}"
        row = {
            "horizon": horizon,
            "all_humans_p25": float(human_h[col].quantile(0.25)),
            "all_humans_p50": float(human_h[col].quantile(0.50)),
            "all_humans_p75": float(human_h[col].quantile(0.75)),
            "all_humans_p90": float(human_h[col].quantile(0.90)),
            "trying_humans_p25": float(trying[col].quantile(0.25)),
            "trying_humans_p50": float(trying[col].quantile(0.50)),
            "trying_humans_p75": float(trying[col].quantile(0.75)),
            "trying_humans_p90": float(trying[col].quantile(0.90)),
            "trying_humans_mean": float(trying[col].mean()),
        }
        human_summary_rows.append(row)
    human_summary = pd.DataFrame(human_summary_rows)
    human_summary.to_csv(HUMAN_CSV, index=False)

    selected = model_summary.head(8).copy()
    if "Gemini-3-Pro" in set(model_summary["model_name"]) and "Gemini-3-Pro" not in set(selected["model_name"]):
        selected = pd.concat([selected, model_summary[model_summary["model_name"] == "Gemini-3-Pro"]], ignore_index=True)
        selected = selected.drop_duplicates(subset=["model_name"], keep="first").reset_index(drop=True)

    plot_histograms(human_h, trying, selected)
    plot_model_lines(model_summary, selected)

    summary = {
        "horizons": HORIZONS,
        "human_counts": {
            "all": int(len(human_h)),
            "trying": int(len(trying)),
            "trying_share": float(len(trying) / len(human_h)),
        },
        "human_horizon_quantiles": human_summary.to_dict(orient="records"),
        "top_models_by_mean_xp_t6000": model_summary.head(10).to_dict(orient="records"),
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        json.dumps(
            {
                "hist_png": str(HIST_PNG),
                "lines_png": str(LINES_PNG),
                "run_csv": str(RUN_CSV),
                "model_csv": str(MODEL_CSV),
                "human_csv": str(HUMAN_CSV),
                "summary_json": str(SUMMARY_JSON),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
