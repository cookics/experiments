from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from plot_human_nao_trajectories import load_achievements
from plot_nle_trajectories import find_nle_csvs
from train_gemini_continuation_value_model import initial_state, update_state_from_bytes


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs"
DB_PATH = OUTPUT_DIR / "human_nao_full_trajectories.sqlite3"
GEMINI_FOLDER = ROOT / "submissions" / "LLM" / "20260203_naive_gemini-3-pro"

BASIC_PNG = OUTPUT_DIR / "human_dataset_basic_panels.png"
COMPARE_PNG = OUTPUT_DIR / "human_trying_vs_gemini_panels.png"
ENGAGEMENT_PNG = OUTPUT_DIR / "human_engagement_probability_hist.png"
CLUSTER_SUMMARY_CSV = OUTPUT_DIR / "human_effort_cluster_summary.csv"
GEMINI_CSV = OUTPUT_DIR / "gemini_final_effort_summary.csv"
PERCENTILES_CSV = OUTPUT_DIR / "gemini_vs_human_percentiles.csv"
SUMMARY_JSON = OUTPUT_DIR / "human_dataset_effort_summary.json"

RNG_SEED = 0

CLUSTER_COLORS = {
    "instant_quit": "#d9d9d9",
    "instant_die": "#bdbdbd",
    "substantive": "#7aa6c2",
    "deep": "#1f4f78",
}
GEMINI_COLORS = ["#355070", "#6d597a", "#b56576", "#457b9d", "#2a9d8f"]


def load_games() -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    try:
        games = pd.read_sql_query("SELECT * FROM games ORDER BY game_key", conn)
    finally:
        conn.close()
    games["game_key"] = games["game_key"].astype(str)
    # Drop obviously inconsistent parsed-progression rows before using progression/xp.
    games = games[games["final_dlvl"].astype(int) <= games["maxlvl"].astype(int)].copy()
    return games


def outcome_family(text: str) -> str:
    s = str(text).lower()
    if "ascended" in s:
        return "ascended"
    if "escaped" in s:
        return "escaped"
    if "quit" in s:
        return "quit"
    if s.startswith("killed"):
        return "killed"
    if s.startswith("petrified"):
        return "petrified"
    if s.startswith("poisoned"):
        return "poisoned"
    if s.startswith("choked"):
        return "choked"
    if "starv" in s:
        return "starved"
    if "drowned" in s:
        return "drowned"
    return "other"


def progression_pct_from_state(state: dict[str, float], achievements: dict[str, float]) -> float:
    candidates = [
        achievements.get(f"Dlvl:{int(state['best_dlvl'])}", 0.0),
        achievements.get(f"Xp:{int(state['best_xl'])}", 0.0),
        achievements.get(f"Home {int(state['best_home'])}", 0.0),
    ]
    return max(candidates) * 100.0


def parse_gemini_final_runs() -> pd.DataFrame:
    ach = load_achievements()
    rows: list[dict[str, float | str]] = []
    for csv_path in sorted(find_nle_csvs(GEMINI_FOLDER), key=lambda p: p.stem):
        df = pd.read_csv(csv_path, usecols=["Observation"])
        state = initial_state()
        last_turn = 0
        for obs in df["Observation"].fillna(""):
            turn = update_state_from_bytes(str(obs).encode("utf-8", errors="ignore"), state)
            if turn is not None and turn >= last_turn:
                last_turn = turn
        rows.append(
            {
                "run_name": csv_path.stem,
                "turns": int(last_turn),
                "points": float(state["best_score"]),
                "final_score": float(state["best_score"]),
                "maxlvl": float(state["best_dlvl"]),
                "final_progression_pct": float(progression_pct_from_state(state, ach)),
                "final_xp": float(state["best_xl"]),
            }
        )
    return pd.DataFrame(rows)


def fit_effort_gmm(games: pd.DataFrame) -> tuple[pd.DataFrame, GaussianMixture, StandardScaler, dict[int, str]]:
    out = games.copy()
    out["quit_flag"] = out["death"].map(outcome_family).eq("quit").astype(float)
    feature_frame = pd.DataFrame(
        {
            "log_turns": np.log1p(out["turns"].clip(lower=0)),
            "log_points": np.log1p(out["points"].clip(lower=0)),
            "maxlvl": out["maxlvl"].astype(float),
            "progress": out["final_progression_pct"].astype(float),
            "final_xp": out["final_xp"].astype(float),
            "quit_flag": out["quit_flag"].astype(float),
        }
    )
    scaler = StandardScaler().fit(feature_frame)
    X = scaler.transform(feature_frame)
    gmm = GaussianMixture(
        n_components=4,
        random_state=RNG_SEED,
        covariance_type="full",
        reg_covar=1e-6,
    ).fit(X)
    cluster_ids = gmm.predict(X)
    probs = gmm.predict_proba(X)
    out["cluster_id"] = cluster_ids

    med_points = out.groupby("cluster_id")["points"].median().sort_values()
    ordered = med_points.index.tolist()
    cluster_name_map = {
        ordered[0]: "instant_quit",
        ordered[1]: "instant_die",
        ordered[2]: "substantive",
        ordered[3]: "deep",
    }
    out["cluster_name"] = out["cluster_id"].map(cluster_name_map)

    engaged = np.zeros(len(out), dtype=np.float64)
    for cluster_id, cluster_name in cluster_name_map.items():
        out[f"prob_{cluster_name}"] = probs[:, cluster_id]
        if cluster_name in {"substantive", "deep"}:
            engaged += probs[:, cluster_id]
    out["engagement_prob"] = engaged
    out["trying_strict"] = out["engagement_prob"] >= 0.5
    return out, gmm, scaler, cluster_name_map


def add_gemini_effort_probs(
    gemini: pd.DataFrame,
    gmm: GaussianMixture,
    scaler: StandardScaler,
    cluster_name_map: dict[int, str],
) -> pd.DataFrame:
    out = gemini.copy()
    features = pd.DataFrame(
        {
            "log_turns": np.log1p(out["turns"].clip(lower=0)),
            "log_points": np.log1p(out["points"].clip(lower=0)),
            "maxlvl": out["maxlvl"].astype(float),
            "progress": out["final_progression_pct"].astype(float),
            "final_xp": out["final_xp"].astype(float),
            "quit_flag": np.zeros(len(out), dtype=np.float64),
        }
    )
    probs = gmm.predict_proba(scaler.transform(features))
    engaged = np.zeros(len(out), dtype=np.float64)
    for cluster_id, cluster_name in cluster_name_map.items():
        out[f"prob_{cluster_name}"] = probs[:, cluster_id]
        if cluster_name in {"substantive", "deep"}:
            engaged += probs[:, cluster_id]
    out["engagement_prob"] = engaged
    return out


def log_series(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.log10(1.0 + np.asarray(values, dtype=np.float64))


def apply_log_ticks(ax, values: np.ndarray) -> None:
    max_raw = float(np.nanmax(values)) if len(values) else 1.0
    candidate_ticks = np.array([0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000, 10000000, 1000000000], dtype=np.float64)
    candidate_ticks = candidate_ticks[candidate_ticks <= max_raw * 1.05]
    if len(candidate_ticks) <= 1:
        return
    tick_positions = np.log10(1.0 + candidate_ticks)
    tick_labels = [f"{int(t):,}" for t in candidate_ticks]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right")


def stacked_cluster_hist(
    ax,
    df: pd.DataFrame,
    value_col: str,
    *,
    bins,
    title: str,
    xlabel: str,
    log_transform: bool = False,
    gemini_values: np.ndarray | None = None,
    gemini_labels: list[str] | None = None,
) -> None:
    order = ["instant_quit", "instant_die", "substantive", "deep"]
    data = []
    for name in order:
        vals = df.loc[df["cluster_name"] == name, value_col].astype(float).to_numpy(dtype=np.float64)
        data.append(log_series(vals) if log_transform else vals)
    ax.hist(
        data,
        bins=bins,
        stacked=True,
        color=[CLUSTER_COLORS[name] for name in order],
        edgecolor="white",
        linewidth=0.35,
        label=order,
    )
    if gemini_values is not None and gemini_labels is not None:
        for color, value, label in zip(GEMINI_COLORS, gemini_values, gemini_labels):
            if not np.isfinite(value):
                continue
            line_value = np.log10(1.0 + float(value)) if log_transform else float(value)
            ax.axvline(line_value, color=color, linewidth=2.1, alpha=0.95)
    ax.set_title(title, fontsize=12.5)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="y", alpha=0.14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if log_transform:
        raw_vals = df[value_col].astype(float).to_numpy(dtype=np.float64)
        if gemini_values is not None:
            raw_vals = np.concatenate([raw_vals, np.asarray(gemini_values, dtype=np.float64)])
        apply_log_ticks(ax, raw_vals)


def overlay_hist(
    ax,
    all_values: np.ndarray,
    engaged_values: np.ndarray,
    *,
    bins,
    title: str,
    xlabel: str,
    log_transform: bool = False,
    gemini_values: np.ndarray | None = None,
    gemini_labels: list[str] | None = None,
) -> None:
    all_plot = log_series(all_values) if log_transform else np.asarray(all_values, dtype=np.float64)
    engaged_plot = log_series(engaged_values) if log_transform else np.asarray(engaged_values, dtype=np.float64)
    ax.hist(all_plot, bins=bins, color="#cfd8dc", alpha=0.7, edgecolor="white", linewidth=0.4, label="All humans")
    ax.hist(engaged_plot, bins=bins, histtype="step", color="#1f4f78", linewidth=2.2, label="Trying humans")
    if gemini_values is not None and gemini_labels is not None:
        for color, value, label in zip(GEMINI_COLORS, gemini_values, gemini_labels):
            if not np.isfinite(value):
                continue
            line_value = np.log10(1.0 + float(value)) if log_transform else float(value)
            ax.axvline(line_value, color=color, linewidth=2.1, alpha=0.95)
    ax.set_title(title, fontsize=12.5)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="y", alpha=0.14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if log_transform:
        raw_vals = np.concatenate([np.asarray(all_values, dtype=np.float64), np.asarray(engaged_values, dtype=np.float64)])
        if gemini_values is not None:
            raw_vals = np.concatenate([raw_vals, np.asarray(gemini_values, dtype=np.float64)])
        apply_log_ticks(ax, raw_vals)


def outcome_panel(ax, games: pd.DataFrame) -> None:
    counts = games["outcome_family"].value_counts()
    order = ["quit", "killed", "escaped", "ascended", "petrified", "poisoned", "choked", "starved", "drowned", "other"]
    counts = counts.reindex([name for name in order if name in counts.index]).fillna(0).astype(int)
    ax.bar(counts.index, counts.values, color="#6c8aa6", edgecolor="white", linewidth=0.5)
    ax.set_title("Outcome family counts", fontsize=12.5)
    ax.set_xlabel("Outcome family")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def percentile_of_value(series: pd.Series, value: float) -> float:
    arr = np.sort(series.astype(float).to_numpy(dtype=np.float64))
    return float(np.searchsorted(arr, float(value), side="right") / len(arr) * 100.0)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    games = load_games()
    games["outcome_family"] = games["death"].map(outcome_family)
    games, gmm, scaler, cluster_name_map = fit_effort_gmm(games)
    gemini = add_gemini_effort_probs(parse_gemini_final_runs(), gmm, scaler, cluster_name_map)

    cluster_summary = (
        games.groupby("cluster_name", sort=False)
        .agg(
            n=("game_key", "size"),
            share=("game_key", lambda s: len(s) / len(games)),
            quit_rate=("outcome_family", lambda s: (s == "quit").mean()),
            turns_median=("turns", "median"),
            points_median=("points", "median"),
            maxlvl_median=("maxlvl", "median"),
            progression_median=("final_progression_pct", "median"),
            xp_median=("final_xp", "median"),
            score_median=("final_score", "median"),
            event_count_median=("event_count", "median"),
        )
        .reset_index()
    )
    cluster_summary.to_csv(CLUSTER_SUMMARY_CSV, index=False)
    gemini.to_csv(GEMINI_CSV, index=False)

    trying = games[games["trying_strict"]].copy()

    percentile_rows = []
    compare_cols = [
        "turns",
        "points",
        "maxlvl",
        "final_progression_pct",
        "final_xp",
        "final_score",
        "engagement_prob",
    ]
    for row in gemini.itertuples(index=False):
        out = {"run_name": str(row.run_name)}
        for col in compare_cols:
            value = float(getattr(row, col))
            out[f"{col}_pct_all_humans"] = percentile_of_value(games[col], value)
            out[f"{col}_pct_trying_humans"] = percentile_of_value(trying[col], value)
        percentile_rows.append(out)
    percentiles = pd.DataFrame(percentile_rows)
    percentiles.to_csv(PERCENTILES_CSV, index=False)

    gem_labels = gemini["run_name"].astype(str).tolist()

    fig, axes = plt.subplots(2, 4, figsize=(22, 11.5))
    fig.patch.set_facecolor("#f4efe8")
    for ax in axes.flat:
        ax.set_facecolor("#fffaf3")

    stacked_cluster_hist(
        axes[0, 0],
        games,
        "turns",
        bins=32,
        title="Turns taken",
        xlabel="log10(1 + turns)",
        log_transform=True,
        gemini_values=gemini["turns"].to_numpy(dtype=np.float64),
        gemini_labels=gem_labels,
    )
    stacked_cluster_hist(
        axes[0, 1],
        games,
        "points",
        bins=32,
        title="Final xlog points",
        xlabel="log10(1 + points)",
        log_transform=True,
        gemini_values=gemini["points"].to_numpy(dtype=np.float64),
        gemini_labels=gem_labels,
    )
    stacked_cluster_hist(
        axes[0, 2],
        games,
        "maxlvl",
        bins=np.arange(0.5, games["maxlvl"].max() + 1.5, 1.0),
        title="Max dungeon level reached",
        xlabel="maxlvl",
        gemini_values=gemini["maxlvl"].to_numpy(dtype=np.float64),
        gemini_labels=gem_labels,
    )
    stacked_cluster_hist(
        axes[0, 3],
        games,
        "final_progression_pct",
        bins=np.linspace(0, 100, 41),
        title="Final BALROG progression",
        xlabel="progression (%)",
        gemini_values=gemini["final_progression_pct"].to_numpy(dtype=np.float64),
        gemini_labels=gem_labels,
    )
    stacked_cluster_hist(
        axes[1, 0],
        games,
        "final_xp",
        bins=np.arange(-0.5, games["final_xp"].max() + 1.5, 1.0),
        title="Final XP level",
        xlabel="XP level",
        gemini_values=gemini["final_xp"].to_numpy(dtype=np.float64),
        gemini_labels=gem_labels,
    )
    stacked_cluster_hist(
        axes[1, 1],
        games,
        "final_score",
        bins=32,
        title="Parsed final score",
        xlabel="log10(1 + final score)",
        log_transform=True,
        gemini_values=gemini["final_score"].to_numpy(dtype=np.float64),
        gemini_labels=gem_labels,
    )
    stacked_cluster_hist(
        axes[1, 2],
        games,
        "event_count",
        bins=32,
        title="Sparse event count",
        xlabel="log10(1 + event count)",
        log_transform=True,
        gemini_values=np.full(len(gemini), np.nan),
        gemini_labels=gem_labels,
    )
    outcome_panel(axes[1, 3], games)

    handles = [
        plt.Line2D([0], [0], color=CLUSTER_COLORS[name], linewidth=8)
        for name in ["instant_quit", "instant_die", "substantive", "deep"]
    ]
    labels = ["instant_quit", "instant_die", "substantive", "deep"]
    handles.extend(
        [plt.Line2D([0], [0], color=color, linewidth=2.1) for color in GEMINI_COLORS]
    )
    labels.extend(gem_labels)
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True, framealpha=0.95)
    fig.suptitle("Human NetHack dataset: final-state distributions", fontsize=22, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(BASIC_PNG, dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("#f4efe8")
    for ax in axes.flat:
        ax.set_facecolor("#fffaf3")

    overlay_hist(
        axes[0, 0],
        games["turns"].to_numpy(dtype=np.float64),
        trying["turns"].to_numpy(dtype=np.float64),
        bins=32,
        title="Turns taken",
        xlabel="log10(1 + turns)",
        log_transform=True,
        gemini_values=gemini["turns"].to_numpy(dtype=np.float64),
        gemini_labels=gem_labels,
    )
    overlay_hist(
        axes[0, 1],
        games["points"].to_numpy(dtype=np.float64),
        trying["points"].to_numpy(dtype=np.float64),
        bins=32,
        title="Final xlog points",
        xlabel="log10(1 + points)",
        log_transform=True,
        gemini_values=gemini["points"].to_numpy(dtype=np.float64),
        gemini_labels=gem_labels,
    )
    overlay_hist(
        axes[1, 0],
        games["maxlvl"].to_numpy(dtype=np.float64),
        trying["maxlvl"].to_numpy(dtype=np.float64),
        bins=np.arange(0.5, games["maxlvl"].max() + 1.5, 1.0),
        title="Max dungeon level reached",
        xlabel="maxlvl",
        gemini_values=gemini["maxlvl"].to_numpy(dtype=np.float64),
        gemini_labels=gem_labels,
    )
    overlay_hist(
        axes[1, 1],
        games["final_progression_pct"].to_numpy(dtype=np.float64),
        trying["final_progression_pct"].to_numpy(dtype=np.float64),
        bins=np.linspace(0, 100, 41),
        title="Final BALROG progression",
        xlabel="progression (%)",
        gemini_values=gemini["final_progression_pct"].to_numpy(dtype=np.float64),
        gemini_labels=gem_labels,
    )
    handles = [
        plt.Line2D([0], [0], color="#cfd8dc", linewidth=8),
        plt.Line2D([0], [0], color="#1f4f78", linewidth=2.2),
    ] + [plt.Line2D([0], [0], color=color, linewidth=2.1) for color in GEMINI_COLORS]
    labels = ["All humans", "Trying humans"] + gem_labels
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, framealpha=0.95)
    fig.suptitle("Trying humans vs Gemini", fontsize=20, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(COMPARE_PNG, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    fig.patch.set_facecolor("#f4efe8")
    ax.set_facecolor("#fffaf3")
    ax.hist(
        games["engagement_prob"].to_numpy(dtype=np.float64),
        bins=np.linspace(0, 1, 41),
        color="#6c8aa6",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.45,
    )
    ax.axvline(0.5, color="#b23a48", linestyle="--", linewidth=2.0, label="Trying cutoff")
    for color, value, label in zip(GEMINI_COLORS, gemini["engagement_prob"], gem_labels):
        if not np.isfinite(value):
            continue
        ax.axvline(float(value), color=color, linewidth=2.0, alpha=0.95, label=label)
    ax.set_title("Human engagement probability from final-run mixture model", fontsize=18, pad=10)
    ax.set_xlabel("Probability(run belongs to substantive/deep human clusters)")
    ax.set_ylabel("Human run count")
    ax.grid(True, axis="y", alpha=0.14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(ENGAGEMENT_PNG, dpi=220)
    plt.close(fig)

    summary = {
        "games_total_after_progression_sanity_filter": int(len(games)),
        "turns_quantiles": games["turns"].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_dict(),
        "points_quantiles": games["points"].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_dict(),
        "progression_quantiles": games["final_progression_pct"].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_dict(),
        "quit_rate": float((games["outcome_family"] == "quit").mean()),
        "escaped_rate": float((games["outcome_family"] == "escaped").mean()),
        "ascended_rate": float((games["outcome_family"] == "ascended").mean()),
        "trying_strict_count": int(games["trying_strict"].sum()),
        "trying_strict_share": float(games["trying_strict"].mean()),
        "cluster_summary_csv": str(CLUSTER_SUMMARY_CSV),
        "gemini_csv": str(GEMINI_CSV),
        "percentiles_csv": str(PERCENTILES_CSV),
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        json.dumps(
            {
                "basic_png": str(BASIC_PNG),
                "compare_png": str(COMPARE_PNG),
                "engagement_png": str(ENGAGEMENT_PNG),
                "cluster_summary_csv": str(CLUSTER_SUMMARY_CSV),
                "gemini_csv": str(GEMINI_CSV),
                "percentiles_csv": str(PERCENTILES_CSV),
                "summary_json": str(SUMMARY_JSON),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
