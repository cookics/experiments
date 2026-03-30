from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.plot_gemini3_vs_humans import update_metrics_from_text
from analysis.plot_human_nao_trajectories import load_achievements
from analysis.plot_nle_trajectories import find_nle_csvs


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "analysis_outputs"
DB_PATH = OUTPUT_DIR / "human_nao_full_trajectories.sqlite3"
GEMINI_FOLDER = ROOT / "submissions" / "LLM" / "20260203_naive_gemini-3-pro"


@dataclass
class SparseCurve:
    run_name: str
    turns: np.ndarray
    values: np.ndarray
    last_turn: int


class FenwickTree:
    def __init__(self, size: int) -> None:
        self.size = size
        self.tree = np.zeros(size + 1, dtype=np.int32)

    def add(self, index_1based: int, delta: int) -> None:
        i = index_1based
        while i <= self.size:
            self.tree[i] += delta
            i += i & -i

    def kth(self, k: int) -> int:
        idx = 0
        bit = 1 << (self.size.bit_length() - 1)
        while bit:
            nxt = idx + bit
            if nxt <= self.size and self.tree[nxt] < k:
                idx = nxt
                k -= int(self.tree[nxt])
            bit >>= 1
        return idx + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    return parser.parse_args()


def output_paths() -> dict[tuple[str, int], Path]:
    return {
        ("progression", 1500): OUTPUT_DIR / "human_best_median_vs_gemini_progression_1500.png",
        ("progression", 5000): OUTPUT_DIR / "human_best_median_vs_gemini_progression_5000.png",
        ("progression", -1): OUTPUT_DIR / "human_best_median_vs_gemini_progression_full.png",
        ("score", 1500): OUTPUT_DIR / "human_best_median_vs_gemini_score_1500.png",
        ("score", 5000): OUTPUT_DIR / "human_best_median_vs_gemini_score_5000.png",
        ("score", -1): OUTPUT_DIR / "human_best_median_vs_gemini_score_full.png",
    }


def csv_paths() -> dict[str, Path]:
    return {
        "progression": OUTPUT_DIR / "human_best_median_progression_full.csv",
        "score": OUTPUT_DIR / "human_best_median_score_full.csv",
    }


def load_human_store(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(str(db_path))
    try:
        games = pd.read_sql_query(
            """
            SELECT game_key, turns
            FROM games
            ORDER BY game_key
            """,
            conn,
        )
        events = pd.read_sql_query(
            """
            SELECT
                game_key,
                turn,
                score,
                progression_pct,
                dlvl,
                xp,
                home_level,
                reached_astral,
                ascended
            FROM events
            ORDER BY turn, event_idx
            """,
            conn,
        )
    finally:
        conn.close()
    return games, events


def suspicious_astral_games(events: pd.DataFrame) -> set[str]:
    astral = events[
        (events["reached_astral"] == 1)
        & (events["ascended"] == 0)
        & (events["turn"] <= 5000)
    ]
    if astral.empty:
        return set()

    grouped = astral.groupby("game_key", as_index=False).agg(
        first_astral=("turn", "min"),
        max_dlvl=("dlvl", "max"),
        max_xp=("xp", "max"),
        max_home=("home_level", "max"),
    )
    suspicious = grouped[
        (grouped["first_astral"] <= 1000)
        & (grouped["max_home"] == 0)
        & (grouped["max_dlvl"] <= 10)
        & (grouped["max_xp"] <= 10)
    ]
    return set(suspicious["game_key"].astype(str).tolist())


def sanitized_progression_values(events: pd.DataFrame, achievements: dict[str, float]) -> np.ndarray:
    dlvl_map = {level: achievements[f"Dlvl:{level}"] * 100.0 for level in range(1, 100) if f"Dlvl:{level}" in achievements}
    xp_map = {level: achievements[f"Xp:{level}"] * 100.0 for level in range(1, 100) if f"Xp:{level}" in achievements}
    home_map = {level: achievements[f"Home {level}"] * 100.0 for level in range(1, 6) if f"Home {level}" in achievements}
    astral_pct = achievements["Astral Plane"] * 100.0
    ascend_pct = achievements["You ascend t"] * 100.0

    suspicious_games = suspicious_astral_games(events)
    suspicious_mask = events["game_key"].astype(str).isin(suspicious_games).to_numpy()

    dlvl_vals = events["dlvl"].map(dlvl_map).fillna(0.0).to_numpy(dtype=np.float64)
    xp_vals = events["xp"].map(xp_map).fillna(0.0).to_numpy(dtype=np.float64)
    home_vals = events["home_level"].map(home_map).fillna(0.0).to_numpy(dtype=np.float64)
    astral_vals = np.where(
        (events["reached_astral"].to_numpy(dtype=np.int8) == 1) & (~suspicious_mask),
        astral_pct,
        0.0,
    )
    ascend_vals = np.where(events["ascended"].to_numpy(dtype=np.int8) == 1, ascend_pct, 0.0)
    return np.maximum.reduce([dlvl_vals, xp_vals, home_vals, astral_vals, ascend_vals])


def compute_human_aggregate_curve(
    games: pd.DataFrame,
    events: pd.DataFrame,
    metric_col: str,
    achievements: dict[str, float] | None = None,
) -> pd.DataFrame:
    game_keys = games["game_key"].astype(str).to_numpy()
    game_turns = games["turns"].to_numpy(dtype=np.int32)
    n_games = len(games)

    game_to_idx = {key: idx for idx, key in enumerate(game_keys)}

    if metric_col == "progression_pct":
        if achievements is None:
            raise ValueError("achievements are required for sanitized progression aggregation")
        event_metric_values = sanitized_progression_values(events, achievements)
    else:
        event_metric_values = events[metric_col].to_numpy(dtype=np.float64)

    metric_values = np.concatenate([np.array([0.0], dtype=np.float64), event_metric_values])
    unique_values = np.unique(metric_values)

    event_turns = events["turn"].to_numpy(dtype=np.int32)
    event_game_idx = events["game_key"].astype(str).map(game_to_idx).to_numpy(dtype=np.int32)
    event_value_idx = np.searchsorted(unique_values, event_metric_values).astype(np.int32)

    death_turns = game_turns + 1
    death_order = np.argsort(death_turns, kind="stable")
    death_turns = death_turns[death_order]
    death_game_idx = death_order.astype(np.int32)

    fenwick = FenwickTree(len(unique_values))
    zero_idx_1based = 1
    fenwick.add(zero_idx_1based, n_games)

    current_value_idx = np.zeros(n_games, dtype=np.int32)
    active = np.ones(n_games, dtype=bool)
    active_count = n_games

    turns_out = [0]
    active_out = [active_count]
    median_out = [0.0]
    best_out = [0.0]

    event_ptr = 0
    death_ptr = 0
    total_events = len(event_turns)
    total_deaths = len(death_turns)

    progress = tqdm(total=total_events + total_deaths, desc=f"Sweeping {metric_col}", unit="update")
    while event_ptr < total_events or death_ptr < total_deaths:
        next_event_turn = int(event_turns[event_ptr]) if event_ptr < total_events else None
        next_death_turn = int(death_turns[death_ptr]) if death_ptr < total_deaths else None

        if next_event_turn is None:
            current_turn = next_death_turn
        elif next_death_turn is None:
            current_turn = next_event_turn
        else:
            current_turn = min(next_event_turn, next_death_turn)

        while death_ptr < total_deaths and int(death_turns[death_ptr]) == current_turn:
            game_idx = int(death_game_idx[death_ptr])
            if active[game_idx]:
                fenwick.add(int(current_value_idx[game_idx]) + 1, -1)
                active[game_idx] = False
                active_count -= 1
            death_ptr += 1
            progress.update(1)

        while event_ptr < total_events and int(event_turns[event_ptr]) == current_turn:
            game_idx = int(event_game_idx[event_ptr])
            if active[game_idx]:
                new_idx = int(event_value_idx[event_ptr])
                old_idx = int(current_value_idx[game_idx])
                if new_idx != old_idx:
                    fenwick.add(old_idx + 1, -1)
                    fenwick.add(new_idx + 1, 1)
                    current_value_idx[game_idx] = new_idx
            event_ptr += 1
            progress.update(1)

        if active_count > 0:
            median_rank = (active_count + 1) // 2
            median_idx = fenwick.kth(median_rank) - 1
            best_idx = fenwick.kth(active_count) - 1
            median_val = float(unique_values[median_idx])
            best_val = float(unique_values[best_idx])
        else:
            median_val = np.nan
            best_val = np.nan

        turns_out.append(int(current_turn))
        active_out.append(int(active_count))
        median_out.append(median_val)
        best_out.append(best_val)

    progress.close()

    return pd.DataFrame(
        {
            "turn": np.array(turns_out, dtype=np.int32),
            "active_runs": np.array(active_out, dtype=np.int32),
            "median": np.array(median_out, dtype=np.float64),
            "best": np.array(best_out, dtype=np.float64),
        }
    )


def parse_gemini_sparse_curve(csv_path: Path, achievements: dict[str, float]) -> tuple[SparseCurve, SparseCurve]:
    df = pd.read_csv(csv_path, usecols=["Observation"])

    progression = 0.0
    score = 0.0
    last_turn = -1

    score_turns = [0]
    score_values = [0.0]
    progression_turns = [0]
    progression_values = [0.0]

    for observation in df["Observation"].fillna(""):
        progression, score, turn = update_metrics_from_text(str(observation), progression, score, achievements)
        if turn is None:
            continue
        if turn < last_turn:
            continue

        prog_pct = progression * 100.0
        if score != score_values[-1]:
            score_turns.append(int(turn))
            score_values.append(float(score))
        if prog_pct != progression_values[-1]:
            progression_turns.append(int(turn))
            progression_values.append(float(prog_pct))
        last_turn = int(turn)

    if last_turn < 0:
        last_turn = 0

    return (
        SparseCurve(
            run_name=csv_path.stem,
            turns=np.array(score_turns, dtype=np.int32),
            values=np.array(score_values, dtype=np.float64),
            last_turn=last_turn,
        ),
        SparseCurve(
            run_name=csv_path.stem,
            turns=np.array(progression_turns, dtype=np.int32),
            values=np.array(progression_values, dtype=np.float64),
            last_turn=last_turn,
        ),
    )


def load_gemini_sparse_curves() -> tuple[list[SparseCurve], list[SparseCurve]]:
    achievements = load_achievements()
    score_curves: list[SparseCurve] = []
    progression_curves: list[SparseCurve] = []
    for csv_path in find_nle_csvs(GEMINI_FOLDER):
        score_curve, progression_curve = parse_gemini_sparse_curve(csv_path, achievements)
        score_curves.append(score_curve)
        progression_curves.append(progression_curve)
    score_curves.sort(key=lambda item: item.run_name)
    progression_curves.sort(key=lambda item: item.run_name)
    return score_curves, progression_curves


def crop_step_curve(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon < 0:
        return df.copy()
    clipped = df[df["turn"] <= horizon].copy()
    if clipped.empty:
        return pd.DataFrame({"turn": [0, horizon], "active_runs": [df["active_runs"].iloc[0], np.nan], "median": [df["median"].iloc[0], df["median"].iloc[0]], "best": [df["best"].iloc[0], df["best"].iloc[0]]})
    if int(clipped["turn"].iloc[-1]) < horizon:
        last = clipped.iloc[-1].copy()
        last["turn"] = horizon
        clipped = pd.concat([clipped, pd.DataFrame([last])], ignore_index=True)
    return clipped


def crop_sparse_curve(curve: SparseCurve, horizon: int) -> SparseCurve:
    if horizon < 0 or curve.last_turn <= horizon:
        return curve
    mask = curve.turns <= horizon
    turns = curve.turns[mask]
    values = curve.values[mask]
    if len(turns) == 0:
        turns = np.array([0], dtype=np.int32)
        values = np.array([0.0], dtype=np.float64)
    return SparseCurve(run_name=curve.run_name, turns=turns, values=values, last_turn=horizon)


def plot_metric(
    human_df: pd.DataFrame,
    gemini_curves: list[SparseCurve],
    output_path: Path,
    title: str,
    y_label: str,
    y_max: float,
    use_log_x: bool,
) -> None:
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#f4efe8")
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.1, 5.4], hspace=0.08)
    ax_top = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0], sharex=ax_top)

    ax_top.set_facecolor("#fffaf3")
    ax.set_facecolor("#fffaf3")

    x = human_df["turn"].to_numpy(dtype=np.int32)
    active_runs = human_df["active_runs"].to_numpy(dtype=np.int32)
    median = human_df["median"].to_numpy(dtype=np.float64)
    best = human_df["best"].to_numpy(dtype=np.float64)

    ax_top.plot(x, active_runs, color="#284b63", linewidth=2.2)
    ax_top.fill_between(x, active_runs, color="#284b63", alpha=0.18)
    ax_top.set_ylabel("Active humans", fontsize=12)
    ax_top.grid(True, alpha=0.16, linewidth=0.8, color="#8c7b6b")
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_title(title, fontsize=20, pad=12)

    ax.plot(x, best, color="#b23a48", linewidth=3.0, drawstyle="steps-post", label="Human pointwise best")
    ax.plot(x, median, color="#1d3557", linewidth=3.0, drawstyle="steps-post", label="Human median")

    palette = ["#7b2cbf", "#9d4edd", "#c77dff", "#f72585", "#4361ee"]
    for color, curve in zip(palette, gemini_curves):
        x_vals = curve.turns
        y_vals = curve.values
        if len(x_vals) == 1 and curve.last_turn > x_vals[0]:
            x_vals = np.array([int(x_vals[0]), int(curve.last_turn)], dtype=np.int32)
            y_vals = np.array([float(y_vals[0]), float(y_vals[0])], dtype=np.float64)
        elif len(x_vals) > 0 and int(x_vals[-1]) < curve.last_turn:
            x_vals = np.append(x_vals, int(curve.last_turn))
            y_vals = np.append(y_vals, float(y_vals[-1]))
        ax.plot(x_vals, y_vals, color=color, linewidth=2.0, alpha=0.95, drawstyle="steps-post", label=f"Gemini {curve.run_name}")

    if use_log_x:
        ax_top.set_xscale("log")
        ax.set_xscale("log")
        ax.set_xlim(1, int(max(np.nanmax(x), max((curve.last_turn for curve in gemini_curves), default=1))))
        ax.set_xlabel("NetHack Turn `T` (log scale)", fontsize=14)
    else:
        ax.set_xlim(0, int(max(np.nanmax(x), max((curve.last_turn for curve in gemini_curves), default=0))))
        ax.set_xlabel("NetHack Turn `T`", fontsize=14)

    ax.set_ylim(0, y_max)
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid(True, alpha=0.12, linewidth=0.8, color="#8c7b6b")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.95, facecolor="#fff7ec", edgecolor="#d7c5ad")
    ax.text(
        0.012,
        0.985,
        "Human best is the pointwise maximum over active human runs at each turn\nHuman median is also computed over active runs only; identity of the best run can switch over time",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        color="#5d554c",
        bbox=dict(facecolor="#fff7ec", edgecolor="#d7c5ad", boxstyle="round,pad=0.4", alpha=0.95),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.db_path = args.db_path.resolve()
    OUTPUT_DIR.mkdir(exist_ok=True)

    games, events = load_human_store(args.db_path)
    achievements = load_achievements()
    progress_df = compute_human_aggregate_curve(
        games=games,
        events=events,
        metric_col="progression_pct",
        achievements=achievements,
    )
    score_df = compute_human_aggregate_curve(games=games, events=events, metric_col="score")

    csv_out = csv_paths()
    progress_df.to_csv(csv_out["progression"], index=False)
    score_df.to_csv(csv_out["score"], index=False)

    gemini_score_curves, gemini_progress_curves = load_gemini_sparse_curves()

    horizons = [1500, 5000, -1]
    out_paths = output_paths()
    for horizon in horizons:
        human_prog = crop_step_curve(progress_df, horizon)
        human_score = crop_step_curve(score_df, horizon)
        gem_prog = [crop_sparse_curve(curve, horizon) for curve in gemini_progress_curves]
        gem_score = [crop_sparse_curve(curve, horizon) for curve in gemini_score_curves]

        progress_y_max = max(
            10.0,
            float(np.nanmax(human_prog["best"].to_numpy(dtype=np.float64))) * 1.05,
            max((float(np.nanmax(curve.values)) for curve in gem_prog), default=0.0) * 1.05,
        )
        score_y_max = max(
            50.0,
            float(np.nanmax(human_score["best"].to_numpy(dtype=np.float64))) * 1.05,
            max((float(np.nanmax(curve.values)) for curve in gem_score), default=0.0) * 1.05,
        )

        label = "full run" if horizon < 0 else f"0-{horizon}"
        use_log_x = horizon < 0

        plot_metric(
            human_df=human_prog,
            gemini_curves=gem_prog,
            output_path=out_paths[("progression", horizon)],
            title=f"Human Best and Median vs Gemini-3-Pro: Progression, {label}",
            y_label="BALROG NLE Progression (%)",
            y_max=progress_y_max,
            use_log_x=use_log_x,
        )
        plot_metric(
            human_df=human_score,
            gemini_curves=gem_score,
            output_path=out_paths[("score", horizon)],
            title=f"Human Best and Median vs Gemini-3-Pro: Score, {label}",
            y_label="NetHack score `S`",
            y_max=score_y_max,
            use_log_x=use_log_x,
        )

    print(
        {
            "games": len(games),
            "events": len(events),
            "progress_csv": str(csv_out["progression"]),
            "score_csv": str(csv_out["score"]),
            "plots": {str(key): str(path) for key, path in out_paths.items()},
        }
    )


if __name__ == "__main__":
    main()

