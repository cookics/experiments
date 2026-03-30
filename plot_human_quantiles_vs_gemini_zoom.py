from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from plot_gemini3_vs_humans import load_gemini_records
from plot_human_nao_trajectories import load_achievements


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs"
DB_PATH = OUTPUT_DIR / "human_nao_full_trajectories.sqlite3"
OUTPUT_PROGRESS_PNG = OUTPUT_DIR / "human_percentiles_vs_gemini_progression_1500.png"
OUTPUT_SCORE_PNG = OUTPUT_DIR / "human_percentiles_vs_gemini_score_1500.png"
OUTPUT_PROGRESS_CSV = OUTPUT_DIR / "human_percentiles_progression_1500.csv"
OUTPUT_SCORE_CSV = OUTPUT_DIR / "human_percentiles_score_1500.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--max-turn", type=int, default=1500)
    parser.add_argument("--progress-png", type=Path, default=OUTPUT_PROGRESS_PNG)
    parser.add_argument("--score-png", type=Path, default=OUTPUT_SCORE_PNG)
    parser.add_argument("--progress-csv", type=Path, default=OUTPUT_PROGRESS_CSV)
    parser.add_argument("--score-csv", type=Path, default=OUTPUT_SCORE_CSV)
    return parser.parse_args()


def load_human_zoom_store(db_path: Path, max_turn: int) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            SELECT game_key, event_idx, turn, score, progression_pct
            FROM events
            WHERE turn <= ?
            ORDER BY game_key, event_idx
            """,
            conn,
            params=(int(max_turn),),
        )
    finally:
        conn.close()
    return games, events


def build_event_index(events: pd.DataFrame) -> dict[str, tuple[int, int]]:
    game_keys = events["game_key"].to_numpy()
    if len(game_keys) == 0:
        return {}
    unique_keys, start_idx, counts = np.unique(game_keys, return_index=True, return_counts=True)
    return {str(key): (int(start), int(count)) for key, start, count in zip(unique_keys, start_idx, counts)}


def build_human_quantiles(
    games: pd.DataFrame,
    events: pd.DataFrame,
    metric_col: str,
    max_turn: int,
) -> pd.DataFrame:
    checkpoints = np.arange(max_turn + 1, dtype=np.int32)
    values = np.full((len(games), len(checkpoints)), np.nan, dtype=np.float32)

    event_index = build_event_index(events)
    turn_arr = events["turn"].to_numpy(dtype=np.int32)
    metric_arr = events[metric_col].to_numpy(dtype=np.float32)

    game_keys = games["game_key"].astype(str).to_numpy()
    game_turns = games["turns"].to_numpy(dtype=np.int32)

    for row_idx, (game_key, final_turn) in enumerate(
        tqdm(zip(game_keys, game_turns), total=len(games), desc=f"Human quantiles {metric_col}", unit="game")
    ):
        active_len = min(max(int(final_turn), 0), max_turn) + 1
        if active_len <= 0:
            continue

        info = event_index.get(game_key)
        if info is None:
            values[row_idx, :active_len] = 0.0
            continue

        start, count = info
        game_event_turns = turn_arr[start : start + count]
        game_event_values = metric_arr[start : start + count]
        active_turns = checkpoints[:active_len]
        positions = np.searchsorted(game_event_turns, active_turns, side="right") - 1
        clamped = np.clip(positions, 0, len(game_event_values) - 1)
        row_values = np.where(positions >= 0, game_event_values[clamped], 0.0)
        values[row_idx, :active_len] = row_values

    active_counts = np.sum(~np.isnan(values), axis=0).astype(np.int32)
    quantile_levels = {
        "q10": 0.10,
        "q25": 0.25,
        "q50": 0.50,
        "q75": 0.75,
        "q90": 0.90,
    }
    out: dict[str, np.ndarray] = {
        "turn": checkpoints,
        "active_runs": active_counts,
    }
    for label, q in quantile_levels.items():
        out[label] = np.nanquantile(values, q, axis=0).astype(np.float32)

    return pd.DataFrame(out)


def gemini_curves_df(gemini_records: list[dict], metric_key: str, max_turn: int) -> pd.DataFrame:
    rows = {"turn": np.arange(max_turn + 1, dtype=np.int32)}
    for record in sorted(gemini_records, key=lambda item: item["run_name"]):
        rows[record["run_name"]] = record[metric_key].reindex(range(max_turn + 1)).ffill().fillna(0.0).to_numpy()
    return pd.DataFrame(rows)


def plot_quantiles_vs_gemini(
    quantiles_df: pd.DataFrame,
    gemini_df: pd.DataFrame,
    output_path: Path,
    title: str,
    y_label: str,
    y_max: float,
    note: str,
) -> None:
    x = quantiles_df["turn"].to_numpy(dtype=np.int32)
    active_runs = quantiles_df["active_runs"].to_numpy(dtype=np.int32)
    q10 = quantiles_df["q10"].to_numpy(dtype=np.float32)
    q25 = quantiles_df["q25"].to_numpy(dtype=np.float32)
    q50 = quantiles_df["q50"].to_numpy(dtype=np.float32)
    q75 = quantiles_df["q75"].to_numpy(dtype=np.float32)
    q90 = quantiles_df["q90"].to_numpy(dtype=np.float32)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#f4efe8")
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.1, 5.4], hspace=0.08)
    ax_top = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0], sharex=ax_top)

    ax_top.set_facecolor("#fffaf3")
    ax.set_facecolor("#fffaf3")

    ax_top.plot(x, active_runs, color="#284b63", linewidth=2.2)
    ax_top.fill_between(x, active_runs, color="#284b63", alpha=0.18)
    ax_top.set_ylabel("Active humans", fontsize=12)
    ax_top.grid(True, alpha=0.16, linewidth=0.8, color="#8c7b6b")
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_title(title, fontsize=20, pad=12)

    ax.fill_between(x, q10, q90, color="#cdb4db", alpha=0.20, label="Human 10-90 pct")
    ax.fill_between(x, q25, q75, color="#95d5b2", alpha=0.30, label="Human 25-75 pct")
    ax.plot(x, q50, color="#1d3557", linewidth=3.0, label="Human median")

    gemini_palette = ["#7b2cbf", "#9d4edd", "#c77dff", "#f72585", "#4361ee"]
    for color, column in zip(gemini_palette, [c for c in gemini_df.columns if c != "turn"]):
        ax.plot(
            gemini_df["turn"],
            gemini_df[column],
            color=color,
            linewidth=2.0,
            alpha=0.95,
            label=f"Gemini {column.replace('.csv', '')}",
        )

    ax.set_xlim(0, int(x.max()))
    ax.set_ylim(0, y_max)
    ax.set_xlabel("NetHack Turn `T`", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid(True, alpha=0.12, linewidth=0.8, color="#8c7b6b")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.95, facecolor="#fff7ec", edgecolor="#d7c5ad")
    ax.text(
        0.012,
        0.985,
        note,
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
    args.progress_png = args.progress_png.resolve()
    args.score_png = args.score_png.resolve()
    args.progress_csv = args.progress_csv.resolve()
    args.score_csv = args.score_csv.resolve()
    OUTPUT_DIR.mkdir(exist_ok=True)

    human_games, human_events = load_human_zoom_store(args.db_path, args.max_turn)

    progress_quantiles = build_human_quantiles(
        games=human_games,
        events=human_events,
        metric_col="progression_pct",
        max_turn=args.max_turn,
    )
    score_quantiles = build_human_quantiles(
        games=human_games,
        events=human_events,
        metric_col="score",
        max_turn=args.max_turn,
    )

    progress_quantiles.to_csv(args.progress_csv, index=False)
    score_quantiles.to_csv(args.score_csv, index=False)

    achievements = load_achievements()
    gemini_records = load_gemini_records(max_turn=args.max_turn, achievements=achievements)
    gemini_progress_df = gemini_curves_df(gemini_records, metric_key="progression_curve", max_turn=args.max_turn)
    gemini_score_df = gemini_curves_df(gemini_records, metric_key="score_curve", max_turn=args.max_turn)

    progress_y_max = max(
        10.0,
        float(np.nanmax(progress_quantiles["q90"].to_numpy(dtype=np.float32))) * 1.08,
        float(gemini_progress_df.drop(columns=["turn"]).to_numpy(dtype=np.float32).max()) * 1.05,
    )
    score_y_max = max(
        50.0,
        float(np.nanmax(score_quantiles["q90"].to_numpy(dtype=np.float32))) * 1.08,
        float(gemini_score_df.drop(columns=["turn"]).to_numpy(dtype=np.float32).max()) * 1.05,
    )

    plot_quantiles_vs_gemini(
        quantiles_df=progress_quantiles,
        gemini_df=gemini_progress_df,
        output_path=args.progress_png,
        title=f"Humans vs Gemini-3-Pro Runs: Progression, Turns 0-{args.max_turn}",
        y_label="BALROG NLE Progression (%)",
        y_max=progress_y_max,
        note="Human curves shown as median with 25-75 and 10-90 percentile bands\nGemini overlay shows all five individual runs",
    )
    plot_quantiles_vs_gemini(
        quantiles_df=score_quantiles,
        gemini_df=gemini_score_df,
        output_path=args.score_png,
        title=f"Humans vs Gemini-3-Pro Runs: Score, Turns 0-{args.max_turn}",
        y_label="NetHack score `S`",
        y_max=score_y_max,
        note="Using shared status-line score `S` for both humans and Gemini\nThis is the comparable cross-dataset score, not Gemini's raw BALROG reward column",
    )

    print(
        {
            "games": len(human_games),
            "events_loaded": len(human_events),
            "progress_png": str(args.progress_png),
            "score_png": str(args.score_png),
            "progress_csv": str(args.progress_csv),
            "score_csv": str(args.score_csv),
        }
    )


if __name__ == "__main__":
    main()
