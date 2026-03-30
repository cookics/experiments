from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs"
DB_PATH = OUTPUT_DIR / "human_nao_full_trajectories.sqlite3"
OUTPUT_SPAGHETTI = OUTPUT_DIR / "human_nao_all_trajectories_spaghetti_logx.png"
OUTPUT_DENSITY = OUTPUT_DIR / "human_nao_trajectory_density_logx.png"
OUTPUT_QUANTILES = OUTPUT_DIR / "human_nao_trajectory_density_quantiles.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--spaghetti-png", type=Path, default=OUTPUT_SPAGHETTI)
    parser.add_argument("--density-png", type=Path, default=OUTPUT_DENSITY)
    parser.add_argument("--quantiles-csv", type=Path, default=OUTPUT_QUANTILES)
    parser.add_argument("--density-checkpoints", type=int, default=640)
    parser.add_argument("--density-y-bins", type=int, default=240)
    return parser.parse_args()


def load_store(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(str(db_path))
    try:
        games = pd.read_sql_query(
            """
            SELECT
                game_key,
                player_name,
                local_gameid,
                turns,
                death,
                final_progression_pct,
                ascended,
                event_count
            FROM games
            ORDER BY game_key
            """,
            conn,
        )
        events = pd.read_sql_query(
            """
            SELECT
                game_key,
                event_idx,
                turn,
                progression_pct
            FROM events
            ORDER BY game_key, event_idx
            """,
            conn,
        )
    finally:
        conn.close()
    return games, events


def build_event_index(events: pd.DataFrame) -> dict[str, tuple[int, int]]:
    game_keys = events["game_key"].to_numpy()
    unique_keys, start_idx, counts = np.unique(game_keys, return_index=True, return_counts=True)
    return {str(key): (int(start), int(count)) for key, start, count in zip(unique_keys, start_idx, counts)}


def build_spaghetti_segments(games: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
    turns_arr = events["turn"].to_numpy(dtype=np.int32)
    prog_arr = events["progression_pct"].to_numpy(dtype=np.float32)
    event_index = build_event_index(events)

    max_segments = (len(events) * 2) + len(games)
    segments = np.empty((max_segments, 2, 2), dtype=np.float32)
    seg_i = 0

    for row in tqdm(games.itertuples(index=False), total=len(games), desc="Building spaghetti segments", unit="game"):
        key = str(row.game_key)
        info = event_index.get(key)
        prev_turn = 0.0
        prev_prog = 0.0
        final_turn = float(max(int(row.turns), 0))

        if info is not None:
            start, count = info
            game_turns = turns_arr[start : start + count]
            game_progs = prog_arr[start : start + count]
            for turn, prog in zip(game_turns, game_progs):
                turn_f = float(turn)
                prog_f = float(prog)

                if turn_f > prev_turn:
                    segments[seg_i, 0] = (prev_turn + 1.0, prev_prog)
                    segments[seg_i, 1] = (turn_f + 1.0, prev_prog)
                    seg_i += 1

                if prog_f != prev_prog:
                    segments[seg_i, 0] = (turn_f + 1.0, prev_prog)
                    segments[seg_i, 1] = (turn_f + 1.0, prog_f)
                    seg_i += 1

                prev_turn = turn_f
                prev_prog = prog_f

        if final_turn > prev_turn:
            segments[seg_i, 0] = (prev_turn + 1.0, prev_prog)
            segments[seg_i, 1] = (final_turn + 1.0, prev_prog)
            seg_i += 1

    return segments[:seg_i]


def make_spaghetti_plot(games: pd.DataFrame, events: pd.DataFrame, output_path: Path) -> None:
    segments = build_spaghetti_segments(games, events)

    fig, ax = plt.subplots(figsize=(22, 13))
    fig.patch.set_facecolor("#f4efe8")
    ax.set_facecolor("#fffaf3")

    collection = LineCollection(
        segments,
        colors=(0.08, 0.27, 0.42, 0.015),
        linewidths=0.25,
        antialiased=False,
        rasterized=True,
    )
    ax.add_collection(collection)

    max_turn = float(games["turns"].max()) + 1.0
    ax.set_xscale("log")
    ax.set_xlim(1.0, max_turn)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("NetHack Turn `T` (log scale)", fontsize=14)
    ax.set_ylabel("BALROG NLE Progression (%)", fontsize=14)
    ax.set_title(f"All {len(games):,} Human NetHack Progress Trajectories", fontsize=20, pad=14)
    ax.grid(True, which="both", alpha=0.15, linewidth=0.8, color="#8c7b6b")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.012,
        0.985,
        "Sparse step curves reconstructed from cached event store\nEach line is one stitched human game",
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


def build_density_matrix(
    games: pd.DataFrame,
    events: pd.DataFrame,
    num_checkpoints: int,
    num_y_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    max_turn = int(games["turns"].max())
    log_turns = np.geomspace(1, max_turn + 1, num=num_checkpoints)
    checkpoints = np.unique(np.floor(log_turns - 1.0).astype(np.int32))
    checkpoints = np.concatenate(([0], checkpoints[checkpoints > 0]))

    y_edges = np.linspace(0.0, 100.0, num_y_bins + 1, dtype=np.float32)
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

    hist = np.zeros((num_y_bins, len(checkpoints)), dtype=np.int32)
    active_counts = np.zeros(len(checkpoints), dtype=np.int32)

    turns_arr = events["turn"].to_numpy(dtype=np.int32)
    prog_arr = events["progression_pct"].to_numpy(dtype=np.float32)
    event_index = build_event_index(events)

    for row in tqdm(games.itertuples(index=False), total=len(games), desc="Accumulating density", unit="game"):
        key = str(row.game_key)
        active_mask = checkpoints <= int(row.turns)
        active_idx = np.flatnonzero(active_mask)
        if active_idx.size == 0:
            continue

        info = event_index.get(key)
        if info is None:
            values = np.zeros(active_idx.size, dtype=np.float32)
        else:
            start, count = info
            game_turns = turns_arr[start : start + count]
            game_progs = prog_arr[start : start + count]
            positions = np.searchsorted(game_turns, checkpoints[active_idx], side="right") - 1
            values = np.where(positions >= 0, game_progs[np.clip(positions, 0, len(game_progs) - 1)], 0.0)

        bins = np.searchsorted(y_edges, values, side="right") - 1
        bins = np.clip(bins, 0, num_y_bins - 1)
        np.add.at(hist, (bins, active_idx), 1)
        active_counts[active_idx] += 1

    quantile_targets = {
        "q10": 0.10,
        "q25": 0.25,
        "q50": 0.50,
        "q75": 0.75,
        "q90": 0.90,
    }
    cumulative = hist.cumsum(axis=0)
    quantile_values: dict[str, np.ndarray] = {}
    for label, q in quantile_targets.items():
        arr = np.full(len(checkpoints), np.nan, dtype=np.float32)
        for idx, active in enumerate(active_counts):
            if active <= 0:
                continue
            target = q * active
            bin_idx = int(np.searchsorted(cumulative[:, idx], target, side="left"))
            bin_idx = min(bin_idx, len(y_centers) - 1)
            arr[idx] = y_centers[bin_idx]
        quantile_values[label] = arr

    quantiles_df = pd.DataFrame(
        {
            "turn": checkpoints,
            "active_runs": active_counts,
            **quantile_values,
        }
    )
    return checkpoints, y_edges, hist, quantiles_df


def make_density_plot(
    games: pd.DataFrame,
    events: pd.DataFrame,
    output_path: Path,
    quantiles_csv: Path,
    num_checkpoints: int,
    num_y_bins: int,
) -> None:
    checkpoints, y_edges, hist, quantiles_df = build_density_matrix(
        games=games,
        events=events,
        num_checkpoints=num_checkpoints,
        num_y_bins=num_y_bins,
    )

    density = hist.astype(np.float32)
    active_counts = quantiles_df["active_runs"].to_numpy(dtype=np.float32)
    nonzero = active_counts > 0
    density[:, nonzero] /= active_counts[nonzero]

    quantiles_csv.parent.mkdir(parents=True, exist_ok=True)
    quantiles_df.to_csv(quantiles_csv, index=False)

    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor("#f4efe8")
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, 6.0], hspace=0.12)
    ax_top = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0], sharex=ax_top)

    ax_top.set_facecolor("#fffaf3")
    ax.set_facecolor("#fffaf3")

    x_edges = np.geomspace(1, int(games["turns"].max()) + 1, num=len(checkpoints) + 1)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        density,
        cmap="magma",
        shading="auto",
        rasterized=True,
    )

    colors = {
        "q10": "#d8c7ff",
        "q25": "#a9d6e5",
        "q50": "#f7ede2",
        "q75": "#ffd166",
        "q90": "#ef476f",
    }
    labels = {
        "q10": "10th pct",
        "q25": "25th pct",
        "q50": "Median",
        "q75": "75th pct",
        "q90": "90th pct",
    }

    x_vals = checkpoints.astype(np.float64) + 1.0
    for key in ["q10", "q25", "q50", "q75", "q90"]:
        ax.plot(
            x_vals,
            quantiles_df[key].to_numpy(dtype=np.float32),
            color=colors[key],
            linewidth=2.0 if key != "q50" else 2.8,
            alpha=0.95,
            label=labels[key],
        )

    ax_top.plot(x_vals, active_counts, color="#284b63", linewidth=2.0)
    ax_top.fill_between(x_vals, active_counts, color="#284b63", alpha=0.18)
    ax_top.set_xscale("log")
    ax_top.set_ylabel("Active runs", fontsize=12)
    ax_top.grid(True, which="both", alpha=0.15, linewidth=0.8, color="#8c7b6b")
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(axis="x", labelbottom=False)

    ax.set_xscale("log")
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("NetHack Turn `T` (log scale)", fontsize=14)
    ax.set_ylabel("BALROG NLE Progression (%)", fontsize=14)
    ax.grid(True, which="both", alpha=0.10, linewidth=0.8, color="#8c7b6b")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor="#fff7ec", edgecolor="#d7c5ad")

    cbar = fig.colorbar(mesh, ax=ax, pad=0.012)
    cbar.set_label("Share of active runs in each progression band", fontsize=12)

    title = f"Human NetHack Trajectory Density Over Time ({len(games):,} games)"
    ax_top.set_title(title, fontsize=20, pad=10)
    ax.text(
        0.012,
        0.985,
        "Heatmap columns are normalized over runs still active at that turn\nPercentile bands are computed from the same full-store distribution",
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
    args.spaghetti_png = args.spaghetti_png.resolve()
    args.density_png = args.density_png.resolve()
    args.quantiles_csv = args.quantiles_csv.resolve()
    OUTPUT_DIR.mkdir(exist_ok=True)

    games, events = load_store(args.db_path)
    make_spaghetti_plot(games=games, events=events, output_path=args.spaghetti_png)
    make_density_plot(
        games=games,
        events=events,
        output_path=args.density_png,
        quantiles_csv=args.quantiles_csv,
        num_checkpoints=args.density_checkpoints,
        num_y_bins=args.density_y_bins,
    )

    print(
        {
            "games": len(games),
            "events": len(events),
            "spaghetti_png": str(args.spaghetti_png),
            "density_png": str(args.density_png),
            "quantiles_csv": str(args.quantiles_csv),
        }
    )


if __name__ == "__main__":
    main()
