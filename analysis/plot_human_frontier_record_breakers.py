from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plot_human_best_median_vs_gemini import (
    DB_PATH,
    OUTPUT_DIR,
    SparseCurve,
    crop_sparse_curve,
    load_gemini_sparse_curves,
    load_human_store,
    sanitized_progression_values,
)
from analysis.plot_human_nao_trajectories import load_achievements


ROOT = Path(__file__).resolve().parent.parent
PROG_MEDIAN_CSV = OUTPUT_DIR / "human_best_median_progression_full.csv"
SCORE_MEDIAN_CSV = OUTPUT_DIR / "human_best_median_score_full.csv"


@dataclass
class RecordHolder:
    game_key: str
    player_name: str
    takeover_turn: int
    takeover_value: float
    turns: int
    points: int
    maxlvl: int
    death: str
    order_idx: int


def load_games_meta(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        games = pd.read_sql_query(
            """
            SELECT game_key, player_name, turns, points, maxlvl, death
            FROM games
            ORDER BY game_key
            """,
            conn,
        )
    finally:
        conn.close()
    return games


def load_median_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(PROG_MEDIAN_CSV), pd.read_csv(SCORE_MEDIAN_CSV)


def crop_step_df(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon < 0:
        return df.copy()
    clipped = df[df["turn"] <= horizon].copy()
    if clipped.empty:
        return pd.DataFrame({"turn": [0, horizon], "value": [0.0, 0.0]})
    if int(clipped["turn"].iloc[-1]) < horizon:
        last = clipped.iloc[-1].copy()
        last["turn"] = horizon
        clipped = pd.concat([clipped, pd.DataFrame([last])], ignore_index=True)
    return clipped


def compute_frontier_curve(events: pd.DataFrame, values: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    frontier_turns = [0]
    frontier_values = [0.0]
    record_rows: list[dict[str, object]] = []
    best_so_far = 0.0

    for game_key, turn, value in zip(
        events["game_key"].astype(str).to_numpy(),
        events["turn"].to_numpy(dtype=np.int32),
        values,
    ):
        value_f = float(value)
        if value_f > best_so_far:
            best_so_far = value_f
            frontier_turns.append(int(turn))
            frontier_values.append(value_f)
            record_rows.append(
                {
                    "game_key": game_key,
                    "turn": int(turn),
                    "value": value_f,
                }
            )

    frontier_df = pd.DataFrame(
        {
            "turn": np.array(frontier_turns, dtype=np.int32),
            "value": np.array(frontier_values, dtype=np.float64),
        }
    )
    records_df = pd.DataFrame(record_rows)
    return frontier_df, records_df


def distinct_record_holders(records_df: pd.DataFrame, games_meta: pd.DataFrame) -> list[RecordHolder]:
    if records_df.empty:
        return []
    first_takeovers = records_df.drop_duplicates(subset=["game_key"], keep="first").copy()
    first_takeovers["order_idx"] = np.arange(1, len(first_takeovers) + 1, dtype=np.int32)
    merged = first_takeovers.merge(games_meta, on="game_key", how="left", sort=False)
    holders: list[RecordHolder] = []
    for row in merged.itertuples(index=False):
        holders.append(
            RecordHolder(
                game_key=str(row.game_key),
                player_name=str(row.player_name),
                takeover_turn=int(row.turn),
                takeover_value=float(row.value),
                turns=int(row.turns),
                points=int(row.points),
                maxlvl=int(row.maxlvl),
                death=str(row.death),
                order_idx=int(row.order_idx),
            )
        )
    return holders


def build_sparse_curves_for_holders(
    events: pd.DataFrame,
    values: np.ndarray,
    holders: list[RecordHolder],
) -> list[SparseCurve]:
    metric_df = events[["game_key", "turn"]].copy()
    metric_df["value"] = values
    curves: list[SparseCurve] = []
    for holder in holders:
        sub = metric_df[metric_df["game_key"].astype(str) == holder.game_key]
        sub = sub[sub["turn"] <= holder.takeover_turn]
        turns = [0]
        vals = [0.0]
        last_val = 0.0
        for row in sub.itertuples(index=False):
            turn = int(row.turn)
            value = float(row.value)
            if value != last_val:
                turns.append(turn)
                vals.append(value)
                last_val = value
        if turns[-1] < holder.takeover_turn:
            turns.append(holder.takeover_turn)
            vals.append(last_val)
        label = f"{holder.player_name} ({holder.game_key})"
        curves.append(
            SparseCurve(
                run_name=label,
                turns=np.array(turns, dtype=np.int32),
                values=np.array(vals, dtype=np.float64),
                last_turn=holder.takeover_turn,
            )
        )
    return curves


def write_holder_manifest(path: Path, holders: list[RecordHolder], metric_name: str) -> None:
    rows = []
    for holder in holders:
        rows.append(
            {
                "order_idx": holder.order_idx,
                "metric": metric_name,
                "game_key": holder.game_key,
                "player_name": holder.player_name,
                "takeover_turn": holder.takeover_turn,
                "takeover_value": holder.takeover_value,
                "turns": holder.turns,
                "points": holder.points,
                "maxlvl": holder.maxlvl,
                "death": holder.death,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_frontier_metric(
    output_path: Path,
    title: str,
    y_label: str,
    horizon: int,
    active_df: pd.DataFrame,
    median_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    gemini_curves: list[SparseCurve],
    y_max: float,
) -> None:
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#f4efe8")
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.1, 5.2], hspace=0.08)
    ax_top = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0], sharex=ax_top)

    ax_top.set_facecolor("#fffaf3")
    ax.set_facecolor("#fffaf3")

    x_active = active_df["turn"].to_numpy(dtype=np.int32)
    y_active = active_df["active_runs"].to_numpy(dtype=np.int32)
    ax_top.plot(x_active, y_active, color="#284b63", linewidth=2.2)
    ax_top.fill_between(x_active, y_active, color="#284b63", alpha=0.18)
    ax_top.set_ylabel("Active humans", fontsize=12)
    ax_top.grid(True, alpha=0.16, linewidth=0.8, color="#8c7b6b")
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_title(title, fontsize=20, pad=12)

    ax.plot(
        frontier_df["turn"].to_numpy(dtype=np.int32),
        frontier_df["value"].to_numpy(dtype=np.float64),
        color="#b23a48",
        linewidth=3.0,
        drawstyle="steps-post",
        label="Human best so far",
    )
    ax.plot(
        median_df["turn"].to_numpy(dtype=np.int32),
        median_df["median"].to_numpy(dtype=np.float64),
        color="#1d3557",
        linewidth=3.0,
        drawstyle="steps-post",
        label="Human median",
    )

    palette = ["#7b2cbf", "#9d4edd", "#c77dff", "#f72585", "#4361ee"]
    for color, curve in zip(palette, gemini_curves):
        x_vals = curve.turns
        y_vals = curve.values
        if len(x_vals) > 0 and int(x_vals[-1]) < curve.last_turn:
            x_vals = np.append(x_vals, int(curve.last_turn))
            y_vals = np.append(y_vals, float(y_vals[-1]))
        ax.plot(
            x_vals,
            y_vals,
            color=color,
            linewidth=2.0,
            alpha=0.95,
            drawstyle="steps-post",
            label=f"Gemini {curve.run_name}",
        )

    x_max = int(
        max(
            np.nanmax(frontier_df["turn"].to_numpy(dtype=np.int32)),
            np.nanmax(median_df["turn"].to_numpy(dtype=np.int32)),
            max((curve.last_turn for curve in gemini_curves), default=0),
        )
    )
    ax.set_xlim(0, x_max if horizon < 0 else horizon)
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
        "Best so far is the cumulative frontier over all humans seen by turn T.\nUnlike pointwise best active, it never drops after a record is set.",
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


def plot_score_frontier_full(
    output_path: Path,
    active_df: pd.DataFrame,
    median_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    gemini_curves: list[SparseCurve],
) -> None:
    fig = plt.figure(figsize=(18, 15))
    fig.patch.set_facecolor("#f4efe8")
    grid = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.0, 3.5, 3.2], hspace=0.12)
    ax_top = fig.add_subplot(grid[0, 0])
    ax_full = fig.add_subplot(grid[1, 0], sharex=ax_top)
    ax_zoom = fig.add_subplot(grid[2, 0], sharex=ax_top)
    for ax in (ax_top, ax_full, ax_zoom):
        ax.set_facecolor("#fffaf3")

    x_active = active_df["turn"].to_numpy(dtype=np.int32)
    y_active = active_df["active_runs"].to_numpy(dtype=np.int32)
    ax_top.plot(x_active, y_active, color="#284b63", linewidth=2.2)
    ax_top.fill_between(x_active, y_active, color="#284b63", alpha=0.18)
    ax_top.set_ylabel("Active humans", fontsize=12)
    ax_top.grid(True, alpha=0.16, linewidth=0.8, color="#8c7b6b")
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_title("Human best-so-far score vs Gemini-3-Pro, full run", fontsize=20, pad=12)

    palette = ["#7b2cbf", "#9d4edd", "#c77dff", "#f72585", "#4361ee"]
    frontier_x = frontier_df["turn"].to_numpy(dtype=np.int32)
    frontier_y = frontier_df["value"].to_numpy(dtype=np.float64)
    median_x = median_df["turn"].to_numpy(dtype=np.int32)
    median_y = median_df["median"].to_numpy(dtype=np.float64)

    for ax in (ax_full, ax_zoom):
        ax.plot(frontier_x, frontier_y, color="#b23a48", linewidth=3.0, drawstyle="steps-post", label="Human best so far")
        ax.plot(median_x, median_y, color="#1d3557", linewidth=3.0, drawstyle="steps-post", label="Human median")
        for color, curve in zip(palette, gemini_curves):
            x_vals = curve.turns
            y_vals = curve.values
            if len(x_vals) > 0 and int(x_vals[-1]) < curve.last_turn:
                x_vals = np.append(x_vals, int(curve.last_turn))
                y_vals = np.append(y_vals, float(y_vals[-1]))
            ax.plot(x_vals, y_vals, color=color, linewidth=1.9, alpha=0.95, drawstyle="steps-post", label=f"Gemini {curve.run_name}")

    x_max = int(max(np.nanmax(frontier_x), np.nanmax(median_x), max((curve.last_turn for curve in gemini_curves), default=0)))
    clip_y = max(
        250_000.0,
        float(frontier_df[frontier_df["turn"] <= 10_000]["value"].max()) * 1.15 if (frontier_df["turn"] <= 10_000).any() else 250_000.0,
    )
    full_y = float(np.nanmax(frontier_y)) * 1.05

    ax_full.set_xlim(0, x_max)
    ax_full.set_ylim(0, full_y)
    ax_full.set_ylabel("NetHack score `S`", fontsize=14)
    ax_full.grid(True, alpha=0.12, linewidth=0.8, color="#8c7b6b")
    ax_full.spines["top"].set_visible(False)
    ax_full.spines["right"].set_visible(False)
    ax_full.text(
        0.012,
        0.985,
        "Full range. This is dominated by a few real monster-score runs, especially Adeon's late-game records.",
        transform=ax_full.transAxes,
        ha="left",
        va="top",
        fontsize=11.0,
        color="#5d554c",
        bbox=dict(facecolor="#fff7ec", edgecolor="#d7c5ad", boxstyle="round,pad=0.4", alpha=0.95),
    )

    ax_zoom.set_xlim(0, x_max)
    ax_zoom.set_ylim(0, clip_y)
    ax_zoom.set_xlabel("NetHack Turn `T`", fontsize=14)
    ax_zoom.set_ylabel("NetHack score `S`\nzoomed", fontsize=14)
    ax_zoom.grid(True, alpha=0.12, linewidth=0.8, color="#8c7b6b")
    ax_zoom.spines["top"].set_visible(False)
    ax_zoom.spines["right"].set_visible(False)
    ax_zoom.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.95, facecolor="#fff7ec", edgecolor="#d7c5ad")
    ax_zoom.text(
        0.012,
        0.985,
        "Zoomed lower panel so the early and mid-game frontier is actually readable on a linear time axis.",
        transform=ax_zoom.transAxes,
        ha="left",
        va="top",
        fontsize=11.0,
        color="#5d554c",
        bbox=dict(facecolor="#fff7ec", edgecolor="#d7c5ad", boxstyle="round,pad=0.4", alpha=0.95),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_record_breakers(
    output_path: Path,
    title: str,
    y_label: str,
    curves: list[SparseCurve],
    full_y_max: float,
    zoom_y_max: float | None = None,
) -> None:
    if zoom_y_max is None:
        fig = plt.figure(figsize=(18, 10))
        fig.patch.set_facecolor("#f4efe8")
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor("#fffaf3")
        axes = [ax]
    else:
        fig = plt.figure(figsize=(18, 13))
        fig.patch.set_facecolor("#f4efe8")
        grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.4, 2.8], hspace=0.12)
        ax = fig.add_subplot(grid[0, 0])
        ax_zoom = fig.add_subplot(grid[1, 0], sharex=ax)
        ax.set_facecolor("#fffaf3")
        ax_zoom.set_facecolor("#fffaf3")
        axes = [ax, ax_zoom]

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0.08, 0.92, max(len(curves), 2))]

    for color, curve in zip(colors, curves):
        label = curve.run_name
        x_vals = curve.turns
        y_vals = curve.values
        if len(x_vals) > 0 and int(x_vals[-1]) < curve.last_turn:
            x_vals = np.append(x_vals, int(curve.last_turn))
            y_vals = np.append(y_vals, float(y_vals[-1]))
        for panel in axes:
            panel.plot(x_vals, y_vals, color=color, linewidth=2.1, alpha=0.96, drawstyle="steps-post", label=label)

    x_max = int(max((curve.last_turn for curve in curves), default=0))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, full_y_max)
    ax.set_title(title, fontsize=20, pad=12)
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid(True, alpha=0.12, linewidth=0.8, color="#8c7b6b")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", ncol=2, fontsize=10, frameon=True, framealpha=0.95, facecolor="#fff7ec", edgecolor="#d7c5ad")
    ax.text(
        0.012,
        0.985,
        "Each line stops at the turn where that human first became the global record holder.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.0,
        color="#5d554c",
        bbox=dict(facecolor="#fff7ec", edgecolor="#d7c5ad", boxstyle="round,pad=0.4", alpha=0.95),
    )

    if zoom_y_max is not None:
        ax_zoom.set_xlim(0, x_max)
        ax_zoom.set_ylim(0, zoom_y_max)
        ax_zoom.set_xlabel("NetHack Turn `T`", fontsize=14)
        ax_zoom.set_ylabel(f"{y_label}\nzoomed", fontsize=14)
        ax_zoom.grid(True, alpha=0.12, linewidth=0.8, color="#8c7b6b")
        ax_zoom.spines["top"].set_visible(False)
        ax_zoom.spines["right"].set_visible(False)
    else:
        ax.set_xlabel("NetHack Turn `T`", fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    games_meta = load_games_meta(DB_PATH)
    games, events = load_human_store(DB_PATH)
    achievements = load_achievements()
    progression_values = sanitized_progression_values(events, achievements)
    score_values = events["score"].to_numpy(dtype=np.float64)

    progress_median_df, score_median_df = load_median_curves()

    progression_frontier_df, progression_records_df = compute_frontier_curve(events, progression_values)
    score_frontier_df, score_records_df = compute_frontier_curve(events, score_values)

    prog_holders = distinct_record_holders(progression_records_df, games_meta)
    score_holders = distinct_record_holders(score_records_df, games_meta)

    write_holder_manifest(OUTPUT_DIR / "human_progression_record_breakers.csv", prog_holders, "progression_pct")
    write_holder_manifest(OUTPUT_DIR / "human_score_record_breakers.csv", score_holders, "score")
    progression_frontier_df.to_csv(OUTPUT_DIR / "human_progression_frontier_full.csv", index=False)
    score_frontier_df.to_csv(OUTPUT_DIR / "human_score_frontier_full.csv", index=False)

    gemini_score_curves, gemini_progress_curves = load_gemini_sparse_curves()

    horizons = [1500, 5000, -1]
    for horizon in horizons:
        prog_frontier_crop = crop_step_df(progression_frontier_df, horizon)
        score_frontier_crop = crop_step_df(score_frontier_df, horizon)
        prog_median_crop = progress_median_df.copy() if horizon < 0 else progress_median_df[progress_median_df["turn"] <= horizon].copy()
        score_median_crop = score_median_df.copy() if horizon < 0 else score_median_df[score_median_df["turn"] <= horizon].copy()
        if horizon >= 0:
            if int(prog_median_crop["turn"].iloc[-1]) < horizon:
                last = prog_median_crop.iloc[-1].copy()
                last["turn"] = horizon
                prog_median_crop = pd.concat([prog_median_crop, pd.DataFrame([last])], ignore_index=True)
            if int(score_median_crop["turn"].iloc[-1]) < horizon:
                last = score_median_crop.iloc[-1].copy()
                last["turn"] = horizon
                score_median_crop = pd.concat([score_median_crop, pd.DataFrame([last])], ignore_index=True)

        gem_prog = [crop_sparse_curve(curve, horizon) for curve in gemini_progress_curves]
        gem_score = [crop_sparse_curve(curve, horizon) for curve in gemini_score_curves]

        label = "full run" if horizon < 0 else f"0-{horizon}"
        if horizon < 0:
            plot_frontier_metric(
                output_path=OUTPUT_DIR / "human_frontier_vs_gemini_progression_full.png",
                title=f"Human best-so-far progression vs Gemini-3-Pro, {label}",
                y_label="BALROG NLE Progression (%)",
                horizon=horizon,
                active_df=progress_median_df,
                median_df=prog_median_crop,
                frontier_df=prog_frontier_crop,
                gemini_curves=gem_prog,
                y_max=max(100.0, float(np.nanmax(prog_frontier_crop["value"])) * 1.05),
            )
            plot_score_frontier_full(
                output_path=OUTPUT_DIR / "human_frontier_vs_gemini_score_full.png",
                active_df=score_median_df,
                median_df=score_median_crop,
                frontier_df=score_frontier_crop,
                gemini_curves=gem_score,
            )
        else:
            plot_frontier_metric(
                output_path=OUTPUT_DIR / f"human_frontier_vs_gemini_progression_{horizon}.png",
                title=f"Human best-so-far progression vs Gemini-3-Pro, {label}",
                y_label="BALROG NLE Progression (%)",
                horizon=horizon,
                active_df=prog_median_crop,
                median_df=prog_median_crop,
                frontier_df=prog_frontier_crop,
                gemini_curves=gem_prog,
                y_max=max(
                    10.0,
                    float(np.nanmax(prog_frontier_crop["value"])) * 1.05,
                    max((float(np.nanmax(curve.values)) for curve in gem_prog), default=0.0) * 1.05,
                ),
            )
            plot_frontier_metric(
                output_path=OUTPUT_DIR / f"human_frontier_vs_gemini_score_{horizon}.png",
                title=f"Human best-so-far score vs Gemini-3-Pro, {label}",
                y_label="NetHack score `S`",
                horizon=horizon,
                active_df=score_median_crop,
                median_df=score_median_crop,
                frontier_df=score_frontier_crop,
                gemini_curves=gem_score,
                y_max=max(
                    50.0,
                    float(np.nanmax(score_frontier_crop["value"])) * 1.05,
                    max((float(np.nanmax(curve.values)) for curve in gem_score), default=0.0) * 1.05,
                ),
            )

    prog_curves = build_sparse_curves_for_holders(events, progression_values, prog_holders)
    score_curves = build_sparse_curves_for_holders(events, score_values, score_holders)

    plot_record_breakers(
        output_path=OUTPUT_DIR / "human_progression_record_breakers_takeover.png",
        title="Human progression record-breaker trajectories up to takeover",
        y_label="BALROG NLE Progression (%)",
        curves=prog_curves,
        full_y_max=max(100.0, max((float(np.nanmax(curve.values)) for curve in prog_curves), default=0.0) * 1.05),
    )
    plot_record_breakers(
        output_path=OUTPUT_DIR / "human_score_record_breakers_takeover.png",
        title="Human score record-breaker trajectories up to takeover",
        y_label="NetHack score `S`",
        curves=score_curves,
        full_y_max=max((float(np.nanmax(curve.values)) for curve in score_curves), default=1.0) * 1.05,
        zoom_y_max=250_000.0,
    )

    print(
        {
            "progression_record_holders": len(prog_holders),
            "score_record_holders": len(score_holders),
            "outputs": [
                "human_frontier_vs_gemini_progression_1500.png",
                "human_frontier_vs_gemini_progression_5000.png",
                "human_frontier_vs_gemini_progression_full.png",
                "human_frontier_vs_gemini_score_1500.png",
                "human_frontier_vs_gemini_score_5000.png",
                "human_frontier_vs_gemini_score_full.png",
                "human_progression_record_breakers_takeover.png",
                "human_score_record_breakers_takeover.png",
                "human_progression_record_breakers.csv",
                "human_score_record_breakers.csv",
                "human_progression_frontier_full.csv",
                "human_score_frontier_full.csv",
            ],
        }
    )


if __name__ == "__main__":
    main()

