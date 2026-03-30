from __future__ import annotations

import argparse
import heapq
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from human_nao_event_cache import HumanMemberEventCache, build_curves_from_cached_members
from human_nao_source import HumanNAODataSource
from plot_gemini3_vs_humans import (
    load_gemini_records,
    parse_human_metric_curves,
    parse_human_metric_curves_from_paths,
)
from plot_human_nao_random_plus_best import build_game_index
from plot_human_nao_trajectories import load_achievements


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_PROGRESS_PNG = OUTPUT_DIR / "gemini3_best_vs_top25_humans_progression_750.png"
OUTPUT_SCORE_PNG = OUTPUT_DIR / "gemini3_best_vs_top25_humans_score_750.png"
OUTPUT_CSV = OUTPUT_DIR / "gemini3_best_vs_top25_humans_750_manifest.csv"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-turn", type=int, default=750)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--candidate-maxlvl", type=int, default=10)
    parser.add_argument("--candidate-points", type=int, default=100000)
    parser.add_argument("--candidate-limit", type=int, default=200)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--use-event-cache", action="store_true")
    parser.add_argument("--clear-event-cache", action="store_true")
    return parser.parse_args()


def select_best_gemini_run(max_turn: int, achievements: dict[str, float]) -> dict:
    gemini_records = load_gemini_records(max_turn=max_turn, achievements=achievements)
    best = max(
        gemini_records,
        key=lambda record: (
            float(record["progression_curve"].loc[max_turn]),
            float(record["score_curve"].loc[max_turn]),
            record["run_name"],
        ),
    )
    return best


def _parse_human_game_from_paths(task: tuple[dict, str, dict[str, float], int]) -> dict | None:
    game, base_dir, achievements, max_turn = task
    member_paths = [str(Path(base_dir) / Path(member_name)) for member_name in game["members"]]
    score_curve, progression_curve = parse_human_metric_curves_from_paths(
        member_paths=member_paths,
        achievements=achievements,
        max_turn=max_turn,
        expected_turns=int(game["turns"]),
    )
    if score_curve is None or progression_curve is None:
        return None

    return {
        "player_name": game["player_name"],
        "local_gameid": int(game["local_gameid"]),
        "death": game["death"],
        "turns": int(game["turns"]),
        "score_curve": score_curve,
        "progression_curve": progression_curve,
        "progression_at_max_turn": float(progression_curve.loc[max_turn]),
        "score_at_max_turn": float(score_curve.loc[max_turn]),
        "final_maxlvl": int(game["maxlvl"]),
        "final_points": int(game["points"]),
    }


def _record_from_curves(game: dict, score_curve: pd.Series, progression_curve: pd.Series, max_turn: int) -> dict:
    return {
        "player_name": game["player_name"],
        "local_gameid": int(game["local_gameid"]),
        "death": game["death"],
        "turns": int(game["turns"]),
        "score_curve": score_curve,
        "progression_curve": progression_curve,
        "progression_at_max_turn": float(progression_curve.loc[max_turn]),
        "score_at_max_turn": float(score_curve.loc[max_turn]),
        "final_maxlvl": int(game["maxlvl"]),
        "final_points": int(game["points"]),
    }


def select_top_humans(
    max_turn: int,
    top_k: int,
    achievements: dict[str, float],
    candidate_maxlvl: int,
    candidate_points: int,
    candidate_limit: int,
    workers: int,
    use_event_cache: bool,
    clear_event_cache: bool,
) -> list[dict]:
    heap: list[tuple[float, float, str, int, dict]] = []

    with HumanNAODataSource() as source:
        stitched_games = build_game_index(source)
        candidates = [
            game
            for game in stitched_games
            if int(game["maxlvl"]) >= candidate_maxlvl or int(game["points"]) >= candidate_points
        ]
        candidates.sort(
            key=lambda game: (int(game["maxlvl"]), int(game["points"]), -int(game["turns"])),
            reverse=True,
        )
        if candidate_limit > 0:
            candidates = candidates[:candidate_limit]
    print(
        f"Candidate shortlist: {len(candidates):,} / {len(stitched_games):,} games "
        f"(maxlvl>={candidate_maxlvl} or points>={candidate_points}, limit={candidate_limit})"
    )

    worker_count = workers if workers > 0 else min(8, max(1, os.cpu_count() or 1))
    use_parallel = False
    source_mode = "zip"
    base_dir = ""

    with HumanNAODataSource() as source:
        source_mode = source.mode
        base_dir = str(source.base_dir)
        use_parallel = source.mode == "dir" and worker_count > 1 and not use_event_cache

        progress = tqdm(total=len(candidates), desc=f"Scanning shortlisted humans to T={max_turn}", unit="game")

        if use_parallel:
            tasks = [(game, base_dir, achievements, max_turn) for game in candidates]
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                iterator = executor.map(_parse_human_game_from_paths, tasks, chunksize=4)
                for record in iterator:
                    progress.update(1)
                    if record is None:
                        continue

                    key = (
                        record["progression_at_max_turn"],
                        record["score_at_max_turn"],
                        record["player_name"],
                        record["local_gameid"],
                    )
                    if len(heap) < top_k:
                        heapq.heappush(heap, (key[0], key[1], key[2], key[3], record))
                    else:
                        if key > (heap[0][0], heap[0][1], heap[0][2], heap[0][3]):
                            heapq.heapreplace(heap, (key[0], key[1], key[2], key[3], record))

                    current_floor = heap[0][0] if heap else 0.0
                    current_best = max((item[0] for item in heap), default=0.0)
                    progress.set_postfix(
                        mode="proc",
                        workers=worker_count,
                        top_floor_pct=f"{current_floor:.2f}",
                        best_pct=f"{current_best:.2f}",
                    )
        elif use_event_cache and source.mode == "dir":
            with HumanMemberEventCache() as cache:
                if clear_event_cache:
                    cache.clear()

                for game in candidates:
                    score_curve, progression_curve = build_curves_from_cached_members(
                        cache=cache,
                        base_dir=Path(base_dir),
                        member_names=list(game["members"]),
                        achievements=achievements,
                        max_turn=max_turn,
                        expected_turns=int(game["turns"]),
                    )
                    progress.update(1)
                    if score_curve is None or progression_curve is None:
                        continue

                    record = _record_from_curves(game, score_curve, progression_curve, max_turn)
                    key = (
                        record["progression_at_max_turn"],
                        record["score_at_max_turn"],
                        record["player_name"],
                        record["local_gameid"],
                    )
                    if len(heap) < top_k:
                        heapq.heappush(heap, (key[0], key[1], key[2], key[3], record))
                    else:
                        if key > (heap[0][0], heap[0][1], heap[0][2], heap[0][3]):
                            heapq.heapreplace(heap, (key[0], key[1], key[2], key[3], record))

                    current_floor = heap[0][0] if heap else 0.0
                    current_best = max((item[0] for item in heap), default=0.0)
                    progress.set_postfix(
                        mode="cache",
                        workers=1,
                        top_floor_pct=f"{current_floor:.2f}",
                        best_pct=f"{current_best:.2f}",
                    )
        else:
            for game in candidates:
                score_curve, progression_curve = parse_human_metric_curves(
                    source,
                    member_names=list(game["members"]),
                    achievements=achievements,
                    max_turn=max_turn,
                    expected_turns=int(game["turns"]),
                )
                progress.update(1)
                if score_curve is None or progression_curve is None:
                    continue
                record = _record_from_curves(game, score_curve, progression_curve, max_turn)

                key = (
                    record["progression_at_max_turn"],
                    record["score_at_max_turn"],
                    record["player_name"],
                    record["local_gameid"],
                )
                if len(heap) < top_k:
                    heapq.heappush(heap, (key[0], key[1], key[2], key[3], record))
                else:
                    if key > (heap[0][0], heap[0][1], heap[0][2], heap[0][3]):
                        heapq.heapreplace(heap, (key[0], key[1], key[2], key[3], record))

                current_floor = heap[0][0] if heap else 0.0
                current_best = max((item[0] for item in heap), default=0.0)
                progress.set_postfix(
                    mode=source_mode,
                    workers=1,
                    top_floor_pct=f"{current_floor:.2f}",
                    best_pct=f"{current_best:.2f}",
                )

        progress.close()

    top_records = [item[-1] for item in heap]
    top_records.sort(
        key=lambda record: (
            record["progression_at_max_turn"],
            record["score_at_max_turn"],
            record["player_name"],
            record["local_gameid"],
        ),
        reverse=True,
    )
    return top_records


def plot_metric(top_humans: list[dict], gemini_best: dict, metric_key: str, y_label: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#f7f1e8")
    ax.set_facecolor("#fffaf3")

    for record in top_humans:
        curve = record[metric_key]
        ax.plot(
            curve.index,
            curve.values,
            color="#78b7b1",
            linewidth=1.2,
            alpha=0.42,
            drawstyle="steps-post",
            zorder=1,
        )

    gemini_curve = gemini_best[metric_key]
    ax.plot(
        gemini_curve.index,
        gemini_curve.values,
        color="#6a1b9a",
        linewidth=3.4,
        alpha=0.98,
        drawstyle="steps-post",
        zorder=4,
        label=f"Gemini best: {gemini_best['run_name']}",
    )

    best_human = top_humans[0]
    human_curve = best_human[metric_key]
    ax.plot(
        human_curve.index,
        human_curve.values,
        color="#b23a48",
        linewidth=2.8,
        alpha=0.98,
        drawstyle="steps-post",
        zorder=3,
        label=f"Top human: {best_human['player_name']}#{best_human['local_gameid']}",
    )

    gemini_final = float(gemini_curve.loc[int(gemini_curve.index.max())])
    human_final = float(human_curve.loc[int(human_curve.index.max())])
    ax.scatter([int(gemini_curve.index.max())], [gemini_final], color="#6a1b9a", s=60, zorder=5)
    ax.scatter([int(human_curve.index.max())], [human_final], color="#b23a48", s=60, zorder=5)

    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel("NetHack Turn `T`", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_xlim(0, int(gemini_curve.index.max()))

    ymax = max(
        [float(gemini_curve.max())]
        + [float(record[metric_key].max()) for record in top_humans]
    )
    ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1)
    ax.grid(True, alpha=0.22, linewidth=0.9, color="#8f7f6d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6c6258")
    ax.spines["bottom"].set_color("#6c6258")

    legend_handles = [
        plt.Line2D([0], [0], color="#78b7b1", linewidth=2.0, alpha=0.6),
        plt.Line2D([0], [0], color="#b23a48", linewidth=2.8),
        plt.Line2D([0], [0], color="#6a1b9a", linewidth=3.4),
    ]
    legend_labels = [
        f"Top {len(top_humans)} humans at T={int(gemini_curve.index.max())}",
        f"Best human in top set: {best_human['player_name']}#{best_human['local_gameid']}",
        f"Gemini-3-Pro best run: {gemini_best['run_name']}",
    ]
    ax.legend(legend_handles, legend_labels, loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_manifest(top_humans: list[dict], gemini_best: dict, max_turn: int) -> None:
    rows = [
        {
            "group": "gemini_best",
            "label": gemini_best["run_name"],
            "death": "",
            "turns": max_turn,
            "score_at_max_turn": float(gemini_best["score_curve"].loc[max_turn]),
            "progression_at_max_turn": float(gemini_best["progression_curve"].loc[max_turn]),
        }
    ]
    for record in top_humans:
        rows.append(
            {
                "group": "human_top",
                "label": f"{record['player_name']}#{record['local_gameid']}",
                "death": record["death"],
                "turns": record["turns"],
                "score_at_max_turn": record["score_at_max_turn"],
                "progression_at_max_turn": record["progression_at_max_turn"],
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    achievements = load_achievements()

    gemini_best = select_best_gemini_run(max_turn=args.max_turn, achievements=achievements)
    top_humans = select_top_humans(
        max_turn=args.max_turn,
        top_k=args.top_k,
        achievements=achievements,
        candidate_maxlvl=args.candidate_maxlvl,
        candidate_points=args.candidate_points,
        candidate_limit=args.candidate_limit,
        workers=args.workers,
        use_event_cache=args.use_event_cache,
        clear_event_cache=args.clear_event_cache,
    )

    plot_metric(
        top_humans=top_humans,
        gemini_best=gemini_best,
        metric_key="progression_curve",
        y_label="BALROG NLE Progression (%)",
        title=f"Gemini-3-Pro Best Run vs Top {args.top_k} Humans at T={args.max_turn} (Progression)",
        output_path=OUTPUT_PROGRESS_PNG,
    )
    plot_metric(
        top_humans=top_humans,
        gemini_best=gemini_best,
        metric_key="score_curve",
        y_label="Status-Line Score `S`",
        title=f"Gemini-3-Pro Best Run vs Top {args.top_k} Humans at T={args.max_turn} (Score)",
        output_path=OUTPUT_SCORE_PNG,
    )
    save_manifest(top_humans=top_humans, gemini_best=gemini_best, max_turn=args.max_turn)

    print(f"Gemini best run: {gemini_best['run_name']}")
    print(f"Gemini best progression at T={args.max_turn}: {float(gemini_best['progression_curve'].loc[args.max_turn]):.6f}%")
    print(f"Top human progression at T={args.max_turn}: {top_humans[0]['progression_at_max_turn']:.6f}%")
    print(f"Saved progression plot to {OUTPUT_PROGRESS_PNG}")
    print(f"Saved score plot to {OUTPUT_SCORE_PNG}")
    print(f"Saved manifest to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
