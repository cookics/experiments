from __future__ import annotations

import argparse
import bz2
import random
import re
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from human_nao_source import HumanNAODataSource
from plot_human_nao_random_plus_best import build_game_index
from plot_human_nao_trajectories import load_achievements
from plot_nle_trajectories import find_nle_csvs


ROOT = Path(__file__).resolve().parent
GEMINI_FOLDER = ROOT / "submissions" / "LLM" / "20260203_naive_gemini-3-pro"
OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_SCORE_PNG = OUTPUT_DIR / "gemini3_vs_25_random_humans_score_3000.png"
OUTPUT_PROGRESS_PNG = OUTPUT_DIR / "gemini3_vs_25_random_humans_progression_3000.png"
OUTPUT_CSV = OUTPUT_DIR / "gemini3_vs_25_random_humans_manifest.csv"

TURN_RE = re.compile(r"\bT:(\d+)")
SCORE_RE = re.compile(r"\bS:(\d+)")
DLVL_RE = re.compile(r"Dlvl:(\d+)")
XP_RE = re.compile(r"(?:Xp|Exp):(\d+)")
HOME_RE = re.compile(r"\bHome ([1-5])\b")
ASTRAL_RE = re.compile(r"\bAstral Plane\b")
ASCEND_RE = re.compile(r"\bYou ascend\b", re.IGNORECASE)
TURN_TOKEN = b"T:"
SCORE_TOKEN = b"S:"
DLVL_TOKEN = b"Dlvl:"
XP_TOKEN = b"Xp:"
EXP_TOKEN = b"Exp:"
HOME_TOKEN = b"Home "
ASTRAL_TOKEN = b"Astral Plane"
ASCEND_TOKEN = b"You ascend"
ASCEND_TOKEN_LOWER = b"you ascend"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-turn", type=int, default=3000)
    parser.add_argument("--human-random", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def update_metrics_from_text(
    text: str,
    progression: float,
    score: float,
    achievements: dict[str, float],
) -> tuple[float, float, int | None]:
    turn_match = TURN_RE.search(text)
    if not turn_match:
        return progression, score, None

    turn = int(turn_match.group(1))
    candidates = [progression]

    dlvl_match = DLVL_RE.search(text)
    xp_match = XP_RE.search(text)
    home_match = HOME_RE.search(text)
    score_match = SCORE_RE.search(text)

    if dlvl_match:
        candidates.append(achievements.get(f"Dlvl:{int(dlvl_match.group(1))}", progression))
    if xp_match:
        candidates.append(achievements.get(f"Xp:{int(xp_match.group(1))}", progression))
    if home_match:
        candidates.append(achievements.get(f"Home {int(home_match.group(1))}", progression))
    if ASTRAL_RE.search(text):
        candidates.append(achievements.get("Astral Plane", progression))
    if ASCEND_RE.search(text):
        candidates.append(achievements.get("You ascend t", progression))

    progression = max(candidates)
    if score_match:
        score = max(score, float(score_match.group(1)))

    return progression, score, turn


def finalize_curve(values_by_turn: dict[int, float], max_turn: int) -> pd.Series | None:
    if not values_by_turn:
        return None
    series = pd.Series(values_by_turn).sort_index()
    return series.reindex(range(max_turn + 1)).ffill().fillna(0.0)


def _last_numeric_value(data: bytes, token: bytes) -> int | None:
    idx = data.rfind(token)
    if idx == -1:
        return None

    start = idx + len(token)
    end = start
    while end < len(data) and 48 <= data[end] <= 57:
        end += 1

    if end == start:
        return None
    return int(data[start:end])


def parse_human_metric_curves(
    source,
    member_names: list[str],
    achievements: dict[str, float],
    max_turn: int,
    expected_turns: int | None,
) -> tuple[pd.Series | None, pd.Series | None]:
    progression = 0.0
    score = 0.0
    last_turn = -1
    progression_by_turn: dict[int, float] = {}
    score_by_turn: dict[int, float] = {}

    for member_name in member_names:
        with source.open(member_name) as raw_member:
            tty = bz2.BZ2File(raw_member)

            while True:
                header = tty.read(12)
                if not header:
                    break

                sec, usec, length = struct.unpack("<iii", header)
                if sec < 0 or usec < 0 or length < 1:
                    break

                data = tty.read(length)
                if not data:
                    break

                if (
                    TURN_TOKEN not in data
                    and SCORE_TOKEN not in data
                    and DLVL_TOKEN not in data
                    and XP_TOKEN not in data
                    and EXP_TOKEN not in data
                    and HOME_TOKEN not in data
                    and ASTRAL_TOKEN not in data
                    and ASCEND_TOKEN not in data
                    and ASCEND_TOKEN_LOWER not in data
                ):
                    continue

                candidates = [progression]
                dlvl = _last_numeric_value(data, DLVL_TOKEN)
                if dlvl is not None:
                    candidates.append(achievements.get(f"Dlvl:{dlvl}", progression))

                xp = _last_numeric_value(data, XP_TOKEN)
                if xp is None:
                    xp = _last_numeric_value(data, EXP_TOKEN)
                if xp is not None:
                    candidates.append(achievements.get(f"Xp:{xp}", progression))

                home = _last_numeric_value(data, HOME_TOKEN)
                if home is not None:
                    candidates.append(achievements.get(f"Home {home}", progression))
                if ASTRAL_TOKEN in data:
                    candidates.append(achievements.get("Astral Plane", progression))
                if ASCEND_TOKEN in data or ASCEND_TOKEN_LOWER in data:
                    candidates.append(achievements.get("You ascend t", progression))
                progression = max(candidates)

                score_match = _last_numeric_value(data, SCORE_TOKEN)
                if score_match is not None:
                    score = max(score, float(score_match))

                turn = _last_numeric_value(data, TURN_TOKEN)
                if turn is None:
                    continue
                if expected_turns is not None and turn > expected_turns:
                    continue
                if turn < last_turn:
                    continue
                last_turn = turn
                progression_by_turn[turn] = progression * 100.0
                score_by_turn[turn] = score

                if turn >= max_turn:
                    break

            if last_turn >= max_turn:
                break

    return finalize_curve(score_by_turn, max_turn), finalize_curve(progression_by_turn, max_turn)


def parse_human_metric_curves_from_paths(
    member_paths: list[Path | str],
    achievements: dict[str, float],
    max_turn: int,
    expected_turns: int | None,
) -> tuple[pd.Series | None, pd.Series | None]:
    progression = 0.0
    score = 0.0
    last_turn = -1
    progression_by_turn: dict[int, float] = {}
    score_by_turn: dict[int, float] = {}

    for member_path in member_paths:
        with Path(member_path).open("rb") as raw_member:
            tty = bz2.BZ2File(raw_member)

            while True:
                header = tty.read(12)
                if not header:
                    break

                sec, usec, length = struct.unpack("<iii", header)
                if sec < 0 or usec < 0 or length < 1:
                    break

                data = tty.read(length)
                if not data:
                    break

                if (
                    TURN_TOKEN not in data
                    and SCORE_TOKEN not in data
                    and DLVL_TOKEN not in data
                    and XP_TOKEN not in data
                    and EXP_TOKEN not in data
                    and HOME_TOKEN not in data
                    and ASTRAL_TOKEN not in data
                    and ASCEND_TOKEN not in data
                    and ASCEND_TOKEN_LOWER not in data
                ):
                    continue

                candidates = [progression]
                dlvl = _last_numeric_value(data, DLVL_TOKEN)
                if dlvl is not None:
                    candidates.append(achievements.get(f"Dlvl:{dlvl}", progression))

                xp = _last_numeric_value(data, XP_TOKEN)
                if xp is None:
                    xp = _last_numeric_value(data, EXP_TOKEN)
                if xp is not None:
                    candidates.append(achievements.get(f"Xp:{xp}", progression))

                home = _last_numeric_value(data, HOME_TOKEN)
                if home is not None:
                    candidates.append(achievements.get(f"Home {home}", progression))
                if ASTRAL_TOKEN in data:
                    candidates.append(achievements.get("Astral Plane", progression))
                if ASCEND_TOKEN in data or ASCEND_TOKEN_LOWER in data:
                    candidates.append(achievements.get("You ascend t", progression))
                progression = max(candidates)

                score_match = _last_numeric_value(data, SCORE_TOKEN)
                if score_match is not None:
                    score = max(score, float(score_match))

                turn = _last_numeric_value(data, TURN_TOKEN)
                if turn is None:
                    continue
                if expected_turns is not None and turn > expected_turns:
                    continue
                if turn < last_turn:
                    continue

                last_turn = turn
                progression_by_turn[turn] = progression * 100.0
                score_by_turn[turn] = score

                if turn >= max_turn:
                    break

            if last_turn >= max_turn:
                break

    return finalize_curve(score_by_turn, max_turn), finalize_curve(progression_by_turn, max_turn)


def parse_llm_metric_curves(
    csv_path: Path,
    achievements: dict[str, float],
    max_turn: int,
) -> tuple[pd.Series | None, pd.Series | None]:
    df = pd.read_csv(csv_path, usecols=["Observation"])

    progression = 0.0
    score = 0.0
    last_turn = -1
    progression_by_turn: dict[int, float] = {}
    score_by_turn: dict[int, float] = {}

    for observation in df["Observation"].fillna(""):
        progression, score, turn = update_metrics_from_text(str(observation), progression, score, achievements)
        if turn is None:
            continue
        if turn < last_turn:
            continue
        if turn > max_turn:
            break
        last_turn = turn
        progression_by_turn[turn] = progression * 100.0
        score_by_turn[turn] = score

    return finalize_curve(score_by_turn, max_turn), finalize_curve(progression_by_turn, max_turn)


def mean_curve(curves: list[pd.Series], max_turn: int) -> pd.Series:
    turns = pd.Index(range(max_turn + 1), name="turn")
    aligned = [curve.reindex(turns).ffill().fillna(0.0) for curve in curves]
    return pd.concat(aligned, axis=1).mean(axis=1)


def select_random_humans(
    human_random: int,
    seed: int,
    max_turn: int,
    achievements: dict[str, float],
) -> list[dict]:
    rng = random.Random(seed)

    with HumanNAODataSource() as source:
        stitched_games = build_game_index(source)
        rng.shuffle(stitched_games)

        records: list[dict] = []
        for game in stitched_games:
            if len(records) >= human_random:
                break

            score_curve, progression_curve = parse_human_metric_curves(
                source,
                member_names=list(game["members"]),
                achievements=achievements,
                max_turn=max_turn,
                expected_turns=int(game["turns"]),
            )
            if score_curve is None or progression_curve is None:
                continue

            records.append(
                {
                    "player_name": game["player_name"],
                    "local_gameid": int(game["local_gameid"]),
                    "death": game["death"],
                    "turns": int(game["turns"]),
                    "score_curve": score_curve,
                    "progression_curve": progression_curve,
                }
            )

    return records


def load_gemini_records(max_turn: int, achievements: dict[str, float]) -> list[dict]:
    records: list[dict] = []
    for csv_path in find_nle_csvs(GEMINI_FOLDER):
        score_curve, progression_curve = parse_llm_metric_curves(csv_path, achievements, max_turn=max_turn)
        if score_curve is None or progression_curve is None:
            continue
        records.append(
            {
                "run_name": csv_path.name,
                "score_curve": score_curve,
                "progression_curve": progression_curve,
            }
        )
    if not records:
        raise SystemExit("Could not parse Gemini-3-Pro NLE curves.")
    return records


def plot_metric(
    human_records: list[dict],
    gemini_records: list[dict],
    metric_key: str,
    y_label: str,
    title: str,
    output_path: Path,
    y_max: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#f7f1e8")
    ax.set_facecolor("#fffaf3")

    for record in human_records:
        curve = record[metric_key]
        ax.plot(
            curve.index,
            curve.values,
            color="#6ea9a1",
            linewidth=1.0,
            alpha=0.28,
            drawstyle="steps-post",
            zorder=1,
        )

    for record in gemini_records:
        curve = record[metric_key]
        ax.plot(
            curve.index,
            curve.values,
            color="#b39ddb",
            linewidth=1.4,
            alpha=0.7,
            drawstyle="steps-post",
            zorder=2,
        )

    gemini_mean = mean_curve([record[metric_key] for record in gemini_records], int(gemini_records[0][metric_key].index.max()))
    ax.plot(
        gemini_mean.index,
        gemini_mean.values,
        color="#6a1b9a",
        linewidth=3.2,
        alpha=0.98,
        drawstyle="steps-post",
        zorder=4,
        label=f"Gemini-3-Pro mean ({float(gemini_mean.iloc[-1]):.2f})",
    )

    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel("NetHack Turn `T`", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_xlim(0, int(gemini_mean.index.max()))
    if y_max is not None:
        ax.set_ylim(0, y_max)
    else:
        ymax = max(
            [float(gemini_mean.max())]
            + [float(record[metric_key].max()) for record in human_records]
            + [float(record[metric_key].max()) for record in gemini_records]
        )
        ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1)
    ax.grid(True, alpha=0.22, linewidth=0.9, color="#8f7f6d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6c6258")
    ax.spines["bottom"].set_color("#6c6258")

    legend_handles = [
        plt.Line2D([0], [0], color="#6ea9a1", linewidth=2.0, alpha=0.55),
        plt.Line2D([0], [0], color="#b39ddb", linewidth=1.8, alpha=0.8),
        plt.Line2D([0], [0], color="#6a1b9a", linewidth=3.2),
    ]
    legend_labels = [
        f"{len(human_records)} random human runs",
        f"{len(gemini_records)} Gemini-3-Pro runs",
        f"Gemini-3-Pro mean ({float(gemini_mean.iloc[-1]):.2f})",
    ]
    ax.legend(legend_handles, legend_labels, loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_manifest(human_records: list[dict], gemini_records: list[dict]) -> None:
    rows = []
    for record in human_records:
        rows.append(
            {
                "group": "human_random",
                "label": f"{record['player_name']}#{record['local_gameid']}",
                "death": record["death"],
                "turns": record["turns"],
                "score_at_1000": float(record["score_curve"].loc[1000]),
                "score_at_3000": float(record["score_curve"].loc[3000]),
                "progression_at_1000": float(record["progression_curve"].loc[1000]),
                "progression_at_3000": float(record["progression_curve"].loc[3000]),
            }
        )
    for record in gemini_records:
        rows.append(
            {
                "group": "gemini_run",
                "label": record["run_name"],
                "death": "",
                "turns": int(record["score_curve"].index.max()),
                "score_at_1000": float(record["score_curve"].loc[1000]),
                "score_at_3000": float(record["score_curve"].loc[3000]),
                "progression_at_1000": float(record["progression_curve"].loc[1000]),
                "progression_at_3000": float(record["progression_curve"].loc[3000]),
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    achievements = load_achievements()

    human_records = select_random_humans(
        human_random=args.human_random,
        seed=args.seed,
        max_turn=args.max_turn,
        achievements=achievements,
    )
    gemini_records = load_gemini_records(max_turn=args.max_turn, achievements=achievements)

    plot_metric(
        human_records=human_records,
        gemini_records=gemini_records,
        metric_key="score_curve",
        y_label="Status-Line Score `S`",
        title="Gemini-3-Pro vs 25 Random Human Runs (Score, First 3,000 Turns)",
        output_path=OUTPUT_SCORE_PNG,
    )
    plot_metric(
        human_records=human_records,
        gemini_records=gemini_records,
        metric_key="progression_curve",
        y_label="BALROG NLE Progression (%)",
        title="Gemini-3-Pro vs 25 Random Human Runs (Progression, First 3,000 Turns)",
        output_path=OUTPUT_PROGRESS_PNG,
    )
    save_manifest(human_records, gemini_records)

    print(f"Saved score plot to {OUTPUT_SCORE_PNG}")
    print(f"Saved progression plot to {OUTPUT_PROGRESS_PNG}")
    print(f"Saved manifest to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
