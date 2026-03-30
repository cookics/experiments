from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "template" / "data.json"
OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_PNG = OUTPUT_DIR / "nle_trajectories.png"
OUTPUT_PROGRESS_PNG = OUTPUT_DIR / "nle_trajectories_progression.png"
OUTPUT_PROGRESS_ZOOM_PNG = OUTPUT_DIR / "nle_trajectories_progression_zoom_3000_10.png"
OUTPUT_CSV = OUTPUT_DIR / "nle_trajectories_manifest.csv"
ACHIEVEMENTS_PATH = ROOT / "_balrog_src" / "balrog" / "environments" / "nle" / "achievements.json"

ENV_KEYS = ["babaisai", "babyai", "crafter", "textworld", "minihack", "nle"]
DLVL_RE = re.compile(r"Dlvl:(\d+)")
XP_RE = re.compile(r"Xp:(\d+)")


def load_data() -> dict:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_achievements() -> dict[str, float]:
    with ACHIEVEMENTS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_nle_csvs(folder: Path) -> list[Path]:
    nle_dir = folder / "nle"
    if not nle_dir.exists():
        return []
    return sorted(nle_dir.rglob("*.csv"))


def load_run_curve(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, usecols=["Step", "Reward"])
    df = df.sort_values("Step")
    df["Reward"] = df["Reward"].fillna(0)
    cumulative_reward = df["Reward"].cumsum()
    steps = df["Step"].astype(int)
    return pd.Series(cumulative_reward.to_numpy(), index=steps.to_numpy(), name=csv_path.name)


def load_progression_curve(csv_path: Path, achievements: dict[str, float]) -> pd.Series:
    df = pd.read_csv(csv_path, usecols=["Step", "Observation"])
    df = df.sort_values("Step")

    progression_values = []
    current_progression = 0.0

    for observation in df["Observation"].fillna(""):
        dlvl_match = DLVL_RE.search(observation)
        xp_match = XP_RE.search(observation)

        candidates = [current_progression]

        if dlvl_match:
            candidates.append(achievements.get(f"Dlvl:{int(dlvl_match.group(1))}", current_progression))
        if xp_match:
            candidates.append(achievements.get(f"Xp:{int(xp_match.group(1))}", current_progression))

        current_progression = max(candidates)
        progression_values.append(current_progression * 100.0)

    steps = df["Step"].astype(int)
    return pd.Series(progression_values, index=steps.to_numpy(), name=csv_path.name)


def infer_lowest_environment(result: dict) -> str | None:
    scored_envs = {
        env: result[env][0]
        for env in ENV_KEYS
        if env in result and isinstance(result[env], list) and result[env]
    }
    if not scored_envs:
        return None
    return min(scored_envs, key=scored_envs.get)


def build_model_records(data: dict, achievements: dict[str, float]) -> list[dict]:
    records: list[dict] = []

    for leaderboard in data.get("leaderboards", []):
        leaderboard_name = leaderboard["name"]
        for result in leaderboard.get("results", []):
            if result.get("trajs") is not True:
                continue

            folder = ROOT / Path(result["folder"])
            csv_paths = find_nle_csvs(folder)
            if not csv_paths:
                continue

            run_curves = [load_run_curve(path) for path in csv_paths]
            progression_curves = [load_progression_curve(path, achievements) for path in csv_paths]
            max_step = max(int(curve.index.max()) for curve in run_curves)

            records.append(
                {
                    "leaderboard": leaderboard_name,
                    "model_name": result["name"],
                    "label": f"{result['name']} [{leaderboard_name}]",
                    "date": result.get("date", ""),
                    "folder": str(folder.relative_to(ROOT)),
                    "nle_progress": result.get("nle", [None])[0],
                    "lowest_environment": infer_lowest_environment(result),
                    "nle_is_lowest": infer_lowest_environment(result) == "nle",
                    "csv_paths": csv_paths,
                    "run_curves": run_curves,
                    "progression_curves": progression_curves,
                    "run_count": len(run_curves),
                    "max_step": max_step,
                }
            )

    return records


def average_curve(curves: list[pd.Series], global_steps: pd.Index) -> pd.Series:
    aligned = []
    for curve in curves:
        aligned_curve = curve.reindex(global_steps).ffill().fillna(0.0)
        aligned.append(aligned_curve)
    return pd.concat(aligned, axis=1).mean(axis=1)


def build_ranked_progression_series(
    records: list[dict], global_steps: pd.Index | None = None, max_step: int | None = None
) -> list[tuple[dict, pd.Series]]:
    ranked = []
    for record in records:
        steps = (
            global_steps
            if global_steps is not None
            else pd.Index(range(min(record["max_step"], max_step) + 1), name="Step")
        )
        mean_curve = average_curve(record["progression_curves"], steps)
        ranked.append((record, mean_curve))

    ranked.sort(
        key=lambda item: (float(item[1].iloc[-1]), item[0]["model_name"].lower()),
        reverse=True,
    )
    return ranked


def prepare_plot_context(records: list[dict]) -> tuple[pd.Index, int]:
    global_max_step = max(record["max_step"] for record in records)
    global_steps = pd.Index(range(global_max_step + 1), name="Step")
    return global_steps, global_max_step


def plot_reward_records(records: list[dict], global_steps: pd.Index, global_max_step: int) -> list[dict]:
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(18, 10))

    manifest_rows = []

    for idx, record in enumerate(sorted(records, key=lambda item: (item["leaderboard"], item["model_name"].lower()))):
        mean_curve = average_curve(record["run_curves"], global_steps)
        color = cmap(idx % 20)
        linestyle = "-" if record["leaderboard"] == "LLM" else "--"

        ax.plot(
            global_steps,
            mean_curve,
            label=record["label"],
            color=color,
            linewidth=2,
            linestyle=linestyle,
            alpha=0.95,
        )

        manifest_rows.append(
            {
                "leaderboard": record["leaderboard"],
                "model_name": record["model_name"],
                "date": record["date"],
                "nle_progress_percent": record["nle_progress"],
                "lowest_environment": record["lowest_environment"],
                "nle_is_lowest": record["nle_is_lowest"],
                "run_count": record["run_count"],
                "max_step": record["max_step"],
                "final_mean_cumulative_reward": float(mean_curve.iloc[-1]),
                "folder": record["folder"],
            }
        )

    ax.set_title("NLE Trajectories for Models with Trajectory Logs")
    ax.set_xlabel("Environment Step (CSV `Step` column)")
    ax.set_ylabel("Mean Cumulative Reward (cumsum of CSV `Reward`)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, global_max_step)
    ax.set_ylim(bottom=0)

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        ncol=1,
    )
    for line in legend.get_lines():
        line.set_linewidth(3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return manifest_rows


def plot_progression_records(records: list[dict], global_steps: pd.Index, global_max_step: int) -> None:
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(18, 10))

    ranked = build_ranked_progression_series(records, global_steps=global_steps)

    for idx, (record, mean_curve) in enumerate(ranked):
        color = cmap(idx % 20)
        linestyle = "-" if record["leaderboard"] == "LLM" else "--"

        ax.plot(
            global_steps,
            mean_curve,
            label=record["label"],
            color=color,
            linewidth=2,
            linestyle=linestyle,
            alpha=0.95,
        )

    ax.set_title("NLE Trajectories in BALROG Progression Percentage")
    ax.set_xlabel("Environment Step (CSV `Step` column)")
    ax.set_ylabel("Mean BALROG NLE Progression (%)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, global_max_step)
    ax.set_ylim(0, 100)

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        ncol=1,
    )
    for line in legend.get_lines():
        line.set_linewidth(3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PROGRESS_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_progression_records_zoomed(records: list[dict]) -> None:
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(18, 10))

    fig.patch.set_facecolor("#f7f1e8")
    ax.set_facecolor("#fffaf3")

    ranked = build_ranked_progression_series(records, max_step=3000)

    for idx, (record, mean_curve) in enumerate(ranked):
        color = cmap(idx % 20)
        linestyle = "-" if record["leaderboard"] == "LLM" else "--"

        ax.plot(
            mean_curve.index,
            mean_curve,
            label=record["label"],
            color=color,
            linewidth=2.4,
            linestyle=linestyle,
            alpha=0.98,
        )

    ax.set_title("NLE Progression Zoomed In (First 3,000 Steps)", fontsize=18, pad=12)
    ax.set_xlabel("Environment Step", fontsize=13)
    ax.set_ylabel("Mean BALROG NLE Progression (%)", fontsize=13)
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.22, linewidth=0.9, color="#8f7f6d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6c6258")
    ax.spines["bottom"].set_color("#6c6258")

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        ncol=1,
    )
    for line in legend.get_lines():
        line.set_linewidth(3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PROGRESS_ZOOM_PNG, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_manifest(manifest_rows: list[dict]) -> None:
    
    pd.DataFrame(manifest_rows).sort_values(
        ["leaderboard", "nle_progress_percent", "model_name"],
        ascending=[True, True, True],
    ).to_csv(OUTPUT_CSV, index=False)

    print(f"Saved plot to {OUTPUT_PNG}")
    print(f"Saved BALROG progression plot to {OUTPUT_PROGRESS_PNG}")
    print(f"Saved zoomed BALROG progression plot to {OUTPUT_PROGRESS_ZOOM_PNG}")
    print(f"Saved manifest to {OUTPUT_CSV}")
    print(f"Plotted {len(manifest_rows)} models on a shared x-axis.")


def main() -> None:
    data = load_data()
    achievements = load_achievements()
    records = build_model_records(data, achievements)
    if not records:
        raise SystemExit("No models with NLE trajectory CSVs were found.")
    OUTPUT_DIR.mkdir(exist_ok=True)
    global_steps, global_max_step = prepare_plot_context(records)
    manifest_rows = plot_reward_records(records, global_steps, global_max_step)
    plot_progression_records(records, global_steps, global_max_step)
    plot_progression_records_zoomed(records)
    save_manifest(manifest_rows)


if __name__ == "__main__":
    main()
