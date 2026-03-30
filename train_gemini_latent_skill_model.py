from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from plot_nle_trajectories import find_nle_csvs


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs"
DB_PATH = OUTPUT_DIR / "human_nao_full_trajectories.sqlite3"
GEMINI_FOLDER = ROOT / "submissions" / "LLM" / "20260203_naive_gemini-3-pro"
HORIZON = 1000
RNG_SEED = 0

METRICS_JSON = OUTPUT_DIR / "gemini_latent_skill_model_metrics.json"
GEMINI_CSV = OUTPUT_DIR / "gemini_latent_skill_percentiles_1000.csv"
IMPORTANCE_CSV = OUTPUT_DIR / "gemini_latent_skill_feature_importance.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "human_latent_skill_predictions_1000.csv"
GEMINI_BAR_PNG = OUTPUT_DIR / "gemini_latent_skill_percentiles_1000.png"
PRED_VS_TRUE_PNG = OUTPUT_DIR / "human_latent_skill_pred_vs_true_1000.png"

TURN_RE = re.compile(r"\bT:(\d+)")
SCORE_RE = re.compile(r"\bS:(\d+)")
DLVL_RE = re.compile(r"Dlvl:(\d+)")
XP_RE = re.compile(r"(?:Xp|Exp):(\d+)")
HOME_RE = re.compile(r"\bHome ([1-5])\b")


def outcome_bonus(death: str) -> float:
    text = str(death).lower()
    if "ascended" in text:
        return 1.0
    if "escaped" in text:
        return 0.5
    return 0.0


def rank_pct(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average")


def make_skill_target(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    games["score_pct"] = rank_pct(np.log1p(games["points"].astype(float)))
    games["depth_pct"] = rank_pct(games["maxlvl"].astype(float))
    games["turns_pct"] = rank_pct(np.log1p(games["turns"].astype(float)))
    games["outcome_bonus"] = games["death"].map(outcome_bonus).astype(float)
    games["skill_raw"] = (
        0.40 * games["depth_pct"]
        + 0.35 * games["score_pct"]
        + 0.15 * games["turns_pct"]
        + 0.10 * games["outcome_bonus"]
    )
    games["skill_percentile"] = rank_pct(games["skill_raw"]) * 100.0
    return games


def load_human_data(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(str(db_path))
    try:
        games = pd.read_sql_query(
            """
            SELECT
                game_key,
                player_name,
                death,
                turns,
                points,
                maxlvl,
                final_score,
                final_progression_pct
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
                dlvl,
                xp,
                home_level
            FROM events
            ORDER BY game_key, turn, event_idx
            """,
            conn,
        )
    finally:
        conn.close()
    return games, events


def find_bad_games(games: pd.DataFrame, events: pd.DataFrame) -> set[str]:
    event_max = events.groupby("game_key", sort=False).agg(
        max_event_dlvl=("dlvl", "max"),
    )
    merged = event_max.merge(games[["game_key", "maxlvl"]], on="game_key", how="left")
    bad_dlvl = set(merged.loc[merged["max_event_dlvl"] > merged["maxlvl"], "game_key"].astype(str))
    return bad_dlvl


def integrate_auc(turns: np.ndarray, values: np.ndarray, horizon: int) -> float:
    total = 0.0
    prev_turn = 0
    prev_val = 0.0
    for turn, value in zip(turns, values):
        turn_i = int(turn)
        value_f = float(value)
        if turn_i > horizon:
            break
        total += (turn_i - prev_turn) * prev_val
        prev_turn = turn_i
        prev_val = value_f
    total += (horizon - prev_turn) * prev_val
    return total / float(max(horizon, 1))


def first_reach(value_turns: list[tuple[int, float]], threshold: float) -> float:
    for turn, value in value_turns:
        if value >= threshold:
            return float(turn)
    return np.nan


def extract_sparse_features(
    *,
    turns: np.ndarray,
    scores: np.ndarray,
    dlvls: np.ndarray,
    xps: np.ndarray,
    homes: np.ndarray,
    last_turn: int,
    horizon: int,
) -> dict[str, float]:
    checkpoints = [25] + list(range(50, horizon + 1, 50))
    dlvl_thresholds = [2, 4, 6, 8, 10, 12, 15, 20, 30]
    xp_thresholds = [2, 4, 6, 8, 10, 12, 15]
    score_thresholds = [50, 100, 200, 500, 1000, 5000, 10000]

    values_by_metric = {
        "score": scores.astype(float),
        "dlvl": dlvls.astype(float),
        "xp": xps.astype(float),
        "home": homes.astype(float),
    }

    feature_row: dict[str, float] = {}
    observed_turns = float(min(max(int(last_turn), 0), horizon))
    feature_row["observed_turns"] = observed_turns
    feature_row["ended_before_horizon"] = 1.0 if int(last_turn) < horizon else 0.0
    feature_row["event_count_early"] = float(np.sum(turns <= horizon))

    turn_score_pairs = [(int(t), float(v)) for t, v in zip(turns, scores) if int(t) <= horizon]
    turn_dlvl_pairs = [(int(t), float(v)) for t, v in zip(turns, dlvls) if int(t) <= horizon]
    turn_xp_pairs = [(int(t), float(v)) for t, v in zip(turns, xps) if int(t) <= horizon]

    for metric_name, values in values_by_metric.items():
        cursor = 0
        current = 0.0
        sampled: list[float] = []
        for checkpoint in checkpoints:
            while cursor < len(turns) and int(turns[cursor]) <= checkpoint:
                current = float(values[cursor])
                cursor += 1
            sampled.append(current)
            feature_row[f"{metric_name}_t{checkpoint}"] = current

        feature_row[f"{metric_name}_final"] = sampled[-1]
        feature_row[f"{metric_name}_auc"] = integrate_auc(turns, values, horizon)
        feature_row[f"{metric_name}_rate"] = sampled[-1] / max(observed_turns, 1.0)

    for threshold in dlvl_thresholds:
        feature_row[f"first_dlvl_{threshold}"] = first_reach(turn_dlvl_pairs, float(threshold))
    for threshold in xp_thresholds:
        feature_row[f"first_xp_{threshold}"] = first_reach(turn_xp_pairs, float(threshold))
    for threshold in score_thresholds:
        feature_row[f"first_score_{threshold}"] = first_reach(turn_score_pairs, float(threshold))

    feature_row["log_score_final"] = math.log1p(max(feature_row["score_final"], 0.0))
    feature_row["depth_plus_xp_final"] = feature_row["dlvl_final"] + feature_row["xp_final"]
    return feature_row


def build_human_feature_table(games: pd.DataFrame, events: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    grouped = events.groupby("game_key", sort=False)
    available = set(events["game_key"].astype(str).unique().tolist())

    for game in games.itertuples(index=False):
        if str(game.game_key) in available:
            sub = grouped.get_group(game.game_key)
            early = sub[sub["turn"] <= horizon]
            turns = early["turn"].to_numpy(dtype=np.int32)
            scores = early["score"].to_numpy(dtype=np.float64)
            dlvls = early["dlvl"].to_numpy(dtype=np.int32)
            xps = early["xp"].to_numpy(dtype=np.int32)
            homes = early["home_level"].to_numpy(dtype=np.int32)
        else:
            turns = np.array([], dtype=np.int32)
            scores = np.array([], dtype=np.float64)
            dlvls = np.array([], dtype=np.int32)
            xps = np.array([], dtype=np.int32)
            homes = np.array([], dtype=np.int32)

        features = extract_sparse_features(
            turns=turns,
            scores=scores,
            dlvls=dlvls,
            xps=xps,
            homes=homes,
            last_turn=int(game.turns),
            horizon=horizon,
        )
        features["game_key"] = str(game.game_key)
        rows.append(features)

    return pd.DataFrame(rows)


def parse_gemini_sparse_events(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    df = pd.read_csv(csv_path, usecols=["Observation"])

    turns: list[int] = []
    scores: list[float] = []
    dlvls: list[int] = []
    xps: list[int] = []
    homes: list[int] = []

    last_turn = -1
    score = 0.0
    dlvl = 0
    xp = 0
    home = 0

    for observation in df["Observation"].fillna(""):
        text = str(observation)
        turn_match = TURN_RE.search(text)
        if not turn_match:
            continue
        turn = int(turn_match.group(1))
        if turn < last_turn:
            continue

        score_match = SCORE_RE.search(text)
        if score_match:
            score = max(score, float(score_match.group(1)))
        dlvl_match = DLVL_RE.search(text)
        if dlvl_match:
            dlvl = max(dlvl, int(dlvl_match.group(1)))
        xp_match = XP_RE.search(text)
        if xp_match:
            xp = max(xp, int(xp_match.group(1)))
        home_match = HOME_RE.search(text)
        if home_match:
            home = max(home, int(home_match.group(1)))

        last_turn = turn
        turns.append(turn)
        scores.append(score)
        dlvls.append(dlvl)
        xps.append(xp)
        homes.append(home)

    if last_turn < 0:
        last_turn = 0

    return (
        np.array(turns, dtype=np.int32),
        np.array(scores, dtype=np.float64),
        np.array(dlvls, dtype=np.int32),
        np.array(xps, dtype=np.int32),
        np.array(homes, dtype=np.int32),
        int(last_turn),
    )


def build_gemini_feature_table(horizon: int) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for csv_path in sorted(find_nle_csvs(GEMINI_FOLDER), key=lambda path: path.stem):
        turns, scores, dlvls, xps, homes, last_turn = parse_gemini_sparse_events(csv_path)
        features = extract_sparse_features(
            turns=turns[turns <= horizon],
            scores=scores[turns <= horizon],
            dlvls=dlvls[turns <= horizon],
            xps=xps[turns <= horizon],
            homes=homes[turns <= horizon],
            last_turn=last_turn,
            horizon=horizon,
        )
        features["run_name"] = csv_path.stem
        features["source_csv"] = str(csv_path)
        rows.append(features)
    return pd.DataFrame(rows)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_rank = pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)
    b_rank = pd.Series(b).rank(method="average").to_numpy(dtype=np.float64)
    return float(np.corrcoef(a_rank, b_rank)[0, 1])


def fit_model(X_train: pd.DataFrame, y_train: np.ndarray, X_valid: pd.DataFrame, y_valid: np.ndarray) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RNG_SEED,
        early_stopping_rounds=50,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    return model


def plot_pred_vs_true(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#f4efe8")
    ax.set_facecolor("#fffaf3")
    ax.scatter(df["true_skill_pct"], df["pred_skill_pct"], s=12, alpha=0.35, color="#355070", edgecolors="none")
    ax.plot([0, 100], [0, 100], linestyle="--", linewidth=1.5, color="#b23a48")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("True latent skill percentile")
    ax.set_ylabel("Predicted from first 1000 turns")
    ax.set_title("Human latent skill model: held-out prediction accuracy")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_gemini_bars(gemini_df: pd.DataFrame, output_path: Path) -> None:
    ordered = gemini_df.sort_values("pred_skill_pct", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#f4efe8")
    ax.set_facecolor("#fffaf3")
    colors = ["#355070", "#6d597a", "#b56576", "#457b9d", "#2a9d8f"]
    ax.bar(
        ordered["run_name"],
        ordered["pred_skill_pct"],
        color=colors[: len(ordered)],
        edgecolor="#1f1f1f",
        linewidth=0.6,
    )
    ax.set_ylabel("Predicted latent skill percentile")
    ax.set_xlabel("Gemini-3-Pro run")
    ax.set_title("Gemini-3-Pro runs on the human latent-skill scale")
    ax.set_ylim(0, max(100.0, float(ordered["pred_skill_pct"].max()) * 1.12))
    ax.grid(True, axis="y", alpha=0.15)
    for idx, row in ordered.iterrows():
        ax.text(idx, float(row["pred_skill_pct"]) + 1.0, f"{row['pred_skill_pct']:.1f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    games, events = load_human_data(DB_PATH)
    bad_games = find_bad_games(games, events)
    clean_games = games[~games["game_key"].astype(str).isin(bad_games)].copy()
    clean_events = events[~events["game_key"].astype(str).isin(bad_games)].copy()

    clean_games = make_skill_target(clean_games)
    human_features = build_human_feature_table(clean_games, clean_events, HORIZON)
    human_df = clean_games.merge(human_features, on="game_key", how="inner")

    feature_cols = [col for col in human_df.columns if col not in {
        "game_key",
        "player_name",
        "death",
        "turns",
        "points",
        "maxlvl",
        "final_score",
        "score_pct",
        "depth_pct",
        "turns_pct",
        "outcome_bonus",
        "skill_raw",
        "skill_percentile",
    }]

    unique_games = human_df["game_key"].to_numpy()
    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(len(unique_games))
    split = int(len(unique_games) * 0.8)
    train_idx = perm[:split]
    test_idx = perm[split:]

    X = human_df[feature_cols]
    y = human_df["skill_percentile"].to_numpy(dtype=np.float64)
    X_train = X.iloc[train_idx]
    y_train = y[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y[test_idx]

    model = fit_model(X_train, y_train, X_test, y_test)
    test_pred = np.clip(model.predict(X_test), 0.0, 100.0)
    full_pred = np.clip(model.predict(X), 0.0, 100.0)

    heldout_df = human_df.iloc[test_idx][["game_key", "skill_percentile"]].copy()
    heldout_df["pred_skill_pct"] = test_pred
    heldout_df = heldout_df.rename(columns={"skill_percentile": "true_skill_pct"})

    baseline_score = X_test["log_score_final"].to_numpy(dtype=np.float64)
    baseline_depth = X_test["dlvl_final"].to_numpy(dtype=np.float64)

    metrics = {
        "horizon": HORIZON,
        "games_total": int(len(games)),
        "games_filtered_out": int(len(bad_games)),
        "games_used": int(len(human_df)),
        "train_games": int(len(train_idx)),
        "test_games": int(len(test_idx)),
        "target_definition": {
            "depth_pct_weight": 0.40,
            "score_pct_weight": 0.35,
            "turns_pct_weight": 0.15,
            "outcome_bonus_weight": 0.10,
            "outcome_bonus": {"ascended": 1.0, "escaped": 0.5, "else": 0.0},
        },
        "test_metrics": {
            "rmse_pct_points": float(mean_squared_error(y_test, test_pred) ** 0.5),
            "mae_pct_points": float(mean_absolute_error(y_test, test_pred)),
            "r2": float(r2_score(y_test, test_pred)),
            "spearman": float(spearman_corr(y_test, test_pred)),
        },
        "baselines": {
            "log_score_final_spearman": float(spearman_corr(y_test, baseline_score)),
            "dlvl_final_spearman": float(spearman_corr(y_test, baseline_depth)),
        },
        "best_iteration": int(getattr(model, "best_iteration", model.n_estimators)),
    }

    with METRICS_JSON.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": model.feature_importances_,
        }
    ).sort_values("importance_gain", ascending=False)
    importance.to_csv(IMPORTANCE_CSV, index=False)

    human_predictions = human_df[["game_key", "player_name", "skill_percentile"]].copy()
    human_predictions["pred_skill_pct"] = full_pred
    human_predictions.to_csv(PREDICTIONS_CSV, index=False)

    gemini_features = build_gemini_feature_table(HORIZON)
    gemini_pred = np.clip(model.predict(gemini_features[feature_cols]), 0.0, 100.0)
    gemini_df = gemini_features[["run_name", "source_csv", "observed_turns", "ended_before_horizon", "score_final", "dlvl_final", "xp_final", "home_final"]].copy()
    gemini_df["pred_skill_pct"] = gemini_pred
    human_pred_sorted = np.sort(full_pred)
    gemini_df["pred_pct_among_human_preds"] = [
        float(np.searchsorted(human_pred_sorted, val, side="right") / len(human_pred_sorted) * 100.0)
        for val in gemini_pred
    ]
    gemini_df.to_csv(GEMINI_CSV, index=False)

    plot_pred_vs_true(heldout_df, PRED_VS_TRUE_PNG)
    plot_gemini_bars(gemini_df, GEMINI_BAR_PNG)

    print(
        json.dumps(
            {
                "metrics_json": str(METRICS_JSON),
                "gemini_csv": str(GEMINI_CSV),
                "importance_csv": str(IMPORTANCE_CSV),
                "predictions_csv": str(PREDICTIONS_CSV),
                "pred_vs_true_png": str(PRED_VS_TRUE_PNG),
                "gemini_bar_png": str(GEMINI_BAR_PNG),
                "gemini_runs": gemini_df[["run_name", "pred_skill_pct", "pred_pct_among_human_preds"]].to_dict(orient="records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
