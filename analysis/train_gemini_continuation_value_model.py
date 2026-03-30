from __future__ import annotations

import bz2
import json
import math
import os
import struct
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

from analysis.plot_human_nao_random_plus_best import build_game_index
from analysis.plot_nle_trajectories import find_nle_csvs
from analysis.train_gemini_latent_skill_model import (
    DB_PATH,
    find_bad_games,
    load_human_data,
    make_skill_target,
    spearman_corr,
)


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "analysis_outputs"
CHECKPOINT_MAX_TURN = 1000
CHECKPOINT_STEP = 50
CHECKPOINTS = list(range(0, CHECKPOINT_MAX_TURN + 1, CHECKPOINT_STEP))
RNG_SEED = 0
HUMAN_SOURCE_BASE = ROOT / "nld-nao"
GEMINI_FOLDER = ROOT / "submissions" / "LLM" / "20260203_naive_gemini-3-pro"
HUMAN_CHECKPOINT_CACHE = OUTPUT_DIR / "human_rich_checkpoints_1000.pkl.gz"

METRICS_JSON = OUTPUT_DIR / "gemini_continuation_model_metrics.json"
CURVES_CSV = OUTPUT_DIR / "gemini_continuation_curves_1000.csv"
HUMAN_BANDS_CSV = OUTPUT_DIR / "human_continuation_bands_1000.csv"
FEATURE_IMPORTANCE_CSV = OUTPUT_DIR / "gemini_continuation_feature_importance.csv"
CURVES_PNG = OUTPUT_DIR / "gemini_continuation_curves_1000.png"
SCATTER_PNG = OUTPUT_DIR / "human_continuation_pred_vs_true_t1000.png"

TURN_TOKEN = b"T:"
ST_TOKEN = b"St:"
DX_TOKEN = b"Dx:"
CO_TOKEN = b"Co:"
IN_TOKEN = b"In:"
WI_TOKEN = b"Wi:"
CH_TOKEN = b"Ch:"
SCORE_TOKEN = b"S:"
DLVL_TOKEN = b"Dlvl:"
GOLD_TOKEN = b"$:"
HP_TOKEN = b"HP:"
PW_TOKEN = b"Pw:"
AC_TOKEN = b"AC:"
XP_TOKEN = b"Xp:"
EXP_TOKEN = b"Exp:"
HOME_TOKEN = b"Home "


def _find_token(data: bytes, token: bytes) -> int:
    return data.rfind(token)


def _parse_int_after_token(data: bytes, token: bytes) -> int | None:
    idx = _find_token(data, token)
    if idx == -1:
        return None
    pos = idx + len(token)
    sign = 1
    if pos < len(data) and data[pos] == 45:
        sign = -1
        pos += 1
    start = pos
    while pos < len(data) and 48 <= data[pos] <= 57:
        pos += 1
    if pos == start:
        return None
    return sign * int(data[start:pos])


def _parse_strength_after_token(data: bytes, token: bytes) -> float | None:
    idx = _find_token(data, token)
    if idx == -1:
        return None
    pos = idx + len(token)
    start = pos
    while pos < len(data) and (
        48 <= data[pos] <= 57 or data[pos] == 47 or data[pos] == 42
    ):
        pos += 1
    if pos == start:
        return None
    return parse_strength(data[start:pos])


def _parse_pair_after_token(data: bytes, token: bytes) -> tuple[int, int] | None:
    idx = _find_token(data, token)
    if idx == -1:
        return None
    pos = idx + len(token)
    start_first = pos
    while pos < len(data) and 48 <= data[pos] <= 57:
        pos += 1
    if pos == start_first:
        return None
    first = int(data[start_first:pos])
    if pos >= len(data) or data[pos] != 40:
        return None
    pos += 1
    start_second = pos
    while pos < len(data) and 48 <= data[pos] <= 57:
        pos += 1
    if pos == start_second:
        return None
    second = int(data[start_second:pos])
    return first, second


def _parse_xp_after_tokens(data: bytes) -> tuple[int, int | None] | None:
    idx = _find_token(data, XP_TOKEN)
    token = XP_TOKEN
    if idx == -1:
        idx = _find_token(data, EXP_TOKEN)
        token = EXP_TOKEN
    if idx == -1:
        return None
    pos = idx + len(token)
    start_level = pos
    while pos < len(data) and 48 <= data[pos] <= 57:
        pos += 1
    if pos == start_level:
        return None
    level = int(data[start_level:pos])
    exp_pts: int | None = None
    if pos < len(data) and data[pos] == 47:
        pos += 1
        start_exp = pos
        while pos < len(data) and 48 <= data[pos] <= 57:
            pos += 1
        if pos > start_exp:
            exp_pts = int(data[start_exp:pos])
    return level, exp_pts


def parse_strength(token: bytes | str) -> float:
    value = token.decode("ascii", errors="ignore") if isinstance(token, bytes) else str(token)
    if "/" not in value:
        try:
            return float(int(value))
        except ValueError:
            return np.nan
    base_str, bonus_str = value.split("/", 1)
    try:
        base = float(int(base_str))
    except ValueError:
        return np.nan
    if bonus_str == "**":
        bonus = 99
    else:
        digits = "".join(ch for ch in bonus_str if ch.isdigit())
        if not digits:
            bonus = 0
        else:
            bonus = int(digits)
    return base + bonus / 100.0


def snapshot_state(game_key: str, checkpoint: int, observed_turn: int, ended_before_checkpoint: int, state: dict[str, float]) -> tuple:
    return (
        game_key,
        checkpoint,
        observed_turn,
        ended_before_checkpoint,
        state["score"],
        state["best_score"],
        state["gold"],
        state["best_gold"],
        state["hp_cur"],
        state["hp_max"],
        state["pw_cur"],
        state["pw_max"],
        state["ac"],
        state["dlvl_cur"],
        state["best_dlvl"],
        state["xl_cur"],
        state["best_xl"],
        state["exp_pts"],
        state["home_cur"],
        state["best_home"],
        state["st"],
        state["dx"],
        state["co"],
        state["int_"],
        state["wi"],
        state["ch"],
    )


def initial_state() -> dict[str, float]:
    return {
        "score": 0.0,
        "best_score": 0.0,
        "gold": 0.0,
        "best_gold": 0.0,
        "hp_cur": np.nan,
        "hp_max": np.nan,
        "pw_cur": np.nan,
        "pw_max": np.nan,
        "ac": np.nan,
        "dlvl_cur": 0.0,
        "best_dlvl": 0.0,
        "xl_cur": 0.0,
        "best_xl": 0.0,
        "exp_pts": 0.0,
        "home_cur": 0.0,
        "best_home": 0.0,
        "st": np.nan,
        "dx": np.nan,
        "co": np.nan,
        "int_": np.nan,
        "wi": np.nan,
        "ch": np.nan,
    }


def update_state_from_bytes(data: bytes, state: dict[str, float]) -> int | None:
    turn = _parse_int_after_token(data, TURN_TOKEN)
    if turn is None:
        return None

    st_value = _parse_strength_after_token(data, ST_TOKEN)
    if st_value is not None:
        state["st"] = st_value
    dx_value = _parse_int_after_token(data, DX_TOKEN)
    if dx_value is not None:
        state["dx"] = float(dx_value)
    co_value = _parse_int_after_token(data, CO_TOKEN)
    if co_value is not None:
        state["co"] = float(co_value)
    int_value = _parse_int_after_token(data, IN_TOKEN)
    if int_value is not None:
        state["int_"] = float(int_value)
    wi_value = _parse_int_after_token(data, WI_TOKEN)
    if wi_value is not None:
        state["wi"] = float(wi_value)
    ch_value = _parse_int_after_token(data, CH_TOKEN)
    if ch_value is not None:
        state["ch"] = float(ch_value)

    score_value = _parse_int_after_token(data, SCORE_TOKEN)
    if score_value is not None:
        state["score"] = float(score_value)
        state["best_score"] = max(state["best_score"], state["score"])

    gold_value = _parse_int_after_token(data, GOLD_TOKEN)
    if gold_value is not None:
        state["gold"] = float(gold_value)
        state["best_gold"] = max(state["best_gold"], state["gold"])

    hp_pair = _parse_pair_after_token(data, HP_TOKEN)
    if hp_pair is not None:
        state["hp_cur"] = float(hp_pair[0])
        state["hp_max"] = float(hp_pair[1])

    pw_pair = _parse_pair_after_token(data, PW_TOKEN)
    if pw_pair is not None:
        state["pw_cur"] = float(pw_pair[0])
        state["pw_max"] = float(pw_pair[1])

    ac_value = _parse_int_after_token(data, AC_TOKEN)
    if ac_value is not None:
        state["ac"] = float(ac_value)

    dlvl_value = _parse_int_after_token(data, DLVL_TOKEN)
    if dlvl_value is not None:
        state["dlvl_cur"] = float(dlvl_value)
        state["best_dlvl"] = max(state["best_dlvl"], state["dlvl_cur"])

    xp_pair = _parse_xp_after_tokens(data)
    if xp_pair is not None:
        state["xl_cur"] = float(xp_pair[0])
        state["best_xl"] = max(state["best_xl"], state["xl_cur"])
        if xp_pair[1] is not None:
            state["exp_pts"] = float(xp_pair[1])

    home_value = _parse_int_after_token(data, HOME_TOKEN)
    if home_value is not None:
        state["home_cur"] = float(home_value)
        state["best_home"] = max(state["best_home"], state["home_cur"])

    return turn


def parse_game_checkpoint_rows(game: dict[str, object]) -> list[tuple]:
    game_key = f"{game['player_name']}#{int(game['local_gameid'])}"
    rows: list[tuple] = []
    state = initial_state()
    last_turn = 0
    checkpoint_idx = 0
    max_turn = min(int(game["turns"]), CHECKPOINT_MAX_TURN)

    while checkpoint_idx < len(CHECKPOINTS) and CHECKPOINTS[checkpoint_idx] == 0:
        rows.append(snapshot_state(game_key, 0, 0, 0, state))
        checkpoint_idx += 1

    for member_name in list(game["members"]):
        member_path = HUMAN_SOURCE_BASE / Path(member_name)
        with member_path.open("rb") as raw_member:
            tty = bz2.BZ2File(raw_member)
            while True:
                header = tty.read(12)
                if not header:
                    break
                sec, usec, length = struct.unpack("<iii", header)
                if sec < 0 or usec < 0 or length < 1:
                    break
                data = tty.read(length)
                if not data or TURN_TOKEN not in data:
                    continue

                turn = update_state_from_bytes(data, state)
                if turn is None:
                    continue
                if turn < last_turn:
                    continue
                if turn > int(game["turns"]):
                    continue

                last_turn = turn
                while checkpoint_idx < len(CHECKPOINTS) and turn >= CHECKPOINTS[checkpoint_idx]:
                    checkpoint = CHECKPOINTS[checkpoint_idx]
                    rows.append(snapshot_state(game_key, checkpoint, turn, 0, state))
                    checkpoint_idx += 1

                if turn >= max_turn:
                    break
        if last_turn >= max_turn:
            break

    while checkpoint_idx < len(CHECKPOINTS):
        checkpoint = CHECKPOINTS[checkpoint_idx]
        rows.append(
            snapshot_state(
                game_key,
                checkpoint,
                last_turn,
                1 if last_turn < checkpoint else 0,
                state,
            )
        )
        checkpoint_idx += 1

    return rows


def parse_game_chunk(games_chunk: list[dict[str, object]]) -> list[tuple]:
    rows: list[tuple] = []
    for game in games_chunk:
        rows.extend(parse_game_checkpoint_rows(game))
    return rows


def chunked(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_human_checkpoint_table(games_index: list[dict[str, object]], workers: int | None = None, chunk_games: int = 128) -> pd.DataFrame:
    expected_keys = {f"{game['player_name']}#{int(game['local_gameid'])}" for game in games_index}
    if HUMAN_CHECKPOINT_CACHE.exists():
        cached = pd.read_pickle(HUMAN_CHECKPOINT_CACHE)
        cached_keys = set(cached["game_key"].astype(str).unique().tolist())
        if cached_keys == expected_keys and int(cached["checkpoint"].max()) >= CHECKPOINT_MAX_TURN:
            return cached

    games_chunks = list(chunked(games_index, chunk_games))
    rows: list[tuple] = []
    max_workers = workers or max(1, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(parse_game_chunk, chunk) for chunk in games_chunks]
        for future in tqdm(futures, desc="Parsing human checkpoints", unit="chunk"):
            rows.extend(future.result())

    columns = [
        "game_key",
        "checkpoint",
        "observed_turn",
        "ended_before_checkpoint",
        "score",
        "best_score",
        "gold",
        "best_gold",
        "hp_cur",
        "hp_max",
        "pw_cur",
        "pw_max",
        "ac",
        "dlvl_cur",
        "best_dlvl",
        "xl_cur",
        "best_xl",
        "exp_pts",
        "home_cur",
        "best_home",
        "st",
        "dx",
        "co",
        "int_",
        "wi",
        "ch",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_pickle(HUMAN_CHECKPOINT_CACHE, compression="gzip")
    return df


def parse_gemini_checkpoint_rows(csv_path: Path) -> list[tuple]:
    df = pd.read_csv(csv_path, usecols=["Observation"])
    state = initial_state()
    rows: list[tuple] = []
    checkpoint_idx = 0
    last_turn = 0

    while checkpoint_idx < len(CHECKPOINTS) and CHECKPOINTS[checkpoint_idx] == 0:
        rows.append(snapshot_state(csv_path.stem, 0, 0, 0, state))
        checkpoint_idx += 1

    for observation in df["Observation"].fillna(""):
        data = str(observation).encode("utf-8", errors="ignore")
        if TURN_TOKEN not in data:
            continue
        turn = update_state_from_bytes(data, state)
        if turn is None:
            continue
        if turn < last_turn:
            continue
        last_turn = turn
        while checkpoint_idx < len(CHECKPOINTS) and turn >= CHECKPOINTS[checkpoint_idx]:
            checkpoint = CHECKPOINTS[checkpoint_idx]
            rows.append(snapshot_state(csv_path.stem, checkpoint, turn, 0, state))
            checkpoint_idx += 1
        if turn >= CHECKPOINT_MAX_TURN:
            break

    while checkpoint_idx < len(CHECKPOINTS):
        checkpoint = CHECKPOINTS[checkpoint_idx]
        rows.append(
            snapshot_state(
                csv_path.stem,
                checkpoint,
                last_turn,
                1 if last_turn < checkpoint else 0,
                state,
            )
        )
        checkpoint_idx += 1
    return rows


def build_gemini_checkpoint_table() -> pd.DataFrame:
    rows: list[tuple] = []
    for csv_path in sorted(find_nle_csvs(GEMINI_FOLDER), key=lambda path: path.stem):
        rows.extend(parse_gemini_checkpoint_rows(csv_path))
    columns = [
        "run_name",
        "checkpoint",
        "observed_turn",
        "ended_before_checkpoint",
        "score",
        "best_score",
        "gold",
        "best_gold",
        "hp_cur",
        "hp_max",
        "pw_cur",
        "pw_max",
        "ac",
        "dlvl_cur",
        "best_dlvl",
        "xl_cur",
        "best_xl",
        "exp_pts",
        "home_cur",
        "best_home",
        "st",
        "dx",
        "co",
        "int_",
        "wi",
        "ch",
    ]
    return pd.DataFrame(rows, columns=columns)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    checkpoint = out["checkpoint"].astype(float).replace(0.0, 1.0)
    observed_turn = out["observed_turn"].astype(float)

    out["hp_ratio"] = out["hp_cur"] / out["hp_max"]
    out["pw_ratio"] = out["pw_cur"] / out["pw_max"]
    out["score_rate"] = out["best_score"] / checkpoint
    out["gold_rate"] = out["best_gold"] / checkpoint
    out["depth_rate"] = out["best_dlvl"] / checkpoint
    out["xp_rate"] = out["best_xl"] / checkpoint
    out["progress_in_horizon"] = observed_turn / checkpoint
    out["log_best_score"] = np.log1p(np.maximum(out["best_score"], 0.0))
    out["log_best_gold"] = np.log1p(np.maximum(out["best_gold"], 0.0))
    out["hp_missing"] = out["hp_cur"].isna().astype(float)
    out["pw_missing"] = out["pw_cur"].isna().astype(float)
    out["ac_missing"] = out["ac"].isna().astype(float)
    out["stat_missing"] = out[["st", "dx", "co", "int_", "wi", "ch"]].isna().all(axis=1).astype(float)
    out["ac_inverted"] = -out["ac"]
    return out


def fit_quantile_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    alpha: float,
) -> xgb.XGBRegressor:
    objective = "reg:quantileerror" if alpha != 0.5 else "reg:squarederror"
    kwargs = {
        "objective": objective,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": RNG_SEED,
        "early_stopping_rounds": 30,
    }
    if objective == "reg:quantileerror":
        kwargs["quantile_alpha"] = alpha
    model = xgb.XGBRegressor(**kwargs)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model


def plot_continuation_curves(gemini_df: pd.DataFrame, human_band_df: pd.DataFrame, output_path: Path) -> None:
    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor("#f4efe8")
    grid = fig.add_gridspec(2, 1, height_ratios=[3.3, 2.2], hspace=0.14)
    ax = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1, 0], sharex=ax)
    ax.set_facecolor("#fffaf3")
    ax2.set_facecolor("#fffaf3")

    x = human_band_df["checkpoint"].to_numpy(dtype=np.int32)
    ax.fill_between(x, human_band_df["p10"], human_band_df["p90"], color="#9fb3c8", alpha=0.28, label="Human test 10-90")
    ax.fill_between(x, human_band_df["p25"], human_band_df["p75"], color="#7d97b0", alpha=0.32, label="Human test 25-75")
    ax.plot(x, human_band_df["median"], color="#1d3557", linewidth=2.8, label="Human test median")

    palette = ["#355070", "#6d597a", "#b56576", "#457b9d", "#2a9d8f"]
    for color, (run_name, sub) in zip(palette, gemini_df.groupby("run_name", sort=True)):
        ax.fill_between(sub["checkpoint"], sub["pred_p10"], sub["pred_p90"], color=color, alpha=0.12)
        ax.plot(sub["checkpoint"], sub["pred_p50"], color=color, linewidth=2.4, label=run_name)
        ax2.plot(sub["checkpoint"], sub["pct_among_test_humans"], color=color, linewidth=2.4, label=run_name)

    ax.set_title("Gemini continuation value from richer state", fontsize=19, pad=10)
    ax.set_ylabel("Predicted final latent skill percentile")
    ax.grid(True, alpha=0.14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.95)

    ax2.axhline(50.0, color="#8d99ae", linestyle="--", linewidth=1.5)
    ax2.set_ylabel("Gemini percentile among\nhuman test states")
    ax2.set_xlabel("NetHack turn checkpoint")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.14)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_holdout_scatter(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.patch.set_facecolor("#f4efe8")
    ax.set_facecolor("#fffaf3")
    ax.scatter(df["true_skill_pct"], df["pred_p50"], s=16, alpha=0.35, color="#355070", edgecolors="none")
    ax.plot([0, 100], [0, 100], color="#b23a48", linestyle="--", linewidth=1.6)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("True final latent skill percentile")
    ax.set_ylabel("Predicted from richer state at T=1000")
    ax.set_title("Continuation model accuracy on unseen players")
    ax.grid(True, alpha=0.14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def quantiles(series: pd.Series) -> dict[str, float]:
    return {
        "p10": float(series.quantile(0.10)),
        "p25": float(series.quantile(0.25)),
        "median": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    games_store, events_store = load_human_data(DB_PATH)
    coarse_bad_games = find_bad_games(games_store, events_store)
    target_games = make_skill_target(games_store[~games_store["game_key"].astype(str).isin(coarse_bad_games)].copy())

    games_index = build_game_index()
    valid_keys = set(target_games["game_key"].astype(str).tolist())
    games_index = [game for game in games_index if f"{game['player_name']}#{int(game['local_gameid'])}" in valid_keys]

    human_checkpoints = build_human_checkpoint_table(games_index)
    parsed_best = human_checkpoints.groupby("game_key", as_index=False)["best_dlvl"].max()
    parsed_best = parsed_best.merge(target_games[["game_key", "maxlvl"]], on="game_key", how="left")
    parse_bad_games = set(parsed_best.loc[parsed_best["best_dlvl"] > parsed_best["maxlvl"], "game_key"].astype(str))

    target_games = target_games[~target_games["game_key"].astype(str).isin(parse_bad_games)].copy()
    human_checkpoints = human_checkpoints[~human_checkpoints["game_key"].astype(str).isin(parse_bad_games)].copy()

    human = human_checkpoints.merge(
        target_games[["game_key", "player_name", "skill_percentile"]],
        on="game_key",
        how="inner",
    )
    human = add_derived_features(human)

    gemini = add_derived_features(build_gemini_checkpoint_table())

    feature_cols = [
        "checkpoint",
        "observed_turn",
        "ended_before_checkpoint",
        "score",
        "best_score",
        "gold",
        "best_gold",
        "hp_cur",
        "hp_max",
        "pw_cur",
        "pw_max",
        "ac",
        "dlvl_cur",
        "best_dlvl",
        "xl_cur",
        "best_xl",
        "exp_pts",
        "home_cur",
        "best_home",
        "st",
        "dx",
        "co",
        "int_",
        "wi",
        "ch",
        "hp_ratio",
        "pw_ratio",
        "score_rate",
        "gold_rate",
        "depth_rate",
        "xp_rate",
        "progress_in_horizon",
        "log_best_score",
        "log_best_gold",
        "hp_missing",
        "pw_missing",
        "ac_missing",
        "stat_missing",
        "ac_inverted",
    ]

    players = human["player_name"].astype(str).drop_duplicates().to_numpy()
    rng = np.random.default_rng(RNG_SEED)
    rng.shuffle(players)
    cut = int(len(players) * 0.8)
    train_players = set(players[:cut])
    train_mask = human["player_name"].astype(str).isin(train_players).to_numpy()

    X_train = human.loc[train_mask, feature_cols]
    y_train = human.loc[train_mask, "skill_percentile"].to_numpy(dtype=np.float64)
    X_test = human.loc[~train_mask, feature_cols]
    y_test = human.loc[~train_mask, "skill_percentile"].to_numpy(dtype=np.float64)

    model_p10 = fit_quantile_model(X_train, y_train, X_test, y_test, 0.10)
    model_p50 = fit_quantile_model(X_train, y_train, X_test, y_test, 0.50)
    model_p90 = fit_quantile_model(X_train, y_train, X_test, y_test, 0.90)

    test_p10 = np.clip(model_p10.predict(X_test), 0.0, 100.0)
    test_p50 = np.clip(model_p50.predict(X_test), 0.0, 100.0)
    test_p90 = np.clip(model_p90.predict(X_test), 0.0, 100.0)
    stacked = np.sort(np.column_stack([test_p10, test_p50, test_p90]), axis=1)
    test_p10, test_p50, test_p90 = stacked[:, 0], stacked[:, 1], stacked[:, 2]

    human_test = human.loc[~train_mask, ["game_key", "player_name", "checkpoint", "skill_percentile"]].copy()
    human_test["pred_p10"] = test_p10
    human_test["pred_p50"] = test_p50
    human_test["pred_p90"] = test_p90

    test_1000 = human_test[human_test["checkpoint"] == CHECKPOINT_MAX_TURN].copy()
    metrics = {
        "checkpoint_max_turn": CHECKPOINT_MAX_TURN,
        "checkpoint_step": CHECKPOINT_STEP,
        "games_total_store": int(len(games_store)),
        "coarse_bad_games_filtered": int(len(coarse_bad_games)),
        "parse_bad_games_filtered": int(len(parse_bad_games)),
        "games_used": int(target_games["game_key"].nunique()),
        "rows_used": int(len(human)),
        "players_total": int(len(players)),
        "players_train": int(len(train_players)),
        "players_test": int(len(players) - len(train_players)),
        "all_test_rows": {
            "mae": float(np.mean(np.abs(y_test - test_p50))),
            "rmse": float(np.sqrt(np.mean((y_test - test_p50) ** 2))),
            "r2": float(1.0 - np.sum((y_test - test_p50) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)),
            "spearman": float(spearman_corr(y_test, test_p50)),
        },
        "checkpoint_1000_test_rows": {
            "mae": float(np.mean(np.abs(test_1000["skill_percentile"] - test_1000["pred_p50"]))),
            "rmse": float(np.sqrt(np.mean((test_1000["skill_percentile"] - test_1000["pred_p50"]) ** 2))),
            "r2": float(
                1.0
                - np.sum((test_1000["skill_percentile"] - test_1000["pred_p50"]) ** 2)
                / np.sum((test_1000["skill_percentile"] - np.mean(test_1000["skill_percentile"])) ** 2)
            ),
            "spearman": float(spearman_corr(test_1000["skill_percentile"].to_numpy(dtype=np.float64), test_1000["pred_p50"].to_numpy(dtype=np.float64))),
        },
        "old_coarse_model_player_split_reference": {
            "r2": 0.8464760621189111,
            "spearman": 0.8693979730536069,
        },
    }

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": model_p50.feature_importances_,
        }
    ).sort_values("importance_gain", ascending=False)
    importance.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    human_band_rows = []
    for checkpoint, sub in human_test.groupby("checkpoint", sort=True):
        q = quantiles(sub["pred_p50"])
        human_band_rows.append({"checkpoint": int(checkpoint), **q})
    human_band_df = pd.DataFrame(human_band_rows)
    human_band_df.to_csv(HUMAN_BANDS_CSV, index=False)

    gemini_pred = gemini[["run_name", "checkpoint", "observed_turn", "ended_before_checkpoint"]].copy()
    gem_p10 = np.clip(model_p10.predict(gemini[feature_cols]), 0.0, 100.0)
    gem_p50 = np.clip(model_p50.predict(gemini[feature_cols]), 0.0, 100.0)
    gem_p90 = np.clip(model_p90.predict(gemini[feature_cols]), 0.0, 100.0)
    gem_stacked = np.sort(np.column_stack([gem_p10, gem_p50, gem_p90]), axis=1)
    gemini_pred["pred_p10"] = gem_stacked[:, 0]
    gemini_pred["pred_p50"] = gem_stacked[:, 1]
    gemini_pred["pred_p90"] = gem_stacked[:, 2]

    percentile_lookup: dict[int, np.ndarray] = {
        int(checkpoint): np.sort(sub["pred_p50"].to_numpy(dtype=np.float64))
        for checkpoint, sub in human_test.groupby("checkpoint", sort=True)
    }
    gemini_pred["pct_among_test_humans"] = [
        float(np.searchsorted(percentile_lookup[int(checkpoint)], pred, side="right") / len(percentile_lookup[int(checkpoint)]) * 100.0)
        for checkpoint, pred in zip(gemini_pred["checkpoint"], gemini_pred["pred_p50"])
    ]
    gemini_pred.to_csv(CURVES_CSV, index=False)

    plot_continuation_curves(gemini_pred, human_band_df, CURVES_PNG)
    plot_holdout_scatter(test_1000.rename(columns={"skill_percentile": "true_skill_pct"}), SCATTER_PNG)

    with METRICS_JSON.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(
        json.dumps(
            {
                "metrics_json": str(METRICS_JSON),
                "curves_csv": str(CURVES_CSV),
                "human_bands_csv": str(HUMAN_BANDS_CSV),
                "feature_importance_csv": str(FEATURE_IMPORTANCE_CSV),
                "curves_png": str(CURVES_PNG),
                "scatter_png": str(SCATTER_PNG),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

