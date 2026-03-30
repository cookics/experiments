from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import umap

from analysis.plot_human_nao_random_plus_best import build_game_index
from analysis.train_gemini_latent_skill_model import (
    DB_PATH,
    find_bad_games,
    load_human_data,
    make_skill_target,
)
from analysis.train_gemini_continuation_value_model import (
    CHECKPOINT_MAX_TURN,
    CURVES_CSV,
    GEMINI_FOLDER,
    HUMAN_CHECKPOINT_CACHE,
    RNG_SEED,
    add_derived_features,
    build_gemini_checkpoint_table,
    build_human_checkpoint_table,
    fit_quantile_model,
)


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "analysis_outputs"
PCA_PNG = OUTPUT_DIR / "continuation_region_pca.png"
UMAP_PNG = OUTPUT_DIR / "continuation_region_umap.png"
POINTS_CSV = OUTPUT_DIR / "continuation_region_points_t1000.csv"
SUMMARY_CSV = OUTPUT_DIR / "continuation_region_summary.csv"
NOTES_JSON = OUTPUT_DIR / "continuation_region_notes.json"

FEATURE_COLS = [
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

SUMMARY_FEATURES = [
    "observed_turn",
    "ended_before_checkpoint",
    "best_score",
    "best_dlvl",
    "best_xl",
    "hp_ratio",
    "ac",
    "depth_rate",
    "xp_rate",
]


def load_model_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    games_store, events_store = load_human_data(DB_PATH)
    coarse_bad_games = find_bad_games(games_store, events_store)
    target_games = make_skill_target(
        games_store[~games_store["game_key"].astype(str).isin(coarse_bad_games)].copy()
    )

    games_index = build_game_index()
    valid_keys = set(target_games["game_key"].astype(str).tolist())
    games_index = [
        game
        for game in games_index
        if f"{game['player_name']}#{int(game['local_gameid'])}" in valid_keys
    ]

    human_checkpoints = build_human_checkpoint_table(games_index)
    parsed_best = human_checkpoints.groupby("game_key", as_index=False)["best_dlvl"].max()
    parsed_best = parsed_best.merge(
        target_games[["game_key", "maxlvl"]],
        on="game_key",
        how="left",
    )
    parse_bad_games = set(
        parsed_best.loc[
            parsed_best["best_dlvl"] > parsed_best["maxlvl"], "game_key"
        ].astype(str)
    )

    target_games = target_games[
        ~target_games["game_key"].astype(str).isin(parse_bad_games)
    ].copy()
    human_checkpoints = human_checkpoints[
        ~human_checkpoints["game_key"].astype(str).isin(parse_bad_games)
    ].copy()

    human = human_checkpoints.merge(
        target_games[["game_key", "player_name", "skill_percentile"]],
        on="game_key",
        how="inner",
    )
    human = add_derived_features(human)

    gemini = add_derived_features(build_gemini_checkpoint_table())
    return human, gemini


def train_median_model(human: pd.DataFrame):
    players = human["player_name"].astype(str).drop_duplicates().to_numpy()
    rng = np.random.default_rng(RNG_SEED)
    rng.shuffle(players)
    cut = int(len(players) * 0.8)
    train_players = set(players[:cut])
    train_mask = human["player_name"].astype(str).isin(train_players).to_numpy()

    X_train = human.loc[train_mask, FEATURE_COLS]
    y_train = human.loc[train_mask, "skill_percentile"].to_numpy(dtype=np.float64)
    X_test = human.loc[~train_mask, FEATURE_COLS]
    y_test = human.loc[~train_mask, "skill_percentile"].to_numpy(dtype=np.float64)

    model = fit_quantile_model(X_train, y_train, X_test, y_test, 0.50)
    return model, train_mask


def label_region(df: pd.DataFrame) -> pd.Series:
    true_skill = df["true_skill_pct"]
    pred_skill = df["pred_skill_pct"]
    abs_error = (pred_skill - true_skill).abs()

    regions = pd.Series("other", index=df.index, dtype=object)
    regions.loc[(true_skill.between(48, 66)) & (abs_error <= 3.0)] = "mid_diagonal"
    regions.loc[(pred_skill.between(74, 86)) & (abs_error >= 8.0)] = "upper_flat_cloud"
    regions.loc[(pred_skill.between(50.5, 53.5)) & (abs_error >= 8.0)] = "shelf_52"
    regions.loc[(pred_skill.between(16, 31)) & (true_skill <= 20)] = "low_floor"
    return regions


def summarize_regions(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for region, sub in df.groupby("region", sort=True):
        row: dict[str, float | str] = {
            "region": region,
            "count": int(len(sub)),
            "true_skill_median": float(sub["true_skill_pct"].median()),
            "pred_skill_median": float(sub["pred_skill_pct"].median()),
            "abs_error_median": float((sub["pred_skill_pct"] - sub["true_skill_pct"]).abs().median()),
        }
        for col in SUMMARY_FEATURES:
            row[f"{col}_median"] = float(sub[col].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["count", "region"], ascending=[False, True])


def make_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    matrix = pipeline.fit_transform(df[FEATURE_COLS])

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(matrix)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=35,
        min_dist=0.12,
        metric="euclidean",
        random_state=RNG_SEED,
        transform_seed=RNG_SEED,
    )
    umap_coords = reducer.fit_transform(matrix)

    out = df.copy()
    out["pca_x"] = pca_coords[:, 0]
    out["pca_y"] = pca_coords[:, 1]
    out["umap_x"] = umap_coords[:, 0]
    out["umap_y"] = umap_coords[:, 1]
    return out


def plot_embedding(
    df: pd.DataFrame,
    gemini_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.2))
    fig.patch.set_facecolor("#f4efe8")
    for ax in axes:
        ax.set_facecolor("#fffaf3")

    scatter = axes[0].scatter(
        df[x_col],
        df[y_col],
        c=df["true_skill_pct"],
        cmap="viridis",
        s=18,
        alpha=0.55,
        edgecolors="none",
    )
    axes[0].set_title("Colored by true final skill")
    cbar = fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("True final skill percentile")

    region_order = ["upper_flat_cloud", "mid_diagonal", "shelf_52", "low_floor", "other"]
    colors = {
        "upper_flat_cloud": "#b23a48",
        "mid_diagonal": "#1d3557",
        "shelf_52": "#7b2cbf",
        "low_floor": "#2a9d8f",
        "other": "#adb5bd",
    }
    labels = {
        "upper_flat_cloud": "Upper flat cloud",
        "mid_diagonal": "Mid diagonal",
        "shelf_52": "52 shelf",
        "low_floor": "Low floor",
        "other": "Other",
    }
    for region in region_order:
        sub = df[df["region"] == region]
        if sub.empty:
            continue
        axes[1].scatter(
            sub[x_col],
            sub[y_col],
            color=colors[region],
            s=20 if region != "other" else 14,
            alpha=0.68 if region != "other" else 0.22,
            edgecolors="none",
            label=labels[region],
        )

    if not gemini_df.empty:
        axes[1].scatter(
            gemini_df[x_col],
            gemini_df[y_col],
            marker="*",
            s=240,
            color="#111111",
            edgecolors="#fffaf3",
            linewidths=0.8,
            label="Gemini runs",
        )
        for row in gemini_df.itertuples(index=False):
            axes[1].text(
                getattr(row, x_col) + 0.1,
                getattr(row, y_col) + 0.1,
                str(row.run_name).replace("NetHackChallenge-v0_", ""),
                fontsize=8.5,
                color="#111111",
            )

    axes[1].set_title("Highlighted geometric regions")
    axes[1].legend(loc="best", frameon=True, framealpha=0.95)

    for ax in axes:
        ax.grid(True, alpha=0.12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    fig.suptitle(title, fontsize=18, y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    human, gemini = load_model_dataset()
    model, train_mask = train_median_model(human)

    human_test = human.loc[~train_mask].copy()
    human_test["pred_skill_pct"] = np.clip(
        model.predict(human_test[FEATURE_COLS]), 0.0, 100.0
    )
    human_test["true_skill_pct"] = human_test["skill_percentile"].astype(float)

    test_1000 = human_test[
        human_test["checkpoint"] == CHECKPOINT_MAX_TURN
    ].copy()
    test_1000["region"] = label_region(test_1000)

    gemini_1000 = gemini[gemini["checkpoint"] == CHECKPOINT_MAX_TURN].copy()
    gemini_1000["pred_skill_pct"] = np.clip(
        model.predict(gemini_1000[FEATURE_COLS]), 0.0, 100.0
    )
    gemini_1000["true_skill_pct"] = np.nan
    gemini_1000["region"] = "gemini"

    embed_input = pd.concat(
        [
            test_1000.assign(dataset="human_test"),
            gemini_1000.assign(dataset="gemini"),
        ],
        ignore_index=True,
        sort=False,
    )
    embedded = make_embeddings(embed_input)

    human_embedded = embedded[embedded["dataset"] == "human_test"].copy()
    gemini_embedded = embedded[embedded["dataset"] == "gemini"].copy()

    plot_embedding(
        human_embedded,
        gemini_embedded,
        x_col="pca_x",
        y_col="pca_y",
        title="Continuation geometry at T=1000 on unseen players: PCA",
        output_path=PCA_PNG,
    )
    plot_embedding(
        human_embedded,
        gemini_embedded,
        x_col="umap_x",
        y_col="umap_y",
        title="Continuation geometry at T=1000 on unseen players: UMAP",
        output_path=UMAP_PNG,
    )

    point_cols = [
        "game_key",
        "player_name",
        "true_skill_pct",
        "pred_skill_pct",
        "region",
        "observed_turn",
        "ended_before_checkpoint",
        "best_score",
        "best_dlvl",
        "best_xl",
        "hp_ratio",
        "ac",
        "depth_rate",
        "xp_rate",
        "pca_x",
        "pca_y",
        "umap_x",
        "umap_y",
    ]
    human_embedded[point_cols].to_csv(POINTS_CSV, index=False)

    summary = summarize_regions(human_embedded)
    summary.to_csv(SUMMARY_CSV, index=False)

    notes = {
        "checkpoint": CHECKPOINT_MAX_TURN,
        "cache_used": str(HUMAN_CHECKPOINT_CACHE),
        "median_model_source_curves": str(CURVES_CSV),
        "counts_by_region": {
            str(region): int(count)
            for region, count in human_embedded["region"].value_counts().items()
        },
        "upper_flat_cloud_true_skill_range": [
            float(human_embedded.loc[human_embedded["region"] == "upper_flat_cloud", "true_skill_pct"].min()),
            float(human_embedded.loc[human_embedded["region"] == "upper_flat_cloud", "true_skill_pct"].max()),
        ],
        "upper_flat_cloud_pred_skill_range": [
            float(human_embedded.loc[human_embedded["region"] == "upper_flat_cloud", "pred_skill_pct"].min()),
            float(human_embedded.loc[human_embedded["region"] == "upper_flat_cloud", "pred_skill_pct"].max()),
        ],
        "gemini_predicted_skill_at_1000": {
            str(row.run_name): float(row.pred_skill_pct)
            for row in gemini_1000.itertuples(index=False)
        },
    }
    with NOTES_JSON.open("w", encoding="utf-8") as handle:
        json.dump(notes, handle, indent=2)

    print(
        json.dumps(
            {
                "pca_png": str(PCA_PNG),
                "umap_png": str(UMAP_PNG),
                "points_csv": str(POINTS_CSV),
                "summary_csv": str(SUMMARY_CSV),
                "notes_json": str(NOTES_JSON),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

