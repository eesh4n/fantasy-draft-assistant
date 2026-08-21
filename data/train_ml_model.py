"""
train_ml_model.py -- Train a real, cross-validated ML model per position
(QB/RB/WR/TE) to REPLACE the hand-weighted z-score composite that used to
be the entire value model in score.py.

-------------------------------------------------------------------------
WHY THIS EXISTS / THE CORE CONSTRAINT
-------------------------------------------------------------------------
score.py's old composite blended two kinds of signal:
  1. Play-by-play-derived per-game rate stats (ppg, rushing_ppg,
     receiving_ppg, red_zone_share, snap_share, efficiency, volume) --
     computable identically for ANY past season via nfl_data_py, because
     they're built from raw seasonal/play-by-play columns using this
     league's own custom scoring formula.
  2. "guide_*"/"playcaller_*" columns hand-transcribed from a single 2026
     draft-guide PDF -- these exist ONLY for the current season. There is
     no historical equivalent, so they CANNOT be used as training features
     for a model trained across multiple past seasons (there's nothing to
     train against in 2019, 2020, etc).

So: this script trains on (1) only -- the 7 play-by-play-derived features
-- and score.py blends the resulting prediction back together with (2) as
a smaller secondary adjustment on top (see score.py's module docstring,
"ML value model" section, for the exact blend weights).

-------------------------------------------------------------------------
FEATURE / TARGET SETUP -- LEAK-FREE FORECASTING
-------------------------------------------------------------------------
For every pair of consecutive seasons (Y, Y+1) with Y in
TRAIN_SEASON_PAIRS_START..2023 (i.e. 2018/2019 through 2023/2024, 6 pairs
across 7 seasons of raw data, 2018-2024):
  - FEATURES come entirely from season Y (ppg, rushing_ppg, receiving_ppg,
    red_zone_share, snap_share, efficiency, volume -- all computed from
    season-Y-only raw stats/play-by-play).
  - TARGET is the player's ACTUAL ppg (this league's custom-scoring
    points/game) in season Y+1.
This is exactly the draft-ranking use case: given a player's LAST season's
profile, predict next season's output. Season Y+1 data never touches the
feature computation for season Y -- no leakage.

Minimum-sample filters (documented, not silently chosen):
  MIN_GAMES_YEAR1 = 4  -- the season-Y feature snapshot needs at least a
                          quarter-season of games, otherwise ppg/efficiency
                          are single-game-noise-dominated garbage inputs.
  MIN_GAMES_YEAR2 = 8  -- the season-Y+1 TARGET needs at least half a
                          season so the label itself isn't small-sample
                          noise (e.g. a Week 1 injury making a starter's
                          "actual next-year ppg" a single-game fluke).
Both are conservative but not extreme -- they cut out the noisiest tail
without shrinking an already-small (a few hundred rows/position) dataset
too aggressively.

-------------------------------------------------------------------------
FEATURE COMPUTATION -- REUSED, NOT REINVENTED
-------------------------------------------------------------------------
All 7 features are computed by IMPORTING and calling
score.compute_stat_group_scores() -- the exact same function score.py
already uses for the CURRENT season -- against each historical season's
raw stat/red-zone data. This guarantees the historical training features
and the current-season inference features are computed by literally the
same code path, which is the single most important thing for a model
trained on Y -> predict on "2024 profile" to be valid (same formula, same
units, same custom scoring rules, on both sides).
compute_stat_group_scores() also computes a full value_score internally
(it always has) -- we ignore that for historical rows; we only harvest the
engineered feature columns and (for the label year) 'ppg' as the target.
It references several guide_*/playcaller_* columns that only exist for
the CURRENT season; _add_null_guide_columns() below stubs those to NaN for
historical data so the function runs unmodified -- the existing <30%-
non-null weight-redistribution logic in compute_stat_group_scores already
handles an all-null column by dropping it, so this has zero effect on the
feature columns we actually care about.

Red-zone share is computed by reusing redzone.py's compute_side() (season-
agnostic; takes a play-by-play frame as an argument) against each
historical season's play-by-play pull, using the exact same red-zone
definition (yardline_100 <= 20) and touch-share denominator.

Raw seasonal stat pulling mirrors pull.py's pull_stats(), parameterized by
season instead of hardcoded to STATS_SEASON.

-------------------------------------------------------------------------
MODEL SELECTION
-------------------------------------------------------------------------
Per position, small n (typically a few hundred player-season rows across
6 season-pairs) argues strongly for a low-variance linear model. We fit,
via K-fold cross-validation on the training set:
  - Ridge regression over an alpha grid
  - ElasticNet over an alpha/l1_ratio grid
  - A shallow, regularized GradientBoostingRegressor (max_depth=2,
    strong min_samples_leaf, subsampling) as a candidate tree ensemble
  - A NAIVE BASELINE: "last season's ppg predicts next season's ppg",
    i.e. just use the season-Y ppg feature directly as the prediction,
    evaluated on the exact same CV folds for a fair comparison.
The tree model is only selected as final if its mean CV R^2 beats the
best linear model's by a non-trivial margin (see TREE_IMPROVEMENT_MARGIN
below) -- otherwise the simpler, lower-variance linear model wins by
default, per this project's stated preference for a robust choice on
small n. All CV R^2/MAE numbers (including the naive baseline) are printed
and saved into each position's metadata file so this choice is auditable.

Final chosen model is refit on the FULL training set (all season-pairs)
inside a Pipeline(StandardScaler -> estimator) and saved to
data/models/{position}_model.joblib, with a sibling
data/models/{position}_metadata.json recording: feature order, model
type + hyperparameters, training-row count, season pairs used, and the
CV performance numbers (including the naive-baseline comparison).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

DATA_DIR = Path(__file__).resolve().parent
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(DATA_DIR))

import score as score_mod  # noqa: E402  -- reuse compute_stat_group_scores
import redzone as redzone_mod  # noqa: E402  -- reuse compute_side / KEEP_COLS

# 7 seasons of raw data -> 6 consecutive (Y, Y+1) training pairs.
HISTORICAL_SEASONS = list(range(2018, 2025))  # 2018..2024 inclusive
TRAIN_PAIR_START_SEASONS = list(range(2018, 2024))  # Y in 2018..2023

MIN_GAMES_YEAR1 = 4
MIN_GAMES_YEAR2 = 8

# The exact feature set each position's model is trained on -- see module
# docstring. QB has no receiving_ppg/red_zone_share (QBs don't receive);
# RB/WR/TE share the identical 6-feature set. This list is also, not
# coincidentally, EXACTLY the set of raw-production components the old
# hand-weighted composite used to weight directly for each position --
# score.py now replaces that whole cluster with this model's single
# ml_predicted_ppg output (see score.py's module docstring for the
# replacement weight math).
FEATURES_BY_POS = {
    "QB": ["ppg", "volume", "rushing_ppg", "snap_share", "efficiency"],
    "RB": ["ppg", "rushing_ppg", "receiving_ppg", "red_zone_share", "snap_share", "efficiency"],
    "WR": ["ppg", "rushing_ppg", "receiving_ppg", "red_zone_share", "snap_share", "efficiency"],
    "TE": ["ppg", "rushing_ppg", "receiving_ppg", "red_zone_share", "snap_share", "efficiency"],
}

# Guide/playcaller columns compute_stat_group_scores() references but that
# only exist for the CURRENT season -- stub to NaN for historical seasons
# so the function runs unmodified. All-null columns get dropped by that
# function's own <30%-non-null weight-redistribution logic, so this has no
# effect on the feature columns we actually harvest.
GUIDE_STUB_COLS = [
    "guide_adj_ppg",
    "pct_pts_lost_to_luck",
    "guide_ol_run_block_rank_2025",
    "guide_proj_volume_rank",
    "playcaller_rb_ppg_rank",
    "pct_rb1_rank",
    "playcaller_wr_ppg_rank",
]

N_SPLITS = 5
RANDOM_STATE = 42
# A tree ensemble only wins over the best linear model if its mean CV R^2
# beats it by at least this much -- otherwise the simpler/lower-variance
# linear model is kept by default (see module docstring "MODEL SELECTION").
TREE_IMPROVEMENT_MARGIN = 0.02

RIDGE_ALPHAS = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
EN_ALPHAS = [0.01, 0.05, 0.1, 0.3, 1.0]
EN_L1_RATIOS = [0.2, 0.5, 0.8]


def pull_seasonal_stats(season: int) -> pd.DataFrame:
    """Mirrors pull.py's pull_stats(), parameterized by season instead of
    the hardcoded STATS_SEASON constant. Same columns, same snap-count
    aggregation, same fantasy-position filter (K excluded here since K
    isn't part of this ML replacement -- see task scope)."""
    import nfl_data_py as nfl

    seasonal = nfl.import_seasonal_data([season])
    seasonal = seasonal[seasonal["season_type"] == "REG"].copy()

    rosters = nfl.import_seasonal_rosters([season])
    rosters = rosters.sort_values("week").drop_duplicates("player_id", keep="last")
    roster_cols = ["player_id", "player_name", "position", "team", "pfr_id"]
    rosters = rosters[roster_cols]

    df = seasonal.merge(rosters, on="player_id", how="left")
    df = df[df["player_name"].notna()]
    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])].copy()

    try:
        snaps = nfl.import_snap_counts([season])
        snaps_agg = snaps.groupby("pfr_player_id")["offense_pct"].mean().reset_index()
        snaps_agg = snaps_agg.rename(columns={"offense_pct": "snap_pct"})
        df = df.merge(snaps_agg, left_on="pfr_id", right_on="pfr_player_id", how="left")
        df = df.drop(columns=["pfr_player_id"], errors="ignore")
    except Exception as e:
        print(f"  [{season}] snap counts unavailable ({e}); continuing without snap_pct")
        df["snap_pct"] = np.nan

    keep_cols = [
        "player_id", "player_name", "position", "team", "games",
        "fantasy_points_ppr", "carries", "rushing_yards", "rushing_tds",
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "completions", "attempts", "passing_yards", "passing_tds",
        "interceptions", "snap_pct",
    ]
    df = df[keep_cols].reset_index(drop=True)
    return df


def pull_redzone_share(season: int) -> pd.DataFrame:
    """Reuses redzone.py's compute_side() (season-agnostic) and KEEP_COLS
    against a play-by-play pull for the given season, replicating
    redzone.py's main() logic exactly (same red-zone definition, same
    touch-share denominator) but scoped to just the red_zone_share output
    column this training script needs."""
    import nfl_data_py as nfl

    print(f"  [{season}] pulling play-by-play for red-zone share...")
    pbp = nfl.import_pbp_data([season], downcast=True)
    pbp = pbp[redzone_mod.KEEP_COLS].copy()

    rush = redzone_mod.compute_side(pbp, "rush_attempt", "rusher_player_id", "rusher_player_name")
    target = redzone_mod.compute_side(pbp, "pass_attempt", "receiver_player_id", "receiver_player_name")

    numeric_cols = ["touches", "rz_touches", "gl_touches"]
    combined = rush[numeric_cols].add(target[numeric_cols], fill_value=0)
    combined = combined.reset_index().rename(columns={"index": "player_id"})
    combined["red_zone_share"] = combined["rz_touches"] / combined["touches"].replace(0, np.nan)
    return combined[["player_id", "red_zone_share"]]


def _add_null_guide_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in GUIDE_STUB_COLS:
        df[col] = np.nan
    return df


def compute_features_for_season(season: int) -> pd.DataFrame:
    """Returns one row per player with this season's engineered features
    (ppg, rushing_ppg, receiving_ppg, red_zone_share, snap_share,
    efficiency, volume), computed via score.compute_stat_group_scores() --
    the exact function score.py uses for the current season."""
    stats = pull_seasonal_stats(season)
    rz = pull_redzone_share(season)
    stats = stats.merge(rz, on="player_id", how="left")
    stats = _add_null_guide_columns(stats)

    scored_parts = []
    for pos, group in stats.groupby("position"):
        group = group.copy()
        has_stats = group["fantasy_points_ppr"].notna() & (group["games"] > 0)
        with_stats = group[has_stats]
        if len(with_stats) == 0:
            continue
        featured = score_mod.compute_stat_group_scores(with_stats)
        scored_parts.append(featured)

    if not scored_parts:
        return pd.DataFrame()
    out = pd.concat(scored_parts, ignore_index=True)
    out["season"] = season
    return out


def build_training_table(pos: str, season_cache: dict) -> pd.DataFrame:
    """Builds the (season-Y features) -> (season-Y+1 actual ppg) training
    table for one position across all TRAIN_PAIR_START_SEASONS pairs."""
    feature_cols = FEATURES_BY_POS[pos]
    rows = []
    for y in TRAIN_PAIR_START_SEASONS:
        y1 = y + 1
        feat_y = season_cache[y]
        feat_y1 = season_cache[y1]

        fy = feat_y[feat_y["position"] == pos].copy()
        fy1 = feat_y1[feat_y1["position"] == pos].copy()

        fy = fy[fy["games"] >= MIN_GAMES_YEAR1]
        fy1 = fy1[fy1["games"] >= MIN_GAMES_YEAR2]

        merged = fy.merge(
            fy1[["player_id", "ppg"]].rename(columns={"ppg": "target_ppg"}),
            on="player_id",
            how="inner",
        )
        merged["season_pair"] = f"{y}->{y1}"
        rows.append(merged[["player_id", "player_name", "season_pair", "target_ppg"] + feature_cols])

    if not rows:
        return pd.DataFrame()
    table = pd.concat(rows, ignore_index=True)
    # Drop rows with any null feature (shouldn't happen often given the
    # games filters above, but efficiency/red_zone_share can be null for a
    # player with 0 carries+targets in a qualifying-games season -- e.g. a
    # QB who kneeled once).
    table = table.dropna(subset=feature_cols + ["target_ppg"]).reset_index(drop=True)
    return table


def _cv_scores(X: np.ndarray, y: np.ndarray, build_estimator):
    """Runs K-fold CV, returns (r2_scores, mae_scores) arrays. build_estimator
    is a zero-arg callable returning a fresh unfit Pipeline/estimator."""
    n_splits = min(N_SPLITS, len(X))
    if n_splits < 2:
        return np.array([np.nan]), np.array([np.nan])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    r2s, maes = [], []
    for train_idx, test_idx in kf.split(X):
        est = build_estimator()
        est.fit(X[train_idx], y[train_idx])
        pred = est.predict(X[test_idx])
        ss_res = np.sum((y[test_idx] - pred) ** 2)
        ss_tot = np.sum((y[test_idx] - y[test_idx].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        mae = np.mean(np.abs(y[test_idx] - pred))
        r2s.append(r2)
        maes.append(mae)
    return np.array(r2s), np.array(maes)


def _cv_naive_baseline(X_df: pd.DataFrame, y: np.ndarray, feature_cols: list):
    """'Last season's ppg predicts next season's ppg' -- no fitting, just
    use the season-Y ppg feature directly as the prediction, evaluated on
    the same-shaped K-fold splits as the real models for a fair,
    apples-to-apples comparison."""
    n_splits = min(N_SPLITS, len(X_df))
    if n_splits < 2:
        return np.array([np.nan]), np.array([np.nan])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    ppg = X_df["ppg"].to_numpy()
    r2s, maes = [], []
    for _, test_idx in kf.split(X_df):
        pred = ppg[test_idx]
        actual = y[test_idx]
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        mae = np.mean(np.abs(actual - pred))
        r2s.append(r2)
        maes.append(mae)
    return np.array(r2s), np.array(maes)


def train_position(pos: str, table: pd.DataFrame) -> dict:
    feature_cols = FEATURES_BY_POS[pos]
    X_df = table[feature_cols]
    X = X_df.to_numpy(dtype=float)
    y = table["target_ppg"].to_numpy(dtype=float)

    results = {}

    # -- Ridge grid --
    best_ridge = None
    for alpha in RIDGE_ALPHAS:
        build = lambda a=alpha: Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=a))])
        r2s, maes = _cv_scores(X, y, build)
        rec = {"alpha": alpha, "cv_r2_mean": float(np.nanmean(r2s)), "cv_mae_mean": float(np.nanmean(maes))}
        if best_ridge is None or rec["cv_r2_mean"] > best_ridge["cv_r2_mean"]:
            best_ridge = rec
    results["ridge"] = best_ridge

    # -- ElasticNet grid --
    best_en = None
    for alpha in EN_ALPHAS:
        for l1_ratio in EN_L1_RATIOS:
            build = lambda a=alpha, l=l1_ratio: Pipeline(
                [("scaler", StandardScaler()), ("model", ElasticNet(alpha=a, l1_ratio=l, max_iter=10000))]
            )
            r2s, maes = _cv_scores(X, y, build)
            rec = {
                "alpha": alpha, "l1_ratio": l1_ratio,
                "cv_r2_mean": float(np.nanmean(r2s)), "cv_mae_mean": float(np.nanmean(maes)),
            }
            if best_en is None or rec["cv_r2_mean"] > best_en["cv_r2_mean"]:
                best_en = rec
    results["elasticnet"] = best_en

    # -- Shallow, regularized GBM --
    def build_gbm():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                max_depth=2, n_estimators=100, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=10, random_state=RANDOM_STATE,
            )),
        ])
    gbm_r2s, gbm_maes = _cv_scores(X, y, build_gbm)
    results["gbm"] = {"cv_r2_mean": float(np.nanmean(gbm_r2s)), "cv_mae_mean": float(np.nanmean(gbm_maes))}

    # -- Naive baseline: last year's ppg predicts next year's ppg --
    base_r2s, base_maes = _cv_naive_baseline(X_df, y, feature_cols)
    results["naive_baseline"] = {
        "cv_r2_mean": float(np.nanmean(base_r2s)), "cv_mae_mean": float(np.nanmean(base_maes)),
    }

    # -- Model selection --
    best_linear_name = "ridge" if results["ridge"]["cv_r2_mean"] >= results["elasticnet"]["cv_r2_mean"] else "elasticnet"
    best_linear_r2 = results[best_linear_name]["cv_r2_mean"]
    if results["gbm"]["cv_r2_mean"] > best_linear_r2 + TREE_IMPROVEMENT_MARGIN:
        chosen = "gbm"
    else:
        chosen = best_linear_name

    if chosen == "ridge":
        final_model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=results["ridge"]["alpha"]))])
        hyperparams = {"alpha": results["ridge"]["alpha"]}
    elif chosen == "elasticnet":
        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=results["elasticnet"]["alpha"], l1_ratio=results["elasticnet"]["l1_ratio"], max_iter=10000)),
        ])
        hyperparams = {"alpha": results["elasticnet"]["alpha"], "l1_ratio": results["elasticnet"]["l1_ratio"]}
    else:
        final_model = build_gbm()
        hyperparams = {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.05, "subsample": 0.8, "min_samples_leaf": 10}

    final_model.fit(X, y)

    model_path = MODELS_DIR / f"{pos}_model.joblib"
    joblib.dump(final_model, model_path)

    metadata = {
        "position": pos,
        "features": feature_cols,
        "model_type": chosen,
        "hyperparams": hyperparams,
        "n_train_rows": int(len(table)),
        "season_pairs": sorted(table["season_pair"].unique().tolist()),
        "min_games_year1": MIN_GAMES_YEAR1,
        "min_games_year2": MIN_GAMES_YEAR2,
        "cv_n_splits": min(N_SPLITS, len(X)),
        "candidates": results,
        "chosen_cv_r2_mean": results[chosen]["cv_r2_mean"],
        "chosen_cv_mae_mean": results[chosen]["cv_mae_mean"],
        "naive_baseline_cv_r2_mean": results["naive_baseline"]["cv_r2_mean"],
        "naive_baseline_cv_mae_mean": results["naive_baseline"]["cv_mae_mean"],
    }
    meta_path = MODELS_DIR / f"{pos}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== {pos} ===")
    print(f"  n_train_rows: {len(table)}  season_pairs: {metadata['season_pairs']}")
    print(f"  ridge:      alpha={results['ridge']['alpha']:<8} CV R2={results['ridge']['cv_r2_mean']:.4f}  MAE={results['ridge']['cv_mae_mean']:.3f}")
    print(f"  elasticnet: alpha={results['elasticnet']['alpha']:<6} l1={results['elasticnet']['l1_ratio']:<4} CV R2={results['elasticnet']['cv_r2_mean']:.4f}  MAE={results['elasticnet']['cv_mae_mean']:.3f}")
    print(f"  gbm (shallow):        CV R2={results['gbm']['cv_r2_mean']:.4f}  MAE={results['gbm']['cv_mae_mean']:.3f}")
    print(f"  naive baseline (last yr ppg): CV R2={results['naive_baseline']['cv_r2_mean']:.4f}  MAE={results['naive_baseline']['cv_mae_mean']:.3f}")
    print(f"  >>> CHOSEN: {chosen}  (CV R2={metadata['chosen_cv_r2_mean']:.4f}, MAE={metadata['chosen_cv_mae_mean']:.3f})")
    print(f"  Saved model -> {model_path}")
    print(f"  Saved metadata -> {meta_path}")

    return metadata


def main():
    print(f"Pulling/computing engineered features for seasons {HISTORICAL_SEASONS}...")
    season_cache = {}
    for season in HISTORICAL_SEASONS:
        print(f"Season {season}...")
        season_cache[season] = compute_features_for_season(season)
        print(f"  -> {len(season_cache[season])} player-rows with usable stats")

    all_metadata = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        table = build_training_table(pos, season_cache)
        if table.empty:
            print(f"WARNING: no training rows for {pos}, skipping.")
            continue
        all_metadata[pos] = train_position(pos, table)

    print("\nDone. Summary:")
    for pos, meta in all_metadata.items():
        print(
            f"  {pos}: {meta['model_type']} | n={meta['n_train_rows']} | "
            f"CV R2={meta['chosen_cv_r2_mean']:.4f} (baseline {meta['naive_baseline_cv_r2_mean']:.4f}) | "
            f"CV MAE={meta['chosen_cv_mae_mean']:.3f} (baseline {meta['naive_baseline_cv_mae_mean']:.3f})"
        )


if __name__ == "__main__":
    sys.exit(main())
