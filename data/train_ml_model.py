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

So: this script trains on (1) -- the play-by-play-derived features -- and
score.py blends the resulting prediction back together with (2) as a
smaller secondary adjustment on top (see score.py's module docstring,
"ML value model" section, for the exact blend weights).

-------------------------------------------------------------------------
ROUND 2 (this revision): REAL iterative improvement + historical analogs
-------------------------------------------------------------------------
This revision responds to two explicit pieces of pushback:

  (a) "keep adjusting and testing until you get the max accuracy you could
      possibly get" -- the first pass was a single Ridge/ElasticNet/GBM
      comparison. This revision:
        - widens the Ridge/ElasticNet hyperparameter grids substantially
        - adds Lasso as a candidate
        - adds a small, tightly-regularized RandomForestRegressor grid
          (max_depth/min_samples_leaf tuned for our small per-position n)
        - adds a simple averaging ensemble of the best linear model + the
          best tree model, evaluated honestly via the SAME k-fold CV (the
          ensemble is refit inside every fold, not just averaged after the
          fact, so its CV score is not inflated)
        - tests two genuinely new, honestly-sourced feature-set variants
          per position (see FEATURE VARIANTS below) instead of just tuning
          hyperparameters on the original 5-7 features
        - selects the winning (feature-set, model-class) combination per
          position with the SAME "don't switch away from the simple
          choice unless it clears a real margin" discipline used before,
          now applied twice: once for feature-set selection (does adding
          engineered features beat the original feature set by a real
          margin, using linear models on both sides so the comparison
          isn't confounded by a flexible model exploiting extra noise
          columns?) and once for model-class selection (does a tree/
          ensemble beat the best linear model on the WINNING feature set
          by a real margin?).
        - every candidate's CV R^2/MAE is recorded in the saved metadata,
          and train_ml_model.py's stdout prints a full before/after table
          at the end so an "honestly, nothing beat the original" outcome
          is visible and auditable, not hidden.

  (b) "all stats and everything I've said so far taken into account IN
      the model" -- the guide_*/playcaller_* signals in score.py's
      post-prediction blend were hand-weighted, not fit. Two of them
      (playcaller_rb_ppg_rank, playcaller_wr_ppg_rank -- "how much does
      this team's offense historically feed the RB/WR spot") and one
      guide_ol_run_block_rank_2025 proxy DO have a genuine, honestly-
      computable historical analog in nfl_data_py:
        - team_rb_ppg_hist / team_wr_ppg_hist: the team's own average
          fantasy RB/WR PPG in THIS SAME historical season, computed
          from the exact same per-player ppg this file already computes
          (score.compute_stat_group_scores' own ppg column, aggregated by
          team). This is a real historical proxy for "how much does this
          offense feed the position," computable identically for every
          season 2018-2024, unlike the current guide's forward-looking
          playcaller-tenure framing -- but it is measuring the same
          underlying thing (team-level positional scoring share).
        - team_ol_rb_ybc_att: team-level offensive-line run-blocking
          proxy -- mean rushing yards-before-contact per carry (PFR's
          `ybc_att`, via nfl_data_py's import_seasonal_pfr('rush', ...)),
          aggregated to the team level. This is a standard, honest proxy
          for O-line push (how many yards a runner gets before being
          touched), computable for every historical season the same way
          it would be computed for the current one -- a genuine analog
          for the guide's OL run-block rank, even though the guide's rank
          is itself an analyst's more holistic judgment call.
      Both are added as CANDIDATE trained-model features (see FEATURE
      VARIANTS below) for RB (both) and WR (team_wr_ppg_hist only, since
      the current blend's playcaller_wr_ppg_rank is WR-only) -- tested via
      the same feature-set-selection discipline as `age`, and ONLY kept in
      the final feature set (and correspondingly removed from score.py's
      hand-picked post-prediction blend, to avoid double-counting the same
      signal once as a model feature and once as a blend term) if they
      clear the improvement margin. If they DON'T clear it, they are
      honestly reported as "tested, not adopted" and the corresponding
      guide_* blend term stays exactly where it was.
      guide_adj_ppg, pct_pts_lost_to_luck ("luck metric"), and
      guide_proj_volume_rank have NO honest historical analog (they are
      either a single analyst's current-season subjective adjustment, a
      current-season-only variance audit, or a purely forward-looking
      projection with nothing to validate against in past seasons) -- they
      stay as post-prediction blend terms in score.py, but see that file's
      module docstring for how their BLEND WEIGHT (specifically
      guide_adj_ppg's, the largest of them) is now fit empirically against
      real known 2025 outcomes instead of hand-picked, using
      data/fit_blend_weight.py.

FEATURE VARIANTS tested per position (all built the SAME way for every
historical season AND for the live 2024 inference season, via the shared
helpers below / score.py's compute_stat_group_scores -- no separate code
path for "training" vs "live" feature computation):
  QB: base (5 features) | base + age
  RB: base (6 features) | base + age | base + team_ctx
      | base + age + team_ctx
      (team_ctx = team_rb_ppg_hist, team_ol_rb_ybc_att)
  WR: base (6 features) | base + age | base + team_ctx
      | base + age + team_ctx
      (team_ctx = team_wr_ppg_hist)
  TE: base (6 features) | base + age
      (TE gets no playcaller-derived blend term in score.py today, so
      there's no team_ctx analog to test for TE)
`age` is real, honestly-available data (nfl_data_py's
import_seasonal_rosters' own `age` column, present for the historical
roster snapshot of every season 2018-2024 the exact same way it's present
for 2024) -- not fabricated. A well-documented real effect (RBs decline
young, WRs/QBs peak later) that the original feature set had no way to
capture at all.

-------------------------------------------------------------------------
FEATURE / TARGET SETUP -- LEAK-FREE FORECASTING (unchanged from before)
-------------------------------------------------------------------------
For every pair of consecutive seasons (Y, Y+1) with Y in
TRAIN_SEASON_PAIRS_START..2023 (i.e. 2018/2019 through 2023/2024, 6 pairs
across 7 seasons of raw data, 2018-2024):
  - FEATURES come entirely from season Y.
  - TARGET is the player's ACTUAL ppg (this league's custom-scoring
    points/game) in season Y+1.
Season Y+1 data never touches the feature computation for season Y -- no
leakage. Minimum-sample filters (MIN_GAMES_YEAR1=4, MIN_GAMES_YEAR2=8) are
unchanged from the original pass -- see the original reasoning preserved
below.
  MIN_GAMES_YEAR1 = 4  -- the season-Y feature snapshot needs at least a
                          quarter-season of games, otherwise ppg/efficiency
                          are single-game-noise-dominated garbage inputs.
  MIN_GAMES_YEAR2 = 8  -- the season-Y+1 TARGET needs at least half a
                          season so the label itself isn't small-sample
                          noise.

-------------------------------------------------------------------------
MODEL SELECTION
-------------------------------------------------------------------------
Per (position, feature-set) combination, we fit via K-fold CV:
  - Ridge over a wider alpha grid
  - Lasso over an alpha grid
  - ElasticNet over a wider alpha/l1_ratio grid
  - A shallow, regularized GradientBoostingRegressor
  - A small, regularized RandomForestRegressor (max_depth/min_samples_leaf
    grid, tuned for small n)
  - A simple averaging ensemble of the best linear model + the best tree
    model found above (refit per-fold, not post-hoc averaged, so its CV
    score is an honest estimate)
  - The NAIVE BASELINE: last season's ppg predicts next season's ppg.
Feature-set selection: the winning feature set is the one whose BEST
LINEAR model (Ridge/Lasso/EN) has the highest CV R^2, and it only wins
over the ORIGINAL base feature set if it beats it by
FEATURE_IMPROVEMENT_MARGIN. Comparing via the best LINEAR model (not
letting a flexible tree model pick the feature set) avoids a tree just
memorizing noise in extra columns and mistaking that for a genuine
feature-engineering win, given our small n.
Model-class selection on the winning feature set: a tree/ensemble only
wins over the best linear model if its mean CV R^2 beats it by at least
TREE_IMPROVEMENT_MARGIN -- same discipline as before.
All CV R^2/MAE numbers for every (feature-set, model) combination are
printed and saved into each position's metadata file.

Final chosen model is refit on the FULL training set (all season-pairs)
inside a Pipeline(StandardScaler -> estimator) [or a small custom
averaging pipeline for the ensemble case] and saved to
data/models/{position}_model.joblib, with a sibling
data/models/{position}_metadata.json recording: feature order, model
type + hyperparameters, training-row count, season pairs used, and the
full CV performance table (every candidate feature-set x model
combination, plus the naive baseline and the previous/original result for
an explicit before/after comparison).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
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

# The ORIGINAL ("base") feature set each position's model was trained on in
# the first pass -- see module docstring. This is always tested as one of
# the feature-set variants below, and is the floor any new variant must
# clear by FEATURE_IMPROVEMENT_MARGIN to be adopted.
BASE_FEATURES_BY_POS = {
    "QB": ["ppg", "volume", "rushing_ppg", "snap_share", "efficiency"],
    "RB": ["ppg", "rushing_ppg", "receiving_ppg", "red_zone_share", "snap_share", "efficiency"],
    "WR": ["ppg", "rushing_ppg", "receiving_ppg", "red_zone_share", "snap_share", "efficiency"],
    "TE": ["ppg", "rushing_ppg", "receiving_ppg", "red_zone_share", "snap_share", "efficiency"],
}

# Extra, honestly-sourced candidate features per position -- see module
# docstring "FEATURE VARIANTS tested per position" for exactly what each
# one is and why it's a genuine historical analog (or, for `age`, just
# real data the original pass never used).
EXTRA_FEATURE_GROUPS_BY_POS = {
    "QB": {"age": ["age"]},
    "RB": {"age": ["age"], "team_ctx": ["team_rb_ppg_hist", "team_ol_rb_ybc_att"]},
    "WR": {"age": ["age"], "team_ctx": ["team_wr_ppg_hist"]},
    "TE": {"age": ["age"]},
}


def _feature_set_variants(pos: str):
    """Yields (variant_name, feature_list) pairs: base, base+each extra
    group individually, and base+all extra groups combined (skipping the
    duplicate when there's only one extra group)."""
    base = BASE_FEATURES_BY_POS[pos]
    groups = EXTRA_FEATURE_GROUPS_BY_POS.get(pos, {})
    yield "base", list(base)
    group_names = list(groups.keys())
    for name in group_names:
        yield f"base+{name}", base + groups[name]
    if len(group_names) > 1:
        all_extra = [f for name in group_names for f in groups[name]]
        yield "base+" + "+".join(group_names), base + all_extra


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
    "real2025_total_pts",
]

N_SPLITS = 5
RANDOM_STATE = 42
# A tree ensemble only wins over the best linear model if its mean CV R^2
# beats it by at least this much -- otherwise the simpler/lower-variance
# linear model is kept by default (see module docstring "MODEL SELECTION").
TREE_IMPROVEMENT_MARGIN = 0.02
# A feature-set variant only wins over the ORIGINAL "base" feature set if
# its best LINEAR model's CV R^2 beats base's best LINEAR model's CV R^2 by
# at least this much -- same "don't chase noise on small n" discipline.
FEATURE_IMPROVEMENT_MARGIN = 0.015

RIDGE_ALPHAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
LASSO_ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
EN_ALPHAS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
EN_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
RF_MAX_DEPTHS = [2, 3, 4]
RF_MIN_LEAF = [5, 10, 20]


def pull_seasonal_stats(season: int) -> pd.DataFrame:
    """Mirrors pull.py's pull_stats(), parameterized by season instead of
    the hardcoded STATS_SEASON constant. Same columns, same snap-count
    aggregation, same fantasy-position filter (K excluded here since K
    isn't part of this ML replacement -- see task scope). Also carries
    through `age` (nfl_data_py's own roster-snapshot age column) -- real
    data, available identically for every historical season and for the
    live 2024 base season (see join.py's matching age pull for live
    inference)."""
    import nfl_data_py as nfl

    seasonal = nfl.import_seasonal_data([season])
    seasonal = seasonal[seasonal["season_type"] == "REG"].copy()

    rosters = nfl.import_seasonal_rosters([season])
    rosters = rosters.sort_values("week").drop_duplicates("player_id", keep="last")
    roster_cols = ["player_id", "player_name", "position", "team", "pfr_id", "age"]
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
        "player_id", "player_name", "position", "team", "games", "age",
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
    efficiency, volume, age, and -- for RB/WR -- the team-context
    historical-analog features), computed via
    score.compute_stat_group_scores(df, season=season) -- the SAME
    function (and, for the team-context features, the SAME
    add_team_context_features()/pull_team_ol_rb_ybc() helpers) score.py
    itself calls for the live inference season. This is the single most
    important invariant for a model trained on season Y -> predict season
    Y+1 to be valid against live 2024-base inference: one code path, no
    separate "historical" vs "live" feature computation logic anywhere."""
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
        featured = score_mod.compute_stat_group_scores(with_stats, season=season)
        scored_parts.append(featured)

    if not scored_parts:
        return pd.DataFrame()
    out = pd.concat(scored_parts, ignore_index=True)
    out["season"] = season
    return out


def build_training_table(pos: str, feature_cols: list, season_cache: dict) -> pd.DataFrame:
    """Builds the (season-Y features) -> (season-Y+1 actual ppg) training
    table for one position across all TRAIN_PAIR_START_SEASONS pairs, for
    a given feature-set variant."""
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
    # games filters above, but efficiency/red_zone_share/team-context can
    # be null for a player with 0 carries+targets, or a team with no PFR
    # match in a given season).
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


def _cv_naive_baseline(X_df: pd.DataFrame, y: np.ndarray):
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


class AveragingEnsemble(BaseEstimator, RegressorMixin):
    """Simple mean-of-predictions ensemble of two already-configured
    (unfit) estimators/pipelines. Each fold refits BOTH sub-models from
    scratch (via clone()), so CV scores reported for this class are an
    honest held-out estimate, not a post-hoc average of already-fit
    models."""

    def __init__(self, est_a, est_b):
        self.est_a = est_a
        self.est_b = est_b

    def fit(self, X, y):
        self.est_a_ = clone(self.est_a).fit(X, y)
        self.est_b_ = clone(self.est_b).fit(X, y)
        return self

    def predict(self, X):
        return (self.est_a_.predict(X) + self.est_b_.predict(X)) / 2.0


def _search_models_for_feature_set(X: np.ndarray, y: np.ndarray) -> dict:
    """Runs the full model-class candidate search (Ridge/Lasso/ElasticNet/
    GBM/RandomForest/averaging-ensemble) for one already-built feature
    matrix. Returns a dict of candidate-name -> {cv_r2_mean, cv_mae_mean,
    (hyperparams)}."""
    results = {}

    best_ridge = None
    for alpha in RIDGE_ALPHAS:
        build = lambda a=alpha: Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=a))])
        r2s, maes = _cv_scores(X, y, build)
        rec = {"alpha": alpha, "cv_r2_mean": float(np.nanmean(r2s)), "cv_mae_mean": float(np.nanmean(maes))}
        if best_ridge is None or rec["cv_r2_mean"] > best_ridge["cv_r2_mean"]:
            best_ridge = rec
    results["ridge"] = best_ridge

    best_lasso = None
    for alpha in LASSO_ALPHAS:
        build = lambda a=alpha: Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=a, max_iter=20000))])
        r2s, maes = _cv_scores(X, y, build)
        rec = {"alpha": alpha, "cv_r2_mean": float(np.nanmean(r2s)), "cv_mae_mean": float(np.nanmean(maes))}
        if best_lasso is None or rec["cv_r2_mean"] > best_lasso["cv_r2_mean"]:
            best_lasso = rec
    results["lasso"] = best_lasso

    best_en = None
    for alpha in EN_ALPHAS:
        for l1_ratio in EN_L1_RATIOS:
            build = lambda a=alpha, l=l1_ratio: Pipeline(
                [("scaler", StandardScaler()), ("model", ElasticNet(alpha=a, l1_ratio=l, max_iter=20000))]
            )
            r2s, maes = _cv_scores(X, y, build)
            rec = {
                "alpha": alpha, "l1_ratio": l1_ratio,
                "cv_r2_mean": float(np.nanmean(r2s)), "cv_mae_mean": float(np.nanmean(maes)),
            }
            if best_en is None or rec["cv_r2_mean"] > best_en["cv_r2_mean"]:
                best_en = rec
    results["elasticnet"] = best_en

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

    best_rf = None
    for max_depth in RF_MAX_DEPTHS:
        for min_leaf in RF_MIN_LEAF:
            build = lambda d=max_depth, m=min_leaf: Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(
                    n_estimators=200, max_depth=d, min_samples_leaf=m,
                    max_features="sqrt", random_state=RANDOM_STATE,
                )),
            ])
            r2s, maes = _cv_scores(X, y, build)
            rec = {
                "max_depth": max_depth, "min_samples_leaf": min_leaf,
                "cv_r2_mean": float(np.nanmean(r2s)), "cv_mae_mean": float(np.nanmean(maes)),
            }
            if best_rf is None or rec["cv_r2_mean"] > best_rf["cv_r2_mean"]:
                best_rf = rec
    results["random_forest"] = best_rf

    # Averaging ensemble of the best linear model found so far + the best
    # tree model found so far (whichever of gbm/random_forest scored
    # higher) -- refit honestly per CV fold via AveragingEnsemble above.
    best_linear_name = max(
        ["ridge", "lasso", "elasticnet"], key=lambda k: results[k]["cv_r2_mean"]
    )
    best_tree_name = "gbm" if results["gbm"]["cv_r2_mean"] >= results["random_forest"]["cv_r2_mean"] else "random_forest"

    def build_linear_from(name):
        if name == "ridge":
            return Ridge(alpha=results["ridge"]["alpha"])
        if name == "lasso":
            return Lasso(alpha=results["lasso"]["alpha"], max_iter=20000)
        return ElasticNet(
            alpha=results["elasticnet"]["alpha"], l1_ratio=results["elasticnet"]["l1_ratio"], max_iter=20000
        )

    def build_tree_from(name):
        if name == "gbm":
            return GradientBoostingRegressor(
                max_depth=2, n_estimators=100, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=10, random_state=RANDOM_STATE,
            )
        return RandomForestRegressor(
            n_estimators=200, max_depth=results["random_forest"]["max_depth"],
            min_samples_leaf=results["random_forest"]["min_samples_leaf"],
            max_features="sqrt", random_state=RANDOM_STATE,
        )

    def build_ensemble():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", AveragingEnsemble(build_linear_from(best_linear_name), build_tree_from(best_tree_name))),
        ])
    ens_r2s, ens_maes = _cv_scores(X, y, build_ensemble)
    results["ensemble"] = {
        "cv_r2_mean": float(np.nanmean(ens_r2s)), "cv_mae_mean": float(np.nanmean(ens_maes)),
        "components": [best_linear_name, best_tree_name],
    }

    return results


def train_position(pos: str, season_cache: dict) -> dict:
    """Full two-stage search for one position: (1) pick the winning
    feature-set variant using each variant's best LINEAR model's CV R^2,
    requiring FEATURE_IMPROVEMENT_MARGIN over the base variant; (2) on the
    winning feature set, pick the winning model class, requiring
    TREE_IMPROVEMENT_MARGIN for a tree/ensemble to beat the best linear
    model. Refits the final choice on the full table and saves it."""
    variant_results = {}  # variant_name -> {feature_cols, table, results}
    for variant_name, feature_cols in _feature_set_variants(pos):
        table = build_training_table(pos, feature_cols, season_cache)
        if table.empty or len(table) < 20:
            print(f"  [{pos}/{variant_name}] skipped (only {len(table)} usable rows)")
            continue
        X = table[feature_cols].to_numpy(dtype=float)
        y = table["target_ppg"].to_numpy(dtype=float)
        results = _search_models_for_feature_set(X, y)
        best_linear_r2 = max(results[k]["cv_r2_mean"] for k in ("ridge", "lasso", "elasticnet"))
        variant_results[variant_name] = {
            "feature_cols": feature_cols,
            "table": table,
            "results": results,
            "best_linear_r2": best_linear_r2,
            "n_rows": len(table),
        }
        print(f"  [{pos}/{variant_name}] n={len(table)}  best_linear_cv_r2={best_linear_r2:.4f}")

    if not variant_results:
        return None

    base_name = "base"
    base_r2 = variant_results[base_name]["best_linear_r2"] if base_name in variant_results else -np.inf
    winning_variant = base_name if base_name in variant_results else max(
        variant_results, key=lambda k: variant_results[k]["best_linear_r2"]
    )
    for name, v in variant_results.items():
        if name == base_name:
            continue
        if v["best_linear_r2"] > base_r2 + FEATURE_IMPROVEMENT_MARGIN and v["best_linear_r2"] > variant_results[winning_variant]["best_linear_r2"]:
            winning_variant = name

    chosen_variant = variant_results[winning_variant]
    feature_cols = chosen_variant["feature_cols"]
    table = chosen_variant["table"]
    results = chosen_variant["results"]
    X = table[feature_cols].to_numpy(dtype=float)
    y = table["target_ppg"].to_numpy(dtype=float)

    base_r2s, base_maes = _cv_naive_baseline(table[["ppg"]], y)
    results["naive_baseline"] = {
        "cv_r2_mean": float(np.nanmean(base_r2s)), "cv_mae_mean": float(np.nanmean(base_maes)),
    }

    best_linear_name = max(["ridge", "lasso", "elasticnet"], key=lambda k: results[k]["cv_r2_mean"])
    best_linear_r2 = results[best_linear_name]["cv_r2_mean"]
    best_tree_name = "gbm" if results["gbm"]["cv_r2_mean"] >= results["random_forest"]["cv_r2_mean"] else "random_forest"
    candidates_beyond_linear = {
        best_tree_name: results[best_tree_name]["cv_r2_mean"],
        "ensemble": results["ensemble"]["cv_r2_mean"],
    }
    best_beyond_name = max(candidates_beyond_linear, key=candidates_beyond_linear.get)
    if candidates_beyond_linear[best_beyond_name] > best_linear_r2 + TREE_IMPROVEMENT_MARGIN:
        chosen_model = best_beyond_name
    else:
        chosen_model = best_linear_name

    if chosen_model == "ridge":
        final_model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=results["ridge"]["alpha"]))])
        hyperparams = {"alpha": results["ridge"]["alpha"]}
    elif chosen_model == "lasso":
        final_model = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=results["lasso"]["alpha"], max_iter=20000))])
        hyperparams = {"alpha": results["lasso"]["alpha"]}
    elif chosen_model == "elasticnet":
        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=results["elasticnet"]["alpha"], l1_ratio=results["elasticnet"]["l1_ratio"], max_iter=20000)),
        ])
        hyperparams = {"alpha": results["elasticnet"]["alpha"], "l1_ratio": results["elasticnet"]["l1_ratio"]}
    elif chosen_model == "gbm":
        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                max_depth=2, n_estimators=100, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=10, random_state=RANDOM_STATE,
            )),
        ])
        hyperparams = {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.05, "subsample": 0.8, "min_samples_leaf": 10}
    elif chosen_model == "random_forest":
        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=200, max_depth=results["random_forest"]["max_depth"],
                min_samples_leaf=results["random_forest"]["min_samples_leaf"],
                max_features="sqrt", random_state=RANDOM_STATE,
            )),
        ])
        hyperparams = {
            "n_estimators": 200, "max_depth": results["random_forest"]["max_depth"],
            "min_samples_leaf": results["random_forest"]["min_samples_leaf"], "max_features": "sqrt",
        }
    else:  # ensemble
        comp_linear, comp_tree = results["ensemble"]["components"]

        def build_linear_from(name):
            if name == "ridge":
                return Ridge(alpha=results["ridge"]["alpha"])
            if name == "lasso":
                return Lasso(alpha=results["lasso"]["alpha"], max_iter=20000)
            return ElasticNet(alpha=results["elasticnet"]["alpha"], l1_ratio=results["elasticnet"]["l1_ratio"], max_iter=20000)

        def build_tree_from(name):
            if name == "gbm":
                return GradientBoostingRegressor(
                    max_depth=2, n_estimators=100, learning_rate=0.05,
                    subsample=0.8, min_samples_leaf=10, random_state=RANDOM_STATE,
                )
            return RandomForestRegressor(
                n_estimators=200, max_depth=results["random_forest"]["max_depth"],
                min_samples_leaf=results["random_forest"]["min_samples_leaf"],
                max_features="sqrt", random_state=RANDOM_STATE,
            )
        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", AveragingEnsemble(build_linear_from(comp_linear), build_tree_from(comp_tree))),
        ])
        hyperparams = {"components": [comp_linear, comp_tree]}

    final_model.fit(X, y)

    model_path = MODELS_DIR / f"{pos}_model.joblib"
    joblib.dump(final_model, model_path)

    # Full before/after comparison table across every feature-set variant
    # tried, for honest reporting -- not just the winner.
    variant_summary = {
        name: {
            "feature_cols": v["feature_cols"],
            "n_rows": v["n_rows"],
            "best_linear_cv_r2": v["best_linear_r2"],
            "candidates": v["results"],
        }
        for name, v in variant_results.items()
    }

    metadata = {
        "position": pos,
        "features": feature_cols,
        "feature_set_variant": winning_variant,
        "model_type": chosen_model,
        "hyperparams": hyperparams,
        "n_train_rows": int(len(table)),
        "season_pairs": sorted(table["season_pair"].unique().tolist()),
        "min_games_year1": MIN_GAMES_YEAR1,
        "min_games_year2": MIN_GAMES_YEAR2,
        "cv_n_splits": min(N_SPLITS, len(X)),
        "candidates": results,
        "chosen_cv_r2_mean": results[chosen_model]["cv_r2_mean"],
        "chosen_cv_mae_mean": results[chosen_model]["cv_mae_mean"],
        "naive_baseline_cv_r2_mean": results["naive_baseline"]["cv_r2_mean"],
        "naive_baseline_cv_mae_mean": results["naive_baseline"]["cv_mae_mean"],
        "feature_set_variants_tried": variant_summary,
    }
    meta_path = MODELS_DIR / f"{pos}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== {pos} ===")
    print(f"  winning feature set: {winning_variant} -> {feature_cols}")
    print(f"  n_train_rows: {len(table)}  season_pairs: {metadata['season_pairs']}")
    for name in ("ridge", "lasso", "elasticnet"):
        r = results[name]
        extra = f"alpha={r['alpha']}" + (f" l1={r.get('l1_ratio')}" if "l1_ratio" in r else "")
        print(f"  {name:<12}{extra:<22} CV R2={r['cv_r2_mean']:.4f}  MAE={r['cv_mae_mean']:.3f}")
    print(f"  gbm (shallow):        CV R2={results['gbm']['cv_r2_mean']:.4f}  MAE={results['gbm']['cv_mae_mean']:.3f}")
    rf = results["random_forest"]
    print(f"  random_forest (depth={rf['max_depth']}, min_leaf={rf['min_samples_leaf']}): CV R2={rf['cv_r2_mean']:.4f}  MAE={rf['cv_mae_mean']:.3f}")
    ens = results["ensemble"]
    print(f"  ensemble ({'+'.join(ens['components'])}): CV R2={ens['cv_r2_mean']:.4f}  MAE={ens['cv_mae_mean']:.3f}")
    print(f"  naive baseline (last yr ppg): CV R2={results['naive_baseline']['cv_r2_mean']:.4f}  MAE={results['naive_baseline']['cv_mae_mean']:.3f}")
    print(f"  >>> CHOSEN: {chosen_model} on '{winning_variant}' feature set (CV R2={metadata['chosen_cv_r2_mean']:.4f}, MAE={metadata['chosen_cv_mae_mean']:.3f})")
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
        print(f"\n--- Searching feature sets + models for {pos} ---")
        meta = train_position(pos, season_cache)
        if meta is None:
            print(f"WARNING: no training rows for {pos}, skipping.")
            continue
        all_metadata[pos] = meta

    print("\nDone. Summary:")
    for pos, meta in all_metadata.items():
        print(
            f"  {pos}: {meta['model_type']} on '{meta['feature_set_variant']}' | n={meta['n_train_rows']} | "
            f"CV R2={meta['chosen_cv_r2_mean']:.4f} (baseline {meta['naive_baseline_cv_r2_mean']:.4f}) | "
            f"CV MAE={meta['chosen_cv_mae_mean']:.3f} (baseline {meta['naive_baseline_cv_mae_mean']:.3f})"
        )


if __name__ == "__main__":
    sys.exit(main())
