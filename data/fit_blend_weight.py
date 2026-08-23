"""
fit_blend_weight.py -- Empirically fit the ml_predicted_ppg <-> guide_adj_ppg
blend weight in score.py's composite, instead of hand-picking it.

-------------------------------------------------------------------------
WHY THIS EXISTS
-------------------------------------------------------------------------
score.py's value_score composite blends the trained model's own prediction
(ml_predicted_ppg) with several guide_*/playcaller_* signals hand-
transcribed from a single 2026 draft-guide PDF (see score.py's module
docstring "ML value model" / guide_adj_ppg sections). Those weights were
originally chosen by hand ("second only to ppg itself... arguably more
informative than the pipeline's own base ppg, so it's weighted
meaningfully").

Most of those guide signals have NO historical equivalent to validate a
weight against -- guide_adj_ppg is one analyst's current-season subjective
adjustment, pct_pts_lost_to_luck is a current-season-only variance audit,
guide_proj_volume_rank is a forward-looking-only projection. There is
genuinely nothing to regress those against for 2018-2023 (the only years
we have multi-year history for).

BUT: this pipeline also has REAL, KNOWN actual 2025 outcomes for a real
subset of players -- data/real2025_stats.csv (real2025_total_pts),
pulled directly from nfl.com's live 2025 stats pages by
real2025_score.py -- for players who have BOTH an ml_predicted_ppg
(computed from their 2024 profile, i.e. exactly the pre-season
information this pipeline had) AND a guide_adj_ppg (the analyst's
pre-season adjusted estimate). That is a genuine, honest held-out
validation set for the SPECIFIC question "how much weight should
ml_predicted_ppg get vs. guide_adj_ppg relative to each other" -- not
fabricated, not circular (both inputs are pre-season; the outcome being
regressed against is the actual season that followed).

-------------------------------------------------------------------------
METHOD
-------------------------------------------------------------------------
real2025_total_pts is a SEASON TOTAL (nfl.com's tables expose no
games-played column -- see real2025_score.py), not a per-game rate, so it
is not directly comparable to ml_predicted_ppg/guide_adj_ppg's PPG units.
Rather than fabricate a games-played denominator, this script validates on
RANK/CORRELATION instead of raw scale: within each position, z-score
ml_predicted_ppg and guide_adj_ppg (the same z-scoring compute_stat_group_
scores() already does), form blended = alpha*z_ml + (1-alpha)*z_guide for
alpha in a fine grid over [0, 1], and grid-search the alpha that maximizes
Spearman rank correlation between `blended` and `real2025_total_pts`
(itself rank-based, so this method never needs a PPG conversion for the
season-total column and is robust to that unit mismatch).

The result -- the fitted alpha per position, its Spearman correlation, and
n -- is written to data/models/blend_weight_fit.json and printed. score.py
then encodes each position's fitted ml:guide RATIO
(alpha / (1 - alpha)) directly into its guide_adj_ppg weight (relative to
ml_predicted_ppg's existing weight), replacing the previous hand-picked
number -- see score.py's module docstring for the exact values used and
this script's printed output for the honesty/confidence caveats (n is
small -- 29-45 players for QB/RB/WR, and TOO SMALL to trust for TE at n=11,
documented explicitly below).

Run this AFTER score.py has been run at least once (so joined.csv has
ml_predicted_ppg populated from the current models).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA_DIR = Path(__file__).resolve().parent

ALPHA_GRID = np.linspace(0.0, 1.0, 41)  # 0.000, 0.025, ..., 1.000
MIN_N_TO_TRUST = 20  # below this, report the fit but flag it as too small
# to adopt -- see module docstring.


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def fit_position(df: pd.DataFrame, pos: str) -> dict:
    sub = df[df["position"] == pos].copy()
    sub = sub.dropna(subset=["ml_predicted_ppg", "guide_adj_ppg", "real2025_total_pts"])
    n = len(sub)
    if n < 5:
        return {"position": pos, "n": n, "status": "too few rows to fit at all"}

    z_ml = zscore(sub["ml_predicted_ppg"])
    z_guide = zscore(sub["guide_adj_ppg"])
    target = sub["real2025_total_pts"]

    best_alpha, best_corr = None, -np.inf
    curve = []
    for alpha in ALPHA_GRID:
        blended = alpha * z_ml + (1 - alpha) * z_guide
        corr, _ = spearmanr(blended, target)
        curve.append({"alpha": round(float(alpha), 3), "spearman": None if np.isnan(corr) else round(float(corr), 4)})
        if not np.isnan(corr) and corr > best_corr:
            best_corr = corr
            best_alpha = alpha

    # Reference points: pure ml, pure guide, and the OLD hand-picked ratio
    # (for context in the printed report).
    corr_pure_ml, _ = spearmanr(z_ml, target)
    corr_pure_guide, _ = spearmanr(z_guide, target)

    trust = "OK" if n >= MIN_N_TO_TRUST else f"TOO SMALL (n={n} < {MIN_N_TO_TRUST}) -- kept hand-picked, not adopted"

    return {
        "position": pos,
        "n": n,
        "best_alpha_on_ml": round(float(best_alpha), 3),
        "best_spearman": round(float(best_corr), 4),
        "spearman_pure_ml": None if np.isnan(corr_pure_ml) else round(float(corr_pure_ml), 4),
        "spearman_pure_guide": None if np.isnan(corr_pure_guide) else round(float(corr_pure_guide), 4),
        "ml_to_guide_ratio": None if best_alpha in (0, 1) else round(float(best_alpha / (1 - best_alpha)), 3),
        "trust": trust,
        "curve": curve,
    }


def main():
    joined_path = DATA_DIR / "joined.csv"
    if not joined_path.exists():
        print(f"ERROR: {joined_path} not found -- run join.py + score.py first.")
        return 1
    df = pd.read_csv(joined_path)
    if "ml_predicted_ppg" not in df.columns:
        print("ERROR: joined.csv has no ml_predicted_ppg column -- run score.py first "
              "(it must run at least once, with the retrained models in place, before "
              "this script can validate against it).")
        return 1

    results = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        res = fit_position(df, pos)
        results[pos] = res
        print(f"\n=== {pos} ===")
        if "status" in res:
            print(f"  {res['status']}")
            continue
        print(f"  n = {res['n']}  ({res['trust']})")
        print(f"  best alpha (weight on ml_predicted_ppg's z-score) = {res['best_alpha_on_ml']}")
        print(f"  Spearman corr @ best alpha vs real2025_total_pts = {res['best_spearman']}")
        print(f"  Spearman corr, pure ml_predicted_ppg              = {res['spearman_pure_ml']}")
        print(f"  Spearman corr, pure guide_adj_ppg                 = {res['spearman_pure_guide']}")
        print(f"  Fitted ml:guide weight ratio = {res['ml_to_guide_ratio']}")

    out_path = DATA_DIR / "models" / "blend_weight_fit.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
