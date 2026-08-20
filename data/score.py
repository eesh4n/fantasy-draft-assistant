"""
score.py -- Compute a within-position value model from joined.csv.

For QB/RB/WR/TE (positions with real seasonal stats):
  - Standardize (z-score) a handful of per-game / rate stats within each
    position group.
  - Combine into a composite "value_score" using weights:
        0.5  fantasy points per game (production, PPR-scored)
        0.2  opportunity/volume per game (carries+targets for RB/WR/TE,
             attempts for QB)
        0.2  snap share (average offensive snap %)
        0.1  efficiency (PPR points per opportunity)
    If a field is missing for the whole position group (e.g. no snap
    data), that weight is redistributed proportionally across the
    remaining available components so weights still sum to 1.0.

For K/DEF (no per-player seasonal stats available from nfl_data_py):
  value_score falls back to a pure ADP-based rank (inverted so lower ADP
  = higher score), since there's no independent production signal to
  blend in. This is documented in data/README.md.

Output: joined.csv + value_score, position_rank, adp_position_rank,
value_gap (adp_position_rank - position_rank; positive = sleeper /
undervalued by ADP relative to actual production, negative = ADP is
pricier than production supports).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

STAT_POSITIONS = {"QB", "RB", "WR", "TE"}
NO_STAT_POSITIONS = {"K", "DEF"}


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def compute_stat_group_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    games = df["games"].replace(0, np.nan)

    df["ppg"] = df["fantasy_points_ppr"] / games

    if df["position"].iloc[0] == "QB":
        df["volume"] = df["attempts"] / games
    else:
        df["volume"] = (df["carries"].fillna(0) + df["targets"].fillna(0)) / games

    df["snap_share"] = df["snap_pct"]  # already 0-1 average

    opportunities = df["volume"] * games  # back out raw opportunity count
    df["efficiency"] = df["fantasy_points_ppr"] / opportunities.replace(0, np.nan)

    components = {
        "ppg": 0.5,
        "volume": 0.2,
        "snap_share": 0.2,
        "efficiency": 0.1,
    }

    available = {}
    for col, weight in components.items():
        non_null_frac = df[col].notna().mean()
        # require a meaningfully populated column to use it at all
        if non_null_frac >= 0.3:
            available[col] = weight

    weight_sum = sum(available.values())
    if weight_sum == 0:
        # degenerate fallback: use ppg alone
        available = {"ppg": 1.0}
        weight_sum = 1.0

    composite = pd.Series(np.zeros(len(df)), index=df.index)
    for col, weight in available.items():
        z = zscore(df[col].fillna(df[col].median()))
        composite += z * (weight / weight_sum)

    df["value_score"] = composite
    return df


def main():
    df = pd.read_csv(DATA_DIR / "joined.csv")

    scored_parts = []
    for pos, group in df.groupby("position"):
        group = group.copy()
        if pos in STAT_POSITIONS:
            has_stats = group["fantasy_points_ppr"].notna() & (group["games"] > 0)
            with_stats = group[has_stats]
            without_stats = group[~has_stats]

            if len(with_stats) > 0:
                with_stats = compute_stat_group_scores(with_stats)
            if len(without_stats) > 0:
                # rookies/no-2024-stats players: ADP-only fallback so they
                # still show up in the file, ranked by ADP within position,
                # scored below the stat-based players in that group.
                without_stats = without_stats.copy()
                min_score = with_stats["value_score"].min() if len(with_stats) else 0
                # rank purely by adp (lower adp = better), map into a
                # value_score range strictly below the stat-based players
                adp_rank = without_stats["adp"].rank(method="min", ascending=True)
                without_stats["value_score"] = min_score - 0.01 * adp_rank

            group = pd.concat([with_stats, without_stats], ignore_index=True)
        else:
            # K / DEF: ADP-only value model (no seasonal per-player stats
            # available). Lower ADP => higher value_score.
            group["value_score"] = -zscore(group["adp"])

        scored_parts.append(group)

    scored = pd.concat(scored_parts, ignore_index=True)

    scored["position_rank"] = (
        scored.groupby("position")["value_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    scored["adp_position_rank"] = (
        scored.groupby("position")["adp"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    scored["value_gap"] = scored["adp_position_rank"] - scored["position_rank"]

    scored = scored.sort_values(["position", "position_rank"]).reset_index(drop=True)

    out_path = DATA_DIR / "joined.csv"
    scored.to_csv(out_path, index=False)
    print(f"Scored {len(scored)} players. Wrote back to {out_path}")

    for pos in sorted(scored["position"].unique()):
        sub = scored[scored["position"] == pos]
        print(f"  {pos}: {len(sub)} players")


if __name__ == "__main__":
    sys.exit(main())
