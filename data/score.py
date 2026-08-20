"""
score.py -- Compute a within-position value model from joined.csv.

All points (ppg, rushing_ppg, receiving_ppg, efficiency) are computed from
this league's EXACT custom scoring rules, not nfl_data_py's generic
fantasy_points_ppr column:
  Passing:   0.04 pt/yard, 4 pt/TD, -1 per interception
  Rushing:   0.1 pt/yard, 6 pt/TD
  Receiving: 1 pt/reception (PPR), 0.1 pt/yard, 6 pt/TD
  (Two-point conversions are part of the league's rules but not reflected
  here -- no such column exists in nfl_data_py's seasonal data, and they're
  rare enough not to meaningfully move rankings. See README.)

For QB/RB/WR/TE (positions with real seasonal stats):
  - Standardize (z-score) a handful of per-game / rate stats within each
    position group.
  - Combine into a composite "value_score" with position-specific weights
    (see below). The two position groups get different components because
    QBs don't receive, and RB/WR/TE touchdown equity and receiving work
    need to be visible separately from raw rushing volume.

  RB/WR/TE components:
        0.30  ppg            -- total PPR points per game (overall anchor)
        0.15  rushing_ppg    -- rushing-only fantasy points/game
                                 (0.1 pt/rush yard + 6 pt/rush TD)
        0.20  receiving_ppg  -- receiving-only fantasy points/game
                                 (1 pt/reception + 0.1 pt/rec yard +
                                 6 pt/rec TD). Weighted ABOVE rushing_ppg
                                 on purpose: this is a PPR league (see
                                 "Value model" below and README), so a
                                 given unit of receiving work is worth more
                                 than the same unit of rushing work, and a
                                 receiving back/slot WR shouldn't get
                                 buried under a between-the-tackles runner
                                 with similar total touches.
        0.10  td_rate        -- (rush TD + rec TD) / (carries + targets):
                                 touchdown production per opportunity, a
                                 proxy for red-zone/goal-line role. This is
                                 NOT a true red-zone stat -- nfl_data_py's
                                 seasonal data has no red-zone-specific
                                 carries/targets column, so this uses ALL
                                 opportunities, not just red-zone ones.
                                 Documented limitation in README.
        0.15  snap_share     -- average offensive snap %
        0.10  efficiency     -- PPR points per opportunity (carries+targets)

  QB components:
        0.45  ppg            -- total PPR points per game
        0.15  volume         -- pass attempts per game
        0.15  rushing_ppg    -- rushing-only fantasy points/game (same
                                 formula as above). This is what the old
                                 model missed entirely: a dual-threat QB
                                 with real rushing production is worth
                                 meaningfully more than a pocket passer
                                 with the same passing line, and lumping
                                 "volume" into pass attempts alone couldn't
                                 see that. rushing_ppg is legitimately near
                                 zero for most pocket passers -- that's a
                                 real (low) value, not missing data, so it
                                 is never redistributed away for QBs.
        0.15  snap_share     -- average offensive snap %
        0.10  efficiency     -- PPR points per opportunity (attempts+carries)

  If a component is missing (non-null) for less than 30% of a position
  group -- i.e. missing for >70% of it -- its weight is redistributed
  proportionally across the remaining available components, so weights
  still sum to 1.0. (In practice this only ever fires for snap_share when
  snap count data failed to pull.)

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

# League's EXACT custom scoring rules (not nfl_data_py's generic
# fantasy_points_ppr column, which uses its own standard formula and does
# not match this league -- e.g. this league's interception penalty is -1,
# not the more common -2). All QB/RB/WR/TE points in this file are computed
# from raw stat columns using these rates. Two-point conversions are part
# of the league's rules but are NOT reflected here -- nfl_data_py's
# seasonal data has no two-point-conversion column, and they're rare
# enough not to meaningfully change rankings. Documented in README.
PASS_YDS_TO_PTS = 0.04
PASS_TD_TO_PTS = 4
INT_TO_PTS = -1
YDS_TO_PTS = 0.1       # rushing AND receiving yards, per this league's rules
TD_TO_PTS = 6          # rushing AND receiving TDs
REC_TO_PTS = 1         # PPR: 1 pt/reception


def compute_passing_pts(df: pd.DataFrame) -> pd.Series:
    return (
        df["passing_yards"].fillna(0) * PASS_YDS_TO_PTS
        + df["passing_tds"].fillna(0) * PASS_TD_TO_PTS
        + df["interceptions"].fillna(0) * INT_TO_PTS
    )


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def compute_stat_group_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    games = df["games"].replace(0, np.nan)
    pos = df["position"].iloc[0]

    # Rushing-only fantasy production, per game. Computed for every
    # position (including QB) so we can separate rushing production from
    # everything else instead of lumping it into one "volume" number.
    rushing_pts = (
        df["rushing_yards"].fillna(0) * YDS_TO_PTS
        + df["rushing_tds"].fillna(0) * TD_TO_PTS
    )
    df["rushing_ppg"] = rushing_pts / games

    df["snap_share"] = df["snap_pct"]  # already 0-1 average

    if pos == "QB":
        passing_pts = compute_passing_pts(df)
        df["ppg"] = (passing_pts + rushing_pts) / games
        df["volume"] = df["attempts"] / games  # passing volume/game
        opportunities = df["attempts"].fillna(0) + df["carries"].fillna(0)
        df["efficiency"] = (passing_pts + rushing_pts) / opportunities.replace(0, np.nan)

        components = {
            "ppg": 0.45,
            "volume": 0.15,
            "rushing_ppg": 0.15,
            "snap_share": 0.15,
            "efficiency": 0.10,
        }
    else:
        # Receiving-only fantasy production, per game -- kept separate from
        # rushing_ppg above so a receiving-work-heavy player (pass-catching
        # RB, possession WR, receiving TE) is visibly rewarded for it,
        # rather than that signal getting averaged away inside one
        # combined "volume" number.
        receiving_pts = (
            df["receptions"].fillna(0) * REC_TO_PTS
            + df["receiving_yards"].fillna(0) * YDS_TO_PTS
            + df["receiving_tds"].fillna(0) * TD_TO_PTS
        )
        df["receiving_ppg"] = receiving_pts / games
        df["ppg"] = (rushing_pts + receiving_pts) / games

        opportunities = df["carries"].fillna(0) + df["targets"].fillna(0)
        df["volume"] = opportunities / games  # combined touches+targets/game
        df["efficiency"] = (rushing_pts + receiving_pts) / opportunities.replace(0, np.nan)

        # Touchdown-rate proxy: TDs scored per opportunity. This is meant
        # to capture touchdown/goal-line equity that pure volume misses
        # (two backs with identical touches can have very different TD
        # profiles). It is an APPROXIMATION -- nfl_data_py's seasonal data
        # has no true red-zone carries/targets column, so this uses ALL
        # opportunities rather than red-zone-specific ones. Explicitly
        # flagged as a limitation in README rather than presented as real
        # red-zone data.
        total_tds = df["rushing_tds"].fillna(0) + df["receiving_tds"].fillna(0)
        df["td_rate"] = total_tds / opportunities.replace(0, np.nan)

        components = {
            "ppg": 0.30,
            "rushing_ppg": 0.15,
            "receiving_ppg": 0.20,
            "td_rate": 0.10,
            "snap_share": 0.15,
            "efficiency": 0.10,
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
            # fantasy_points_ppr is used ONLY as an existence check here --
            # "did this player have a 2024 stat row at all" -- not as a
            # points value. Every actual point calculation in this file
            # uses the league's custom scoring (compute_passing_pts, plus
            # the rushing/receiving math above), never this column's value.
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
