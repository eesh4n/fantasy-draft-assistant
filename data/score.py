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
        0.10  red_zone_share -- REAL red-zone usage: (red-zone rush
                                 attempts + red-zone targets) / (total
                                 carries + targets), computed from 2024
                                 play-by-play data (yardline_100 <= 20) by
                                 redzone.py and joined in by join.py via
                                 player_id (gsis_id). Per an analyst's
                                 claim that red-zone touches are one of
                                 the two most predictive stats for RB
                                 touchdown production (~65% of RB TDs),
                                 this replaces the old td_rate proxy,
                                 which only measured TDs-per-opportunity
                                 over ALL opportunities (not red-zone-
                                 specific) because nfl_data_py's seasonal
                                 aggregate data has no red-zone columns.
                                 See redzone.py's docstring for the exact
                                 red-zone/goal-line definitions used and
                                 the goal_line_share stat (not currently
                                 in the composite, but surfaced in
                                 stats.goal_line_share for detail views).
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

For K (real per-kicker stats from 2024 play-by-play, via
data/kicker_stats.csv / pull_kicker_stats.py -- nfl_data_py's seasonal
aggregate table itself has zero kicker rows, so this is a systematic
stats-based build, not named-player overrides):
  An analyst's video identified what actually drives kicker fantasy value,
  in priority order: (1) overall FG accuracy, (2) FG accuracy from 50+
  yards specifically, (3) dome/weather-independent home stadium, (4) team
  offense quality (more scoring drives = more kick attempts). Composite:
        0.40  fg_pct         -- overall FG accuracy (z-scored)
        0.25  fg_pct_50plus  -- 50+ yard FG accuracy (z-scored; falls back
                                 to fg_pct when null for a kicker with <3
                                 attempts from 50+, see pull_kicker_stats.py
                                 MIN_50PLUS_ATTEMPTS -- reusing fg_pct
                                 keeps the weight meaningful instead of
                                 silently dropping to 0 for most kickers,
                                 since long-range attempts are infrequent)
        0.10  is_dome         -- 0/1, weather-independent home stadium
        0.10  team_offense_ppg -- z-scored team scoring rate (more drives
                                 = more kick volume)
        0.15  adp_anchor      -- inverted z-score of ADP, blended in so a
                                 kicker with almost no 2024 attempts (e.g.
                                 an injury replacement/new signing with a
                                 thin or all-null stat line) doesn't get
                                 wildly mis-ranked by noisy/absent
                                 accuracy numbers alone; ADP already
                                 reflects real-world expert judgment about
                                 projected role. Accuracy (0.40+0.25=0.65)
                                 dominates by design per the analyst's
                                 stated priority order; dome/offense are
                                 clearly secondary (0.10 each); ADP is a
                                 stabilizer, not the primary signal (unlike
                                 the pure-ADP fallback this replaces).
  Kickers with no 2024 play-by-play match at all (no fg_pct) skip straight
  to the same ADP-only fallback described below for DEF, same as
  QB/RB/WR/TE's "without_stats" path.

For DEF (real per-team stats from 2024 play-by-play, via
data/defense_stats.csv / build_defense_stats.py -- nfl_data_py's seasonal
aggregate table has zero team-defense rows, so this is a systematic
stats-based build joined in as def_-prefixed columns, not named-team
overrides):
  An analyst's video identified what actually drives defense fantasy
  value: pressure rate (drives sacks/turnovers), an "adjusted" PPG that
  strips fluke special-teams/defensive TDs, strength of opposing offenses
  faced, and offseason roster change. Of those, this pipeline implements
  the first two directly (a real pressure-rate proxy, and a proper
  custom-scoring adjusted PPG built from real component stats rather than
  a vague "subtract the TDs" heuristic); the last two (opponent-offense
  strength faced, offseason roster improvement/decline) are NOT
  implemented -- neither is derivable from 2024 play-by-play data alone,
  and are flagged as real gaps in build_defense_stats.py and README.md
  rather than faked. Composite:
        0.60  def_custom_adjusted_ppg  -- this league's EXACT custom DEF
                                           scoring rules (2/INT, 2/FR,
                                           1/FF, 2/safety, 2/blocked kick,
                                           6/def TD, 1/sack, 6/ST TD,
                                           1/ST FF, 1/ST FR, plus the
                                           points-allowed tier bonus),
                                           averaged per game. Dominates
                                           the weighting since it's the
                                           single number that directly
                                           reflects what this league
                                           actually pays for defenses,
                                           unlike a generic fantasy-points
                                           column. See
                                           build_defense_stats.py for the
                                           full computation and the
                                           documented 21-27-points-
                                           allowed-tier assumption
                                           (0, pending user confirmation).
        0.20  def_pressure_rate_proxy  -- sacks per opponent pass attempt
                                           faced. This IS the "pressure
                                           rate" the analyst called out --
                                           true pressure rate (pass-rush
                                           snaps -> pressures) isn't in
                                           public nflverse pbp data, so
                                           this is the best available
                                           substitute, weighted as a
                                           secondary signal since it's
                                           explicitly a proxy.
        0.20  turnover/sack volume     -- z-score of def_sacks +
                                           def_interceptions +
                                           def_forced_fumbles (season
                                           totals). A secondary signal on
                                           raw turnover/pressure event
                                           volume -- overlaps somewhat
                                           with def_custom_adjusted_ppg by
                                           construction (these events also
                                           feed into it), but kept at low
                                           weight so a defense that
                                           generates events consistently
                                           isn't fully penalized by one or
                                           two bad points-allowed games
                                           dragging the adjusted-PPG
                                           number down.
  If def_custom_adjusted_ppg is null for a team (e.g. defense_stats.csv
  hasn't been generated yet -- build_defense_stats.py not run), that
  team falls back to the pure ADP-based rank (inverted so lower ADP =
  higher score) previously used for all of DEF. Documented in
  data/README.md.

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
# K has real per-kicker stats now (data/kicker_stats.csv), handled by its
# own branch in main() below -- not part of STAT_POSITIONS since its
# columns/composite are entirely different from QB/RB/WR/TE's.
NO_STAT_POSITIONS = {"DEF"}

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

        # REAL red-zone usage share, not a TD-rate proxy. red_zone_share
        # comes from data/redzone_stats.csv (built by redzone.py from 2024
        # play-by-play data: red-zone rush attempts + red-zone targets,
        # divided by total carries+targets) and is joined into joined.csv
        # by join.py on player_id. This captures touchdown/goal-line
        # equity directly from actual red-zone opportunity share, per an
        # analyst's claim that red-zone touches are one of the two most
        # predictive stats for RB touchdown production (~65% of RB TDs) --
        # replacing the old td_rate proxy (TDs per opportunity over ALL
        # opportunities, not red-zone-specific, since nfl_data_py's
        # seasonal aggregate data has no red-zone columns).
        # goal_line_share (yardline_100 <= 5, a stricter cut) is also
        # computed by redzone.py and carried through to stats.
        # output/players.json, but is not part of the composite weight --
        # red_zone_share is the broader, less noisy signal for volume-
        # weighted RB/WR/TE comparisons; goal_line_share is exposed for
        # detail views/spot-checking instead.
        components = {
            "ppg": 0.30,
            "rushing_ppg": 0.15,
            "receiving_ppg": 0.20,
            "red_zone_share": 0.10,
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


# K composite weights -- see module docstring for the full reasoning behind
# each weight. Accuracy (fg_pct + fg_pct_50plus) intentionally dominates
# per the analyst's stated priority order; dome/team-offense are secondary;
# adp_anchor is a stabilizer against noisy/thin stat lines, not the primary
# signal (unlike the pure-ADP fallback this replaces for K).
K_COMPONENTS = {
    "fg_pct": 0.40,
    "fg_pct_50plus": 0.25,
    "is_dome": 0.10,
    "team_offense_ppg": 0.10,
    "adp_anchor": 0.15,
}


# Bayesian-shrinkage pseudo-counts for fg_pct / fg_pct_50plus, used only to
# stabilize the COMPOSITE (raw fg_pct/fg_pct_50plus in stats/players.json
# are untouched). Without this, a kicker with a handful of attempts and a
# lucky 100% (e.g. 5-for-5) out-z-scores a proven, high-volume kicker at
# 93% on 44 attempts -- confirmed happening in practice (Spencer Shrader,
# 5/5, ranked #1 overall above Chris Boswell, 41/44). Shrinking each
# kicker's rate toward the position's attempt-weighted average FG%,
# proportional to a virtual prior of FG_PCT_PRIOR_WEIGHT / FG50_PRIOR_WEIGHT
# attempts at that average, fixes this the standard way (a "regressed"
# batting-average-style estimator) without discarding the signal entirely
# the way a hard attempts cutoff would. 50+ yard attempts are rarer per
# kicker, so it gets a smaller prior weight than overall fg_pct.
FG_PCT_PRIOR_WEIGHT = 10
FG50_PRIOR_WEIGHT = 4


def _shrink_pct(made: pd.Series, attempts: pd.Series, prior_weight: float) -> pd.Series:
    attempts = attempts.fillna(0)
    made = made.fillna(0)
    league_avg = made.sum() / attempts.sum() if attempts.sum() > 0 else 0.0
    return (made + prior_weight * league_avg) / (attempts + prior_weight)


def compute_kicker_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Real per-kicker composite for K, replacing the old ADP-only
    fallback. Requires fg_pct to be non-null (i.e. the kicker had a 2024
    play-by-play match via kicker_stats.csv) -- callers should route
    kickers without a match to the ADP-only fallback instead, same as
    QB/RB/WR/TE's "without_stats" path.
    """
    df = df.copy()

    # Shrink both accuracy rates toward the group's attempt-weighted mean
    # before z-scoring, so low-volume kickers' noisy percentages don't
    # dominate the ranking (see FG_PCT_PRIOR_WEIGHT comment above). The
    # RAW fg_pct/fg_pct_50plus values (not these shrunk ones) are what get
    # written to joined.csv/players.json -- this shrinkage only affects
    # the composite's internal ranking math.
    fg_pct_shrunk = _shrink_pct(df["fg_made"], df["fg_attempts"], FG_PCT_PRIOR_WEIGHT)

    # fg_pct_50plus falls back to fg_pct when null (too few 50+ attempts to
    # trust the rate -- see pull_kicker_stats.py's MIN_50PLUS_ATTEMPTS).
    # This keeps the 0.25 weight meaningful for the majority of kickers who
    # simply don't see many 50+ yard tries in a season, instead of that
    # weight effectively vanishing. Also shrunk toward the group mean for
    # the same small-sample reason as fg_pct above.
    fg50_made = df["fg_made_50plus"].fillna(0)
    fg50_attempts = df["fg_attempts_50plus"].fillna(0)
    fg_pct_50plus_shrunk = _shrink_pct(fg50_made, fg50_attempts, FG50_PRIOR_WEIGHT)
    # kickers with literally zero 50+ attempts get no information from that
    # bucket at all -- fall back fully to their (shrunk) overall fg_pct.
    fg_pct_50plus_shrunk = fg_pct_50plus_shrunk.where(fg50_attempts > 0, fg_pct_shrunk)

    df["is_dome_numeric"] = df["is_dome"].astype(float)
    # Lower ADP = better; invert so higher adp_anchor = more valuable,
    # matching every other z-scored component here.
    df["adp_anchor"] = -df["adp"]

    z_fg_pct = zscore(fg_pct_shrunk)
    z_fg_pct_50plus = zscore(fg_pct_50plus_shrunk)
    z_dome = zscore(df["is_dome_numeric"].fillna(df["is_dome_numeric"].median()))
    z_offense = zscore(df["team_offense_ppg"].fillna(df["team_offense_ppg"].median()))
    z_adp = zscore(df["adp_anchor"].fillna(df["adp_anchor"].median()))

    composite = (
        z_fg_pct * K_COMPONENTS["fg_pct"]
        + z_fg_pct_50plus * K_COMPONENTS["fg_pct_50plus"]
        + z_dome * K_COMPONENTS["is_dome"]
        + z_offense * K_COMPONENTS["team_offense_ppg"]
        + z_adp * K_COMPONENTS["adp_anchor"]
    )
    df["value_score"] = composite
    return df


# DEF composite weights -- see module docstring for the full reasoning
# behind each weight. def_custom_adjusted_ppg dominates because it's this
# league's exact custom scoring; the other two are secondary/proxy
# signals.
DEF_COMPONENTS = {
    "def_custom_adjusted_ppg": 0.60,
    "def_pressure_rate_proxy": 0.20,
    "def_turnover_volume": 0.20,
}


def compute_defense_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Real per-team composite for DEF, replacing the old pure-ADP
    fallback. Requires def_custom_adjusted_ppg to be non-null (i.e. the
    team had a defense_stats.csv row) -- callers should route teams
    without a match to the ADP-only fallback instead, same as
    QB/RB/WR/TE's "without_stats" path.
    """
    df = df.copy()

    df["def_turnover_volume"] = (
        df["def_sacks"].fillna(0)
        + df["def_interceptions"].fillna(0)
        + df["def_forced_fumbles"].fillna(0)
    )

    z_adjusted_ppg = zscore(
        df["def_custom_adjusted_ppg"].fillna(df["def_custom_adjusted_ppg"].median())
    )
    z_pressure = zscore(
        df["def_pressure_rate_proxy"].fillna(df["def_pressure_rate_proxy"].median())
    )
    z_turnover = zscore(df["def_turnover_volume"])

    composite = (
        z_adjusted_ppg * DEF_COMPONENTS["def_custom_adjusted_ppg"]
        + z_pressure * DEF_COMPONENTS["def_pressure_rate_proxy"]
        + z_turnover * DEF_COMPONENTS["def_turnover_volume"]
    )
    df["value_score"] = composite
    return df


def main():
    df = pd.read_csv(DATA_DIR / "joined.csv")

    scored_parts = []
    for pos, group in df.groupby("position"):
        group = group.copy()
        if pos == "K":
            # Real stats-based composite for kickers with a 2024
            # play-by-play match (fg_pct non-null); everyone else (no
            # match, e.g. an ADP-ranked replacement/rookie with 0 2024
            # attempts) falls back to the same ADP-only treatment as
            # QB/RB/WR/TE's "without_stats" path / DEF below.
            has_stats = group["fg_pct"].notna()
            with_stats = group[has_stats]
            without_stats = group[~has_stats]

            if len(with_stats) > 0:
                with_stats = compute_kicker_scores(with_stats)
            if len(without_stats) > 0:
                without_stats = without_stats.copy()
                min_score = with_stats["value_score"].min() if len(with_stats) else 0
                adp_rank = without_stats["adp"].rank(method="min", ascending=True)
                without_stats["value_score"] = min_score - 0.01 * adp_rank

            group = pd.concat([with_stats, without_stats], ignore_index=True)
        elif pos in STAT_POSITIONS:
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
            # DEF: real stats-based composite for teams with a
            # defense_stats.csv match (def_custom_adjusted_ppg non-null);
            # any team without a match falls back to the ADP-only
            # treatment (lower ADP => higher value_score), same pattern as
            # K/QB/RB/WR/TE's "without_stats" path.
            has_stats = group["def_custom_adjusted_ppg"].notna()
            with_stats = group[has_stats]
            without_stats = group[~has_stats]

            if len(with_stats) > 0:
                with_stats = compute_defense_scores(with_stats)
            if len(without_stats) > 0:
                without_stats = without_stats.copy()
                min_score = with_stats["value_score"].min() if len(with_stats) else 0
                adp_rank = without_stats["adp"].rank(method="min", ascending=True)
                without_stats["value_score"] = min_score - 0.01 * adp_rank

            group = pd.concat([with_stats, without_stats], ignore_index=True)

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
