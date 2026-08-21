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

  guide_adj_ppg (ALL FOUR positions -- QB/RB/WR/TE):
        REAL analyst data hand-transcribed from Joel Smyth's Draft Guide
        2026 (data/guide_adjusted_ppg.csv), joined onto joined.csv by name
        (join.py, reusing the same normalize_name + last-name/first-name
        fuzzy-match approach already used for the raw_stats join). This is
        the analyst's own context-adjusted 2025 PPG estimate (e.g.
        stripping out backup-QB games, adding back missed-time production)
        -- a full season MORE RECENT than this pipeline's base 2024
        play-by-play stats, and a human-refined production read rather
        than a raw counting stat. Given both of those, it's arguably more
        informative than the pipeline's own base `ppg` (stale 2024 raw
        stats), so it's weighted meaningfully -- second only to `ppg`
        itself for every position. The guide only covers ~30-50 players
        per position (a big-board preview, not every player); for players
        with no guide match, guide_adj_ppg is null and gets dropped by the
        existing missing-data threshold below, same as any other missing
        component -- NOT defaulted to 0 (which would wrongly penalize
        players simply outside the guide's coverage).

        New weights, redistributed proportionally FROM the original
        weights above to make room for guide_adj_ppg while keeping every
        group's total at 1.0:
          QB:      ppg .38, volume .13, rushing_ppg .13, snap_share .13,
                    efficiency .08, guide_adj_ppg .15
          WR/TE:   ppg .25, rushing_ppg .12, receiving_ppg .16,
                    red_zone_share .08, snap_share .12, efficiency .08,
                    guide_adj_ppg .19
          RB:      see below -- RB carves further room out of the WR/TE
                    split to also fit its two guide-sourced components.

  RB-only additions (on top of guide_adj_ppg above):
        0.07  ol_run_block_rank -- REAL 2025 offensive-line run-blocking
                                    rank for the RB's team (1 = best in the
                                    NFL), from data/guide_ol_stats.csv,
                                    joined onto RB rows by team in join.py.
                                    Inverted before z-scoring (lower/better
                                    rank -> higher z-score) so it points the
                                    same direction as every other component
                                    here. Real context for a runner's
                                    efficiency/opportunity quality, but kept
                                    to a modest weight since it's a team-
                                    level signal, not the player's own
                                    production.
        0.11  proj_volume_rank    -- the analyst's own projected 2026 RB
                                    volume rank (data/guide_rb_volume.csv,
                                    joined by name in join.py), "fully
                                    weighted with targets + goal-line
                                    attempts" per the guide. This is
                                    legitimately distinct from the existing
                                    red_zone_share component: red_zone_share
                                    is backward-looking (real 2024 red-zone
                                    opportunity share from play-by-play),
                                    while proj_volume_rank is the analyst's
                                    forward-looking 2026 projection -- real
                                    additional signal, not a duplicate.
                                    Inverted before z-scoring (lower rank =
                                    better) like ol_run_block_rank above.
                                    Weighted just under guide_adj_ppg since
                                    it's the analyst's own explicit "how
                                    much volume will this back get" call,
                                    but still meaningfully below it since
                                    it's a rank (ordinal), not a rate.
        Resulting RB weights: ppg .20, rushing_ppg .10, receiving_ppg .13,
          red_zone_share .07, snap_share .10, efficiency .07,
          guide_adj_ppg .15, ol_run_block_rank .07, proj_volume_rank .11
          (sums to 1.0).
        adj_volume_25_rank and confidence from guide_rb_volume.csv are NOT
        part of the composite -- they're carried through to
        stats.adj_volume_25_rank / stats.guide_volume_confidence in
        build_json.py as informational/detail-view fields only.
        ol_run_block_rank/proj_volume_rank are null for non-RB rows (WR/TE/
        QB never get these columns populated) and for RBs whose team/name
        didn't match the guide files -- same missing-data handling as
        everything else in this file.

  RB-only playcaller/scheme additions, hand-transcribed from the
  "Playcaller Table" page of Joel Smyth's Draft Guide 2026
  (data/guide_playcaller_stats.csv), joined onto RB rows by team in
  join.py:
        0.04  playcaller_rb_ppg_rank -- REAL rank (1=best) of this team's
                                    average fantasy RB PPG across the
                                    current playcaller's tenure (up to their
                                    last 5 seasons). A team-level offensive-
                                    system signal, distinct from
                                    ol_run_block_rank (blocking quality) and
                                    proj_volume_rank (the analyst's player-
                                    volume projection) -- this captures how
                                    much the SCHEME itself has historically
                                    fed the RB spot. Inverted before
                                    z-scoring (lower/better rank -> higher
                                    value) like every other rank column
                                    here.
        0.04  pct_rb1_rank        -- REAL rank (1=best) of %RB1, the share
                                    of the team's RB fantasy points that go
                                    to the starter (higher %RB1 = more of a
                                    true bellcow role, lower = committee).
                                    Rank 1 = highest %RB1 = most bellcow-
                                    friendly, so a player on a high-%RB1
                                    team should score BETTER -- inverted
                                    before z-scoring the same direction as
                                    playcaller_rb_ppg_rank above.
        Both are kept to a small, modest weight (0.04 each) since they are
        team-level context, not the player's own production, same
        philosophy as ol_run_block_rank's 0.07.

        Redistribution math: the previous RB weights (ppg .20,
        rushing_ppg .10, receiving_ppg .13, red_zone_share .07,
        snap_share .10, efficiency .07, guide_adj_ppg .15,
        ol_run_block_rank .07, proj_volume_rank .11 -- sums to 1.0) are each
        scaled down by a factor of (1 - 0.08) = 0.92 to free up exactly
        0.08 of total weight (0.04 + 0.04) for the two new components,
        preserving their relative proportions:
          ppg:               .20 * .92 = .184
          rushing_ppg:       .10 * .92 = .092
          receiving_ppg:     .13 * .92 = .1196
          red_zone_share:    .07 * .92 = .0644
          snap_share:        .10 * .92 = .092
          efficiency:        .07 * .92 = .0644
          guide_adj_ppg:     .15 * .92 = .138
          ol_run_block_rank: .07 * .92 = .0644
          proj_volume_rank:  .11 * .92 = .1012
          playcaller_rb_ppg_rank: .04
          pct_rb1_rank:           .04
          (sum = 0.92 + 0.08 = 1.0, confirmed by summing the scaled block:
          .184+.092+.1196+.0644+.092+.0644+.138+.0644+.1012 = .92)
        playcaller_rb_ppg_rank/pct_rb1_rank are null for non-RB rows and for
        RBs whose team didn't match guide_playcaller_stats.csv (the four
        teams with a brand-new, 0-season playcaller -- PHI/SEA/WAS/BAL --
        which the guide itself leaves blank rather than assigning a made-up
        rank) -- same missing-data threshold handling as everything else in
        this file. personnel/run_scheme/motion_rank/formation_spread/
        rb_screen_rank from the same guide file are NOT part of the
        composite -- too qualitative/hard to weight confidently under time
        pressure -- and are carried through to build_json.py purely as
        informational/detail-view fields.

  WR-only playcaller addition, same source file, joined onto WR rows by
  team in join.py:
        0.05  playcaller_wr_ppg_rank -- REAL rank (1=best) of this team's
                                    average fantasy WR PPG across the
                                    current playcaller's tenure. Inverted
                                    before z-scoring like the RB version
                                    above.
        Redistribution math: the WR/TE weights (ppg .25, rushing_ppg .12,
        receiving_ppg .16, red_zone_share .08, snap_share .12,
        efficiency .08, guide_adj_ppg .19 -- sums to 1.0) are each scaled
        down by a factor of (1 - 0.05) = 0.95 to free up 0.05 for
        playcaller_wr_ppg_rank:
          ppg:              .25 * .95 = .2375
          rushing_ppg:      .12 * .95 = .114
          receiving_ppg:    .16 * .95 = .152
          red_zone_share:   .08 * .95 = .076
          snap_share:       .12 * .95 = .114
          efficiency:       .08 * .95 = .076
          guide_adj_ppg:    .19 * .95 = .1805
          playcaller_wr_ppg_rank: .05
          (sum = 0.95 + 0.05 = 1.0, confirmed:
          .2375+.114+.152+.076+.114+.076+.1805 = .95)
        Applies to WR only -- TE keeps the original (unmodified) WR/TE
        weights above, since guide_playcaller_stats.csv's WR PPG rank
        column is specifically a WR-context signal, not joined onto TE
        rows in join.py. playcaller_wr_ppg_rank is null for non-WR rows and
        for WRs on the four brand-new-playcaller teams (same as the RB
        case above), handled by the same missing-data threshold.

  If a component is missing (non-null) for less than 30% of a position
  group -- i.e. missing for >70% of it -- its weight is redistributed
  proportionally across the remaining available components, so weights
  still sum to 1.0. (Previously this only ever fired for snap_share when
  snap count data failed to pull; now it's also the normal path for
  guide_adj_ppg/ol_run_block_rank/proj_volume_rank on the ~50-80% of
  players per position who aren't covered by the guide.)

For K, there are now THREE tiers, checked in this order in main():
  1. Guide-matched kickers (data/guide_kicker_stats.csv, hand-transcribed
     REAL 2025 data from Joel Smyth's Draft Guide 2026 -- see
     compute_kicker_scores_with_guide()): real data beats proxy, so these
     kickers get a new composite where guide_ppg_25/guide_fg_acc dominate.
  2. Kickers with a 2024 play-by-play proxy match (data/kicker_stats.csv)
     but NO guide match: UNCHANGED from before -- same
     compute_kicker_scores() / K_COMPONENTS as always. This is the "proxy
     is a reasonable fallback for anyone the guide doesn't rank" case; the
     old logic is deliberately left untouched (not de-weighted) here, only
     de-weighted when it's blended alongside guide data in tier 1.
  3. Neither: same ADP-only fallback as DEF/QB/RB/WR/TE's "without_stats"
     path.

  Tier 2, UNCHANGED (real per-kicker stats from 2024 play-by-play, via
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
  Kickers with no 2024 play-by-play match at all (no fg_pct) AND no guide
  match skip straight to the same ADP-only fallback described below for
  DEF, same as QB/RB/WR/TE's "without_stats" path.

  Tier 1 (real data beats proxy -- see compute_kicker_scores_with_guide()
  and GUIDE_K_COMPONENTS below): guide_ppg_25 (real 2025 PPG) and
  guide_fg_acc (real 2025 FG%) are REAL, more recent, human-verified
  numbers -- not a 2024-pbp reconstruction -- so they take over as the
  dominant signal (0.25 + 0.20 = 0.45 of the composite). The tier-2 proxy
  components (fg_pct, fg_pct_50plus, is_dome, team_offense_ppg) are kept,
  not deleted, but DE-WEIGHTED from their tier-2 weights (0.40/0.25/0.10/
  0.10 -> 0.10/0.07/0.02/0.05) to a secondary/tie-breaking role in this
  blended tier, since the guide doesn't cover 100% of kickers and this
  proxy logic remains the fallback for whoever it misses (tier 2 above).
  guide_off_rank (team's projected 2026 offensive rank, 1=best -- inverted
  before z-scoring), guide_go_pct_rank (team's 4th-down go-for-it rate rank
  in FG range; NOT inverted -- a HIGH rank number here means a
  conservative team that kicks more field goals, which is good for a
  kicker, per the guide file's own column semantics), guide_is_50plus_good,
  and guide_is_value are secondary analyst-judgment signals. See
  GUIDE_K_COMPONENTS for the exact weights (sums to 1.0 including the
  de-weighted tier-2 proxy terms and a small adp_anchor stabilizer).

For DEF (REAL analyst data from Joel Smyth's Draft Guide 2026, hand-
transcribed into data/guide_def_stats.csv, now PRIMARY; the play-by-play-
derived proxy in data/defense_stats.csv / build_defense_stats.py is kept
as a SECONDARY/tie-breaking signal, not deleted, since the guide doesn't
cover every situation the proxy could theoretically be regenerated for):
  Real data beats proxy: pr_roe_rank (real Pressure Rate over Expected
  rank -- the actual metric the old def_pressure_rate_proxy could only
  approximate via sacks-per-dropback) and adj_ppg_25_rank (the analyst's
  own 2025 adjusted-PPG rank, built from real box scores rather than this
  pipeline's 2024-pbp custom-scoring reconstruction) are both REAL,
  analyst-computed numbers now available from guide_def_stats.csv, so they
  take over as the dominant signal. The old def_custom_adjusted_ppg /
  def_pressure_rate_proxy pair is de-weighted (0.60+0.20 -> 0.15+0.10) to a
  secondary/tie-breaking role rather than removed -- guide_def_stats.csv
  only covers this season's 32 teams as ranked by one analyst, so keeping
  the pbp-derived proxy alive is a reasonable fallback/blend, not dead
  weight. bottom10_offense and trend are two additional REAL analyst
  judgments (strength of opposing offenses faced, offseason roster
  trajectory) that build_defense_stats.py's docstring explicitly flagged
  as gaps it could NOT derive from play-by-play data alone -- the guide
  fills both gaps directly, so they're folded in as a small flag-based
  bonus/malus (not z-scored/weighted like the rank-based components, since
  they're binary/categorical, not continuous). Composite:
        0.35  guide_pr_roe_rank      -- REAL pressure-rate rank (1=best),
                                         inverted (z-score of -rank) so a
                                         lower/better rank scores higher.
                                         Co-primary signal: this is the
                                         actual metric the old proxy only
                                         approximated.
        0.35  guide_adj_ppg_25_rank  -- REAL analyst adjusted-PPG rank
                                         (1=best), inverted the same way.
                                         Co-primary signal alongside
                                         pressure rank -- both come
                                         straight from the analyst's own
                                         published numbers, not a 2024-pbp
                                         reconstruction.
        0.15  def_custom_adjusted_ppg -- DE-WEIGHTED (was 0.60). Still
                                         this league's exact custom
                                         scoring rules averaged per game,
                                         kept as a secondary/tie-breaking
                                         signal and as the fallback engine
                                         for any team the guide doesn't
                                         cover.
        0.10  def_pressure_rate_proxy -- DE-WEIGHTED (was 0.20). Sacks per
                                         opponent pass attempt faced --
                                         now redundant with the REAL
                                         pr_roe_rank above, kept at low
                                         weight purely as a tie-breaker.
        0.05  turnover/sack volume    -- DE-WEIGHTED (was 0.20). Same
                                         def_sacks + def_interceptions +
                                         def_forced_fumbles z-score as
                                         before, now a minor tie-breaker.
  Plus two small additive (not z-scored) modifiers layered on top of the
  weighted composite above, each on the same rough +/-1-z-unit scale so
  neither dominates the rank-based signal:
        -0.15  bottom10_offense flag -- REAL analyst red flag: this
                                         defense faces a lot of bad
                                         offenses, per Joel Smyth's Draft
                                         Guide 2026 (their stated
                                         reasoning, not re-derived here).
                                         Applied when bottom10_offense==1.
        +0.05 / -0.05  trend         -- REAL offseason-trajectory signal:
                                         up = +0.05, down = -0.05,
                                         flat = 0. Fills the "offseason
                                         roster change" gap
                                         build_defense_stats.py flagged
                                         as undoable from pbp data alone.
  If guide_pr_roe_rank is null for a team (not in guide_def_stats.csv),
  that team's composite runs on def_custom_adjusted_ppg /
  def_pressure_rate_proxy / turnover volume alone (their now-secondary
  weights, renormalized) rather than losing the 0.70 of composite weight
  the guide columns would otherwise own -- see compute_defense_scores().
  If def_custom_adjusted_ppg is ALSO null for a team (e.g.
  defense_stats.csv hasn't been generated -- build_defense_stats.py not
  run), that team falls back further to the pure ADP-based rank (inverted
  so lower ADP = higher score) previously used for all of DEF. Documented
  in data/README.md.

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
            "ppg": 0.36,
            "volume": 0.12,
            "rushing_ppg": 0.12,
            "snap_share": 0.12,
            "efficiency": 0.08,
            "guide_adj_ppg": 0.14,
            # Real per-player "Luck Metric" (data/guide_luck_metric.csv):
            # pct of season fantasy points lost/gained to variance (OT
            # points, penalty-nullified plays, dropped TDs, tackled-at-1,
            # etc). Positive = unlucky = positive-regression candidate, so
            # this is added WITHOUT inverting -- a higher (more positive)
            # value should raise the composite. Small weight since it only
            # covers a top/bottom-25 subset (~50 players total across all
            # 4 positions) and the existing >=30%-non-null threshold below
            # will drop it entirely for positions where coverage is too
            # sparse to trust, same protection as every other optional
            # component in this file.
            "pct_pts_lost_to_luck": 0.06,
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
        if pos == "RB":
            # RB carves out room for two RB-only guide-sourced components
            # (ol_run_block_rank, proj_volume_rank) on top of guide_adj_ppg
            # -- see module docstring "RB-only additions" for the full
            # reasoning and the redistribution math.
            #
            # ol_run_block_rank: REAL 2025 team run-blocking rank (1=best),
            # from data/guide_ol_stats.csv joined by team in join.py.
            # Inverted (lower/better rank -> higher value) so it points the
            # same direction as every other z-scored component.
            df["ol_run_block_rank"] = -df["guide_ol_run_block_rank_2025"]

            # proj_volume_rank: the analyst's own forward-looking 2026 RB
            # volume projection (targets + goal-line attempts weighted),
            # from data/guide_rb_volume.csv joined by name in join.py.
            # Distinct from red_zone_share (backward-looking, from 2024
            # play-by-play) -- this is a forward projection, genuinely
            # additional signal. Inverted like ol_run_block_rank above.
            df["proj_volume_rank"] = -df["guide_proj_volume_rank"]

            # playcaller_rb_ppg_rank / pct_rb1_rank: REAL team-level
            # playcaller/scheme context from data/guide_playcaller_stats.csv
            # (joined by team in join.py) -- see module docstring "RB-only
            # playcaller/scheme additions" for the full reasoning and
            # redistribution math. Both are 1=best ranks, inverted before
            # z-scoring so a lower/better rank scores higher, matching every
            # other rank-based component here.
            df["playcaller_rb_ppg_rank_inv"] = -df["playcaller_rb_ppg_rank"]
            df["pct_rb1_rank_inv"] = -df["pct_rb1_rank"]

            components = {
                "ppg": 0.174,
                "rushing_ppg": 0.087,
                "receiving_ppg": 0.113,
                "red_zone_share": 0.061,
                "snap_share": 0.087,
                "efficiency": 0.061,
                "guide_adj_ppg": 0.130,
                "ol_run_block_rank": 0.061,
                "proj_volume_rank": 0.096,
                "playcaller_rb_ppg_rank_inv": 0.038,
                "pct_rb1_rank_inv": 0.038,
                "pct_pts_lost_to_luck": 0.054,  # see QB block above for reasoning
            }
        elif pos == "WR":
            # playcaller_wr_ppg_rank: REAL team-level playcaller context
            # from data/guide_playcaller_stats.csv (joined by team in
            # join.py) -- see module docstring "WR-only playcaller
            # addition" for the redistribution math. Inverted before
            # z-scoring like the RB version above.
            df["playcaller_wr_ppg_rank_inv"] = -df["playcaller_wr_ppg_rank"]

            components = {
                "ppg": 0.223,
                "rushing_ppg": 0.107,
                "receiving_ppg": 0.143,
                "red_zone_share": 0.071,
                "snap_share": 0.107,
                "efficiency": 0.071,
                "guide_adj_ppg": 0.170,
                "playcaller_wr_ppg_rank_inv": 0.047,
                "pct_pts_lost_to_luck": 0.061,  # see QB block above for reasoning
            }
        else:
            components = {
                "ppg": 0.235,
                "rushing_ppg": 0.113,
                "receiving_ppg": 0.150,
                "red_zone_share": 0.075,
                "snap_share": 0.113,
                "efficiency": 0.075,
                "guide_adj_ppg": 0.179,
                "pct_pts_lost_to_luck": 0.06,  # see QB block above for reasoning
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


# Guide-blended K composite weights -- see module docstring "Tier 1" for
# the full reasoning. guide_ppg_25/guide_fg_acc (REAL 2025 numbers from
# Joel Smyth's Draft Guide 2026) dominate; the tier-2 play-by-play proxy
# terms (fg_pct/fg_pct_50plus/is_dome/team_offense_ppg) are kept but
# DE-WEIGHTED to secondary/tie-breaking weight rather than removed, per
# "real data beats proxy, but the proxy is a reasonable fallback." Sums to
# 1.0.
GUIDE_K_COMPONENTS = {
    "guide_ppg_25": 0.25,
    "guide_fg_acc": 0.20,
    "guide_off_rank_inv": 0.08,
    "guide_go_pct_rank": 0.05,
    "guide_is_50plus_good": 0.05,
    "guide_is_value": 0.05,
    "guide_is_dome": 0.03,
    "fg_pct": 0.10,           # de-weighted from 0.40 (tier 2)
    "fg_pct_50plus": 0.07,    # de-weighted from 0.25 (tier 2)
    "is_dome": 0.02,          # de-weighted from 0.10 (tier 2); mostly
                               # redundant with guide_is_dome above
    "team_offense_ppg": 0.05,  # de-weighted from 0.10 (tier 2)
    "adp_anchor": 0.05,        # de-weighted from 0.15 (tier 2) -- still a
                               # stabilizer, now a minor one since real
                               # guide data is doing most of the work
}


def compute_kicker_scores_with_guide(df: pd.DataFrame) -> pd.DataFrame:
    """Blended real-guide-data + play-by-play-proxy composite for K rows
    matched in data/guide_kicker_stats.csv (guide_off_rank non-null; see
    join.py). A kicker in this subset may or may not ALSO have a
    kicker_stats.csv proxy match -- if not, the proxy components below are
    simply null and get median-imputed via the same missing-data handling
    used everywhere else in this file (compute_stat_group_scores'
    fillna(median) + <30%-non-null weight redistribution pattern), so a
    guide-only kicker's score still leans entirely on real guide signal
    instead of being dragged toward a meaningless imputed proxy value.
    """
    df = df.copy()

    # Same Bayesian shrinkage as the proxy-only path above, reused here so
    # the de-weighted fg_pct/fg_pct_50plus terms are still stabilized
    # against small samples the same way.
    fg_pct_shrunk = _shrink_pct(df["fg_made"], df["fg_attempts"], FG_PCT_PRIOR_WEIGHT)
    fg50_made = df["fg_made_50plus"].fillna(0)
    fg50_attempts = df["fg_attempts_50plus"].fillna(0)
    fg_pct_50plus_shrunk = _shrink_pct(fg50_made, fg50_attempts, FG50_PRIOR_WEIGHT)
    fg_pct_50plus_shrunk = fg_pct_50plus_shrunk.where(fg50_attempts > 0, fg_pct_shrunk)
    df["fg_pct"] = fg_pct_shrunk
    df["fg_pct_50plus"] = fg_pct_50plus_shrunk

    df["is_dome"] = df["is_dome"].astype(float)
    df["guide_is_dome"] = df["guide_is_dome"].astype(float)
    df["adp_anchor"] = -df["adp"]

    # guide_off_rank: 1 = best projected 2026 offense. Invert so a better
    # (lower-number) offense rank scores higher, matching every other
    # z-scored component here.
    df["guide_off_rank_inv"] = -df["guide_off_rank"]
    # guide_go_pct_rank is used AS-IS, not inverted -- per
    # guide_kicker_stats.csv's own column semantics, a HIGH rank number
    # here means a conservative team that goes for the field goal more
    # often in FG range, which is good for the kicker (more attempts).

    available = {}
    for col, weight in GUIDE_K_COMPONENTS.items():
        non_null_frac = df[col].notna().mean()
        if non_null_frac >= 0.3:
            available[col] = weight

    weight_sum = sum(available.values())
    if weight_sum == 0:
        available = {"guide_ppg_25": 1.0}
        weight_sum = 1.0

    composite = pd.Series(np.zeros(len(df)), index=df.index)
    for col, weight in available.items():
        z = zscore(df[col].fillna(df[col].median()))
        composite += z * (weight / weight_sum)

    df["value_score"] = composite
    return df


# DEF composite weights -- see module docstring for the full reasoning
# behind each weight. guide_pr_roe_rank/guide_adj_ppg_25_rank (REAL analyst
# data from Joel Smyth's Draft Guide 2026) are now co-primary; the old
# play-by-play-derived proxy trio is kept but DE-WEIGHTED to a secondary/
# tie-breaking role (real data beats proxy, but the proxy stays alive as a
# fallback for any team the guide doesn't cover). Sums to 1.0.
DEF_COMPONENTS = {
    "guide_pr_roe_rank": 0.35,       # REAL pressure-rate rank (co-primary)
    "guide_adj_ppg_25_rank": 0.35,   # REAL adjusted-PPG rank (co-primary)
    "def_custom_adjusted_ppg": 0.15,  # de-weighted from 0.60
    "def_pressure_rate_proxy": 0.10,  # de-weighted from 0.20
    "def_turnover_volume": 0.05,      # de-weighted from 0.20
}

# Additive (not z-scored/weighted) modifiers layered on top of the
# DEF_COMPONENTS composite -- both are REAL analyst judgments from the
# guide that build_defense_stats.py's docstring flagged as gaps it could
# NOT derive from 2024 play-by-play data alone (opponent-offense strength
# faced, offseason roster trajectory). Binary/categorical, so they're
# applied as flat bonuses/maluses on the same rough +/-1-z-unit scale as
# the weighted components, rather than being z-scored themselves.
DEF_BOTTOM10_OFFENSE_MALUS = -0.15
DEF_TREND_MODIFIER = {"up": 0.05, "down": -0.05, "flat": 0.0}


def compute_defense_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Real per-team composite for DEF, replacing the old pure-ADP
    fallback. Requires def_custom_adjusted_ppg to be non-null (i.e. the
    team had a defense_stats.csv row) -- callers should route teams
    without a match to the ADP-only fallback instead, same as
    QB/RB/WR/TE's "without_stats" path. guide_def_stats.csv columns
    (guide_pr_roe_rank etc.) may be null for a team not in the guide --
    handled the same way as any other missing component (median-imputed,
    or its weight redistributed if missing for >70% of this group).
    """
    df = df.copy()

    df["def_turnover_volume"] = (
        df["def_sacks"].fillna(0)
        + df["def_interceptions"].fillna(0)
        + df["def_forced_fumbles"].fillna(0)
    )

    # guide_pr_roe_rank / guide_adj_ppg_25_rank: 1 = best. Invert so a
    # better (lower-number) rank scores higher, matching every other
    # z-scored component here.
    df["guide_pr_roe_rank_inv"] = -df["guide_pr_roe_rank"]
    df["guide_adj_ppg_25_rank_inv"] = -df["guide_adj_ppg_25_rank"]

    components = {
        "guide_pr_roe_rank_inv": DEF_COMPONENTS["guide_pr_roe_rank"],
        "guide_adj_ppg_25_rank_inv": DEF_COMPONENTS["guide_adj_ppg_25_rank"],
        "def_custom_adjusted_ppg": DEF_COMPONENTS["def_custom_adjusted_ppg"],
        "def_pressure_rate_proxy": DEF_COMPONENTS["def_pressure_rate_proxy"],
        "def_turnover_volume": DEF_COMPONENTS["def_turnover_volume"],
    }

    available = {}
    for col, weight in components.items():
        non_null_frac = df[col].notna().mean()
        if non_null_frac >= 0.3:
            available[col] = weight

    weight_sum = sum(available.values())
    if weight_sum == 0:
        available = {"def_custom_adjusted_ppg": 1.0}
        weight_sum = 1.0

    composite = pd.Series(np.zeros(len(df)), index=df.index)
    for col, weight in available.items():
        z = zscore(df[col].fillna(df[col].median()))
        composite += z * (weight / weight_sum)

    # Flat additive modifiers -- see DEF_BOTTOM10_OFFENSE_MALUS /
    # DEF_TREND_MODIFIER comments above. Teams with no guide match get 0
    # for both (fillna(0) / unmapped trend -> 0), i.e. no modifier applied.
    bottom10 = df["guide_bottom10_offense"].fillna(0)
    composite = composite + bottom10 * DEF_BOTTOM10_OFFENSE_MALUS
    trend_modifier = df["guide_trend"].map(DEF_TREND_MODIFIER).fillna(0.0)
    composite = composite + trend_modifier

    df["value_score"] = composite
    return df


def main():
    df = pd.read_csv(DATA_DIR / "joined.csv")

    scored_parts = []
    for pos, group in df.groupby("position"):
        group = group.copy()
        if pos == "K":
            # Three tiers -- see module docstring "For K, there are now
            # THREE tiers":
            #   1. guide-matched (guide_off_rank non-null): real-data-
            #      dominant blended composite, regardless of whether a
            #      proxy (fg_pct) match also exists.
            #   2. proxy-only (fg_pct non-null, no guide match): UNCHANGED
            #      compute_kicker_scores() -- the guide doesn't cover every
            #      kicker, so this stays a reasonable fallback.
            #   3. neither: ADP-only fallback, same as everywhere else.
            has_guide = group["guide_off_rank"].notna()
            has_stats = group["fg_pct"].notna()

            guide_rows = group[has_guide]
            proxy_only_rows = group[~has_guide & has_stats]
            no_stats_rows = group[~has_guide & ~has_stats]

            scored_pieces = []
            if len(guide_rows) > 0:
                scored_pieces.append(compute_kicker_scores_with_guide(guide_rows))
            if len(proxy_only_rows) > 0:
                scored_pieces.append(compute_kicker_scores(proxy_only_rows))

            with_stats = (
                pd.concat(scored_pieces, ignore_index=True)
                if scored_pieces
                else proxy_only_rows.copy()
            )

            if len(no_stats_rows) > 0:
                no_stats_rows = no_stats_rows.copy()
                min_score = with_stats["value_score"].min() if len(with_stats) else 0
                adp_rank = no_stats_rows["adp"].rank(method="min", ascending=True)
                no_stats_rows["value_score"] = min_score - 0.01 * adp_rank

            group = pd.concat([with_stats, no_stats_rows], ignore_index=True)
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
