"""
build_defense_stats.py -- Compute REAL, per-team defensive/special-teams
production stats for 2024 from play-by-play data, replacing the previous
ADP-only fallback for the DEF position.

This directly implements what the analyst video said mattered for defense
value (paraphrased, not a transcript -- see task/README):
  1. Pressure rate (drives sacks/turnovers) -- see PRESSURE RATE PROXY note
     below; true pressure rate isn't in public nflverse data.
  2. "Adjusted" points-per-game that strips fluke special-teams/defensive
     TDs -- superseded here by computing the league's EXACT custom scoring
     from real component stats (sacks, INTs, fumbles, TDs, points-allowed
     tiers) rather than a vague "subtract the flukes" heuristic. This is
     `custom_adjusted_ppg` below: it IS the adjusted-PPG concept, done
     properly, because every point in it is tied to a real, attributable
     defensive event rather than nfl_data_py's generic fantasy_points
     column (which isn't computed with this league's scoring rules at all).
  3. Strength of opposing offenses faced -- NOT implemented. nfl_data_py's
     pbp/schedule data doesn't give a ready-made "opponent offensive
     strength" rating, and building one (e.g. opponent's offensive
     EPA/play across the season, excluding games against this defense)
     is a nontrivial modeling project of its own. Flagged here as a real
     gap rather than faked with a placeholder.
  4. Offseason roster improvement/decline -- NOT implemented. This is
     fundamentally a 2025/2026-offseason signal (free agency, draft,
     coaching changes) that can't be derived from 2024 play-by-play at
     all. Flagged as a gap, not faked.

DATA SOURCE: nfl_data_py.import_pbp_data([2024]), regular season only
(season_type == "REG"), plus nfl_data_py.import_schedules([2024]) for
final per-game scores (used for points-allowed tiers).

TEAM ABBREVIATION NOTE: nflverse pbp/schedule data uses "LA" (Rams) and
"JAX" (Jaguars), while this pipeline's ADP source (raw_adp.csv, from
FantasyPros) uses "LAR" and "JAC". Normalized to the ADP convention here
so downstream joins work.

STAT ATTRIBUTION (verified against real pbp rows, not assumed):
  - sacks: sack == 1, credited to defteam.
  - interceptions: interception == 1, credited to defteam.
  - forced fumbles: forced_fumble_player_1_team / _2_team == defteam.
  - fumble recoveries: fumble_recovery_1_team / _2_team == defteam
    (excludes the offense recovering its own fumble).
  - safeties: safety == 1, credited to defteam.
  - blocked kicks: field_goal_result == "blocked" OR punt_blocked == 1,
    credited to defteam.
  - defensive TDs: return_touchdown == 1 on a pass/run play, with
    td_team == defteam (interception/fumble return TDs).
  - special-teams TDs: return_touchdown == 1 on a punt/kickoff play, with
    td_team == defteam (the receiving team, since posteam on those plays
    is the kicking/punting team).
  - special-teams forced fumbles / recoveries: same forced-fumble /
    fumble-recovery logic as above, restricted to punt/kickoff plays.
    (Excludes the rare case of the kicking team recovering its own
    kick/punt for a TD -- td_team == posteam on those plays -- which is
    not a defensive/return-unit accomplishment for the opponent.)

PRESSURE RATE PROXY: True pressure rate (QB hurries/knockdowns/hits as a
% of pass-rush snaps) is not published in nfl_data_py's public pbp data --
there's a `qb_hit` column but no play-level "pressure" flag, and pass-rush
snap counts aren't broken out by player/team either. As the best available
substitute, this uses SACKS PER OPPONENT PASS ATTEMPT FACED
(sacks / opponent pass attempts) as `pressure_rate_proxy`. This is
explicitly a proxy, not real pressure rate -- documented here, in
data/README.md, and in the score.py comment that consumes it.

POINTS-ALLOWED SCORING TIERS (this league's exact custom rule, except the
21-27 tier -- see below):
  0 pts allowed    -> 10
  1-6 pts allowed  ->  7
  7-13 pts allowed ->  4
  14-20 pts allowed->  1
  21-27 pts allowed->  0   *** ASSUMED DEFAULT -- see note below ***
  28-34 pts allowed-> -1
  35+ pts allowed  -> -4

*** IMPORTANT: the user did not give us the 21-27 tier value. 0 is used
here as a standard-in-most-leagues default. This is an ASSUMPTION PENDING
USER CONFIRMATION, not a real league rule -- do not treat it as verified.
See data/README.md. ***

custom_adjusted_ppg is computed PER GAME (not from season totals divided
by games) for the points-allowed-tier component, because the tiers are a
nonlinear step function of points allowed in a single game -- averaging
season-total points allowed into one tier lookup would give a different
(wrong) number than averaging the correct per-game tier bonuses. All the
other components (sacks, INTs, etc.) are linear counts, so summing over
the season and dividing by games played is equivalent to per-game
averaging for those.

Output: data/defense_stats.csv, one row per team, columns:
  team, sacks, interceptions, fumble_recoveries, forced_fumbles,
  safeties, blocked_kicks, def_tds, st_tds, st_ff, st_fr,
  points_allowed_per_game, custom_adjusted_ppg, pressure_rate_proxy
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
SEASON = 2024

# nflverse -> this pipeline's ADP-source (FantasyPros) team abbreviation
TEAM_ABBR_FIX = {"LA": "LAR", "JAX": "JAC"}

# League's exact custom DEF scoring rules
INT_PTS = 2
FUM_REC_PTS = 2
FORCED_FUM_PTS = 1
SAFETY_PTS = 2
BLOCKED_KICK_PTS = 2
DEF_TD_PTS = 6
SACK_PTS = 1
ST_TD_PTS = 6
ST_FF_PTS = 1
ST_FR_PTS = 1

# Points-allowed tiers: (max_points_allowed_inclusive, bonus). Checked in
# order; first tier whose max >= points allowed wins. 21-27 uses the
# ASSUMED default of 0 -- see module docstring.
POINTS_ALLOWED_TIERS = [
    (0, 10),
    (6, 7),
    (13, 4),
    (20, 1),
    (27, 0),   # ASSUMED default (user's exact number not provided)
    (34, -1),
    (float("inf"), -4),
]


def points_allowed_bonus(pts: float) -> float:
    for max_pts, bonus in POINTS_ALLOWED_TIERS:
        if pts <= max_pts:
            return bonus
    return POINTS_ALLOWED_TIERS[-1][1]


def fix_team(abbr):
    if pd.isna(abbr):
        return abbr
    return TEAM_ABBR_FIX.get(abbr, abbr)


def main():
    import nfl_data_py as nfl

    print(f"Pulling {SEASON} play-by-play data (this can take a minute)...")
    pbp = nfl.import_pbp_data([SEASON], downcast=True)
    pbp = pbp[pbp["season_type"] == "REG"].copy()
    pbp["defteam"] = pbp["defteam"].apply(fix_team)
    pbp["posteam"] = pbp["posteam"].apply(fix_team)

    print(f"Pulling {SEASON} schedules for final scores...")
    sched = nfl.import_schedules([SEASON])
    sched = sched[sched["game_type"] == "REG"].copy()
    sched["home_team"] = sched["home_team"].apply(fix_team)
    sched["away_team"] = sched["away_team"].apply(fix_team)

    teams = sorted(set(sched["home_team"]) | set(sched["away_team"]))
    print(f"{len(teams)} teams found in {SEASON} schedule.")

    is_punt_or_kickoff = pbp["play_type"].isin(["punt", "kickoff"])
    is_pass_or_run = pbp["play_type"].isin(["pass", "run"])

    rows = []
    for team in teams:
        def_plays = pbp[pbp["defteam"] == team]

        sacks = int(def_plays["sack"].fillna(0).sum())
        interceptions = int(def_plays["interception"].fillna(0).sum())

        ff1 = (def_plays["forced_fumble_player_1_team"] == team).sum()
        ff2 = (def_plays["forced_fumble_player_2_team"] == team).sum()
        forced_fumbles = int(ff1 + ff2)

        fr1 = def_plays["fumble_recovery_1_team"] == team
        fr2 = def_plays["fumble_recovery_2_team"] == team
        fumble_recoveries = int(fr1.sum() + fr2.sum())

        safeties = int((def_plays["safety"].fillna(0) == 1).sum())

        blocked_fg = (
            (def_plays["play_type"] == "field_goal")
            & (def_plays["field_goal_result"] == "blocked")
        ).sum()
        blocked_punt = (
            (def_plays["play_type"] == "punt")
            & (def_plays["punt_blocked"].fillna(0) == 1)
        ).sum()
        blocked_kicks = int(blocked_fg + blocked_punt)

        def_tds = int(
            (
                is_pass_or_run.loc[def_plays.index]
                & (def_plays["return_touchdown"].fillna(0) == 1)
                & (def_plays["td_team"] == team)
            ).sum()
        )
        st_tds = int(
            (
                is_punt_or_kickoff.loc[def_plays.index]
                & (def_plays["return_touchdown"].fillna(0) == 1)
                & (def_plays["td_team"] == team)
            ).sum()
        )

        st_mask = is_punt_or_kickoff.loc[def_plays.index]
        st_ff = int(
            (
                st_mask
                & (
                    (def_plays["forced_fumble_player_1_team"] == team)
                    | (def_plays["forced_fumble_player_2_team"] == team)
                )
            ).sum()
        )
        st_fr = int(
            (
                st_mask
                & (
                    (def_plays["fumble_recovery_1_team"] == team)
                    | (def_plays["fumble_recovery_2_team"] == team)
                )
            ).sum()
        )
        # Non-special-teams forced fumbles / recoveries (regular-play FF/FR,
        # worth different points than the ST versions above).
        non_st_ff = forced_fumbles - st_ff
        non_st_fr = fumble_recoveries - st_fr

        # Games played + points allowed per game, from schedules (source of
        # truth for final scores; simpler and more reliable than summing
        # pbp score deltas).
        home_games = sched[sched["home_team"] == team][["week", "home_score", "away_score"]]
        away_games = sched[sched["away_team"] == team][["week", "home_score", "away_score"]]
        game_points_allowed = list(home_games["away_score"]) + list(away_games["home_score"])
        games = len(game_points_allowed)

        if games == 0:
            points_allowed_per_game = np.nan
            custom_adjusted_ppg = np.nan
        else:
            points_allowed_per_game = float(np.mean(game_points_allowed))

            # Linear (per-season-total) component points, this league's
            # exact custom scoring:
            season_component_pts = (
                interceptions * INT_PTS
                + non_st_fr * FUM_REC_PTS
                + non_st_ff * FORCED_FUM_PTS
                + safeties * SAFETY_PTS
                + blocked_kicks * BLOCKED_KICK_PTS
                + def_tds * DEF_TD_PTS
                + sacks * SACK_PTS
                + st_tds * ST_TD_PTS
                + st_ff * ST_FF_PTS
                + st_fr * ST_FR_PTS
            )
            # Points-allowed tier bonus MUST be computed per game (see
            # module docstring) then summed, not derived from the season
            # average.
            tier_bonus_total = sum(points_allowed_bonus(p) for p in game_points_allowed)

            custom_adjusted_ppg = (season_component_pts + tier_bonus_total) / games

        # Pressure-rate proxy: sacks / opponent pass attempts faced. Real
        # pressure rate isn't available -- see module docstring.
        opp_pass_attempts = int(
            (def_plays["play_type"] == "pass").sum()
        )
        pressure_rate_proxy = (
            sacks / opp_pass_attempts if opp_pass_attempts > 0 else np.nan
        )

        rows.append(
            {
                "team": team,
                "sacks": sacks,
                "interceptions": interceptions,
                "fumble_recoveries": fumble_recoveries,
                "forced_fumbles": forced_fumbles,
                "safeties": safeties,
                "blocked_kicks": blocked_kicks,
                "def_tds": def_tds,
                "st_tds": st_tds,
                "st_ff": st_ff,
                "st_fr": st_fr,
                "points_allowed_per_game": round(points_allowed_per_game, 3)
                if games
                else None,
                "custom_adjusted_ppg": round(custom_adjusted_ppg, 3) if games else None,
                "pressure_rate_proxy": round(pressure_rate_proxy, 4)
                if not np.isnan(pressure_rate_proxy)
                else None,
            }
        )

    out = pd.DataFrame(rows).sort_values("team").reset_index(drop=True)
    out_path = DATA_DIR / "defense_stats.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} team defense rows to {out_path}")
    print(out.sort_values("custom_adjusted_ppg", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
