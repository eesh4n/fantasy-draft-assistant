"""
pull_kicker_stats.py -- Build data/kicker_stats.csv: real, systematic
per-kicker production stats for the 2024 season, computed from play-by-play
data (nfl_data_py's seasonal aggregate data has zero kicker rows, so K has
historically been ADP-only -- see data/README.md / join.py docstring).

An analyst's video (paraphrased) identified the signals that actually drive
kicker fantasy value:
  1. Overall field-goal accuracy
  2. Field-goal accuracy from 50+ yards specifically (long-range reliability)
  3. Whether the kicker's home stadium is a dome / fixed-roof (weather-
     independent kicking conditions)
  4. Whether the team's offense is good/high-scoring (more scoring drives =
     more kick attempts, both FG and PAT)
  5. How often the team's coach goes for it on 4th down in FG range instead
     of kicking (aggressive coaches reduce kick volume)

This script computes (1), (2), (3), (4) from real 2024 data via
`nfl_data_py.import_pbp_data([2024])` and `import_schedules([2024])`.
Signal (5) -- 4th-down aggressiveness in FG range -- is computed too, as
`team_go_for_it_rate_fg_range`, but is NOT currently consumed by score.py's
composite (see score.py comment for why: it's a team-coaching signal that
would need to be validated against actual FG-attempt-suppression before
being weighted, and the analyst's stated priority order puts it last: "an
aggressive coach reduces kick volume" is a second-order modifier on the
team-offense signal already captured by team_offense_ppg). It's written to
this CSV so it's available for use.

Data sources (2024 season -- the base season this pipeline uses; 2025/2026
play-by-play isn't published yet, same reasoning as pull.py's STATS_SEASON):
  - nfl.import_pbp_data([2024]): play-by-play rows, filtered to
    field_goal_attempt==1 and extra_point_attempt==1 plays, plus 4th-down
    plays in FG range for the go-for-it rate.
  - nfl.import_schedules([2024]): REG season final scores, used for
    team_offense_ppg (points scored per game, both home and away games).

Dome / fixed-roof mapping (signal 3): nfl_data_py's own `roof` column on
import_schedules is per-GAME weather observation (e.g. "outdoors" vs
"closed" vs "dome" logged inconsistently game to game -- e.g. it shows MIN
as "outdoors" for one game despite US Bank Stadium being a fixed roof), not
a reliable per-STADIUM classification, and the task calls for the kicker's
HOME stadium type, not per-game roof state. So this uses a static,
hand-maintained team -> is_dome mapping of current (2024/2025 season) NFL
stadiums with a fixed or normally-closed retractable roof (weather-
independent kicking conditions). Verify/update this list if any team moves
stadiums or changes roof policy in a future season:
  ARI (State Farm Stadium, retractable, usually closed)
  ATL (Mercedes-Benz Stadium, retractable)
  DAL (AT&T Stadium, retractable)
  DET (Ford Field, fixed dome)
  HOU (NRG Stadium, retractable)
  IND (Lucas Oil Stadium, retractable)
  LV  (Allegiant Stadium, fixed dome)
  LA  (SoFi Stadium -- Rams; fixed translucent roof, not open-air)
  LAC (SoFi Stadium -- Chargers; same building as LA)
  MIN (U.S. Bank Stadium, fixed roof)
  NO  (Caesars Superdome, fixed dome)
All other teams play outdoors and are_dome=False.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
STATS_SEASON = 2024

# Minimum 50+ yard FG attempts required before we trust fg_pct_50plus as a
# real rate rather than noise (e.g. 1-for-1 = 100% is not a meaningful
# "elite from 50+" signal). 3 is a judgment call: enough attempts that a
# single make/miss doesn't swing the rate by 33+ points, while still being
# low enough that most attempting kickers clear the bar (most starting
# kickers see at least a handful of 50+ tries across a season).
MIN_50PLUS_ATTEMPTS = 3

DOME_TEAMS = {
    "ARI", "ATL", "DAL", "DET", "HOU", "IND", "LV", "LA", "LAC", "MIN", "NO",
}


def compute_team_offense_ppg(sched: pd.DataFrame) -> pd.Series:
    reg = sched[sched["game_type"] == "REG"].copy()
    home = reg[["home_team", "home_score"]].rename(
        columns={"home_team": "team", "home_score": "points"}
    )
    away = reg[["away_team", "away_score"]].rename(
        columns={"away_team": "team", "away_score": "points"}
    )
    all_games = pd.concat([home, away], ignore_index=True).dropna(subset=["points"])
    return all_games.groupby("team")["points"].mean()


def compute_go_for_it_rate(pbp: pd.DataFrame) -> pd.Series:
    """4th-down go-for-it rate when in field-goal range (yardline <= 40,
    i.e. roughly a <=57-yard attempt) and not already trailing/leading by a
    margin that makes the decision obvious garbage time (kept simple: no
    score-differential filter, this is a rough team-tendency proxy, not a
    win-probability model)."""
    reg = pbp[pbp["season_type"] == "REG"].copy()
    fg_range = reg[
        (reg["down"] == 4)
        & (reg["yardline_100"] <= 40)
        & (reg["play_type"].isin(["field_goal", "run", "pass", "qb_kneel", "qb_spike", "punt"]))
    ].copy()
    is_go = fg_range["play_type"].isin(["run", "pass", "qb_kneel", "qb_spike"])
    fg_range["is_go"] = is_go
    rate = fg_range.groupby("posteam")["is_go"].mean()
    return rate


def main():
    import nfl_data_py as nfl

    print(f"Pulling {STATS_SEASON} play-by-play data for kicker stats...")
    pbp = nfl.import_pbp_data([STATS_SEASON], downcast=True, cache=False)
    pbp = pbp[pbp["season_type"] == "REG"].copy()

    print(f"Pulling {STATS_SEASON} schedules for team offense PPG...")
    sched = nfl.import_schedules([STATS_SEASON])
    offense_ppg = compute_team_offense_ppg(sched)
    go_for_it_rate = compute_go_for_it_rate(pbp)

    fg = pbp[pbp["field_goal_attempt"] == 1].copy()
    fg = fg[fg["kicker_player_name"].notna()]
    fg["made"] = (fg["field_goal_result"] == "made").astype(int)
    fg["is_50plus"] = fg["kick_distance"] >= 50

    pat = pbp[pbp["extra_point_attempt"] == 1].copy()
    pat = pat[pat["kicker_player_name"].notna()]
    pat["made"] = (pat["extra_point_result"] == "good").astype(int)

    # Group by (kicker_player_id, kicker_player_name, posteam) -- posteam as
    # a proxy for "team" since pbp has no separate kicker roster table here.
    # A kicker who was traded mid-season will show one row per team; keep
    # the team with the most attempts as their "team" for the dome/offense
    # lookup, matching how the rest of the pipeline treats team as a single
    # value per player.
    fg_group = (
        fg.groupby(["kicker_player_id", "kicker_player_name", "posteam"])
        .agg(
            fg_attempts=("made", "count"),
            fg_made=("made", "sum"),
        )
        .reset_index()
    )
    fg_50_group = (
        fg[fg["is_50plus"]]
        .groupby(["kicker_player_id", "kicker_player_name", "posteam"])
        .agg(
            fg_attempts_50plus=("made", "count"),
            fg_made_50plus=("made", "sum"),
        )
        .reset_index()
    )
    pat_group = (
        pat.groupby(["kicker_player_id", "kicker_player_name", "posteam"])
        .agg(
            pat_attempts=("made", "count"),
            pat_made=("made", "sum"),
        )
        .reset_index()
    )

    # Pick each kicker's primary team = the (id, name) with the most total
    # kicking events (FG+PAT attempts), then aggregate stats across all
    # teams they played for that season (handles in-season trades cleanly
    # instead of splitting one kicker into two rows).
    all_events = pd.concat(
        [
            fg[["kicker_player_id", "kicker_player_name", "posteam"]],
            pat[["kicker_player_id", "kicker_player_name", "posteam"]],
        ],
        ignore_index=True,
    )
    team_counts = (
        all_events.groupby(["kicker_player_id", "kicker_player_name", "posteam"])
        .size()
        .reset_index(name="n")
    )
    primary_team = (
        team_counts.sort_values("n", ascending=False)
        .drop_duplicates(["kicker_player_id", "kicker_player_name"], keep="first")
        [["kicker_player_id", "kicker_player_name", "posteam"]]
        .rename(columns={"posteam": "team"})
    )

    def agg_across_teams(g, cols):
        return g.groupby(["kicker_player_id", "kicker_player_name"])[cols].sum().reset_index()

    fg_totals = agg_across_teams(fg_group, ["fg_attempts", "fg_made"])
    fg_50_totals = agg_across_teams(fg_50_group, ["fg_attempts_50plus", "fg_made_50plus"])
    pat_totals = agg_across_teams(pat_group, ["pat_attempts", "pat_made"])

    out = primary_team.merge(fg_totals, on=["kicker_player_id", "kicker_player_name"], how="left")
    out = out.merge(fg_50_totals, on=["kicker_player_id", "kicker_player_name"], how="left")
    out = out.merge(pat_totals, on=["kicker_player_id", "kicker_player_name"], how="left")

    for c in ["fg_attempts", "fg_made", "fg_attempts_50plus", "fg_made_50plus", "pat_attempts", "pat_made"]:
        out[c] = out[c].fillna(0).astype(int)

    out["fg_pct"] = np.where(out["fg_attempts"] > 0, out["fg_made"] / out["fg_attempts"], np.nan)
    out["fg_pct_50plus"] = np.where(
        out["fg_attempts_50plus"] >= MIN_50PLUS_ATTEMPTS,
        out["fg_made_50plus"] / out["fg_attempts_50plus"].replace(0, np.nan),
        np.nan,
    )
    out["pat_pct"] = np.where(out["pat_attempts"] > 0, out["pat_made"] / out["pat_attempts"], np.nan)

    out["is_dome"] = out["team"].isin(DOME_TEAMS)
    out["team_offense_ppg"] = out["team"].map(offense_ppg)
    out["team_go_for_it_rate_fg_range"] = out["team"].map(go_for_it_rate)

    out = out.rename(columns={"kicker_player_name": "kicker_name"})
    out_cols = [
        "kicker_name",
        "team",
        "fg_attempts",
        "fg_made",
        "fg_pct",
        "fg_attempts_50plus",
        "fg_made_50plus",
        "fg_pct_50plus",
        "pat_attempts",
        "pat_made",
        "pat_pct",
        "is_dome",
        "team_offense_ppg",
        "team_go_for_it_rate_fg_range",
    ]
    out = out[out_cols].sort_values("fg_attempts", ascending=False).reset_index(drop=True)

    out_path = DATA_DIR / "kicker_stats.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} kickers to {out_path}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
