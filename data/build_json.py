"""
build_json.py -- Emit the final data/players.json contract consumed by the
UI. Schema (hard contract):

{
  "id": "string, unique, slug of name+team",
  "name": "string",
  "position": "QB|RB|WR|TE|K|DEF",
  "team": "string, team abbreviation",
  "value_score": number,
  "position_rank": integer (1-indexed, by value_score within position),
  "adp": number,
  "adp_position_rank": integer (1-indexed, by adp within position),
  "value_gap": integer (adp_position_rank - position_rank),
  "stats": {                          // raw inputs behind value_score, for
    "ppg": number|null,               // a "why this ranking" detail view.
    "volume": number|null,            // null for K/DEF (ADP-only model)
    "snap_share": number|null,        // and for rookies with no 2024 stats.
    "efficiency": number|null,
    "rushing_ppg": number|null,       // rushing-only fantasy pts/game
                                       // (QB/RB/WR/TE)
    "receiving_ppg": number|null,     // receiving-only fantasy pts/game
                                       // (RB/WR/TE only; null for QB, who
                                       // don't receive)
    "red_zone_share": number|null,    // REAL red-zone usage share: (red-
                                       // zone rush attempts + red-zone
                                       // targets) / (total carries +
                                       // targets), from 2024 play-by-play
                                       // data (data/redzone_stats.csv,
                                       // built by data/redzone.py;
                                       // RB/WR/TE only). Replaces the old
                                       // td_rate approximation.
    "goal_line_share": number|null,   // same, but for goal-line usage
                                       // specifically (yardline_100 <= 5,
                                       // a stricter cut than red zone's
                                       // <= 20). Not part of value_score,
                                       // surfaced for detail views only
                                       // (RB/WR/TE only).
    "fg_pct": number|null,            // K only. Real 2024 overall FG
                                       // accuracy (fg_made/fg_attempts)
                                       // from play-by-play data
                                       // (data/kicker_stats.csv, built by
                                       // data/pull_kicker_stats.py). null
                                       // for non-K positions and for K
                                       // rows with no 2024 pbp match
                                       // (ADP-only fallback).
    "fg_pct_50plus": number|null,     // K only. FG accuracy specifically
                                       // from 50+ yards; null if the
                                       // kicker had fewer than
                                       // MIN_50PLUS_ATTEMPTS (3) attempts
                                       // from that range in 2024 (too
                                       // noisy to trust as a rate) --
                                       // score.py falls back to fg_pct in
                                       // that case for the composite, but
                                       // this raw field stays null so the
                                       // UI can show "not enough attempts"
                                       // rather than a misleading number.
    "is_dome": boolean|null,          // K only. Whether the kicker's 2024
                                       // team's home stadium has a fixed
                                       // or normally-closed roof (see
                                       // data/pull_kicker_stats.py for the
                                       // team list and reasoning).
    "team_offense_ppg": number|null,  // K only. The kicker's 2024 team's
                                       // average points scored per game
                                       // (REG season, home+away), a proxy
                                       // for scoring-drive volume /
                                       // kick-attempt opportunity.
    // -- DEF-only fields below (null for every other position). Sourced
    // from data/defense_stats.csv (built by data/build_defense_stats.py
    // from 2024 play-by-play data) -- see that file's docstring for exact
    // stat attribution and the documented points-allowed-tier assumption.
    "sacks": number|null,
    "interceptions": number|null,     // DEF interceptions -- NOT the same
                                       // stat as a QB's thrown picks; no
                                       // collision since a given row only
                                       // ever populates one position's
                                       // fields.
    "fumble_recoveries": number|null,
    "forced_fumbles": number|null,
    "safeties": number|null,
    "blocked_kicks": number|null,
    "def_tds": number|null,
    "st_tds": number|null,
    "st_ff": number|null,
    "st_fr": number|null,
    "points_allowed_per_game": number|null,
    "custom_adjusted_ppg": number|null, // this league's EXACT custom DEF
                                       // scoring, averaged per game -- the
                                       // dominant input to DEF value_score
    "pressure_rate_proxy": number|null  // sacks per opponent pass attempt
                                       // faced; a PROXY for real pressure
                                       // rate, which isn't in public
                                       // nflverse data -- see
                                       // build_defense_stats.py
  }
}
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent


def clean_num(v):
    """None for NaN/missing, else a rounded float."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), 4)


def slugify(name: str, team: str) -> str:
    base = f"{name}-{team}".lower()
    base = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    return base


def main():
    df = pd.read_csv(DATA_DIR / "joined.csv")

    # Only keep rows with a usable ADP (unranked players aren't useful for
    # a draft assistant) and a team.
    df = df[df["adp"].notna()].copy()
    df["team"] = df["team"].fillna("FA")

    players = []
    seen_ids = {}
    for _, row in df.iterrows():
        name = str(row["player_name"])
        team = str(row["team"])
        pid = slugify(name, team)
        if pid in seen_ids:
            seen_ids[pid] += 1
            pid = f"{pid}-{seen_ids[pid]}"
        else:
            seen_ids[pid] = 0

        players.append(
            {
                "id": pid,
                "name": name,
                "position": row["position"],
                "team": team,
                "value_score": round(float(row["value_score"]), 4),
                "position_rank": int(row["position_rank"]),
                "adp": round(float(row["adp"]), 2),
                "adp_position_rank": int(row["adp_position_rank"]),
                "value_gap": int(row["value_gap"]),
                "stats": {
                    "ppg": clean_num(row.get("ppg")),
                    "volume": clean_num(row.get("volume")),
                    "snap_share": clean_num(row.get("snap_share")),
                    "efficiency": clean_num(row.get("efficiency")),
                    "rushing_ppg": clean_num(row.get("rushing_ppg")),
                    "receiving_ppg": clean_num(row.get("receiving_ppg")),
                    "red_zone_share": clean_num(row.get("red_zone_share")),
                    "goal_line_share": clean_num(row.get("goal_line_share")),
                    "fg_pct": clean_num(row.get("fg_pct")),
                    "fg_pct_50plus": clean_num(row.get("fg_pct_50plus")),
                    "is_dome": (
                        bool(row.get("is_dome"))
                        if pd.notna(row.get("is_dome"))
                        else None
                    ),
                    "team_offense_ppg": clean_num(row.get("team_offense_ppg")),
                    "sacks": clean_num(row.get("def_sacks")),
                    "interceptions": clean_num(row.get("def_interceptions")),
                    "fumble_recoveries": clean_num(row.get("def_fumble_recoveries")),
                    "forced_fumbles": clean_num(row.get("def_forced_fumbles")),
                    "safeties": clean_num(row.get("def_safeties")),
                    "blocked_kicks": clean_num(row.get("def_blocked_kicks")),
                    "def_tds": clean_num(row.get("def_def_tds")),
                    "st_tds": clean_num(row.get("def_st_tds")),
                    "st_ff": clean_num(row.get("def_st_ff")),
                    "st_fr": clean_num(row.get("def_st_fr")),
                    "points_allowed_per_game": clean_num(row.get("def_points_allowed_per_game")),
                    "custom_adjusted_ppg": clean_num(row.get("def_custom_adjusted_ppg")),
                    "pressure_rate_proxy": clean_num(row.get("def_pressure_rate_proxy")),
                },
            }
        )

    players.sort(key=lambda p: p["adp"])

    out_path = DATA_DIR / "players.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(players, f, indent=2)

    print(f"Wrote {len(players)} players to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
