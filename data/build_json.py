"""
build_json.py -- Emit the final data/players.json contract consumed by the
UI. Schema (hard contract):

{
  "id": "string, unique, slug of name+team",
  "name": "string",
  "position": "QB|RB|WR|TE|K|DEF",
  "team": "string, team abbreviation",
  "value_score": number,             // For QB/RB/WR/TE, now PRIMARILY
                                       // derived from a trained ML model's
                                       // ml_predicted_ppg (see stats.
                                       // ml_predicted_ppg below and score.
                                       // py's "ML value model" docstring
                                       // section) -- NOT a hand-weighted
                                       // formula anymore. K/DEF are
                                       // unchanged (own real-data models,
                                       // out of scope for the ML swap).
  "position_rank": integer (1-indexed, by value_score within position),
  "adp": number,
  "adp_position_rank": integer (1-indexed, by adp within position),
  "value_gap": integer (adp_position_rank - position_rank),
  "stats": {                          // raw inputs behind value_score, for
    "ppg": number|null,               // a "why this ranking" detail view.
    "volume": number|null,            // null for K/DEF (ADP-only model)
    "snap_share": number|null,        // and for rookies with no 2024 stats.
    "efficiency": number|null,
    "ml_predicted_ppg": number|null,  // QB/RB/WR/TE only. THE PRIMARY
                                       // signal in value_score now (see
                                       // score.py's module docstring "ML
                                       // value model"). Output of a real,
                                       // per-position scikit-learn model
                                       // (Ridge/ElasticNet, cross-validated
                                       // against a naive baseline and a
                                       // shallow GBM in train_ml_model.py,
                                       // trained on 7 seasons of nfl_data_py
                                       // history, 2018-2024) applied to this
                                       // player's CURRENT ppg/rushing_ppg/
                                       // receiving_ppg/red_zone_share/
                                       // snap_share/efficiency (RB/WR/TE) or
                                       // ppg/volume/rushing_ppg/snap_share/
                                       // efficiency (QB) profile -- i.e. the
                                       // model's own projection of this
                                       // player's NEXT-season PPG. Replaces
                                       // the old hand-weighted composite of
                                       // those raw components. null only if
                                       // no model file exists for this
                                       // position (train_ml_model.py not
                                       // yet run) or the player has no 2024
                                       // stat line at all.
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
    "guide_ppg_25": number|null,      // K only. REAL 2025 PPG for this
                                       // kicker, hand-transcribed from Joel
                                       // Smyth's Draft Guide 2026
                                       // (data/guide_kicker_stats.csv,
                                       // joined by last-name+team in
                                       // join.py). Now the dominant input
                                       // to K value_score for guide-matched
                                       // kickers (see score.py). null for
                                       // kickers with no guide match
                                       // (deep bench/practice squad) or
                                       // when the guide itself left it
                                       // blank (e.g. an injury replacement
                                       // with no 2025 games).
    "guide_fg_acc": number|null,      // K only. REAL 2025 field-goal
                                       // accuracy (0-1) from the same guide
                                       // file. Co-dominant with
                                       // guide_ppg_25 in value_score for
                                       // guide-matched kickers. Same
                                       // null-handling as guide_ppg_25.
    "guide_is_value": boolean|null,   // K only. The analyst's own "this is
                                       // a value pick" flag from the guide.
                                       // A secondary input to value_score.
    "guide_adj_ppg": number|null,     // QB/RB/WR/TE only. REAL analyst
                                       // context-adjusted 2025 PPG estimate
                                       // (e.g. stripping backup-QB games,
                                       // adding back missed-time production)
                                       // hand-transcribed from Joel Smyth's
                                       // Draft Guide 2026
                                       // (data/guide_adjusted_ppg.csv), a
                                       // full season more recent than this
                                       // pipeline's base 2024 stats. Now a
                                       // meaningfully-weighted component of
                                       // value_score (see score.py). null
                                       // for the ~50-70% of players per
                                       // position the guide doesn't cover
                                       // (it's a big-board preview, not
                                       // every player).
    "guide_adj_ppg_reason": string|null, // QB/RB/WR/TE only. Short analyst
                                       // note explaining the adjustment
                                       // above (e.g. "in complete games",
                                       // "with Joe Burrow") -- display-only,
                                       // not parsed/used in scoring.
    "ol_run_block_rank": number|null, // RB only. REAL 2025 offensive-line
                                       // run-blocking rank for the RB's
                                       // team (1 = best in the NFL), from
                                       // data/guide_ol_stats.csv (joined by
                                       // team in join.py). Lower is better.
                                       // A modest-weight component of RB
                                       // value_score (inverted internally --
                                       // see score.py); shown here as the
                                       // original 1-32 rank for readability.
    "proj_volume_rank": number|null,  // RB only. The analyst's own
                                       // projected 2026 RB volume rank
                                       // (targets + goal-line attempts
                                       // weighted), from
                                       // data/guide_rb_volume.csv (joined
                                       // by name in join.py). Lower is
                                       // better. Forward-looking, distinct
                                       // from the backward-looking
                                       // red_zone_share above -- a
                                       // meaningfully-weighted component of
                                       // RB value_score (inverted
                                       // internally -- see score.py).
    "adj_volume_25_rank": number|null, // RB only. Informational -- the
                                       // analyst's volume-only 2025 rank
                                       // (a volume analogue of
                                       // guide_adj_ppg). NOT part of
                                       // value_score. May be null for
                                       // rookies/small-sample players even
                                       // when proj_volume_rank is present.
    "guide_volume_confidence": string|null, // RB only. Informational --
                                       // the analyst's own confidence
                                       // ("high"/"medium"/"low") in the
                                       // proj_volume_rank projection above.
                                       // NOT part of value_score.
    "playcaller_rb_ppg_rank": number|null, // RB only. REAL rank (1=best)
                                       // of this team's average fantasy RB
                                       // PPG across the current
                                       // playcaller's tenure, hand-
                                       // transcribed from the Playcaller
                                       // Table page of Joel Smyth's Draft
                                       // Guide 2026
                                       // (data/guide_playcaller_stats.csv,
                                       // joined by team in join.py). A
                                       // small-weight component of RB
                                       // value_score (inverted internally
                                       // -- see score.py). null for the
                                       // four teams with a brand-new,
                                       // 0-season playcaller (no track
                                       // record to rank).
    "pct_rb1_rank": number|null,      // RB only. REAL rank (1=best) of
                                       // %RB1 -- the share of the team's RB
                                       // fantasy points going to the
                                       // starter (higher %RB1 = bellcow,
                                       // lower = committee). Same source/
                                       // join as playcaller_rb_ppg_rank. A
                                       // small-weight component of RB
                                       // value_score (inverted internally).
    "rb_screen_rank": number|null,    // RB only. Informational -- 2025 RB
                                       // screen-pass usage rank (1=most),
                                       // same source file. NOT part of
                                       // value_score (too qualitative to
                                       // weight confidently).
    "playcaller_wr_ppg_rank": number|null, // WR only. REAL rank (1=best)
                                       // of this team's average fantasy WR
                                       // PPG across the current
                                       // playcaller's tenure, same source/
                                       // join as playcaller_rb_ppg_rank
                                       // above. A small-weight component of
                                       // WR value_score (inverted
                                       // internally -- see score.py). null
                                       // for the four brand-new-playcaller
                                       // teams.
    "personnel": string|null,         // RB/WR. Informational -- the
                                       // playcaller's preferred personnel
                                       // grouping (e.g. "High 11P",
                                       // "High 21P", "--" if no standout
                                       // grouping), from
                                       // data/guide_playcaller_stats.csv.
                                       // NOT part of value_score.
    "run_scheme": string|null,        // RB only. Informational -- "Gap",
                                       // "Zone", or "Balanced" run-scheme
                                       // lean, same source file. NOT part
                                       // of value_score.
    "motion_rank": number|null,       // RB/WR. Informational -- 2025
                                       // pre-snap motion-usage rank
                                       // (1=most), same source file. NOT
                                       // part of value_score.
    "formation_spread": string|null,  // RB only. Informational --
                                       // condensed/even/spread formation
                                       // tendency (e.g. "Very Condensed",
                                       // "Even", "Very Spread"), same
                                       // source file. NOT part of
                                       // value_score.
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
    "pressure_rate_proxy": number|null, // sacks per opponent pass attempt
                                       // faced; a PROXY for real pressure
                                       // rate, which isn't in public
                                       // nflverse data -- see
                                       // build_defense_stats.py. DE-WEIGHTED
                                       // secondary/tie-breaking signal now
                                       // that guide_pressure_rank exists.
    "guide_pressure_rank": number|null, // DEF only. REAL Pressure Rate
                                       // over Expected rank (1=best), hand-
                                       // transcribed from Joel Smyth's
                                       // Draft Guide 2026
                                       // (data/guide_def_stats.csv, joined
                                       // by team in join.py). This is the
                                       // ACTUAL metric pressure_rate_proxy
                                       // above could only approximate --
                                       // now a co-primary input to DEF
                                       // value_score (see score.py). null
                                       // for a team not in the guide.
    "guide_adj_ppg_rank": number|null, // DEF only. The analyst's own 2025
                                       // adjusted-PPG rank (1=best), same
                                       // source/join as guide_pressure_rank
                                       // above. Co-primary with it in DEF
                                       // value_score.
    "guide_bottom10_offense": boolean|null, // DEF only. REAL analyst red
                                       // flag: this defense faces a lot of
                                       // bad offenses. Applied as a small
                                       // malus in DEF value_score (see
                                       // score.py).
    "guide_trend": string|null        // DEF only. "up"/"down"/"flat" --
                                       // the analyst's offseason roster
                                       // improvement/decline read. Applied
                                       // as a small bonus/penalty in DEF
                                       // value_score (see score.py).
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


def load_chart_notes():
    """data/guide_chart_notes.csv -- qualitative reads from the guide's
    scatter-plot charts (QB volume vs ADP, RB/WR efficiency, QB rushing,
    fantasy shootout, RB's dream QB). These charts have no labeled
    per-player axis values, so they can't be digitized into value_score --
    this is descriptive positioning ("elite in both rushing and receiving
    efficiency"), not an opinion/target-fade call, kept purely as a display
    note. Returns {player_name: (chart_source, chart_note)}, first match
    wins if a player appears on multiple charts.
    """
    path = DATA_DIR / "guide_chart_notes.csv"
    if not path.exists():
        return {}
    notes_df = pd.read_csv(path)
    result = {}
    for _, r in notes_df.iterrows():
        result.setdefault(r["player_name"], (r["chart_source"], r["chart_note"]))
    return result


def load_stat_notes():
    """data/guide_stat_notes.csv -- player-specific reasoning paraphrased
    from the guide's "Top 50 Stats" section (target-share splits, red-zone
    competition, efficiency trends, injury/schedule context, etc). This is
    the analyst's actual analytical reasoning, not just a ranking number --
    kept as a display-only note, same as chart notes, not scored.
    Returns {player_name: stat_note}.
    """
    path = DATA_DIR / "guide_stat_notes.csv"
    if not path.exists():
        return {}
    notes_df = pd.read_csv(path)
    return dict(zip(notes_df["player_name"], notes_df["stat_note"]))


def main():
    df = pd.read_csv(DATA_DIR / "joined.csv")
    chart_notes = load_chart_notes()
    stat_notes = load_stat_notes()

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
                    "ml_predicted_ppg": clean_num(row.get("ml_predicted_ppg")),
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
                    "guide_ppg_25": clean_num(row.get("guide_ppg_25")),
                    "guide_fg_acc": clean_num(row.get("guide_fg_acc")),
                    "guide_is_value": (
                        bool(row.get("guide_is_value"))
                        if pd.notna(row.get("guide_is_value"))
                        else None
                    ),
                    "guide_adj_ppg": clean_num(row.get("guide_adj_ppg")),
                    "guide_adj_ppg_reason": (
                        str(row.get("guide_reason"))
                        if pd.notna(row.get("guide_reason"))
                        else None
                    ),
                    "pct_pts_lost_to_luck": clean_num(row.get("pct_pts_lost_to_luck")),
                    "pts_lost_to_luck": clean_num(row.get("pts_lost_to_luck")),
                    "ol_run_block_rank": clean_num(row.get("guide_ol_run_block_rank_2025")),
                    "proj_volume_rank": clean_num(row.get("guide_proj_volume_rank")),
                    "adj_volume_25_rank": clean_num(row.get("guide_adj_volume_25_rank")),
                    "guide_volume_confidence": (
                        str(row.get("guide_confidence"))
                        if pd.notna(row.get("guide_confidence"))
                        else None
                    ),
                    "playcaller_rb_ppg_rank": clean_num(row.get("playcaller_rb_ppg_rank")),
                    "pct_rb1_rank": clean_num(row.get("pct_rb1_rank")),
                    "rb_screen_rank": clean_num(row.get("rb_screen_rank")),
                    "run_scheme": (
                        str(row.get("run_scheme"))
                        if pd.notna(row.get("run_scheme"))
                        else None
                    ),
                    "formation_spread": (
                        str(row.get("formation_spread"))
                        if pd.notna(row.get("formation_spread"))
                        else None
                    ),
                    "playcaller_wr_ppg_rank": clean_num(row.get("playcaller_wr_ppg_rank")),
                    "personnel": (
                        str(row.get("personnel"))
                        if pd.notna(row.get("personnel"))
                        else None
                    ),
                    "motion_rank": clean_num(row.get("motion_rank")),
                    "chart_note": (
                        chart_notes[name][1] if name in chart_notes else None
                    ),
                    "chart_source": (
                        chart_notes[name][0] if name in chart_notes else None
                    ),
                    "stat_note": stat_notes.get(name),
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
                    "guide_pressure_rank": clean_num(row.get("guide_pr_roe_rank")),
                    "guide_adj_ppg_rank": clean_num(row.get("guide_adj_ppg_25_rank")),
                    "guide_bottom10_offense": (
                        bool(row.get("guide_bottom10_offense"))
                        if pd.notna(row.get("guide_bottom10_offense"))
                        else None
                    ),
                    "guide_trend": (
                        str(row.get("guide_trend"))
                        if pd.notna(row.get("guide_trend"))
                        else None
                    ),
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
