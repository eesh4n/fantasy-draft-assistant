"""
redzone.py -- Compute REAL per-player red-zone/goal-line usage stats from
2024 play-by-play data (nfl_data_py.import_pbp_data), to replace the crude
td_rate approximation previously used in score.py.

Background: an analyst video (paraphrased) claimed red-zone targets and
goal-line carries are the two most predictive stats for RB touchdown
production (~65% of RB TDs). The rest of this pipeline only pulls
nfl_data_py's SEASONAL aggregate data, which has no red-zone-specific
columns -- so td_rate in score.py was previously computed as
(rush TDs + rec TDs) / (all carries + all targets), i.e. touchdown rate
over ALL opportunities, not just red-zone ones. This script fixes that by
going to the play-by-play (every single play) dataset, which does carry
per-play field position and per-play rusher/receiver attribution.

Definitions used (play-by-play grain, 2024 season, REG + POST included --
nfl_data_py's import_pbp_data doesn't split season_type at the row level
the way import_seasonal_data does, so no filtering is applied there; see
note below):
  - "red zone" play:  yardline_100 <= 20  (yardline_100 = yards from the
    opponent's end zone, so <=20 means inside the opponent 20-yard line --
    the standard definition of "red zone").
  - "goal line" play: yardline_100 <= 5. nfl_data_py's own `goal_to_go`
    flag does NOT mean "at/inside the 5" -- it means "this is a 1st-and-
    goal situation," which can trigger from well outside the 5 (e.g. 1st
    and goal from the 8, or even further out after a penalty). Since the
    analyst's exact goal-line cutoff wasn't specified, this uses the
    standard, commonly-cited yardline_100 <= 5 threshold rather than
    goal_to_go, so "goal line" here specifically means "inside the 5,"
    not "any 1st-and-goal down."
  - A rush attempt is attributed to `rusher_player_id`.
  - A target is attributed to `receiver_player_id` (present on every
    pass play with an intended receiver, including incompletions --
    absent only on sacks/scrambles/spikes/throwaways with no intended
    target, which correctly aren't targets).
  - "touches" for the share denominator = rush attempts + targets (same
    definition score.py already uses for RB/WR/TE "volume"/opportunities).

Output: data/redzone_stats.csv, one row per player (matched on gsis_id --
the same `player_id` format used in raw_stats.csv / joined.csv -- with
player_name carried through for readability / as a fallback join key):
  player_id         -- gsis_id (00-00xxxxx), preferred join key
  player_name        -- as it appears in pbp data, for reference
  red_zone_touches   -- red-zone rush attempts + red-zone targets
  goal_line_touches  -- goal-line rush attempts + goal-line targets
  total_touches      -- all rush attempts + all targets (denominator)
  red_zone_share     -- red_zone_touches / total_touches
  goal_line_share    -- goal_line_touches / total_touches

Runtime note: play-by-play is ~50k rows x ~370 columns for a single
season, much bigger than the seasonal aggregates used elsewhere in this
pipeline. To keep this cheap, we pull the full frame once (nfl_data_py's
`columns=` filter goes through its own parquet-column-pruned read path,
which errors out for this release's schema -- silently falls back to an
empty frame -- so it's not used here) and immediately narrow to the ~10
columns actually needed before doing any further work.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
PBP_SEASON = 2024  # matches STATS_SEASON in pull.py

RED_ZONE_YARDLINE = 20
GOAL_LINE_YARDLINE = 5

KEEP_COLS = [
    "season",
    "week",
    "yardline_100",
    "rush_attempt",
    "pass_attempt",
    "rusher_player_id",
    "rusher_player_name",
    "receiver_player_id",
    "receiver_player_name",
]


def pull_pbp() -> pd.DataFrame:
    import nfl_data_py as nfl

    print(f"Pulling play-by-play data for {PBP_SEASON} (this is a big pull)...")
    pbp = nfl.import_pbp_data([PBP_SEASON], downcast=True)
    print(f"  raw pbp shape: {pbp.shape}")
    pbp = pbp[KEEP_COLS].copy()
    return pbp


def compute_side(pbp: pd.DataFrame, attempt_col: str, id_col: str, name_col: str) -> pd.DataFrame:
    """Compute touches/red-zone/goal-line counts for one side (rush or
    target), keyed by player id. Returns a DataFrame indexed by player_id
    with columns: player_name, touches, rz_touches, gl_touches.
    """
    side = pbp[pbp[id_col].notna()].copy()
    if attempt_col == "rush_attempt":
        side = side[side[attempt_col] == 1]
    # targets: any play with a non-null receiver_player_id is a target,
    # regardless of pass_attempt flag (sacks/scrambles never have a
    # receiver_player_id in the first place, so no extra filter needed).

    side["is_rz"] = side["yardline_100"] <= RED_ZONE_YARDLINE
    side["is_gl"] = side["yardline_100"] <= GOAL_LINE_YARDLINE

    grouped = side.groupby(id_col).agg(
        player_name=(name_col, "first"),
        touches=(id_col, "count"),
        rz_touches=("is_rz", "sum"),
        gl_touches=("is_gl", "sum"),
    )
    return grouped


def main():
    pbp = pull_pbp()

    rush = compute_side(pbp, "rush_attempt", "rusher_player_id", "rusher_player_name")
    target = compute_side(pbp, "pass_attempt", "receiver_player_id", "receiver_player_name")

    numeric_cols = ["touches", "rz_touches", "gl_touches"]
    combined = rush[numeric_cols].add(target[numeric_cols], fill_value=0)
    # player_name is non-numeric -- recover it separately, preferring
    # whichever side has it (players with both rushes and targets will
    # have the same name on both sides anyway).
    names = rush["player_name"].combine_first(target["player_name"])
    combined["player_name"] = names

    combined = combined.reset_index().rename(columns={"index": "player_id"})
    combined = combined.rename(
        columns={
            "rz_touches": "red_zone_touches",
            "gl_touches": "goal_line_touches",
            "touches": "total_touches",
        }
    )

    combined["red_zone_share"] = combined["red_zone_touches"] / combined["total_touches"].replace(0, np.nan)
    combined["goal_line_share"] = combined["goal_line_touches"] / combined["total_touches"].replace(0, np.nan)

    combined = combined[
        [
            "player_id",
            "player_name",
            "red_zone_touches",
            "goal_line_touches",
            "total_touches",
            "red_zone_share",
            "goal_line_share",
        ]
    ].sort_values("red_zone_touches", ascending=False)

    out_path = DATA_DIR / "redzone_stats.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {len(combined)} players to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
