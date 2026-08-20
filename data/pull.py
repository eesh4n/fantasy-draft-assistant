"""
pull.py -- Pull raw data for the fantasy draft assistant.

Sources:
1. Seasonal per-player stats from `nfl_data_py` (nflverse data).
   As of this run, nfl_data_py's hosted release only has *complete* seasonal
   stats through the 2024 season (import_seasonal_data([2025]) 404s -- the
   2025 season's final seasonal rollup hasn't been published to the
   nflverse data repo yet). So "last season" here = 2024. This is
   clearly documented in data/README.md.

2. ADP / consensus rankings. FantasyPros' CSV export endpoint
   (`?export=xls` on the rankings pages) requires a logged-in session --
   fetching it anonymously just returns the HTML page. However, that same
   HTML page embeds the full consensus rankings as a JSON blob
   (`var ecrData = {...};`) used to render the on-page table, and this is
   served with no login required. We parse that JSON directly instead of
   trying to hit the gated CSV export. This gives us, for every ranked
   player: expert consensus rank (ECR), average expert rank, position
   rank, and team -- i.e. everything a CSV export would have given us,
   for PPR draft rankings specifically.
   URL: https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent
STATS_SEASON = 2024  # last season with complete seasonal data in nfl_data_py
ADP_URL = "https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def pull_stats() -> pd.DataFrame:
    import nfl_data_py as nfl

    print(f"Pulling seasonal stats for {STATS_SEASON}...")
    seasonal = nfl.import_seasonal_data([STATS_SEASON])
    seasonal = seasonal[seasonal["season_type"] == "REG"].copy()

    print(f"Pulling {STATS_SEASON} rosters for name/position/team...")
    rosters = nfl.import_seasonal_rosters([STATS_SEASON])
    # one row per player: prefer the most recent week's listing
    rosters = rosters.sort_values("week").drop_duplicates("player_id", keep="last")
    roster_cols = ["player_id", "player_name", "position", "team", "pfr_id"]
    rosters = rosters[roster_cols]

    df = seasonal.merge(rosters, on="player_id", how="left")
    df = df[df["player_name"].notna()]

    # Only fantasy-relevant positions
    fantasy_positions = {"QB", "RB", "WR", "TE", "K"}
    df = df[df["position"].isin(fantasy_positions)].copy()

    print("Pulling snap counts to compute average offensive snap share...")
    try:
        snaps = nfl.import_snap_counts([STATS_SEASON])
        snaps_agg = (
            snaps.groupby("pfr_player_id")["offense_pct"].mean().reset_index()
        )
        snaps_agg = snaps_agg.rename(columns={"offense_pct": "snap_pct"})
        df = df.merge(
            snaps_agg, left_on="pfr_id", right_on="pfr_player_id", how="left"
        )
        df = df.drop(columns=["pfr_player_id"], errors="ignore")
    except Exception as e:
        print(f"  snap counts unavailable ({e}); continuing without snap_pct")
        df["snap_pct"] = pd.NA

    keep_cols = [
        "player_id",
        "player_name",
        "position",
        "team",
        "games",
        "fantasy_points",
        "fantasy_points_ppr",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "snap_pct",
    ]
    df = df[keep_cols].reset_index(drop=True)

    # DEF/DST is not part of nfl_data_py's per-player seasonal data (it's a
    # team unit, not a player). We add a lightweight synthetic DEF stat line
    # per team using the seasonal defense-relevant columns nfl_data_py
    # exposes via schedules/scoring is out of scope for this quick pull, so
    # DEF rows get their fantasy signal entirely from ADP in join.py/score.py
    # (value model falls back to ADP-only ranking for DEF -- documented in
    # README).
    return df


def pull_adp() -> pd.DataFrame:
    print(f"Fetching FantasyPros PPR consensus rankings page: {ADP_URL}")
    resp = requests.get(ADP_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    html = resp.text

    match = re.search(r"var ecrData = (\{.*?\});", html, re.DOTALL)
    if not match:
        raise RuntimeError(
            "Could not find embedded ecrData JSON on FantasyPros rankings "
            "page -- page structure may have changed."
        )
    ecr = json.loads(match.group(1))
    players = ecr.get("players", [])
    if not players:
        raise RuntimeError("ecrData JSON had no players -- aborting ADP pull.")

    rows = []
    for p in players:
        pos_rank_raw = p.get("pos_rank", "")
        pos_match = re.match(r"([A-Z]+)(\d+)", pos_rank_raw or "")
        position = pos_match.group(1) if pos_match else None
        rows.append(
            {
                "player_name": p.get("player_name"),
                "team": p.get("player_team_id"),
                "position": position,
                "pos_rank_label": pos_rank_raw,
                "ecr_rank": p.get("rank_ecr"),
                "adp": p.get("rank_ave"),  # avg expert rank, used as ADP proxy
                "rank_min": p.get("rank_min"),
                "rank_max": p.get("rank_max"),
                "rank_std": p.get("rank_std"),
                "bye_week": p.get("player_bye_week"),
                "owned_avg_pct": p.get("player_owned_avg"),
            }
        )
    df = pd.DataFrame(rows)
    df["adp"] = pd.to_numeric(df["adp"], errors="coerce")
    # DST rows on FantasyPros are position "DST"; normalize to "DEF"
    df["position"] = df["position"].replace({"DST": "DEF"})
    df = df[df["position"].isin(["QB", "RB", "WR", "TE", "K", "DEF"])].copy()
    df = df.sort_values("adp").reset_index(drop=True)
    return df


def main():
    stats_df = pull_stats()
    stats_path = DATA_DIR / "raw_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Wrote {len(stats_df)} rows to {stats_path}")

    adp_df = pull_adp()
    adp_path = DATA_DIR / "raw_adp.csv"
    adp_df.to_csv(adp_path, index=False)
    print(f"Wrote {len(adp_df)} rows to {adp_path}")


if __name__ == "__main__":
    sys.exit(main())
