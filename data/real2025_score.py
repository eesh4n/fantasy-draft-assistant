"""
real2025_score.py -- Compute real 2025 season fantasy points per player
from raw NFL.com season-total stats (data/real2025_passing.csv,
real2025_rushing.csv, real2025_receiving.csv -- pulled directly from
nfl.com's live 2025 stats pages, since nflverse/nfl_data_py still has no
2025 data published as of this pipeline run -- see data/README.md).

These are REAL, actual 2025 season totals (not projections, not last
year's stats) for every player who cleared a meaningful stat threshold in
passing (50 QBs), rushing (75 backs, includes rushing QBs), and receiving
(50 pass-catchers). NOT full team-by-team exhaustive coverage -- deep
bench/inactive players are not included -- but covers every player who
would realistically be drafted or matter in this league.

Uses this league's EXACT custom scoring rules (same constants as
score.py): 0.04 pt/pass yard, 4 pt/pass TD, -1/INT, 0.1 pt/rush-or-rec
yard, 6 pt/TD, 1 pt/reception (PPR). A player appearing in multiple
categories (e.g. a rushing QB, or a receiving RB) has their points summed
across all categories they appear in.

No games-played column is exposed by NFL.com's stat tables, so this is a
SEASON TOTAL, not a per-game rate -- real signal, but not directly
comparable to this pipeline's ppg-style columns without normalizing.
Output: real2025_total_pts (raw) and real2025_total_pts_rank (rank within
position, computed after this file is joined onto joined.csv by name).
"""
import re
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

PASS_YDS_TO_PTS = 0.04
PASS_TD_TO_PTS = 4
INT_TO_PTS = -1
YDS_TO_PTS = 0.1
TD_TO_PTS = 6
REC_TO_PTS = 1


def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", name)
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def main():
    pts = {}  # norm_name -> total points

    pass_path = DATA_DIR / "real2025_passing.csv"
    if pass_path.exists():
        df = pd.read_csv(pass_path)
        for _, row in df.iterrows():
            n = normalize_name(row["Player"])
            p = (
                float(row["Pass Yds"]) * PASS_YDS_TO_PTS
                + float(row["TD"]) * PASS_TD_TO_PTS
                + float(row["INT"]) * INT_TO_PTS
            )
            pts[n] = pts.get(n, 0.0) + p

    rush_path = DATA_DIR / "real2025_rushing.csv"
    if rush_path.exists():
        df = pd.read_csv(rush_path)
        for _, row in df.iterrows():
            n = normalize_name(row["Player"])
            p = float(row["Rush Yds"]) * YDS_TO_PTS + float(row["TD"]) * TD_TO_PTS
            pts[n] = pts.get(n, 0.0) + p

    recv_path = DATA_DIR / "real2025_receiving.csv"
    if recv_path.exists():
        df = pd.read_csv(recv_path)
        for _, row in df.iterrows():
            n = normalize_name(row["Player"])
            p = (
                float(row["Rec"]) * REC_TO_PTS
                + float(row["Yds"]) * YDS_TO_PTS
                + float(row["TD"]) * TD_TO_PTS
            )
            pts[n] = pts.get(n, 0.0) + p

    out = pd.DataFrame(
        [{"norm_name": k, "real2025_total_pts": round(v, 2)} for k, v in pts.items()]
    )
    out = out.sort_values("real2025_total_pts", ascending=False).reset_index(drop=True)
    out_path = DATA_DIR / "real2025_stats.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} players (real 2025 season totals) to {out_path}")
    print(out.head(10).to_string())


if __name__ == "__main__":
    sys.exit(main())
