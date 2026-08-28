"""
build_schedule_strength.py -- Real strength-of-schedule per team, derived
from the actual 2026 regular-season schedule (schedule_2026.csv, pulled
from ESPN's schedule grid) crossed with each opponent's real defensive
scoring rate (defense_stats.csv's points_allowed_per_game, already in the
pipeline for DEF scoring).

Output: data/schedule_strength.csv with one row per team:
  team, sos_ros_avg_pa, sos_ros_rank, sos_playoff_avg_pa, sos_playoff_rank

sos_*_avg_pa = the average points_allowed_per_game of a team's opponents
over that window. LOWER means a tougher schedule (opponents allow fewer
points => harder for this team's offense to score against them).
sos_*_rank = 1 (toughest schedule) .. 32 (easiest schedule), so a lower
rank number always means "harder schedule" consistently with how ADP/
position_rank work elsewhere in this pipeline.

Playoff window = weeks 15-17, the standard fantasy-playoff stretch.
This is a TEAM-level signal (not per-player), used as supplementary
schedule context in the trade calculator -- it does not feed value_score,
since this pipeline has no weekly player-level projection model to blend
it into honestly.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
PLAYOFF_WEEKS = ["w15", "w16", "w17"]
ALL_WEEKS = [f"w{i}" for i in range(1, 19)]


def main():
    schedule = pd.read_csv(DATA_DIR / "schedule_2026.csv")
    defense = pd.read_csv(DATA_DIR / "defense_stats.csv")
    pa_by_team = dict(zip(defense["team"], defense["points_allowed_per_game"]))

    rows = []
    for _, row in schedule.iterrows():
        team = row["team"]

        def avg_opp_pa(weeks):
            opp_pas = []
            for w in weeks:
                opp = row[w]
                if opp == "BYE" or pd.isna(opp):
                    continue
                if opp in pa_by_team:
                    opp_pas.append(pa_by_team[opp])
            return sum(opp_pas) / len(opp_pas) if opp_pas else None

        rows.append({
            "team": team,
            "sos_ros_avg_pa": avg_opp_pa(ALL_WEEKS),
            "sos_playoff_avg_pa": avg_opp_pa(PLAYOFF_WEEKS),
        })

    out = pd.DataFrame(rows)
    # rank ascending on avg_pa: lowest avg_pa (stingiest opponents) = rank 1 = toughest
    out["sos_ros_rank"] = out["sos_ros_avg_pa"].rank(method="min", ascending=True).astype(int)
    out["sos_playoff_rank"] = out["sos_playoff_avg_pa"].rank(method="min", ascending=True).astype(int)

    out_path = DATA_DIR / "schedule_strength.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} teams to {out_path}")
    print(out.sort_values("sos_playoff_rank")[["team", "sos_playoff_rank", "sos_ros_rank"]].head(6).to_string(index=False))


if __name__ == "__main__":
    main()
