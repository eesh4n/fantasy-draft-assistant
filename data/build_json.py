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
  "value_gap": integer (adp_position_rank - position_rank)
}
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent


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
            }
        )

    players.sort(key=lambda p: p["adp"])

    out_path = DATA_DIR / "players.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(players, f, indent=2)

    print(f"Wrote {len(players)} players to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
