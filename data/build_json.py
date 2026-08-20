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
    "td_rate": number|null            // TDs per opportunity, a red-zone
                                       // proxy (RB/WR/TE only; see
                                       // data/README.md for why this is
                                       // an approximation, not real
                                       // red-zone data)
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
                    "td_rate": clean_num(row.get("td_rate")),
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
