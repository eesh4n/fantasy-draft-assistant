"""
join.py -- Join raw seasonal stats with raw ADP/consensus rankings on
player name, handling name mismatches with normalization + fuzzy matching.

Note: nfl_data_py's seasonal stats (raw_stats.csv) only cover offensive
skill positions with tracked play-by-play stats (QB/RB/WR/TE) -- it has no
rows for K or DEF (kickers' FG stats and team defense stats aren't part of
that dataset). For K and DEF, we carry the ADP data through with stats
columns left null; score.py falls back to an ADP-only value model for
those two groups (documented in data/README.md).
"""
import re
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

DATA_DIR = Path(__file__).resolve().parent

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = name.lower().strip()
    n = n.replace(".", "").replace("'", "").replace("-", " ")
    n = re.sub(r"\s+", " ", n)
    tokens = [t for t in n.split(" ") if t not in SUFFIXES]
    return " ".join(tokens).strip()


def main():
    stats = pd.read_csv(DATA_DIR / "raw_stats.csv")
    adp = pd.read_csv(DATA_DIR / "raw_adp.csv")

    stats["norm_name"] = stats["player_name"].apply(normalize_name)
    adp["norm_name"] = adp["player_name"].apply(normalize_name)

    # DEF rows have no stats counterpart -- keep them ADP-only.
    adp_players = adp[adp["position"] != "DEF"].copy()
    adp_def = adp[adp["position"] == "DEF"].copy()

    # 1) Exact match on (normalized name, position)
    stats_key = stats.copy()
    stats_key["match_key"] = stats_key["norm_name"] + "|" + stats_key["position"]
    adp_players["match_key"] = (
        adp_players["norm_name"] + "|" + adp_players["position"]
    )

    exact = adp_players.merge(
        stats_key, on="match_key", how="left", suffixes=("_adp", "_stats")
    )

    matched_mask = exact["player_id"].notna()
    matched = exact[matched_mask].copy()
    unmatched_adp = exact[~matched_mask].copy()

    # 2) Fuzzy match the leftovers within the same position group.
    #
    # Guardrail against same-surname/different-player false positives
    # (e.g. "Kevin Coleman" vs "Keon Coleman" -- both real players, both
    # WRs, different people): require the LAST name token to match
    # exactly, and the FIRST name token to be a strong partial match
    # (handles nicknames/truncation like "Scotty"/"Scott",
    # "Joshua"/"Josh", "Kenny"/"Kenneth", "Gabe"/"Gabriel" -- these all
    # score >=85 on partial_ratio) while "Kevin"/"Keon" scores ~67 and is
    # correctly rejected. This is intentionally stricter than a plain
    # whole-name token_sort_ratio, which would accept the Coleman case.
    def name_parts(norm_name: str):
        toks = norm_name.split(" ")
        if not toks:
            return "", ""
        return toks[0], toks[-1]

    fuzzy_rows = []
    still_unmatched = []
    for pos in unmatched_adp["position_adp"].unique():
        pos_stats = stats[stats["position"] == pos].reset_index(drop=True)
        pos_unmatched = unmatched_adp[unmatched_adp["position_adp"] == pos]
        if pos_stats.empty:
            # e.g. K -- nfl_data_py's seasonal stats have no kicker rows
            still_unmatched.extend(pos_unmatched.to_dict("records"))
            continue
        for _, row in pos_unmatched.iterrows():
            target_first, target_last = name_parts(row["norm_name_adp"])
            best = None
            best_score = 0
            for idx, stats_row_candidate in pos_stats.iterrows():
                cand_first, cand_last = name_parts(stats_row_candidate["norm_name"])
                if cand_last != target_last:
                    continue
                first_score = fuzz.partial_ratio(target_first, cand_first)
                if first_score >= 85 and first_score > best_score:
                    best_score = first_score
                    best = (stats_row_candidate["norm_name"], first_score, idx)
            result = best
            if result:
                best_name, score, idx = result
                stats_row = pos_stats.iloc[idx]
                merged = row.to_dict()
                for col in stats.columns:
                    merged[f"{col}_stats" if col in adp.columns else col] = (
                        stats_row[col]
                    )
                # explicit overwrite of stats-side columns used downstream
                for col in [
                    "player_id",
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
                ]:
                    merged[col] = stats_row[col]
                merged["_fuzzy_score"] = score
                fuzzy_rows.append(merged)
            else:
                still_unmatched.append(row.to_dict())

    fuzzy_df = pd.DataFrame(fuzzy_rows)
    unmatched_df = pd.DataFrame(still_unmatched)

    combined = pd.concat([matched, fuzzy_df], ignore_index=True, sort=False)

    # normalize column names post merge (exact-match branch used _adp/_stats
    # suffixes; fuzzy branch mirrors that naming)
    def col(df_, base):
        for cand in (f"{base}_adp", base):
            if cand in df_.columns:
                return df_[cand]
        return pd.Series([None] * len(df_))

    out = pd.DataFrame(
        {
            "player_name": col(combined, "player_name"),
            "team": col(combined, "team_adp").fillna(col(combined, "team_stats")),
            "position": col(combined, "position"),
            "adp": combined["adp"],
            "ecr_rank": combined["ecr_rank"],
            "bye_week": combined["bye_week"],
            "player_id": combined["player_id"],
            "games": combined["games"],
            "fantasy_points": combined["fantasy_points"],
            "fantasy_points_ppr": combined["fantasy_points_ppr"],
            "carries": combined["carries"],
            "rushing_yards": combined["rushing_yards"],
            "rushing_tds": combined["rushing_tds"],
            "targets": combined["targets"],
            "receptions": combined["receptions"],
            "receiving_yards": combined["receiving_yards"],
            "receiving_tds": combined["receiving_tds"],
            "completions": combined["completions"],
            "attempts": combined["attempts"],
            "passing_yards": combined["passing_yards"],
            "passing_tds": combined["passing_tds"],
            "interceptions": combined["interceptions"],
            "snap_pct": combined["snap_pct"],
        }
    )

    # Append DEF (ADP-only, no per-player stats) and any still-unmatched
    # ADP players (also carried through ADP-only rather than dropped, so we
    # don't silently lose fantasy-relevant players e.g. rookies with no
    # 2024 stats).
    def_rows = pd.DataFrame(
        {
            "player_name": adp_def["player_name"],
            "team": adp_def["team"],
            "position": adp_def["position"],
            "adp": adp_def["adp"],
            "ecr_rank": adp_def["ecr_rank"],
            "bye_week": adp_def["bye_week"],
        }
    )

    leftover_cols = {
        "player_name": "player_name_adp" if "player_name_adp" in unmatched_df.columns else "player_name",
    }
    if not unmatched_df.empty:
        def ucol(base):
            for cand in (f"{base}_adp", base):
                if cand in unmatched_df.columns:
                    return unmatched_df[cand]
            return pd.Series([None] * len(unmatched_df))

        leftover = pd.DataFrame(
            {
                "player_name": ucol("player_name"),
                "team": ucol("team"),
                "position": ucol("position"),
                "adp": unmatched_df["adp"] if "adp" in unmatched_df.columns else None,
                "ecr_rank": unmatched_df["ecr_rank"] if "ecr_rank" in unmatched_df.columns else None,
                "bye_week": unmatched_df["bye_week"] if "bye_week" in unmatched_df.columns else None,
            }
        )
    else:
        leftover = pd.DataFrame(
            columns=["player_name", "team", "position", "adp", "ecr_rank", "bye_week"]
        )

    final = pd.concat([out, def_rows, leftover], ignore_index=True, sort=False)
    final = final.dropna(subset=["player_name", "position"]).reset_index(drop=True)
    final = final.drop_duplicates(subset=["player_name", "team", "position"])

    out_path = DATA_DIR / "joined.csv"
    final.to_csv(out_path, index=False)

    n_matched = len(matched) + len(fuzzy_df)
    print(f"Exact matches: {len(matched)}")
    print(f"Fuzzy matches: {len(fuzzy_df)}")
    print(f"ADP-only (no stats match, e.g. DEF or rookies): {len(unmatched_df) + len(def_rows)}")
    print(f"Wrote {len(final)} rows to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
