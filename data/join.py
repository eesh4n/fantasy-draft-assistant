"""
join.py -- Join raw seasonal stats with raw ADP/consensus rankings on
player name, handling name mismatches with normalization + fuzzy matching.

Note: nfl_data_py's seasonal stats (raw_stats.csv) only cover offensive
skill positions with tracked play-by-play stats (QB/RB/WR/TE) -- it has no
rows for K or DEF (kickers' FG stats and team defense stats aren't part of
that dataset).

For K, real per-kicker production stats (FG accuracy, 50+ yard FG
accuracy, dome/weather-independence, team offense scoring rate) are pulled
separately by pull_kicker_stats.py into data/kicker_stats.csv (built from
2024 play-by-play data, since nfl_data_py's seasonal aggregate table has no
kicker rows at all). This script merges that file onto K rows by
normalized name so score.py can use a real composite for K instead of the
ADP-only fallback that still applies when kicker_stats.csv is missing.
Kickers with no play-by-play match (e.g. an ADP-ranked rookie/backup with
no 2024 attempts) simply keep null kicker_stats columns and fall through
the existing "without_stats" ADP-only path in score.py.

For DEF, real per-team defensive/special-teams production stats (sacks,
INTs, fumble recoveries/forced fumbles, safeties, blocked kicks,
defensive/special-teams TDs, points allowed, and this league's exact
custom-scoring adjusted PPG) are computed separately by
build_defense_stats.py into data/defense_stats.csv (also from 2024
play-by-play data). This script merges that file onto DEF rows by team,
using a `def_` column prefix (def_sacks, def_interceptions, etc.) so it
never collides with the `interceptions` column already used elsewhere in
this same joined output for QB/RB/WR/TE (thrown picks, a different stat).
Run build_defense_stats.py before this script.

Also merges in data/redzone_stats.csv (real per-player red-zone/goal-line
usage computed from 2024 play-by-play data by redzone.py -- run that
script first) on player_id (gsis_id), replacing what used to be a crude
td-rate approximation computed entirely inside score.py.
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

    # Bring in real red-zone/goal-line usage (data/redzone_stats.csv,
    # produced by redzone.py from 2024 play-by-play data -- see that file's
    # docstring). Joined on player_id (gsis_id), the same id format used by
    # both raw_stats.csv and redzone_stats.csv, since it's a more reliable
    # key than name matching (no fuzzy-match risk). Rows with no player_id
    # (DEF, unmatched ADP-only rows) or no red-zone play-by-play appearance
    # (e.g. a player with 0 carries/targets in 2024) simply get nulls here,
    # same as any other missing stat -- score.py already handles that via
    # its non-null-fraction threshold.
    redzone_path = DATA_DIR / "redzone_stats.csv"
    if redzone_path.exists():
        redzone = pd.read_csv(redzone_path)
        redzone = redzone[
            ["player_id", "red_zone_touches", "goal_line_touches", "red_zone_share", "goal_line_share"]
        ]
        final = final.merge(redzone, on="player_id", how="left")
    else:
        print(f"WARNING: {redzone_path} not found -- run redzone.py first. "
              f"Continuing without red-zone columns (will be all-null).")
        for col in ["red_zone_touches", "goal_line_touches", "red_zone_share", "goal_line_share"]:
            final[col] = pd.NA

    # Bring in real kicker production stats (data/kicker_stats.csv, produced
    # by pull_kicker_stats.py from 2024 play-by-play data -- see that
    # file's docstring). Joined on normalized name (kicker_stats.csv has no
    # gsis player_id since it's built straight from pbp's
    # kicker_player_name, not a roster table), restricted to K rows so we
    # never accidentally match a non-kicker with the same surname. Kickers
    # with no 2024 pbp attempts (e.g. an ADP-ranked replacement/rookie) get
    # nulls here and fall through score.py's existing ADP-only path for K,
    # same treatment DEF still gets everywhere.
    kicker_path = DATA_DIR / "kicker_stats.csv"
    kicker_cols = [
        "fg_attempts", "fg_made", "fg_pct", "fg_attempts_50plus",
        "fg_made_50plus", "fg_pct_50plus", "pat_attempts", "pat_made",
        "pat_pct", "is_dome", "team_offense_ppg", "team_go_for_it_rate_fg_range",
    ]
    if kicker_path.exists():
        # kicker_stats.csv comes straight from play-by-play's
        # kicker_player_name, which is ABBREVIATED ("B.Aubrey", not
        # "Brandon Aubrey" -- pbp has no full-name field for kickers). A
        # plain normalize_name() match against ADP's full names would
        # match nothing (confirmed: 0/34 on first attempt), so match on
        # (last name, first initial) instead, same shape as the
        # first/last-name guardrail already used above for the fuzzy WR/RB
        # match, just built for initial-format names.
        def last_name_and_initial(norm_name: str):
            toks = [t for t in norm_name.split(" ") if t]
            if not toks:
                return "", ""
            return toks[-1], toks[0][0] if toks[0] else ""

        kicker_stats = pd.read_csv(kicker_path)
        kicker_stats["k_norm"] = kicker_stats["kicker_name"].apply(
            lambda n: normalize_name(n.replace(".", " ")) if isinstance(n, str) else ""
        )
        kicker_parts = kicker_stats["k_norm"].apply(last_name_and_initial)
        kicker_stats["last_name"] = kicker_parts.apply(lambda t: t[0])
        kicker_stats["first_initial"] = kicker_parts.apply(lambda t: t[1])
        kicker_stats["match_key"] = kicker_stats["last_name"] + "|" + kicker_stats["first_initial"]
        kicker_stats = kicker_stats.drop_duplicates("match_key", keep="first")

        final["norm_name"] = final["player_name"].apply(normalize_name)
        final_parts = final["norm_name"].apply(last_name_and_initial)
        final["last_name"] = final_parts.apply(lambda t: t[0])
        final["first_initial"] = final_parts.apply(lambda t: t[1])
        final["match_key"] = final["last_name"] + "|" + final["first_initial"]

        is_k = final["position"] == "K"
        k_rows = final[is_k].merge(
            kicker_stats[["match_key"] + kicker_cols], on="match_key", how="left"
        )
        non_k_rows = final[~is_k].copy()
        for col in kicker_cols:
            non_k_rows[col] = pd.NA
        final = pd.concat([k_rows, non_k_rows], ignore_index=True, sort=False)
        final = final.drop(columns=["norm_name", "last_name", "first_initial", "match_key"])
    else:
        print(f"WARNING: {kicker_path} not found -- run pull_kicker_stats.py "
              f"first. Continuing without kicker stat columns (will be all-null, "
              f"K falls back to ADP-only in score.py).")
        for col in kicker_cols:
            final[col] = pd.NA

    # Bring in real team defense stats (data/defense_stats.csv, produced by
    # build_defense_stats.py from 2024 play-by-play data -- see that
    # file's docstring for the full stat-attribution methodology and the
    # documented 21-27-points-allowed-tier assumption). Joined on team,
    # restricted to DEF rows, with a def_ prefix on every merged column so
    # nothing collides with the offensive `interceptions` column (QB picks
    # thrown) already present for QB/RB/WR/TE rows above.
    defense_path = DATA_DIR / "defense_stats.csv"
    defense_cols = [
        "sacks", "interceptions", "fumble_recoveries", "forced_fumbles",
        "safeties", "blocked_kicks", "def_tds", "st_tds", "st_ff", "st_fr",
        "points_allowed_per_game", "custom_adjusted_ppg", "pressure_rate_proxy",
    ]
    def_prefixed_cols = [f"def_{c}" for c in defense_cols]
    if defense_path.exists():
        defense_stats = pd.read_csv(defense_path)
        defense_stats = defense_stats.rename(
            columns={c: f"def_{c}" for c in defense_cols}
        )
        is_def = final["position"] == "DEF"
        def_rows_merged = final[is_def].merge(
            defense_stats[["team"] + def_prefixed_cols], on="team", how="left"
        )
        non_def_rows = final[~is_def].copy()
        for col in def_prefixed_cols:
            non_def_rows[col] = pd.NA
        final = pd.concat([def_rows_merged, non_def_rows], ignore_index=True, sort=False)
    else:
        print(f"WARNING: {defense_path} not found -- run build_defense_stats.py "
              f"first. Continuing without defense stat columns (will be all-null, "
              f"DEF falls back to ADP-only in score.py).")
        for col in def_prefixed_cols:
            final[col] = pd.NA

    out_path = DATA_DIR / "joined.csv"
    final.to_csv(out_path, index=False)

    n_matched = len(matched) + len(fuzzy_df)
    print(f"Exact matches: {len(matched)}")
    print(f"Fuzzy matches: {len(fuzzy_df)}")
    print(f"ADP-only (no stats match, e.g. DEF or rookies): {len(unmatched_df) + len(def_rows)}")
    print(f"Red-zone stats matched: {final['red_zone_share'].notna().sum()} of {len(final)} rows")
    n_k = (final["position"] == "K").sum()
    n_k_matched = final.loc[final["position"] == "K", "fg_pct"].notna().sum()
    print(f"Kicker stats matched: {n_k_matched} of {n_k} K rows")
    n_def = (final["position"] == "DEF").sum()
    n_def_matched = final.loc[final["position"] == "DEF", "def_custom_adjusted_ppg"].notna().sum()
    print(f"Defense stats matched: {n_def_matched} of {n_def} DEF rows")
    print(f"Wrote {len(final)} rows to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
