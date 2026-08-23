"""
retrain_with_real2025.py -- Retrain the ML model with a REAL 2024->2025
season pair added to the historical training set, using actual 2025
outcomes (data/real2025_stats.csv, pulled from nfl.com) as the target
instead of the usual "next season's real ppg from nfl_data_py" (which
doesn't exist for 2025 yet).

Every other pair (2018->2019 ... 2023->2024) is built exactly as before
via train_ml_model.py's own functions -- unchanged. Only this one new
pair is added on top, using this pipeline's already-computed 2024 base
features (joined.csv) as X and real 2025 season points / 17 games (a
standard-season approximation -- no per-player 2025 games-played column
exists in the scraped source) as y.

This is a genuine out-of-sample validation AND retrain: the model has
never seen 2025 outcomes before, so comparing its pre-existing 2024-based
prediction against this real data (done before running this script, see
chat) is honest, and adding the real pair here should make the retrained
model measurably better calibrated against actual near-term outcomes,
not just historical cross-validation.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_ml_model as tm

DATA_DIR = Path(__file__).resolve().parent
GAMES_IN_SEASON = 17  # standard-season approximation, see module docstring


def build_real_2025_pair(pos: str, feature_cols: list) -> pd.DataFrame:
    joined = pd.read_csv(DATA_DIR / "joined.csv")
    sub = joined[joined["position"] == pos].copy()
    sub = sub[sub["real2025_total_pts"].notna()]
    sub = sub[sub["games"] >= tm.MIN_GAMES_YEAR1]  # same 2024-side games floor as every other pair

    # int_rate isn't a materialized column in joined.csv (only the raw
    # interceptions/attempts columns it's derived from are) -- compute it
    # here identically to score.py's compute_stat_group_scores QB branch
    # so the real-2025 pair isn't silently dropped for any feature-set
    # variant that includes it (e.g. QB's turnover_durability group).
    if "int_rate" in feature_cols and "int_rate" not in sub.columns:
        sub["int_rate"] = sub["interceptions"].fillna(0) / sub["attempts"].replace(0, np.nan)

    missing = [c for c in feature_cols if c not in sub.columns]
    if missing:
        return pd.DataFrame()
    sub = sub.dropna(subset=feature_cols)
    if sub.empty:
        return pd.DataFrame()
    sub["target_ppg"] = sub["real2025_total_pts"] / GAMES_IN_SEASON
    sub["season_pair"] = "2024->2025 (REAL)"
    return sub[["player_name", "season_pair", "target_ppg"] + feature_cols].reset_index(drop=True)


def main():
    print("Building historical season cache (2018-2024, unchanged)...")
    season_cache = {s: tm.compute_features_for_season(s) for s in tm.HISTORICAL_SEASONS}

    report = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        print(f"\n=== {pos} ===")
        variant_results = {}
        for variant_name, feature_cols in tm._feature_set_variants(pos):
            hist_table = tm.build_training_table(pos, feature_cols, season_cache)
            real_pair = build_real_2025_pair(pos, feature_cols)
            table = pd.concat([hist_table, real_pair], ignore_index=True) if not real_pair.empty else hist_table
            if table.empty or len(table) < 20:
                continue
            X = table[feature_cols].to_numpy(dtype=float)
            y = table["target_ppg"].to_numpy(dtype=float)
            results = tm._search_models_for_feature_set(X, y)
            best_linear_r2 = max(results[k]["cv_r2_mean"] for k in ("ridge", "lasso", "elasticnet"))
            variant_results[variant_name] = {
                "feature_cols": feature_cols, "table": table, "results": results,
                "best_linear_r2": best_linear_r2, "n_rows": len(table),
                "n_real_2025": len(real_pair),
            }
            print(f"  [{variant_name}] n={len(table)} (incl. {len(real_pair)} real-2025 rows)  best_linear_cv_r2={best_linear_r2:.4f}")

        if not variant_results:
            print(f"  SKIPPED {pos}: no usable variant")
            continue

        base_name = "base"
        base_r2 = variant_results[base_name]["best_linear_r2"] if base_name in variant_results else -np.inf
        winning_variant = base_name if base_name in variant_results else max(variant_results, key=lambda k: variant_results[k]["best_linear_r2"])
        for name, v in variant_results.items():
            if name == base_name:
                continue
            if v["best_linear_r2"] > base_r2 + tm.FEATURE_IMPROVEMENT_MARGIN and v["best_linear_r2"] > variant_results[winning_variant]["best_linear_r2"]:
                winning_variant = name

        chosen = variant_results[winning_variant]
        feature_cols = chosen["feature_cols"]
        table = chosen["table"]
        results = chosen["results"]
        X = table[feature_cols].to_numpy(dtype=float)
        y = table["target_ppg"].to_numpy(dtype=float)

        base_r2s, base_maes = tm._cv_naive_baseline(table[["ppg"]], y)
        results["naive_baseline"] = {"cv_r2_mean": float(np.nanmean(base_r2s)), "cv_mae_mean": float(np.nanmean(base_maes))}

        best_linear_name = max(["ridge", "lasso", "elasticnet"], key=lambda k: results[k]["cv_r2_mean"])
        best_linear_r2 = results[best_linear_name]["cv_r2_mean"]
        best_tree_name = "gbm" if results["gbm"]["cv_r2_mean"] >= results["random_forest"]["cv_r2_mean"] else "random_forest"
        candidates = {best_tree_name: results[best_tree_name]["cv_r2_mean"], "ensemble": results["ensemble"]["cv_r2_mean"]}
        best_beyond = max(candidates, key=candidates.get)
        chosen_model = best_beyond if candidates[best_beyond] > best_linear_r2 + tm.TREE_IMPROVEMENT_MARGIN else best_linear_name

        # Build + fit final model (reuses train_ml_model's exact construction logic per model type)
        if chosen_model == "ridge":
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            final_model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=results["ridge"]["alpha"]))])
            hyperparams = {"alpha": results["ridge"]["alpha"]}
        elif chosen_model == "lasso":
            from sklearn.linear_model import Lasso
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            final_model = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=results["lasso"]["alpha"], max_iter=20000))])
            hyperparams = {"alpha": results["lasso"]["alpha"]}
        elif chosen_model == "elasticnet":
            from sklearn.linear_model import ElasticNet
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            final_model = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=results["elasticnet"]["alpha"], l1_ratio=results["elasticnet"]["l1_ratio"], max_iter=20000))])
            hyperparams = {"alpha": results["elasticnet"]["alpha"], "l1_ratio": results["elasticnet"]["l1_ratio"]}
        else:
            # gbm/random_forest/ensemble: fall back to best linear for simplicity/stability
            # given the added real-2025 rows shrink per-position n further.
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            final_model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=results["ridge"]["alpha"]))])
            hyperparams = {"alpha": results["ridge"]["alpha"], "note": f"forced ridge instead of {chosen_model} for stability with added real-2025 rows"}
            chosen_model = "ridge"

        final_model.fit(X, y)
        model_path = tm.MODELS_DIR / f"{pos}_model.joblib"
        joblib.dump(final_model, model_path)

        metadata = {
            "position": pos, "feature_set": winning_variant, "features": feature_cols,
            "model_type": chosen_model, "hyperparams": hyperparams,
            "cv_r2_mean": results[chosen_model]["cv_r2_mean"] if chosen_model in results else best_linear_r2,
            "cv_mae_mean": results[chosen_model]["cv_mae_mean"] if chosen_model in results else results[best_linear_name]["cv_mae_mean"],
            "naive_baseline_r2": results["naive_baseline"]["cv_r2_mean"],
            "naive_baseline_mae": results["naive_baseline"]["cv_mae_mean"],
            "n_rows": len(table), "n_real_2025_rows": chosen["n_real_2025"],
            "retrained_with_real_2025": True,
        }
        import json
        with open(tm.MODELS_DIR / f"{pos}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        report[pos] = metadata
        print(f"  WINNER: {chosen_model} on '{winning_variant}' -- CV R2={metadata['cv_r2_mean']:.4f} MAE={metadata['cv_mae_mean']:.2f} (baseline R2={metadata['naive_baseline_r2']:.4f})")

    print("\n=== FINAL SUMMARY ===")
    for pos, m in report.items():
        print(f"{pos}: {m['model_type']} | n={m['n_rows']} (+{m['n_real_2025_rows']} real-2025) | CV R2={m['cv_r2_mean']:.4f} MAE={m['cv_mae_mean']:.2f} | baseline R2={m['naive_baseline_r2']:.4f}")


if __name__ == "__main__":
    main()
