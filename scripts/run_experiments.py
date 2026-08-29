"""
run_experiments.py — Intersectional Fairness Experiment Pipeline
=================================================================

Executes the full nested cross-validation experiment and saves raw results
to disk. This script should be run ONCE from the terminal. The Streamlit
dashboard (pages/2_🤖_Modelos.py) loads those pre-computed results and
renders only visualizations — it never trains models at runtime.

Pipeline (faithful to the reference paper):
  Outer CV: K=3 folds  →  unbiased performance estimate
  Inner CV: K=3 folds  →  hyperparameter tuning (RandomizedSearchCV, n_iter=30)
  Models: RandomForestClassifier, GradientBoostingClassifier
  Optimization metrics: accuracy, recall, precision, roc_auc, average_precision

Fairness metrics computed per outer fold (see utils/bias_metrics.py):
  - Intersectional AAOD (per subgroup + Max across subgroups)
  - Sensitivity Gap (Max TPR - Min TPR, viable subgroups N>=30)
  - Post-training Disparate Impact (per subgroup vs. reference group)

Output files (data/results/):
  <dataset_key>_results.parquet          — 1 row per (model x opt_metric)
  <dataset_key>_subgroup_results.parquet — 1 row per (model x opt_metric x subgroup x fold)

Usage
-----
  python scripts/run_experiments.py
  python scripts/run_experiments.py --dry-run       # 1 outer fold only, fast
  python scripts/run_experiments.py --datasets "Adult" "COMPAS"

"""

import os
import sys
import argparse
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    roc_auc_score, average_precision_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Add project root to path so we can import data_module and utils
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from data_module import DATASETS
from utils.bias_metrics import calculate_model_fairness_metrics

# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------

# Datasets to run, with the intersectional attribute combinations to evaluate.
# Each key must match a key in DATASETS (data_module/__init__.py).
EXPERIMENT_CONFIG = {
    "Adult 🇺🇸": {
        "attr_combinations": [
            ["sex", "race"],
        ]
    },
    "COMPAS 🇺🇸": {
        "attr_combinations": [
            ["sex", "race"],
        ]
    },
    "Dropout 🇵🇹": {
        "attr_combinations": [
            ["Gender", "Age_Group"],
        ]
    },
}

OPTIMIZATION_METRICS = ["accuracy", "recall", "precision", "roc_auc", "average_precision"]

MODELS = {
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "param_dist": {
            "clf__n_estimators": [50, 100, 200],
            "clf__max_depth": [5, 10, None],
            "clf__min_samples_leaf": [1, 5, 10],
            "clf__max_features": ["sqrt", "log2"],
        },
    },
    "GradientBoosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_dist": {
            "clf__n_estimators": [50, 100],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__subsample": [0.7, 1.0],
        },
    },
}

OUTER_K = 3
INNER_K = 3
N_ITER_SEARCH = 30

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: Build preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessor(df_train, feature_cols):
    """
    Dynamically builds a ColumnTransformer that:
      - One-hot encodes all categorical (object / category) columns.
      - Standard-scales all numeric columns.
    """
    cat_cols = [c for c in feature_cols if df_train[c].dtype in ["object", "category"]]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols,
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_group_series(df, attr_cols):
    """
    Creates a single string label per row combining all intersectional attributes.
    Example: "Male & White", "Female & Black"
    """
    return df[attr_cols].astype(str).apply(" & ".join, axis=1)


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run_dataset_experiment(dataset_key, attr_combination, dry_run=False):
    """
    Runs the full nested CV experiment for ONE dataset + ONE attribute combination.
    Returns two DataFrames: aggregate results and per-subgroup-per-fold results.
    """
    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_key}")
    print(f"Attrs   : {attr_combination}")
    print(f"{'='*60}")

    dataset_info = DATASETS[dataset_key]

    # Load data (no UF filter for experiments — we use the full dataset)
    print("  Loading data...")
    if dataset_info.get("supports_uf", False):
        df = dataset_info["loader"](["Todos"])
    else:
        df = dataset_info["loader"]()

    target_col = dataset_info["target"]
    favorable_val = dataset_info["favorable_val"]

    # Ensure all required columns are present
    missing = [c for c in attr_combination if c not in df.columns]
    if missing:
        print(f"  [SKIP] Missing columns: {missing}")
        return None, None

    # Drop rows where any attr or target is null
    cols_needed = attr_combination + [target_col]
    df = df.dropna(subset=cols_needed).copy()

    # Build intersectional group labels (preserved alongside data, not used as features)
    group_series_full = build_group_series(df, attr_combination)

    # Feature columns = all columns except target and the sensitive attrs we're auditing.
    # Sensitive attrs are EXCLUDED from features to simulate a "fairness-unaware" model,
    # consistent with the paper's setup: the model does not see the protected attribute.
    feature_cols = [c for c in df.columns if c != target_col and c not in attr_combination]
    X = df[feature_cols].copy()
    y_raw = df[target_col].copy()

    # Encode target if needed
    if y_raw.dtype == "object" or y_raw.dtype.name == "category":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw), index=y_raw.index)
        favorable_encoded = int(le.transform([favorable_val])[0])
    else:
        y = y_raw.copy()
        favorable_encoded = favorable_val

    agg_rows = []
    subgroup_rows = []

    run_ts = datetime.now(timezone.utc).isoformat()

    outer_cv = StratifiedKFold(n_splits=OUTER_K, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=42)

    for model_name, model_cfg in MODELS.items():
        for opt_metric in OPTIMIZATION_METRICS:
            print(f"  -> {model_name} / opt={opt_metric}", end="", flush=True)

            fold_global_metrics = []
            fold_subgroup_dfs = []

            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                if dry_run and fold_idx >= 1:
                    break  # Only 1 outer fold in dry-run mode

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                groups_test = group_series_full.iloc[test_idx].reset_index(drop=True)

                # Build pipeline
                preprocessor = build_preprocessor(X_train, feature_cols)
                pipe = Pipeline([
                    ("pre", preprocessor),
                    ("clf", model_cfg["estimator"]),
                ])

                # Inner CV: hyperparameter search
                n_iter = 5 if dry_run else N_ITER_SEARCH
                search = RandomizedSearchCV(
                    pipe,
                    param_distributions=model_cfg["param_dist"],
                    n_iter=n_iter,
                    cv=inner_cv,
                    scoring=opt_metric,
                    n_jobs=-1,
                    random_state=42,
                    refit=True,
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_

                # Predict on outer test fold
                y_pred = best_model.predict(X_test)
                y_test_reset = y_test.reset_index(drop=True)

                # --- Global performance metrics ---
                acc = accuracy_score(y_test_reset, y_pred)
                rec = recall_score(y_test_reset, y_pred, pos_label=favorable_encoded, zero_division=0)
                prec = precision_score(y_test_reset, y_pred, pos_label=favorable_encoded, zero_division=0)

                roc = np.nan
                prc = np.nan
                if hasattr(best_model, "predict_proba"):
                    y_proba = best_model.predict_proba(X_test)
                    classes = list(best_model.classes_)
                    if favorable_encoded in classes:
                        fav_col = classes.index(favorable_encoded)
                        y_score = y_proba[:, fav_col]
                        try:
                            roc = roc_auc_score(y_test_reset, y_score)
                            prc = average_precision_score(
                                y_test_reset, y_score, pos_label=favorable_encoded
                            )
                        except ValueError:
                            pass  # Only one class in test fold

                fold_global_metrics.append({
                    "fold": fold_idx,
                    "accuracy": acc,
                    "recall": rec,
                    "precision": prec,
                    "roc_auc": roc,
                    "pr_auc": prc,
                    "best_params": str(search.best_params_),
                })

                # --- Intersectional fairness metrics ---
                sg_df, agg = calculate_model_fairness_metrics(
                    y_true=y_test_reset,
                    y_pred=pd.Series(y_pred),
                    groups_series=groups_test,
                    favorable_val=favorable_encoded,
                )
                sg_df["fold"] = fold_idx
                sg_df["model"] = model_name
                sg_df["opt_metric"] = opt_metric
                sg_df["dataset"] = dataset_key
                sg_df["attrs"] = " & ".join(attr_combination)
                sg_df["reference_group"] = agg["reference_group"]
                sg_df["run_timestamp"] = run_ts
                fold_subgroup_dfs.append(sg_df)

                print(".", end="", flush=True)

            print()  # newline after fold dots

            # Aggregate global metrics across folds
            gm_df = pd.DataFrame(fold_global_metrics)
            subgroup_df_all = (
                pd.concat(fold_subgroup_dfs, ignore_index=True) if fold_subgroup_dfs else pd.DataFrame()
            )

            agg_row = {
                "dataset": dataset_key,
                "attrs": " & ".join(attr_combination),
                "model": model_name,
                "opt_metric": opt_metric,
                "accuracy_mean": gm_df["accuracy"].mean(),
                "accuracy_std": gm_df["accuracy"].std(),
                "recall_mean": gm_df["recall"].mean(),
                "recall_std": gm_df["recall"].std(),
                "precision_mean": gm_df["precision"].mean(),
                "precision_std": gm_df["precision"].std(),
                "roc_auc_mean": gm_df["roc_auc"].mean(),
                "roc_auc_std": gm_df["roc_auc"].std(),
                "pr_auc_mean": gm_df["pr_auc"].mean(),
                "pr_auc_std": gm_df["pr_auc"].std(),
                "run_timestamp": run_ts,
                "outer_folds_run": len(gm_df),
                "dry_run": dry_run,
            }

            # Aggregate fairness metrics: mean across outer folds
            if not subgroup_df_all.empty:
                fold_agg = subgroup_df_all.groupby("fold").apply(
                    lambda g: pd.Series({
                        "max_aaod": g["aaod"].max(),
                        "sensitivity_gap": (
                            g[g["n"] >= 30]["tpr"].max() - g[g["n"] >= 30]["tpr"].min()
                            if len(g[g["n"] >= 30]) >= 2 else np.nan
                        ),
                        "reference_group": g["reference_group"].iloc[0],
                    }), include_groups=False
                ).reset_index()

                agg_row["max_aaod_mean"] = fold_agg["max_aaod"].mean()
                agg_row["max_aaod_std"] = fold_agg["max_aaod"].std()
                agg_row["sensitivity_gap_mean"] = fold_agg["sensitivity_gap"].mean()
                agg_row["sensitivity_gap_std"] = fold_agg["sensitivity_gap"].std()
                agg_row["reference_group"] = fold_agg["reference_group"].mode()[0]
            else:
                agg_row.update({
                    "max_aaod_mean": np.nan, "max_aaod_std": np.nan,
                    "sensitivity_gap_mean": np.nan, "sensitivity_gap_std": np.nan,
                    "reference_group": None,
                })

            agg_rows.append(agg_row)
            subgroup_rows.append(subgroup_df_all)

    agg_df = pd.DataFrame(agg_rows)
    subgroup_df_final = (
        pd.concat(subgroup_rows, ignore_index=True) if subgroup_rows else pd.DataFrame()
    )
    return agg_df, subgroup_df_final


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run intersectional fairness experiments.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only 1 outer fold and 5 inner iterations (fast, for testing).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Limit to specific dataset keys.",
    )
    args = parser.parse_args()

    configs_to_run = EXPERIMENT_CONFIG
    if args.datasets:
        configs_to_run = {k: v for k, v in EXPERIMENT_CONFIG.items() if k in args.datasets}
        if not configs_to_run:
            print(f"[ERROR] No matching datasets found. Available: {list(EXPERIMENT_CONFIG.keys())}")
            sys.exit(1)

    all_agg = []
    all_subgroup = []

    for dataset_key, config in configs_to_run.items():
        # Sanitize dataset key for filenames
        safe_key = (
            dataset_key
            .replace(" ", "_")
            .replace("🇺🇸", "us")
            .replace("🇧🇷", "br")
            .replace("🇵🇹", "pt")
            .replace("🧪", "sim")
            .lower()
        )
        safe_key = "".join(c for c in safe_key if c.isalnum() or c == "_").strip("_")

        for attr_combination in config["attr_combinations"]:
            attrs_str = "_".join(attr_combination).lower()
            out_prefix = f"{safe_key}_{attrs_str}"

            agg_df, subgroup_df = run_dataset_experiment(
                dataset_key=dataset_key,
                attr_combination=attr_combination,
                dry_run=args.dry_run,
            )

            if agg_df is not None and not agg_df.empty:
                out_agg = os.path.join(RESULTS_DIR, f"{out_prefix}_results.parquet")
                out_sub = os.path.join(RESULTS_DIR, f"{out_prefix}_subgroup_results.parquet")

                agg_df.to_parquet(out_agg, index=False)
                print(f"  [SAVED] {out_agg}")

                if not subgroup_df.empty:
                    subgroup_df.to_parquet(out_sub, index=False)
                    print(f"  [SAVED] {out_sub}")

                all_agg.append(agg_df)
                all_subgroup.append(subgroup_df)

    if all_agg:
        consolidated_agg = pd.concat(all_agg, ignore_index=True)
        consolidated_sub = pd.concat(
            [s for s in all_subgroup if not s.empty], ignore_index=True
        )

        suffix = "_dryrun" if args.dry_run else ""
        consolidated_agg.to_parquet(
            os.path.join(RESULTS_DIR, f"all_results{suffix}.parquet"), index=False
        )
        consolidated_sub.to_parquet(
            os.path.join(RESULTS_DIR, f"all_subgroup_results{suffix}.parquet"), index=False
        )
        print(f"\n[DONE] Consolidated results saved to {RESULTS_DIR}")
        print(f"       Rows in agg results     : {len(consolidated_agg)}")
        print(f"       Rows in subgroup results : {len(consolidated_sub)}")
    else:
        print("\n[WARNING] No results were generated.")


if __name__ == "__main__":
    main()
