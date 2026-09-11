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

import gc

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Add project root to path so we can import data_module and utils
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from data_module import DATASETS
from utils.bias_metrics import calculate_model_fairness_metrics
from utils.fair_networks import FairMLPClassifier, AdversarialFairMLPClassifier

# ---------------------------------------------------------------------------
# Configuration Flags
# ---------------------------------------------------------------------------
INCLUDE_ADVERSARIAL_MODEL = False

# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------

# Datasets to run, with the intersectional attribute combinations to evaluate.
# Dynamically pulls all active datasets from data_module/__init__.py
EXPERIMENT_CONFIG = {}
for ds_key, ds_info in DATASETS.items():
    protected = ds_info.get('protected_attributes', [])
    if len(protected) >= 2:
        # Pega as duas primeiras variáveis protegidas como o par interseccional principal
        EXPERIMENT_CONFIG[ds_key] = {
            "attr_combinations": [
                [protected[0], protected[1]],
            ]
        }

OPTIMIZATION_METRICS = ["accuracy", "recall", "precision", "roc_auc", "average_precision"]

MODELS = {
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=1),
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

from sklearn.preprocessing import OrdinalEncoder

def build_preprocessor(df_train, feature_cols, sensitive_cols=None, pass_sensitive=False):
    """
    Dynamically builds a ColumnTransformer that:
      - One-hot encodes all categorical (object / category / string) columns.
      - Standard-scales all numeric columns.
      - If pass_sensitive=True, Ordinal Encodes sensitive columns and passes them through.
    """
    cat_cols = df_train[feature_cols].select_dtypes(exclude=['number']).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
            cat_cols,
        ))
        
    if pass_sensitive and sensitive_cols:
        transformers.append((
            "sensitive",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32),
            sensitive_cols
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_group_series(df, attr_cols):
    """
    Creates a single string label per row combining all intersectional attributes.
    Example: "Male & White", "Female & Black"
    """
    if not attr_cols:
        return pd.Series(["All"] * len(df), index=df.index)
    res = df[attr_cols[0]].astype(str)
    for col in attr_cols[1:]:
        res = res + " & " + df[col].astype(str)
    return res


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run_dataset_experiment(dataset_key, attr_combination, dry_run=False, n_jobs=2):
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
    raw_loader = getattr(dataset_info["loader"], "__wrapped__", dataset_info["loader"])
    if dataset_info.get("supports_uf", False):
        df = raw_loader(["Todos"])
    else:
        df = raw_loader()

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
    # Sensitive attrs are kept in X. Standard models drop them, fair neural models use them for loss.
    feature_cols = [c for c in df.columns if c != target_col and c not in attr_combination]
    X = df[feature_cols + attr_combination].copy()
    y_raw = df[target_col].copy()

    # Binary classification: Map favorable_val to 1 (positive outcome) and others to 0.
    # This guarantees consistent positive label (pos_label=1) for all scikit-learn scorers
    # (recall, precision, average_precision, roc_auc) and PyTorch models regardless of raw dtype.
    y = (y_raw == favorable_val).astype(int)
    favorable_encoded = 1

    # Determine global sensitive attribute dimensions for PyTorch models
    global_sens_dims = None
    if attr_combination:
        global_sens_dims = []
        for attr in attr_combination:
            global_sens_dims.append(df[attr].nunique())

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

                # Identify if model requires sensitive attributes
                is_fair_nn = model_name in ["FairMLP", "AdversarialFairMLP"]
                
                # Build pipeline
                preprocessor = build_preprocessor(
                    X_train, 
                    feature_cols, 
                    sensitive_cols=attr_combination, 
                    pass_sensitive=is_fair_nn
                )
                
                estimator = model_cfg["estimator"]
                if is_fair_nn:
                    # Dynamically set the number of sensitive attributes and their categories
                    estimator.set_params(
                        num_sensitive_attrs=len(attr_combination),
                        sensitive_dims_list=global_sens_dims
                    )

                pipe = Pipeline([
                    ("pre", preprocessor),
                    ("clf", estimator),
                ])

                # Inner CV: hyperparameter search
                is_large = len(X_train) > 200_000
                param_dist = dict(model_cfg["param_dist"])
                if is_large and model_name == "RandomForest":
                    # Evita max_depth=None em bases massivas (1.65M linhas), o que gerava árvores
                    # com 17 milhões de nós e estourava a RAM com ArrayMemoryError de 134 MiB por nó.
                    param_dist["clf__max_depth"] = [5, 10, 15]
                    param_dist["clf__n_estimators"] = [50, 100]

                n_iter = 5 if dry_run else (10 if is_large else N_ITER_SEARCH)
                if is_fair_nn and not dry_run:
                    n_iter = min(n_iter, 4) # Less searches for NNs to save time

                eff_n_jobs = 1 if is_large else n_jobs

                search = RandomizedSearchCV(
                    pipe,
                    param_distributions=param_dist,
                    n_iter=n_iter,
                    cv=inner_cv,
                    scoring=opt_metric,
                    n_jobs=eff_n_jobs,
                    random_state=42,
                    refit=True,
                    error_score="raise",
                )
                try:
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_

                    # Predict on outer test fold
                    y_pred = best_model.predict(X_test)
                except Exception as e:
                    print(f" [Fold {fold_idx} Failed: {str(e)[:50]}]", end="", flush=True)
                    fold_global_metrics.append({
                        "fold": fold_idx,
                        "accuracy": np.nan, "recall": np.nan, "precision": np.nan,
                        "roc_auc": np.nan, "pr_auc": np.nan, "best_params": "FAILED",
                    })
                    del preprocessor, pipe
                    gc.collect()
                    continue

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
                        # CORREÇÃO DO BUG: garantindo que a classe favorável vira a classe positiva (1)
                        # Isso previne que a pontuação de roc_auc_score inverta caso favorável seja 0 (como em COMPAS).
                        fav_col = classes.index(favorable_encoded)
                        y_score = y_proba[:, fav_col]
                        try:
                            y_true_bin = (y_test_reset == favorable_encoded).astype(int)
                            roc = roc_auc_score(y_true_bin, y_score)
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

                del search, best_model, pipe, preprocessor, X_train, X_test, y_train, y_test
                gc.collect()

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
# Lightweight Lambda Sweep for FairMLP (Non-nested simple 3-fold CV)
# ---------------------------------------------------------------------------

def run_lambda_sweep(dataset_key, attr_combination, dry_run=False):
    """
    Executes a fast, non-nested 3-fold CV sweep over fixed fairness penalty lambdas
    for FairMLPClassifier: [0.0, 0.5, 1.0, 2.0, 4.0].
    """
    print(f"\n  --- FairMLP Lambda Sweep for {dataset_key} ({' & '.join(attr_combination)}) ---")
    dataset_info = DATASETS[dataset_key]

    raw_loader = getattr(dataset_info["loader"], "__wrapped__", dataset_info["loader"])
    if dataset_info.get("supports_uf", False):
        df = raw_loader(["Todos"])
    else:
        df = raw_loader()

    target_col = dataset_info["target"]
    favorable_val = dataset_info["favorable_val"]

    missing = [c for c in attr_combination if c not in df.columns]
    if missing:
        print(f"  [SKIP] Missing columns for lambda sweep: {missing}")
        return None, None

    cols_needed = attr_combination + [target_col]
    df = df.dropna(subset=cols_needed).copy()

    group_series_full = build_group_series(df, attr_combination)
    feature_cols = [c for c in df.columns if c != target_col and c not in attr_combination]
    X = df[feature_cols + attr_combination].copy()
    y_raw = df[target_col].copy()

    # Binary classification: Map favorable_val to 1 (positive outcome) and others to 0.
    y = (y_raw == favorable_val).astype(int)
    favorable_encoded = 1

    global_sens_dims = [df[attr].nunique() for attr in attr_combination]

    lambdas = [0.0, 0.5, 1.0, 2.0, 4.0]
    if dry_run:
        lambdas = [0.0, 1.0]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    run_ts = datetime.now(timezone.utc).isoformat()

    sweep_agg_rows = []
    sweep_subgroup_rows = []

    for lam in lambdas:
        print(f"    -> FairMLP (λ={lam})", end="", flush=True)
        fold_metrics = []
        fold_subgroups = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            if dry_run and fold_idx >= 1:
                break

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            groups_test = group_series_full.iloc[test_idx].reset_index(drop=True)

            preprocessor = build_preprocessor(
                X_train,
                feature_cols,
                sensitive_cols=attr_combination,
                pass_sensitive=True,
            )

            is_large_data = len(X_train) > 200_000
            clf = FairMLPClassifier(
                hidden_dims=(128, 64),
                lr=0.001,
                epochs=15 if not dry_run else 3,
                batch_size=1024 if is_large_data else 256,
                lambda_fairness=lam,
                num_sensitive_attrs=len(attr_combination),
                sensitive_dims_list=global_sens_dims,
                random_state=42 + fold_idx,
            )

            pipe = Pipeline([
                ("pre", preprocessor),
                ("clf", clf),
            ])

            try:
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                y_proba = pipe.predict_proba(X_test)
            except Exception as e:
                print(f" [Fold {fold_idx} Failed: {str(e)[:50]}]", end="", flush=True)
                del preprocessor, pipe
                gc.collect()
                continue

            y_test_reset = y_test.reset_index(drop=True)

            acc = accuracy_score(y_test_reset, y_pred)
            rec = recall_score(y_test_reset, y_pred, pos_label=favorable_encoded, zero_division=0)
            prec = precision_score(y_test_reset, y_pred, pos_label=favorable_encoded, zero_division=0)

            roc = np.nan
            prc = np.nan
            classes = list(clf.classes_)
            if favorable_encoded in classes:
                fav_col = classes.index(favorable_encoded)
                y_score = y_proba[:, fav_col]
                try:
                    y_true_bin = (y_test_reset == favorable_encoded).astype(int)
                    roc = roc_auc_score(y_true_bin, y_score)
                    prc = average_precision_score(y_test_reset, y_score, pos_label=favorable_encoded)
                except ValueError:
                    pass

            fold_metrics.append({
                "accuracy": acc,
                "recall": rec,
                "precision": prec,
                "roc_auc": roc,
                "pr_auc": prc,
            })

            # Intersectional fairness metrics
            sg_df, agg = calculate_model_fairness_metrics(
                y_true=y_test_reset,
                y_pred=pd.Series(y_pred),
                groups_series=groups_test,
                favorable_val=favorable_encoded,
            )
            sg_df["fold"] = fold_idx
            sg_df["model"] = "FairMLP"
            sg_df["opt_metric"] = f"lambda_{lam}"
            sg_df["lambda_fairness"] = lam
            sg_df["dataset"] = dataset_key
            sg_df["attrs"] = " & ".join(attr_combination)
            sg_df["reference_group"] = agg["reference_group"]
            sg_df["run_timestamp"] = run_ts
            fold_subgroups.append(sg_df)

            del pipe, clf, preprocessor, X_train, X_test, y_train, y_test
            gc.collect()

            print(".", end="", flush=True)

        print()  # newline after fold dots

        if fold_metrics:
            m_df = pd.DataFrame(fold_metrics)
            s_df = pd.concat(fold_subgroups, ignore_index=True) if fold_subgroups else pd.DataFrame()

            if not s_df.empty:
                fold_agg = s_df.groupby("fold").apply(
                    lambda g: pd.Series({
                        "max_aaod": g["aaod"].max(),
                        "sensitivity_gap": (
                            g[g["n"] >= 30]["tpr"].max() - g[g["n"] >= 30]["tpr"].min()
                            if len(g[g["n"] >= 30]) >= 2 else np.nan
                        ),
                        "reference_group": g["reference_group"].iloc[0],
                    }), include_groups=False
                ).reset_index()

                max_aaod_mean = fold_agg["max_aaod"].mean()
                max_aaod_std = fold_agg["max_aaod"].std()
                sens_gap_mean = fold_agg["sensitivity_gap"].mean()
                sens_gap_std = fold_agg["sensitivity_gap"].std()
                ref_group = fold_agg["reference_group"].mode()[0]
            else:
                max_aaod_mean, max_aaod_std = np.nan, np.nan
                sens_gap_mean, sens_gap_std = np.nan, np.nan
                ref_group = None

            sweep_agg_rows.append({
                "dataset": dataset_key,
                "attrs": " & ".join(attr_combination),
                "model": "FairMLP",
                "opt_metric": f"lambda_{lam}",
                "accuracy_mean": m_df["accuracy"].mean(),
                "accuracy_std": m_df["accuracy"].std(),
                "recall_mean": m_df["recall"].mean(),
                "recall_std": m_df["recall"].std(),
                "precision_mean": m_df["precision"].mean(),
                "precision_std": m_df["precision"].std(),
                "roc_auc_mean": m_df["roc_auc"].mean(),
                "roc_auc_std": m_df["roc_auc"].std(),
                "pr_auc_mean": m_df["pr_auc"].mean(),
                "pr_auc_std": m_df["pr_auc"].std(),
                "run_timestamp": run_ts,
                "outer_folds_run": len(m_df),
                "dry_run": dry_run,
                "max_aaod_mean": max_aaod_mean,
                "max_aaod_std": max_aaod_std,
                "sensitivity_gap_mean": sens_gap_mean,
                "sensitivity_gap_std": sens_gap_std,
                "reference_group": ref_group,
                "lambda_fairness": lam,
            })

            sweep_subgroup_rows.extend(fold_subgroups)

    sweep_agg_df = pd.DataFrame(sweep_agg_rows)
    sweep_sub_df = pd.concat(sweep_subgroup_rows, ignore_index=True) if sweep_subgroup_rows else pd.DataFrame()
    return sweep_agg_df, sweep_sub_df


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
        "--skip-existing",
        action="store_true",
        help="Skip datasets and models that already have parquet files saved in data/results/.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Limit to specific dataset keys.",
    )
    parser.add_argument(
        "--skip-lambda",
        action="store_true",
        help="Skip the neural FairMLP lambda sweep and run only standard models.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for hyperparameter search (default: 1, robust on Windows).",
    )
    args = parser.parse_args()

    configs_to_run = EXPERIMENT_CONFIG
    if args.datasets:
        selected = {}
        for req in args.datasets:
            req_norm = req.lower().replace(" ", "").replace("_", "").replace("🇺🇸", "us").replace("🇧🇷", "br").replace("🇵🇹", "pt")
            for k, v in EXPERIMENT_CONFIG.items():
                k_norm = k.lower().replace(" ", "").replace("_", "").replace("🇺🇸", "us").replace("🇧🇷", "br").replace("🇵🇹", "pt")
                if req_norm in k_norm or k_norm in req_norm or req.lower() in k.lower():
                    selected[k] = v
        configs_to_run = selected
        if not configs_to_run:
            print(f"[ERROR] No matching datasets found for {args.datasets}. Available: {list(EXPERIMENT_CONFIG.keys())}")
            sys.exit(1)
        else:
            print(f"[INFO] Datasets selected to run: {list(configs_to_run.keys())}")

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
            out_agg = os.path.join(RESULTS_DIR, f"{out_prefix}_results.parquet")
            out_sub = os.path.join(RESULTS_DIR, f"{out_prefix}_subgroup_results.parquet")
            sweep_path = os.path.join(RESULTS_DIR, f"{out_prefix}_lambda_sweep.parquet")

            # 1) Standard Models (RF, GBM)
            if args.skip_existing and os.path.exists(out_agg) and os.path.exists(out_sub):
                print(f"\n[RESUME] Loading existing standard model results for {out_prefix}")
                agg_df = pd.read_parquet(out_agg)
                subgroup_df = pd.read_parquet(out_sub)
            else:
                agg_df, subgroup_df = run_dataset_experiment(
                    dataset_key=dataset_key,
                    attr_combination=attr_combination,
                    dry_run=args.dry_run,
                    n_jobs=args.n_jobs,
                )

                if agg_df is not None and not agg_df.empty:
                    agg_df.to_parquet(out_agg, index=False)
                    print(f"  [SAVED] {out_agg}")

                    if not subgroup_df.empty:
                        subgroup_df.to_parquet(out_sub, index=False)
                        print(f"  [SAVED] {out_sub}")

            if agg_df is not None and not agg_df.empty:
                all_agg.append(agg_df)
                all_subgroup.append(subgroup_df)
                
            # 2) Lightweight Lambda Sweep (FairMLP)
            if not args.skip_lambda:
                if args.skip_existing and os.path.exists(sweep_path):
                    print(f"\n[RESUME] Loading existing lambda sweep for {out_prefix}")
                    sweep_df = pd.read_parquet(sweep_path)
                    all_agg.append(sweep_df)
                else:
                    sweep_df, sweep_sub = run_lambda_sweep(dataset_key, attr_combination, dry_run=args.dry_run)
                    if sweep_df is not None and not sweep_df.empty:
                        sweep_df.to_parquet(sweep_path, index=False)
                        print(f"  [SAVED] {sweep_path}")
                        all_agg.append(sweep_df)
                        if sweep_sub is not None and not sweep_sub.empty:
                            all_subgroup.append(sweep_sub)

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
