"""
run_experiments.py — Intersectional Fairness Experiment Pipeline
=================================================================

Executes the nested cross-validation experiment and saves raw results to disk.
The Streamlit dashboard (pages/2_🤖_Modelos.py) loads those pre-computed results
and renders only visualizations — it never trains models at runtime.

Pipeline (faithful to the reference paper):
  Outer CV: K=3 folds  →  unbiased performance estimate
  Inner CV: K=3 folds  →  hyperparameter tuning (RandomizedSearchCV, n_iter=30)
  Models: RandomForestClassifier, GradientBoostingClassifier
  Optimization metrics: accuracy, recall, precision, roc_auc, average_precision
  FairMLP: non-nested 3-fold CV over fixed fairness penalties (lambda sweep)

Fairness metrics computed per outer fold (see utils/bias_metrics.py):
  - Intersectional AAOD (per subgroup + Max across subgroups)
  - Sensitivity Gap (Max TPR - Min TPR, viable subgroups N>=30)
  - Post-training Disparate Impact (per subgroup vs. reference group)

Two execution modes
-------------------
1) Sequential (default) — everything in one process, as used for the thesis proposal.
   Output files (--results-dir, default data/results/):
     <prefix>_results.parquet          — 1 row per (model x opt_metric)
     <prefix>_subgroup_results.parquet — 1 row per (model x opt_metric x subgroup x fold)
     <prefix>_lambda_sweep.parquet     — 1 row per lambda

     python scripts/run_experiments.py
     python scripts/run_experiments.py --dry-run       # 1 outer fold only, fast
     python scripts/run_experiments.py --datasets "Adult" "COMPAS"

2) Sharded (--shard-dir) — each unit of work
       (dataset, model, configuration, outer fold, seed)
   is independent and resumable, so units can run as separate cluster jobs.
   Each unit writes a small shard; scripts/collect_results.py merges the shards
   into the same file layout as mode 1.

     # list one argument line per unit (e.g. to build a Slurm job array)
     python scripts/run_experiments.py --print-units --datasets compas_us --seeds 42 43 44

     # run a single unit
     python scripts/run_experiments.py --shard-dir runs/exp1/shards \
         --datasets compas_us --only standard --models RandomForest \
         --opt-metrics accuracy --folds 0 --seeds 42

     # merge
     python scripts/collect_results.py --shard-dir runs/exp1/shards --out-dir runs/exp1/results

Seeds: the outer and inner CV partitions are always generated with random_state=42,
so every seed is evaluated on the same test instances. The seed controls the model
(estimator initialisation, hyperparameter sampling, network weights and batch order).
"""

import os
import sys
import argparse
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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

from sklearn.impute import SimpleImputer

from utils.bias_metrics import calculate_model_fairness_metrics
from utils.fair_networks import FairMLPClassifier, AdversarialFairMLPClassifier
from utils.experiment_io import (
    build_agg_row, make_safe_key, make_out_prefix,
    shard_stem, shard_done, write_shard,
)

# ---------------------------------------------------------------------------
# Configuration Flags
# ---------------------------------------------------------------------------
INCLUDE_ADVERSARIAL_MODEL = False

# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------

# Dataset registry and the intersectional attribute combinations to evaluate.
# Filled by load_registry():
#   "dashboard" -> data_module.DATASETS        (bases da qualificação; importa o Streamlit)
#   "thesis"    -> utils.thesis_datasets       (bases da tese; sem Streamlit)
DATASETS = {}
EXPERIMENT_CONFIG = {}

# Smoke tests only (--sample-rows): random subsample of each dataset.
SAMPLE_ROWS = None


# Tamanho mínimo do grupo de referência do DI/AAOD. None = comportamento da qualificação
# (qualquer subgrupo); a tese usa 30, o mesmo limiar de viabilidade da diferença de sensibilidade.
REF_MIN_N = None


def load_registry(name="dashboard"):
    global DATASETS, EXPERIMENT_CONFIG, REF_MIN_N
    if name == "thesis":
        from utils.thesis_datasets import THESIS_DATASETS as registry
        REF_MIN_N = 30
    else:
        from data_module import DATASETS as registry
        REF_MIN_N = None

    DATASETS = registry
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

LAMBDA_MODEL_NAME = "FairMLP"
LAMBDAS = [0.0, 0.5, 1.0, 2.0, 4.0]
LAMBDAS_DRY_RUN = [0.0, 1.0]

OUTER_K = 3
INNER_K = 3
N_ITER_SEARCH = 30

# Semente das partições de validação cruzada (fixa) e semente padrão dos modelos.
CV_SEED = 42
DEFAULT_SEED = 42

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")

# ---------------------------------------------------------------------------
# Helper: Build preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessor(df_train, feature_cols, sensitive_cols=None, pass_sensitive=False,
                       impute_numeric=False):
    """
    Dynamically builds a ColumnTransformer that:
      - One-hot encodes all categorical (object / category / string) columns.
      - Standard-scales all numeric columns.
      - If impute_numeric=True (dataset has missing numeric values), imputes the median
        (fitted on the training fold) and adds a missing-value indicator per column.
      - If pass_sensitive=True, Ordinal Encodes sensitive columns and passes them through.
    """
    cat_cols = df_train[feature_cols].select_dtypes(exclude=['number']).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    transformers = []
    if num_cols:
        if impute_numeric:
            num_transformer = Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ])
        else:
            num_transformer = StandardScaler()
        transformers.append(("num", num_transformer, num_cols))
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
# Data preparation (shared by all models)
# ---------------------------------------------------------------------------

def prepare_dataset(dataset_key, attr_combination):
    """
    Loads one dataset and builds everything the fold functions need.
    Returns None if a required column is missing.
    """
    dataset_info = DATASETS[dataset_key]

    # Load data (no UF filter for experiments — we use the full dataset)
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
        return None

    # Drop rows where any attr or target is null
    cols_needed = attr_combination + [target_col]
    df = df.dropna(subset=cols_needed).copy()

    if SAMPLE_ROWS and len(df) > SAMPLE_ROWS:
        df = df.sample(n=SAMPLE_ROWS, random_state=CV_SEED)

    # Build intersectional group labels (preserved alongside data, not used as features)
    group_series_full = build_group_series(df, attr_combination)

    # Feature columns = all columns except target and the sensitive attrs we're auditing,
    # unless the registry lists the input variables explicitly (feature_columns).
    # Sensitive attrs are kept in X. Standard models drop them, fair neural models use them for loss.
    if dataset_info.get("feature_columns"):
        absent = [c for c in dataset_info["feature_columns"] if c not in df.columns]
        if absent:
            print(f"  [SKIP] feature_columns not found in data: {absent}")
            return None
        feature_cols = [c for c in dataset_info["feature_columns"]
                        if c != target_col and c not in attr_combination]
    else:
        feature_cols = [c for c in df.columns if c != target_col and c not in attr_combination]
    X = df[feature_cols + attr_combination].copy()

    # Missing numeric values (none in the qualification datasets) are imputed inside each fold.
    num_feature_cols = X[feature_cols].select_dtypes(include=['number']).columns
    impute_numeric = bool(X[num_feature_cols].isna().any().any())
    y_raw = df[target_col].copy()

    # Binary classification: Map favorable_val to 1 (positive outcome) and others to 0.
    # This guarantees consistent positive label (pos_label=1) for all scikit-learn scorers
    # (recall, precision, average_precision, roc_auc) and PyTorch models regardless of raw dtype.
    y = (y_raw == favorable_val).astype(int)

    # Global sensitive attribute dimensions for PyTorch models
    sens_dims = [df[attr].nunique() for attr in attr_combination] if attr_combination else None

    # Outer CV partitions: fixed seed, identical for every model, configuration and seed.
    outer_cv = StratifiedKFold(n_splits=OUTER_K, shuffle=True, random_state=CV_SEED)
    splits = list(outer_cv.split(X, y))

    return {
        "dataset_key": dataset_key,
        "attr_combination": attr_combination,
        "attrs_str": " & ".join(attr_combination),
        "X": X,
        "y": y,
        "groups": group_series_full,
        "feature_cols": feature_cols,
        "sens_dims": sens_dims,
        "favorable_encoded": 1,
        "splits": splits,
        "impute_numeric": impute_numeric,
    }


def _global_metrics(y_true, y_pred, y_proba, classes, favorable_encoded):
    """Accuracy, recall, precision, ROC-AUC and PR-AUC for one test fold."""
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, pos_label=favorable_encoded, zero_division=0)
    prec = precision_score(y_true, y_pred, pos_label=favorable_encoded, zero_division=0)

    roc = np.nan
    prc = np.nan
    if y_proba is not None and favorable_encoded in classes:
        # Garante que a classe favorável é a classe positiva (1). Isso previne que o
        # roc_auc_score inverta caso a classe favorável original seja 0 (como no COMPAS).
        fav_col = classes.index(favorable_encoded)
        y_score = y_proba[:, fav_col]
        try:
            y_true_bin = (y_true == favorable_encoded).astype(int)
            roc = roc_auc_score(y_true_bin, y_score)
            prc = average_precision_score(y_true, y_score, pos_label=favorable_encoded)
        except ValueError:
            pass  # Only one class in test fold

    return {"accuracy": acc, "recall": rec, "precision": prec, "roc_auc": roc, "pr_auc": prc}


def _annotate_subgroups(sg_df, agg, prep, fold_idx, model_name, opt_metric, run_ts):
    sg_df["fold"] = fold_idx
    sg_df["model"] = model_name
    sg_df["opt_metric"] = opt_metric
    sg_df["dataset"] = prep["dataset_key"]
    sg_df["attrs"] = prep["attrs_str"]
    sg_df["reference_group"] = agg["reference_group"]
    sg_df["run_timestamp"] = run_ts
    return sg_df


# ---------------------------------------------------------------------------
# One outer fold: standard models (nested CV)
# ---------------------------------------------------------------------------

def run_standard_fold(prep, model_name, opt_metric, fold_idx, run_ts,
                      dry_run=False, n_jobs=1, seed=DEFAULT_SEED):
    """
    Tunes (inner CV) and evaluates ONE model/opt_metric on ONE outer fold.
    Returns (metrics_dict, subgroup_df). On failure, metrics are NaN with
    best_params == "FAILED" and subgroup_df is None.
    """
    model_cfg = MODELS[model_name]
    X, y = prep["X"], prep["y"]
    feature_cols = prep["feature_cols"]
    attr_combination = prep["attr_combination"]
    favorable_encoded = prep["favorable_encoded"]
    train_idx, test_idx = prep["splits"][fold_idx]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_test = prep["groups"].iloc[test_idx].reset_index(drop=True)

    # Identify if model requires sensitive attributes
    is_fair_nn = model_name in ["FairMLP", "AdversarialFairMLP"]

    preprocessor = build_preprocessor(
        X_train,
        feature_cols,
        sensitive_cols=attr_combination,
        pass_sensitive=is_fair_nn,
        impute_numeric=prep["impute_numeric"],
    )

    estimator = clone(model_cfg["estimator"])
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=seed)
    if is_fair_nn:
        # Dynamically set the number of sensitive attributes and their categories
        estimator.set_params(
            num_sensitive_attrs=len(attr_combination),
            sensitive_dims_list=prep["sens_dims"],
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
        n_iter = min(n_iter, 4)  # Less searches for NNs to save time

    eff_n_jobs = 1 if is_large else n_jobs
    inner_cv = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=CV_SEED)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=inner_cv,
        scoring=opt_metric,
        n_jobs=eff_n_jobs,
        random_state=seed,
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
        gc.collect()
        return {
            "fold": fold_idx,
            "accuracy": np.nan, "recall": np.nan, "precision": np.nan,
            "roc_auc": np.nan, "pr_auc": np.nan, "best_params": "FAILED",
        }, None

    y_test_reset = y_test.reset_index(drop=True)

    y_proba = None
    classes = []
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test)
        classes = list(best_model.classes_)

    metrics = {"fold": fold_idx}
    metrics.update(_global_metrics(y_test_reset, y_pred, y_proba, classes, favorable_encoded))
    metrics["best_params"] = str(search.best_params_)

    # --- Intersectional fairness metrics ---
    sg_df, agg = calculate_model_fairness_metrics(
        y_true=y_test_reset,
        y_pred=pd.Series(y_pred),
        groups_series=groups_test,
        favorable_val=favorable_encoded,
        min_ref_n=REF_MIN_N,
    )
    sg_df = _annotate_subgroups(sg_df, agg, prep, fold_idx, model_name, opt_metric, run_ts)

    del search, best_model, pipe, preprocessor, X_train, X_test, y_train, y_test
    gc.collect()

    print(".", end="", flush=True)
    return metrics, sg_df


# ---------------------------------------------------------------------------
# One outer fold: FairMLP with a fixed fairness penalty (non-nested)
# ---------------------------------------------------------------------------

def run_lambda_fold(prep, lam, fold_idx, run_ts, dry_run=False, seed=DEFAULT_SEED):
    """
    Trains and evaluates FairMLPClassifier with penalty `lam` on ONE fold.
    Returns (metrics_dict, subgroup_df), or (None, None) on failure.
    """
    X, y = prep["X"], prep["y"]
    feature_cols = prep["feature_cols"]
    attr_combination = prep["attr_combination"]
    favorable_encoded = prep["favorable_encoded"]
    train_idx, test_idx = prep["splits"][fold_idx]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_test = prep["groups"].iloc[test_idx].reset_index(drop=True)

    preprocessor = build_preprocessor(
        X_train,
        feature_cols,
        sensitive_cols=attr_combination,
        pass_sensitive=True,
        impute_numeric=prep["impute_numeric"],
    )

    is_large_data = len(X_train) > 200_000
    clf = FairMLPClassifier(
        hidden_dims=(128, 64),
        lr=0.001,
        epochs=15 if not dry_run else 3,
        batch_size=1024 if is_large_data else 256,
        lambda_fairness=lam,
        num_sensitive_attrs=len(attr_combination),
        sensitive_dims_list=prep["sens_dims"],
        random_state=seed + fold_idx,
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
        return None, None

    y_test_reset = y_test.reset_index(drop=True)
    metrics = _global_metrics(y_test_reset, y_pred, y_proba, list(clf.classes_), favorable_encoded)

    # Intersectional fairness metrics
    sg_df, agg = calculate_model_fairness_metrics(
        y_true=y_test_reset,
        y_pred=pd.Series(y_pred),
        groups_series=groups_test,
        favorable_val=favorable_encoded,
        min_ref_n=REF_MIN_N,
    )
    sg_df["fold"] = fold_idx
    sg_df["model"] = LAMBDA_MODEL_NAME
    sg_df["opt_metric"] = f"lambda_{lam}"
    sg_df["lambda_fairness"] = lam
    sg_df["dataset"] = prep["dataset_key"]
    sg_df["attrs"] = prep["attrs_str"]
    sg_df["reference_group"] = agg["reference_group"]
    sg_df["run_timestamp"] = run_ts

    del pipe, clf, preprocessor, X_train, X_test, y_train, y_test
    gc.collect()

    print(".", end="", flush=True)
    return metrics, sg_df


# ---------------------------------------------------------------------------
# Sequential mode: full experiment for one dataset
# ---------------------------------------------------------------------------

def run_dataset_experiment(dataset_key, attr_combination, dry_run=False, n_jobs=2, seed=DEFAULT_SEED):
    """
    Runs the full nested CV experiment for ONE dataset + ONE attribute combination.
    Returns two DataFrames: aggregate results and per-subgroup-per-fold results.
    """
    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_key}")
    print(f"Attrs   : {attr_combination}")
    print(f"{'='*60}")

    print("  Loading data...")
    prep = prepare_dataset(dataset_key, attr_combination)
    if prep is None:
        return None, None

    agg_rows = []
    subgroup_rows = []
    run_ts = datetime.now(timezone.utc).isoformat()

    for model_name in MODELS:
        for opt_metric in OPTIMIZATION_METRICS:
            print(f"  -> {model_name} / opt={opt_metric}", end="", flush=True)

            fold_global_metrics = []
            fold_subgroup_dfs = []

            for fold_idx in range(len(prep["splits"])):
                if dry_run and fold_idx >= 1:
                    break  # Only 1 outer fold in dry-run mode

                metrics, sg_df = run_standard_fold(
                    prep, model_name, opt_metric, fold_idx, run_ts,
                    dry_run=dry_run, n_jobs=n_jobs, seed=seed,
                )
                fold_global_metrics.append(metrics)
                if sg_df is not None:
                    fold_subgroup_dfs.append(sg_df)

            print()  # newline after fold dots

            gm_df = pd.DataFrame(fold_global_metrics)
            subgroup_df_all = (
                pd.concat(fold_subgroup_dfs, ignore_index=True) if fold_subgroup_dfs else pd.DataFrame()
            )

            agg_rows.append(build_agg_row(
                dataset_key, prep["attrs_str"], model_name, opt_metric,
                gm_df, subgroup_df_all, run_ts, dry_run,
            ))
            subgroup_rows.append(subgroup_df_all)

    agg_df = pd.DataFrame(agg_rows)
    subgroup_df_final = (
        pd.concat(subgroup_rows, ignore_index=True) if subgroup_rows else pd.DataFrame()
    )
    return agg_df, subgroup_df_final


def run_lambda_sweep(dataset_key, attr_combination, dry_run=False, seed=DEFAULT_SEED):
    """
    Executes a fast, non-nested 3-fold CV sweep over fixed fairness penalty lambdas
    for FairMLPClassifier: [0.0, 0.5, 1.0, 2.0, 4.0].
    """
    print(f"\n  --- FairMLP Lambda Sweep for {dataset_key} ({' & '.join(attr_combination)}) ---")
    prep = prepare_dataset(dataset_key, attr_combination)
    if prep is None:
        return None, None

    lambdas = LAMBDAS_DRY_RUN if dry_run else LAMBDAS
    run_ts = datetime.now(timezone.utc).isoformat()

    sweep_agg_rows = []
    sweep_subgroup_rows = []

    for lam in lambdas:
        print(f"    -> FairMLP (λ={lam})", end="", flush=True)
        fold_metrics = []
        fold_subgroups = []

        for fold_idx in range(len(prep["splits"])):
            if dry_run and fold_idx >= 1:
                break

            metrics, sg_df = run_lambda_fold(prep, lam, fold_idx, run_ts, dry_run=dry_run, seed=seed)
            if metrics is None:
                continue
            fold_metrics.append(metrics)
            fold_subgroups.append(sg_df)

        print()  # newline after fold dots

        if fold_metrics:
            m_df = pd.DataFrame(fold_metrics)
            s_df = pd.concat(fold_subgroups, ignore_index=True) if fold_subgroups else pd.DataFrame()

            sweep_agg_rows.append(build_agg_row(
                dataset_key, prep["attrs_str"], LAMBDA_MODEL_NAME, f"lambda_{lam}",
                m_df, s_df, run_ts, dry_run, lambda_fairness=lam,
            ))
            sweep_subgroup_rows.extend(fold_subgroups)

    sweep_agg_df = pd.DataFrame(sweep_agg_rows)
    sweep_sub_df = pd.concat(sweep_subgroup_rows, ignore_index=True) if sweep_subgroup_rows else pd.DataFrame()
    return sweep_agg_df, sweep_sub_df


# ---------------------------------------------------------------------------
# Sharded mode: independent, resumable units of work
# ---------------------------------------------------------------------------

def enumerate_units(configs_to_run, args):
    """
    Lists every unit of work selected by the CLI filters, in canonical order.
    A unit = (dataset, attrs, kind, model, configuration, outer fold, seed).
    """
    kinds = ["standard", "lambda"] if args.only == "all" else [args.only]
    folds = args.folds if args.folds is not None else list(range(OUTER_K))
    if args.dry_run:
        folds = [f for f in folds if f == 0]
    models = args.models if args.models is not None else list(MODELS.keys())
    opt_metrics = args.opt_metrics if args.opt_metrics is not None else OPTIMIZATION_METRICS
    if args.lambdas is not None:
        lambdas = args.lambdas
    else:
        lambdas = LAMBDAS_DRY_RUN if args.dry_run else LAMBDAS

    units = []
    for dataset_key, config in configs_to_run.items():
        for attr_combination in config["attr_combinations"]:
            out_prefix = make_out_prefix(dataset_key, attr_combination)
            base = {
                "dataset_key": dataset_key,
                "attr_combination": attr_combination,
                "out_prefix": out_prefix,
            }
            if "standard" in kinds:
                for model in models:
                    if model not in MODELS:
                        continue
                    m_idx = list(MODELS.keys()).index(model)
                    for opt_metric in opt_metrics:
                        o_idx = OPTIMIZATION_METRICS.index(opt_metric)
                        for seed in args.seeds:
                            for fold in folds:
                                units.append(dict(
                                    base, kind="standard", model=model, config=opt_metric,
                                    lam=None, fold=fold, seed=seed, order=m_idx * 1000 + o_idx,
                                ))
            if "lambda" in kinds and (args.models is None or LAMBDA_MODEL_NAME in args.models):
                for l_idx, lam in enumerate(sorted(lambdas)):
                    for seed in args.seeds:
                        for fold in folds:
                            units.append(dict(
                                base, kind="lambda", model=LAMBDA_MODEL_NAME, config=f"lambda_{lam}",
                                lam=lam, fold=fold, seed=seed, order=100_000 + l_idx,
                            ))
    return units


def format_unit_args(unit):
    """Command-line arguments that select exactly this unit (one line per unit)."""
    parts = ["--datasets", make_safe_key(unit["dataset_key"]), "--only", unit["kind"]]
    if unit["kind"] == "standard":
        parts += ["--models", unit["model"], "--opt-metrics", unit["config"]]
    else:
        parts += ["--lambdas", str(unit["lam"])]
    parts += ["--folds", str(unit["fold"]), "--seeds", str(unit["seed"])]
    return " ".join(parts)


def run_units(units, args):
    """Runs the selected units, skipping those whose shard already exists."""
    pending = [u for u in units if not shard_done(args.shard_dir, _stem(u))]
    print(f"[INFO] Units selected: {len(units)} | already done: {len(units) - len(pending)} "
          f"| to run: {len(pending)}")
    if not pending:
        return

    prep_cache = {}
    for unit in pending:
        key = unit["out_prefix"]
        if key not in prep_cache:
            prep_cache.clear()  # keep at most one dataset in memory
            gc.collect()
            print(f"\n[DATA] Loading {unit['dataset_key']} ({' & '.join(unit['attr_combination'])})")
            prep_cache[key] = prepare_dataset(unit["dataset_key"], unit["attr_combination"])
        prep = prep_cache[key]
        if prep is None:
            continue

        run_ts = datetime.now(timezone.utc).isoformat()
        print(f"  -> {unit['model']} / {unit['config']} / fold={unit['fold']} / seed={unit['seed']} ",
              end="", flush=True)

        if unit["kind"] == "standard":
            metrics, sg_df = run_standard_fold(
                prep, unit["model"], unit["config"], unit["fold"], run_ts,
                dry_run=args.dry_run, n_jobs=args.n_jobs, seed=unit["seed"],
            )
            failed = metrics["best_params"] == "FAILED"
        else:
            metrics, sg_df = run_lambda_fold(
                prep, unit["lam"], unit["fold"], run_ts, dry_run=args.dry_run, seed=unit["seed"],
            )
            failed = metrics is None
            if failed:
                metrics = {m: np.nan for m in ["accuracy", "recall", "precision", "roc_auc", "pr_auc"]}
            metrics["best_params"] = "FAILED" if failed else "fixed"
        print()

        metrics_row = {
            "out_prefix": unit["out_prefix"],
            "dataset": unit["dataset_key"],
            "attrs": prep["attrs_str"],
            "kind": unit["kind"],
            "model": unit["model"],
            "opt_metric": unit["config"],
            "lambda_fairness": unit["lam"] if unit["lam"] is not None else np.nan,
            "fold": unit["fold"],
            "seed": unit["seed"],
            "order": unit["order"],
            "accuracy": metrics["accuracy"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "best_params": metrics["best_params"],
            "failed": bool(failed),
            "run_timestamp": run_ts,
            "dry_run": bool(args.dry_run),
        }
        if sg_df is not None:
            sg_df = sg_df.copy()
            sg_df["seed"] = unit["seed"]
            sg_df["out_prefix"] = unit["out_prefix"]
            sg_df["kind"] = unit["kind"]
        write_shard(args.shard_dir, _stem(unit), metrics_row, sg_df)

    print(f"\n[DONE] Shards saved to {args.shard_dir}")
    print("       Merge them with: python scripts/collect_results.py "
          f"--shard-dir \"{args.shard_dir}\" --out-dir <results dir>")


def _stem(unit):
    return shard_stem(unit["out_prefix"], unit["model"], unit["config"], unit["fold"], unit["seed"])


# ---------------------------------------------------------------------------
# Sequential mode driver (unchanged behaviour)
# ---------------------------------------------------------------------------

def run_sequential(configs_to_run, args):
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    seed = args.seeds[0]

    all_agg = []
    all_subgroup = []

    for dataset_key, config in configs_to_run.items():
        for attr_combination in config["attr_combinations"]:
            out_prefix = make_out_prefix(dataset_key, attr_combination)
            out_agg = os.path.join(results_dir, f"{out_prefix}_results.parquet")
            out_sub = os.path.join(results_dir, f"{out_prefix}_subgroup_results.parquet")
            sweep_path = os.path.join(results_dir, f"{out_prefix}_lambda_sweep.parquet")

            # 1) Standard Models (RF, GBM)
            agg_df, subgroup_df = None, None
            if args.only in ("all", "standard"):
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
                        seed=seed,
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
            if args.only in ("all", "lambda"):
                if args.skip_existing and os.path.exists(sweep_path):
                    print(f"\n[RESUME] Loading existing lambda sweep for {out_prefix}")
                    sweep_df = pd.read_parquet(sweep_path)
                    all_agg.append(sweep_df)
                else:
                    sweep_df, sweep_sub = run_lambda_sweep(
                        dataset_key, attr_combination, dry_run=args.dry_run, seed=seed)
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
            os.path.join(results_dir, f"all_results{suffix}.parquet"), index=False
        )
        consolidated_sub.to_parquet(
            os.path.join(results_dir, f"all_subgroup_results{suffix}.parquet"), index=False
        )
        print(f"\n[DONE] Consolidated results saved to {results_dir}")
        print(f"       Rows in agg results     : {len(consolidated_agg)}")
        print(f"       Rows in subgroup results : {len(consolidated_sub)}")
    else:
        print("\n[WARNING] No results were generated.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _norm(name):
    return (name.lower().replace(" ", "").replace("_", "")
            .replace("🇺🇸", "us").replace("🇧🇷", "br").replace("🇵🇹", "pt"))


def select_configs(requested):
    """Selects datasets by exact safe key (e.g. 'compas_us') or by fuzzy name match."""
    if not requested:
        return EXPERIMENT_CONFIG
    selected = {}
    for req in requested:
        exact = [k for k in EXPERIMENT_CONFIG if make_safe_key(k) == req]
        if exact:
            for k in exact:
                selected[k] = EXPERIMENT_CONFIG[k]
            continue
        req_norm = _norm(req)
        for k, v in EXPERIMENT_CONFIG.items():
            k_norm = _norm(k)
            if req_norm in k_norm or k_norm in req_norm or req.lower() in k.lower():
                selected[k] = v
    return selected


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
        help="Sequential mode: skip datasets that already have parquet files in the results dir.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Limit to specific datasets (name fragment, or exact safe key such as 'compas_us').",
    )
    parser.add_argument(
        "--skip-lambda",
        action="store_true",
        help="Skip the neural FairMLP lambda sweep and run only standard models (same as --only standard).",
    )
    parser.add_argument(
        "--only",
        choices=["all", "standard", "lambda"],
        default="all",
        help="Which part to run: nested-CV standard models, the FairMLP lambda sweep, or both.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for hyperparameter search (default: 1, robust on Windows).",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="Sequential mode: where consolidated parquet files are written (default: data/results).",
    )
    # --- Unit selection (sharded mode) ---
    parser.add_argument("--shard-dir", default=None,
                        help="Enable sharded mode: write one resumable shard per unit into this directory.")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Sharded mode: subset of {list(MODELS.keys()) + [LAMBDA_MODEL_NAME]}.")
    parser.add_argument("--opt-metrics", nargs="+", default=None, choices=OPTIMIZATION_METRICS,
                        help="Sharded mode: subset of optimization metrics for the standard models.")
    parser.add_argument("--lambdas", nargs="+", type=float, default=None,
                        help="Sharded mode: fairness penalties for the FairMLP sweep.")
    parser.add_argument("--folds", nargs="+", type=int, default=None, choices=list(range(OUTER_K)),
                        help="Sharded mode: outer folds to run (0-based).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[DEFAULT_SEED],
                        help="Model seed(s). CV partitions stay fixed. More than one seed requires --shard-dir.")
    parser.add_argument("--print-units", action="store_true",
                        help="Print one argument line per selected unit and exit (no training).")
    parser.add_argument("--registry", choices=["dashboard", "thesis"], default="dashboard",
                        help="Dataset registry: 'dashboard' (qualification datasets) or 'thesis' "
                             "(utils/thesis_datasets.py, no Streamlit dependency).")
    parser.add_argument("--sample-rows", type=int, default=None,
                        help="Smoke test: use a random subsample of each dataset. Requires --dry-run.")
    parser.add_argument("--overwrite-qualification", action="store_true",
                        help="Allow the sequential mode to overwrite existing files in data/results.")
    args = parser.parse_args()

    if args.skip_lambda:
        if args.only == "lambda":
            parser.error("--skip-lambda and --only lambda are contradictory.")
        args.only = "standard"

    if args.sample_rows and not args.dry_run:
        parser.error("--sample-rows is for smoke tests only and requires --dry-run.")
    global SAMPLE_ROWS
    SAMPLE_ROWS = args.sample_rows

    load_registry(args.registry)

    configs_to_run = select_configs(args.datasets)
    if args.datasets:
        if not configs_to_run:
            print(f"[ERROR] No matching datasets found for {args.datasets}. "
                  f"Available: {list(EXPERIMENT_CONFIG.keys())}")
            sys.exit(1)
        elif not args.print_units:
            print(f"[INFO] Datasets selected to run: {list(configs_to_run.keys())}")

    if args.print_units:
        for unit in enumerate_units(configs_to_run, args):
            print(format_unit_args(unit))
        return

    if args.shard_dir:
        run_units(enumerate_units(configs_to_run, args), args)
        return

    # Sequential mode writes one consolidated file per dataset, so a partial selection
    # would silently overwrite complete results with incomplete ones.
    partial = [name for name, value in (
        ("--models", args.models), ("--opt-metrics", args.opt_metrics),
        ("--lambdas", args.lambdas), ("--folds", args.folds),
    ) if value is not None]
    if len(args.seeds) > 1:
        partial.append("--seeds (more than one)")
    if partial:
        parser.error(f"{', '.join(partial)} require(s) --shard-dir.")

    # data/results holds the qualification results: never overwrite them by accident.
    writes_default = os.path.abspath(args.results_dir) == os.path.abspath(RESULTS_DIR)
    if writes_default and args.registry == "thesis":
        parser.error("--registry thesis must not write to data/results. "
                     "Use --shard-dir or a different --results-dir.")
    if writes_default and not args.overwrite_qualification and os.path.isdir(RESULTS_DIR) \
            and any(f.endswith("_results.parquet") for f in os.listdir(RESULTS_DIR)):
        parser.error("data/results already holds the qualification results. Use --results-dir <other folder> "
                     "(or --shard-dir), or pass --overwrite-qualification to replace them on purpose.")

    run_sequential(configs_to_run, args)


if __name__ == "__main__":
    main()
