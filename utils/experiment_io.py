"""
experiment_io.py — Agregação e persistência dos resultados dos experimentos
===========================================================================

Funções leves (apenas numpy/pandas) compartilhadas por:
  - scripts/run_experiments.py   (execução, local ou em cluster)
  - scripts/collect_results.py   (junta os shards gerados por jobs independentes)

Um "shard" é o resultado bruto de UMA unidade de trabalho:
    (dataset, par de atributos, modelo, configuração, fold externo, seed)
Cada unidade grava dois arquivos em --shard-dir:
    <stem>.subgroups.parquet   métricas por subgrupo naquele fold
    <stem>.metrics.parquet     1 linha com as métricas globais do fold
O arquivo .metrics é gravado por último e de forma atômica: sua existência
marca a unidade como concluída, o que torna a execução retomável.
"""

import glob
import os

import numpy as np
import pandas as pd

METRIC_COLS = ["accuracy", "recall", "precision", "roc_auc", "pr_auc"]


# ---------------------------------------------------------------------------
# Nomes de arquivo
# ---------------------------------------------------------------------------

def make_safe_key(dataset_key):
    """Converte a chave do dataset (com emojis/espaços) em um nome seguro para arquivos."""
    safe_key = (
        dataset_key
        .replace(" ", "_")
        .replace("🇺🇸", "us")
        .replace("🇧🇷", "br")
        .replace("🇵🇹", "pt")
        .replace("🧪", "sim")
        .lower()
    )
    return "".join(c for c in safe_key if c.isalnum() or c == "_").strip("_")


def make_out_prefix(dataset_key, attr_combination):
    attrs_str = "_".join(attr_combination).lower()
    return f"{make_safe_key(dataset_key)}_{attrs_str}"


# ---------------------------------------------------------------------------
# Agregação entre folds (mesmas fórmulas usadas na qualificação)
# ---------------------------------------------------------------------------

def fold_fairness_summary(subgroup_df):
    """Max AAOD, Sensitivity Gap (subgrupos com n>=30) e grupo de referência por fold."""
    return subgroup_df.groupby("fold").apply(
        lambda g: pd.Series({
            "max_aaod": g["aaod"].max(),
            "sensitivity_gap": (
                g[g["n"] >= 30]["tpr"].max() - g[g["n"] >= 30]["tpr"].min()
                if len(g[g["n"] >= 30]) >= 2 else np.nan
            ),
            "reference_group": g["reference_group"].iloc[0],
        }), include_groups=False
    ).reset_index()


def build_agg_row(dataset_key, attrs_str, model, opt_metric, gm_df, subgroup_df,
                  run_ts, dry_run, lambda_fairness=None):
    """
    Monta a linha agregada (média e desvio-padrão entre folds) de uma configuração.
    gm_df: 1 linha por fold com as colunas de METRIC_COLS.
    subgroup_df: métricas por subgrupo de todos os folds (pode ser vazio).
    """
    row = {
        "dataset": dataset_key,
        "attrs": attrs_str,
        "model": model,
        "opt_metric": opt_metric,
    }
    for m in METRIC_COLS:
        row[f"{m}_mean"] = gm_df[m].mean()
        row[f"{m}_std"] = gm_df[m].std()
    row["run_timestamp"] = run_ts
    row["outer_folds_run"] = len(gm_df)
    row["dry_run"] = dry_run

    if subgroup_df is not None and not subgroup_df.empty:
        fold_agg = fold_fairness_summary(subgroup_df)
        row["max_aaod_mean"] = fold_agg["max_aaod"].mean()
        row["max_aaod_std"] = fold_agg["max_aaod"].std()
        row["sensitivity_gap_mean"] = fold_agg["sensitivity_gap"].mean()
        row["sensitivity_gap_std"] = fold_agg["sensitivity_gap"].std()
        row["reference_group"] = fold_agg["reference_group"].mode()[0]
    else:
        row.update({
            "max_aaod_mean": np.nan, "max_aaod_std": np.nan,
            "sensitivity_gap_mean": np.nan, "sensitivity_gap_std": np.nan,
            "reference_group": None,
        })

    if lambda_fairness is not None:
        row["lambda_fairness"] = lambda_fairness
    return row


# ---------------------------------------------------------------------------
# Shards: escrita e leitura
# ---------------------------------------------------------------------------

def shard_stem(out_prefix, model, config, fold, seed):
    return f"{out_prefix}__{model}__{config}__fold{fold}__seed{seed}"


def shard_paths(shard_dir, stem):
    return (
        os.path.join(shard_dir, f"{stem}.metrics.parquet"),
        os.path.join(shard_dir, f"{stem}.subgroups.parquet"),
    )


def shard_done(shard_dir, stem):
    return os.path.exists(shard_paths(shard_dir, stem)[0])


def write_shard(shard_dir, stem, metrics_row, subgroup_df):
    """Grava os subgrupos primeiro e as métricas por último (marcador de conclusão)."""
    os.makedirs(shard_dir, exist_ok=True)
    metrics_path, sub_path = shard_paths(shard_dir, stem)
    if subgroup_df is not None and not subgroup_df.empty:
        subgroup_df.to_parquet(sub_path, index=False, compression="zstd")
    tmp_path = metrics_path + ".tmp"
    pd.DataFrame([metrics_row]).to_parquet(tmp_path, index=False, compression="zstd")
    os.replace(tmp_path, metrics_path)


def _read_many(paths):
    frames = [pd.read_parquet(p) for p in paths]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_shards(shard_dir, out_dir, expected_folds=None, verbose=True):
    """
    Junta todos os shards de shard_dir e grava, em out_dir, arquivos com o mesmo
    nome e esquema dos resultados da execução sequencial:
        <prefix>_results.parquet, <prefix>_subgroup_results.parquet,
        <prefix>_lambda_sweep.parquet, all_results.parquet, all_subgroup_results.parquet
    Diferença: há uma coluna extra `seed` (uma linha agregada por seed).

    Retorna um dicionário com contagens e a lista de configurações incompletas.
    """
    metric_files = sorted(glob.glob(os.path.join(shard_dir, "*.metrics.parquet")))
    if not metric_files:
        raise FileNotFoundError(f"Nenhum shard (*.metrics.parquet) encontrado em {shard_dir}")

    metrics = _read_many(metric_files)
    subgroups = _read_many(sorted(glob.glob(os.path.join(shard_dir, "*.subgroups.parquet"))))

    os.makedirs(out_dir, exist_ok=True)
    incomplete = []
    all_agg, all_sub = [], []
    group_cols = ["order", "model", "opt_metric", "seed"]

    for out_prefix, m_ds in metrics.groupby("out_prefix", sort=True):
        dataset_key = m_ds["dataset"].iloc[0]
        attrs_str = m_ds["attrs"].iloc[0]
        s_ds = subgroups[subgroups["out_prefix"] == out_prefix] if not subgroups.empty else subgroups

        for kind, out_name in (("standard", "results"), ("lambda", "lambda_sweep")):
            m_kind = m_ds[m_ds["kind"] == kind]
            if m_kind.empty:
                continue
            s_kind = s_ds[s_ds["kind"] == kind] if not s_ds.empty else s_ds

            agg_rows, sub_frames = [], []
            for (order, model, opt_metric, seed), g in m_kind.groupby(group_cols, sort=True):
                g = g.sort_values("fold")
                n_folds = g["fold"].nunique()
                if expected_folds and n_folds < expected_folds:
                    incomplete.append(f"{out_prefix} / {model} / {opt_metric} / seed={seed}: "
                                      f"{n_folds} de {expected_folds} folds")

                # Varredura de lambda: folds que falharam são descartados (como na execução sequencial).
                # Modelos padrão: folds que falharam permanecem como NaN.
                gm_df = g[~g["failed"]] if kind == "lambda" else g
                if gm_df.empty:
                    continue

                if s_kind.empty:
                    sg = pd.DataFrame()
                else:
                    sg = s_kind[
                        (s_kind["model"] == model)
                        & (s_kind["opt_metric"] == opt_metric)
                        & (s_kind["seed"] == seed)
                    ].sort_values("fold", kind="stable")

                lam = g["lambda_fairness"].iloc[0] if kind == "lambda" else None
                row = build_agg_row(
                    dataset_key, attrs_str, model, opt_metric, gm_df, sg,
                    run_ts=g["run_timestamp"].min(),
                    dry_run=bool(g["dry_run"].any()),
                    lambda_fairness=lam,
                )
                row["seed"] = seed
                agg_rows.append(row)
                if not sg.empty:
                    drop = ["out_prefix", "kind"]
                    if kind == "standard" and "lambda_fairness" in sg.columns:
                        drop.append("lambda_fairness")
                    sub_frames.append(sg.drop(columns=drop))

            if not agg_rows:
                continue
            agg_df = pd.DataFrame(agg_rows)
            agg_path = os.path.join(out_dir, f"{out_prefix}_{out_name}.parquet")
            agg_df.to_parquet(agg_path, index=False, compression="zstd")
            all_agg.append(agg_df)
            if verbose:
                print(f"  [SAVED] {agg_path}  ({len(agg_df)} linhas)")

            if sub_frames:
                sub_df = pd.concat(sub_frames, ignore_index=True)
                all_sub.append(sub_df)
                if kind == "standard":
                    sub_path = os.path.join(out_dir, f"{out_prefix}_subgroup_results.parquet")
                    sub_df.to_parquet(sub_path, index=False, compression="zstd")
                    if verbose:
                        print(f"  [SAVED] {sub_path}  ({len(sub_df)} linhas)")

    summary = {
        "shards": len(metric_files),
        "agg_rows": 0,
        "subgroup_rows": 0,
        "incomplete": incomplete,
    }
    if all_agg:
        consolidated_agg = pd.concat(all_agg, ignore_index=True)
        consolidated_agg.to_parquet(
            os.path.join(out_dir, "all_results.parquet"), index=False, compression="zstd")
        summary["agg_rows"] = len(consolidated_agg)
    if all_sub:
        consolidated_sub = pd.concat(all_sub, ignore_index=True)
        consolidated_sub.to_parquet(
            os.path.join(out_dir, "all_subgroup_results.parquet"), index=False, compression="zstd")
        summary["subgroup_rows"] = len(consolidated_sub)
    return summary
