"""Resultados processados dos experimentos da tese que o painel exibe.

Os arquivos em data/results_thesis/ saem de scripts/recompute_fairness.py, aplicado sobre o
que scripts/collect_results.py juntou no PCAD (runs/, fora do git). Os resultados da
qualificação continuam em data/results/ e não são tocados.
"""
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "results_thesis")
SIH_DIR = os.path.join(ROOT, "sih2024")

MODEL_LABELS = {"RandomForest": "Random Forest", "XGBoost": "XGBoost", "FairMLP": "FairMLP"}
OPT_LABELS = {"accuracy": "Accuracy", "recall": "Recall", "precision": "Precision",
              "roc_auc": "AUROC", "average_precision": "AUPRC"}


def sih_available():
    return os.path.exists(os.path.join(SIH_DIR, "all_results.parquet"))


def load_sih_results():
    agg = pd.read_parquet(os.path.join(SIH_DIR, "all_results.parquet"))
    sub = pd.read_parquet(os.path.join(SIH_DIR, "all_subgroup_results.parquet"))
    for df in (agg, sub):
        df["capitulo_n"] = df["dataset"].str.extract(r"cap(\d+)$")[0].astype(int)
        df["config"] = [config_label(m, o, l) for m, o, l in
                        zip(df["model"], df["opt_metric"], df["lambda_fairness"])]
    return agg, sub


def config_label(model, opt_metric, lam):
    if model == "FairMLP":
        return f"FairMLP · λ={lam:g}"
    return f"{MODEL_LABELS.get(model, model)} · {OPT_LABELS.get(opt_metric, opt_metric)}"


def results_key(dataset_name, year=None, group=None):
    """Chave dos resultados processados para a base selecionada na aba Dados, ou None se ela
    ainda não tiver resultados. Novas bases entram aqui conforme forem processadas."""
    if dataset_name.startswith("SIH") and str(year) == "2024" and group is not None and sih_available():
        return f"sih_v2_2024_cap{int(group):02d}"
    return None


def load_dataset_results(key):
    """Linhas por semente (uma por configuração x semente) e subgrupos da base `key`."""
    if key is None:
        return None, None
    if key.startswith("sih_v2_2024_"):
        agg, sub = load_sih_results()
        a, s = agg[agg["dataset"] == key], sub[sub["dataset"] == key]
        return (a, s) if len(a) else (None, None)
    return None, None


def sih_model_subgroup_n(sub, capitulo_n):
    """N de cada subgrupo (raça/cor & sexo) na amostra de modelagem do capítulo.
    Vazio se o capítulo não tiver resultados."""
    d = sub[sub["capitulo_n"] == capitulo_n]
    return (d.groupby("subgroup")[["n_capitulo"]].first()
            .reset_index().rename(columns={"n_capitulo": "n_modelo"}))
