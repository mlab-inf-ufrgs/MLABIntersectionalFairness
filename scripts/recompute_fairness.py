"""Recalcula as métricas de justiça a partir dos resultados por subgrupo já salvos,
sem treinar nada de novo.

Por que existe
--------------
Nos capítulos do SIH com poucos óbitos, subgrupos pequenos (quase sempre Indígena) ficam
sem nenhum óbito numa dobra de teste. A FPR fica indefinida, `calculate_model_fairness_metrics`
grava 0, e o AAOD contra a referência vira 0,5 automaticamente: um artefato, não um viés.

O que faz
---------
1. Reconstrói as contagens de cada subgrupo em cada dobra (positivos = desfecho favorável,
   negativos = desfavorável; no SIH, Alta e Óbito) a partir de n, TPR, FPR, precisão e taxa
   favorável, que estão nos parquets.
2. Marca TPR/FPR como indefinidas (NaN) quando a classe correspondente não existe na dobra.
3. Elegibilidade (opcional, --min-n; padrão 0 = todos os subgrupos, que é o usado na tese):
   só entram nas métricas de justiça os subgrupos com pelo menos MIN_N pessoas no capítulo
   (soma das dobras de teste = amostra inteira). Os modelos continuam treinados e avaliados
   com todos; o corte valeria apenas para max_aaod, sensitivity_gap e referência. Decidido em
   22/11/2026 não usar corte: ele tiraria da avaliação justamente os grupos com menos dados.
4. Por dobra: referência = maior taxa favorável entre os elegíveis com n >= 30 na dobra e
   TPR/FPR definidas; AAOD = NaN quando falta TPR ou FPR; max_aaod e gap só entre elegíveis.
5. Acrescenta recall e precisão da classe desfavorável (no SIH, detecção de óbito), somando
   as contagens dos subgrupos.

Os parquets originais não são alterados. Saída em <results-dir>/n{MIN_N}/:
    all_results.parquet           mesmas linhas de configuração, métricas de justiça
                                  recalculadas + recall/precisão da classe desfavorável
    all_subgroup_results.parquet  linhas por subgrupo e dobra, com contagens, N do capítulo,
                                  elegibilidade e AAOD recalculado

Uso:
    python scripts/recompute_fairness.py --results-dir runs/sih2024/results
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REF_MIN_N_FOLD = 30          # mesmo mínimo do grupo de referência nos experimentos da tese
CONFIG_KEYS = ["dataset", "attrs", "model", "opt_metric", "seed", "lambda_fairness"]


def reconstruct_counts(s: pd.DataFrame) -> pd.DataFrame:
    """tp, fp, fn, tn por linha (subgrupo x dobra) a partir das taxas salvas."""
    n = s["n"].astype(float)
    pred_pos = (s["favorable_rate"] * n).round()
    tp = (s["precision"] * pred_pos).round()
    fp = pred_pos - tp
    # Negativos: com FPR > 0, sai direto de fp / FPR. Com FPR = 0 e fp = 0, ou não há
    # negativos ou todos foram acertados; o TPR resolve: positivos = tp / TPR.
    pos_from_tpr = np.where(s["tpr"] > 0, tp / s["tpr"].where(s["tpr"] > 0), np.nan)
    neg = np.where(s["fpr"] > 0, fp / s["fpr"].where(s["fpr"] > 0), n - pos_from_tpr)
    # TPR = 0 e FPR = 0: ninguém predito favorável; sem como separar, assume tudo negativo
    # só se também não houve predição favorável (tp = fp = 0), que é o único caso possível.
    neg = np.where(np.isnan(neg) & (pred_pos == 0), np.nan, neg)
    neg = pd.Series(np.round(neg), index=s.index)
    if "n_neg" in s.columns:          # shards gerados com undefined_as_nan já trazem as contagens
        neg = s["n_neg"].astype(float).where(s["n_neg"].notna(), neg)
    pos = n - neg
    out = s.drop(columns=[c for c in ("n_pos", "n_neg") if c in s.columns])
    out["tp"], out["fp"] = tp, fp
    out["fn"], out["tn"] = pos - tp, neg - fp
    out["n_pos"], out["n_neg"] = pos, neg
    return out


def fold_metrics(g: pd.DataFrame):
    """Recalcula referência, AAOD, max_aaod e gap para uma dobra de uma configuração."""
    g = g.copy()
    g["tpr_def"] = g["tpr"].where(g["n_pos"] > 0)
    g["fpr_def"] = g["fpr"].where(g["n_neg"] > 0)
    elig = g[g["elegivel"]]
    cand = elig[(elig["n"] >= REF_MIN_N_FOLD) & elig["tpr_def"].notna() & elig["fpr_def"].notna()]
    if cand.empty:
        g["aaod_n"] = np.nan
        return g, {"max_aaod": np.nan, "sensitivity_gap": np.nan, "reference_group": None}
    ref = cand.loc[cand["favorable_rate"].idxmax()]
    g["aaod_n"] = 0.5 * ((g["fpr_def"] - ref["fpr_def"]).abs() + (g["tpr_def"] - ref["tpr_def"]).abs())
    g.loc[~g["elegivel"], "aaod_n"] = np.nan
    e = g[g["elegivel"]]
    tprs = e.loc[e["n"] >= REF_MIN_N_FOLD, "tpr_def"].dropna()
    agg = {
        "max_aaod": e["aaod_n"].max(),
        "sensitivity_gap": tprs.max() - tprs.min() if len(tprs) >= 2 else np.nan,
        "reference_group": ref["subgroup"],
    }
    return g, agg


def attribute_metrics(g: pd.DataFrame, attrs):
    """Métricas de justiça por atributo isolado (ex.: só raça/cor), somando as contagens dos
    subgrupos interseccionais da dobra. Mesmas regras: taxas indefinidas = NaN, referência =
    maior taxa favorável entre grupos com n >= 30 e taxas definidas."""
    parts = g["subgroup"].str.split(" & ", expand=True)
    out = {}
    for i, attr in enumerate(attrs):
        m = g.assign(nivel=parts[i]).groupby("nivel")[["n", "tp", "fp", "fn", "tn"]].sum()
        pos, neg = m["tp"] + m["fn"], m["fp"] + m["tn"]
        tpr = (m["tp"] / pos).where(pos > 0)
        fpr = (m["fp"] / neg).where(neg > 0)
        fav = (m["tp"] + m["fp"]) / m["n"]
        cand = m.index[(m["n"] >= REF_MIN_N_FOLD) & tpr.notna() & fpr.notna()]
        if len(cand) == 0:
            out[f"max_aaod_{attr}"] = out[f"sensitivity_gap_{attr}"] = np.nan
            continue
        ref = fav[cand].idxmax()
        aaod = 0.5 * ((fpr - fpr[ref]).abs() + (tpr - tpr[ref]).abs())
        t = tpr[(m["n"] >= REF_MIN_N_FOLD)].dropna()
        out[f"max_aaod_{attr}"] = aaod.max()
        out[f"sensitivity_gap_{attr}"] = t.max() - t.min() if len(t) >= 2 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="runs/sih2024/results")
    ap.add_argument("--min-n", type=int, default=0)
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    res = pd.read_parquet(rdir / "all_results.parquet")
    sub = pd.read_parquet(rdir / "all_subgroup_results.parquet")
    for df in (res, sub):
        df["lambda_fairness"] = df["lambda_fairness"].fillna(-1.0)   # NaN não agrupa

    sub = reconstruct_counts(sub)
    # N do subgrupo no capítulo = soma das dobras de teste de uma configuração (as dobras
    # cobrem a amostra inteira, então é o mesmo valor para qualquer modelo e semente).
    sub["n_capitulo"] = sub.groupby(CONFIG_KEYS + ["subgroup"])["n"].transform("sum")
    sub["elegivel"] = sub["n_capitulo"] >= args.min_n

    sub_rows, fold_rows = [], []
    for key, g in sub.groupby(CONFIG_KEYS + ["fold"], sort=False):
        g2, agg = fold_metrics(g)
        sub_rows.append(g2)
        cnt = g[["tp", "fp", "fn", "tn"]].sum()
        attrs = key[CONFIG_KEYS.index("attrs")].split(" & ")
        fold_rows.append({**dict(zip(CONFIG_KEYS + ["fold"], key)), **agg,
                          **attribute_metrics(g, attrs),
                          "recall_desfav": cnt["tn"] / (cnt["tn"] + cnt["fp"]) if cnt["tn"] + cnt["fp"] else np.nan,
                          "precision_desfav": cnt["tn"] / (cnt["tn"] + cnt["fn"]) if cnt["tn"] + cnt["fn"] else np.nan})
    sub_out = pd.concat(sub_rows, ignore_index=True)
    folds = pd.DataFrame(fold_rows)

    num = ["max_aaod", "sensitivity_gap", "recall_desfav", "precision_desfav"]
    num += [c for c in folds.columns if c.startswith(("max_aaod_", "sensitivity_gap_"))]
    agg = folds.groupby(CONFIG_KEYS)[num].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    ref_mode = (folds.dropna(subset=["reference_group"]).groupby(CONFIG_KEYS)["reference_group"]
                .agg(lambda x: x.value_counts().index[0]).rename("reference_group"))
    agg = agg.join(ref_mode).reset_index()

    keep = [c for c in res.columns if not c.startswith(("max_aaod", "sensitivity_gap")) and c != "reference_group"]
    out = res[keep].merge(agg, on=CONFIG_KEYS, how="left")
    for df in (out, sub_out):
        df["lambda_fairness"] = df["lambda_fairness"].replace(-1.0, np.nan)
    out["min_n_subgrupo"] = args.min_n

    odir = rdir / f"n{args.min_n}"
    odir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(odir / "all_results.parquet", index=False)
    sub_out.drop(columns=["tpr_def", "fpr_def"]).to_parquet(odir / "all_subgroup_results.parquet", index=False)

    excl = (sub_out.loc[~sub_out["elegivel"], ["dataset", "subgroup", "n_capitulo"]]
            .drop_duplicates().sort_values(["dataset", "subgroup"]))
    bad = int(((sub_out["n_neg"] < 0) | (sub_out["n_pos"] < 0) | (sub_out["tp"] + sub_out["fp"] + sub_out["fn"] + sub_out["tn"] != sub_out["n"])).sum())
    print(f"[OK] {len(out)} configurações, {len(sub_out)} linhas de subgrupo -> {odir}")
    print(f"     contagens inconsistentes: {bad}")
    print(f"     subgrupos fora (N < {args.min_n} no capítulo): {len(excl)}")
    print(excl.to_string(index=False))


if __name__ == "__main__":
    main()
