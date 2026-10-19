"""
Tabelas de relatório compartilhadas pelas páginas Modelos e Principais Achados.

Reúne em um só lugar:
- o Pre-DI por subgrupo interseccional (calculado sobre os rótulos da base);
- a comparação de desempenho e equidade entre todos os modelos e configurações;
- a tabela por subgrupo com Pre-DI e Post-DI lado a lado;
- os geradores de código LaTeX dessas tabelas, no formato usado no texto da tese
  (vírgula decimal, \\hline, paisagem para tabelas largas).
"""
import numpy as np
import pandas as pd

OPT_METRIC_ORDER = ["accuracy", "recall", "precision", "roc_auc", "average_precision"]
OPT_METRIC_LABELS = {
    "accuracy": "Acurácia",
    "recall": "Revocação",
    "precision": "Precisão",
    "roc_auc": "AUROC",
    "average_precision": "AUPRC",
}
MODEL_ORDER = ["RandomForest", "GradientBoosting", "FairMLP"]
MODEL_LABELS = {
    "RandomForest": "Random Forest",
    "GradientBoosting": "Gradient Boosting",
    "FairMLP": "FairMLP",
}

# (coluna de média, coluna de desvio, rótulo, maior é melhor)
COMPARISON_METRICS = [
    ("accuracy_mean", "accuracy_std", "Acurácia", True),
    ("recall_mean", "recall_std", "Revocação", True),
    ("precision_mean", "precision_std", "Precisão", True),
    ("roc_auc_mean", "roc_auc_std", "AUROC", True),
    ("pr_auc_mean", "pr_auc_std", "AUPRC", True),
    ("max_aaod_mean", "max_aaod_std", "Max AAOD", False),
    ("sensitivity_gap_mean", "sensitivity_gap_std", "Sens. Gap", False),
]


def dataset_short_name(dataset_key):
    """'SINASC (DATASUS) 🇧🇷' -> 'SINASC'."""
    return dataset_key.split()[0]


def config_label(model, opt_metric, lambda_fairness=np.nan):
    if model == "FairMLP":
        lam = lambda_fairness
        if pd.isna(lam):
            lam = float(str(opt_metric).replace("lambda_", ""))
        return f"λ = {lam:.1f}".replace(".", ",")
    return f"Otim.: {OPT_METRIC_LABELS.get(opt_metric, opt_metric)}"


# ---------------------------------------------------------------------------
# Cálculo das tabelas
# ---------------------------------------------------------------------------
def pretrain_subgroup_table(df, group_cols, target_col, favorable_val):
    """Taxa favorável, Pre-DI e Pre-SPD por subgrupo interseccional (rótulos da base completa).

    O subgrupo de referência é o de maior taxa favorável, como no Post-DI. As chaves dos
    subgrupos seguem o mesmo formato usado em scripts/run_experiments.py ("A & B").
    """
    fav = (df[target_col] == favorable_val)
    g = fav.groupby([df[c] for c in group_cols], observed=True).agg(favorable_rate="mean", n="size").reset_index()
    key = g[group_cols[0]].astype(str)
    for c in group_cols[1:]:
        key = key + " & " + g[c].astype(str)
    g["subgroup"] = key
    ref_rate = g["favorable_rate"].max()
    g["pre_di"] = g["favorable_rate"] / ref_rate
    g["pre_spd"] = g["favorable_rate"] - ref_rate
    return (g[["subgroup", "n", "favorable_rate", "pre_di", "pre_spd"]]
            .sort_values("favorable_rate", ascending=False).reset_index(drop=True))


def model_comparison_table(agg_df, dataset_key, attrs):
    """Uma linha por modelo/configuração (RF e GB por métrica de otimização; FairMLP por λ)."""
    d = agg_df[(agg_df["dataset"] == dataset_key) & (agg_df["attrs"] == attrs)].copy()
    if d.empty:
        return d
    if "lambda_fairness" not in d.columns:
        d["lambda_fairness"] = np.nan
    d["_model_order"] = d["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(len(MODEL_ORDER))
    d["_opt_order"] = d["opt_metric"].map({m: i for i, m in enumerate(OPT_METRIC_ORDER)})
    d["_opt_order"] = d["_opt_order"].fillna(d["lambda_fairness"].fillna(0) + len(OPT_METRIC_ORDER))
    d = d.sort_values(["_model_order", "_opt_order"]).reset_index(drop=True)
    d["Modelo"] = d["model"].map(lambda m: MODEL_LABELS.get(m, m))
    d["Configuração"] = [config_label(m, o, l) for m, o, l in zip(d["model"], d["opt_metric"], d["lambda_fairness"])]
    return d.drop(columns=["_model_order", "_opt_order"])


def model_comparison_display(comp_df, decimals=4):
    """Versão formatada (média ± desvio) da tabela comparativa, para st.dataframe."""
    out = comp_df[["Modelo", "Configuração"]].copy()
    for mean_col, std_col, label, higher in COMPARISON_METRICS:
        arrow = "↑" if higher else "↓"
        out[f"{label} {arrow}"] = [
            "N/A" if pd.isna(m) else (f"{m:.{decimals}f}" if pd.isna(s) else f"{m:.{decimals}f} ± {s:.{decimals}f}")
            for m, s in zip(comp_df[mean_col], comp_df[std_col])
        ]
    return out


def subgroup_pre_post_table(sub_df, dataset_key, attrs, model, opt_metric, pre_df):
    """Métricas pós-treino por subgrupo (média entre folds) com a taxa e o Pre-DI dos dados ao lado."""
    s = sub_df[
        (sub_df["dataset"] == dataset_key) & (sub_df["attrs"] == attrs)
        & (sub_df["model"] == model) & (sub_df["opt_metric"] == opt_metric)
    ]
    if s.empty:
        return None
    post = s.groupby("subgroup").agg(
        n_mean=("n", "mean"),
        pred_rate=("favorable_rate", "mean"),
        post_di=("post_di", "mean"),
        post_di_std=("post_di", "std"),
        tpr=("tpr", "mean"),
        fpr=("fpr", "mean"),
        aaod=("aaod", "mean"),
    ).reset_index()
    pre = pre_df[["subgroup", "favorable_rate", "pre_di"]].rename(columns={"favorable_rate": "data_rate"})
    out = pre.merge(post, on="subgroup", how="inner")
    return out.sort_values("data_rate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def _num(x, decimals):
    return "N/A" if pd.isna(x) else f"{x:.{decimals}f}".replace(".", ",")


def _int(x):
    return f"{x:,.0f}".replace(",", ".")


def _tex_escape(text):
    text = str(text)
    for a, b in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")]:
        text = text.replace(a, b)
    return text.replace(">", "$>$").replace("<", "$<$").replace("λ", "$\\lambda$")


def latex_model_comparison(comp_df, dataset_key, attrs, decimals=4):
    """Tabela comparativa de todos os modelos (paisagem), com o melhor valor de cada coluna em negrito."""
    ds = dataset_short_name(dataset_key)
    best = {}
    for mean_col, _, _, higher in COMPARISON_METRICS:
        vals = comp_df[mean_col].round(decimals)
        if vals.notna().any() and vals.nunique() > 1:
            best[mean_col] = vals.max() if higher else vals.min()

    up, down = r"$\uparrow$", r"$\downarrow$"
    header = " & ".join(
        [r"\textit{Modelo}", r"\textit{Configuração}"]
        + [rf"\textit{{{label}}} {up if higher else down}" for _, _, label, higher in COMPARISON_METRICS]
    )
    lines = [
        r"\begin{landscape}",
        r"\begin{table}[H]",
        r"    \centering",
        r"    \small",
        rf"    \caption{{{ds}: desempenho preditivo e equidade interseccional de todos os modelos e configurações avaliados (média $\pm$ desvio-padrão entre 3 \textit{{folds}} externos). Para Random Forest e Gradient Boosting, a configuração indica a métrica usada na seleção de hiperparâmetros; para a FairMLP, o valor de $\lambda$. Em negrito, o melhor valor de cada coluna.}}",
        r"    \vspace{2mm}",
        r"    \setlength{\tabcolsep}{4pt}",
        r"    \begin{tabular}{ll|ccccc|cc}",
        r"        \hline",
        f"        {header} \\\\",
    ]
    for model in comp_df["Modelo"].unique():
        block = comp_df[comp_df["Modelo"] == model]
        lines.append(r"        \hline")
        for i, (_, row) in enumerate(block.iterrows()):
            first = rf"\multirow{{{len(block)}}}{{*}}{{{model}}}" if i == 0 else ""
            cells = [first, _tex_escape(row["Configuração"])]
            for mean_col, std_col, _, _ in COMPARISON_METRICS:
                m, s = row[mean_col], row[std_col]
                txt = _num(m, decimals)
                if mean_col in best and not pd.isna(m) and round(m, decimals) == best[mean_col]:
                    txt = rf"\textbf{{{txt}}}"
                if not pd.isna(m) and not pd.isna(s):
                    txt = f"{txt} $\\pm$ {_num(s, decimals)}"
                cells.append(txt)
            lines.append("        " + " & ".join(cells) + r" \\")
    pair = _tex_escape(attrs.replace(" & ", " × "))
    lines += [
        r"        \hline",
        rf"        \multicolumn{{9}}{{l}}{{\footnotesize $\uparrow$: maior é melhor; $\downarrow$: menor é melhor. Métricas de equidade calculadas sobre os subgrupos de {pair}.}} \\",
        r"    \end{tabular}",
        rf"    \label{{tab:{ds.lower()}_comparacao_modelos}}",
        r"\end{table}",
        r"\end{landscape}",
    ]
    return "\n".join(lines)


def latex_subgroup_pre_post(table, dataset_key, model, opt_metric, lambda_fairness=np.nan, decimals=3):
    """Tabela por subgrupo com Pre-DI e Post-DI lado a lado; o subgrupo de maior AAOD vai em negrito."""
    ds = dataset_short_name(dataset_key)
    model_label = MODEL_LABELS.get(model, model)
    if model == "FairMLP":
        cfg = config_label(model, opt_metric, lambda_fairness).replace("λ", r"$\lambda$")
        config_txt = f"{model_label}, {cfg}"
        label_suffix = "mlp"
    else:
        config_txt = f"{model_label} otimizado por {OPT_METRIC_LABELS.get(opt_metric, opt_metric)}"
        label_suffix = model.lower()
    worst = table["aaod"].max()
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        rf"    \caption{{{ds}: Disparate Impact antes e depois do treinamento e métricas de equidade por subgrupo ({config_txt}; média entre 3 \textit{{folds}} externos). Taxa nos dados e Pre-DI referem-se aos rótulos da base completa; as demais colunas, às predições nos \textit{{folds}} de teste.}}",
        r"    \vspace{2mm}",
        r"    \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{l|r|cc|cc|ccc}",
        r"        \hline",
        r"        \multicolumn{1}{c|}{\textit{Subgrupo}} & \textit{N (teste/fold)} & \textit{Taxa nos dados} & \textit{Pre-DI} & \textit{Taxa de predição} & \textit{Post-DI} & \textit{TPR} & \textit{FPR} & \textit{AAOD} \\",
        r"        \hline",
    ]
    for _, r in table.iterrows():
        cells = [
            _tex_escape(r["subgroup"]), _int(r["n_mean"]),
            _num(r["data_rate"], decimals), _num(r["pre_di"], decimals),
            _num(r["pred_rate"], decimals), _num(r["post_di"], decimals),
            _num(r["tpr"], decimals), _num(r["fpr"], decimals), _num(r["aaod"], decimals),
        ]
        if worst > 0 and r["aaod"] == worst:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        lines.append("        " + " & ".join(cells) + r" \\")
    lines += [
        r"        \hline",
        r"    \end{tabular}",
        r"    }",
        rf"    \label{{tab:{ds.lower()}_subgrupos_pre_pos_{label_suffix}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)
