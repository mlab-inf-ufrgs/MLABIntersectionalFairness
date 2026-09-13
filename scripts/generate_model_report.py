#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_model_report.py

Gera análise consolidada dos resultados dos modelos de machine learning:
1. Figura 1 no estilo publicação (Boxplots + Stripplots com cores por Dataset e marcadores por Métrica Otimizada).
2. Figura 2 de Fairness Interseccional (ROC-AUC vs Max AAOD).
3. Tabelas comparativas em LaTeX (booktabs) e Markdown (com médias e desvios padrão).
4. Relatório executivo impresso no terminal.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
ALL_AGG_PATH = os.path.join(RESULTS_DIR, "all_results.parquet")
ALL_SUB_PATH = os.path.join(RESULTS_DIR, "all_subgroup_results.parquet")


def clean_dataframe(df):
    """Padroniza rótulos e nomes para exibição acadêmica."""
    df = df.copy()

    # Limpeza dos nomes dos datasets
    dataset_clean_map = {
        "Adult 🇺🇸": "Adult",
        "COMPAS 🇺🇸": "COMPAS",
        "SINASC (DATASUS) 🇧🇷": "SINASC",
        "Dropout 🇵🇹": "Dropout",
        "Heart 🇺🇸": "Heart",
        "Intersectional bias 🧪": "Intersectional bias",
    }
    df["dataset_clean"] = df["dataset"].map(lambda x: dataset_clean_map.get(x, x.split()[0]))

    # Limpeza dos modelos
    model_clean_map = {
        "RandomForest": "Random Forest",
        "GradientBoosting": "Gradient Boosting",
        "FairMLP": "FairMLP",
    }
    df["model_clean"] = df["model"].map(lambda x: model_clean_map.get(x, x))

    # Limpeza da métrica de otimização
    metric_clean_map = {
        "accuracy": "accuracy",
        "recall": "recall",
        "precision": "precision",
        "roc_auc": "roc_auc",
        "average_precision": "pr_auc",
        "pr_auc": "pr_auc",
        "mcc": "mcc",
        "specificity": "specificity",
    }
    df["opt_metric_clean"] = df["opt_metric"].map(lambda x: metric_clean_map.get(x, x))

    return df


def plot_performance_distribution(df,
                                  metric_left="accuracy_mean",
                                  metric_right="recall_mean",
                                  title_left="Model Accuracy Distribution",
                                  title_right="Model Recall Distribution",
                                  ylabel_left="Accuracy",
                                  ylabel_right="Recall",
                                  save_filename="fig_overall_performance.png"):
    """
    Gera figura com 2 painéis lado a lado no estilo exato do anexo:
    Boxplot cinza de fundo + pontos com jitter (cor = Dataset, marcador = Métrica Otimizada).
    """
    # Filtra apenas modelos com métricas válidas e ordena
    models = [m for m in ["Gradient Boosting", "Random Forest"] if m in df["model_clean"].values]
    if not models:
        models = sorted(df["model_clean"].unique())

    # Paleta de cores para os datasets (consistente com estilo da figura de referência)
    dataset_colors = {
        "Adult": "#1f77b4",          # Azul
        "COMPAS": "#ff7f0e",         # Laranja
        "SINASC": "#2ca02c",         # Verde
        "Dropout": "#00cc44",        # Verde claro
        "Heart": "#d62728",          # Vermelho
        "Intersectional bias": "#9467bd", # Roxo
    }

    # Marcadores para as métricas de otimização
    opt_markers = {
        "accuracy": "o",      # Círculo
        "mcc": "s",           # Quadrado
        "pr_auc": "^",        # Triângulo cima
        "precision": "D",     # Losango
        "recall": "v",        # Triângulo baixo
        "roc_auc": "<",       # Triângulo esquerda
        "specificity": ">",   # Triângulo direita
    }

    # Configurações globais de plot acadêmico
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#444444"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8), dpi=300)

    panels = [
        (ax1, metric_left, title_left, ylabel_left),
        (ax2, metric_right, title_right, ylabel_right),
    ]

    datasets_present = [d for d in dataset_colors.keys() if d in df["dataset_clean"].unique()]
    for d in df["dataset_clean"].unique():
        if d not in datasets_present:
            datasets_present.append(d)

    opt_metrics_present = [m for m in opt_markers.keys() if m in df["opt_metric_clean"].unique()]
    for m in df["opt_metric_clean"].unique():
        if m not in opt_metrics_present:
            opt_metrics_present.append(m)

    np.random.seed(42)

    for ax, metric_col, title, ylabel in panels:
        # 1. Boxplot de fundo
        box_data = [df[df["model_clean"] == m][metric_col].dropna() for m in models]
        bp = ax.boxplot(
            box_data,
            positions=range(len(models)),
            widths=0.65,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="#eeeeee", edgecolor="#aaaaaa", linewidth=1.0),
            medianprops=dict(color="#333333", linewidth=1.5),
            whiskerprops=dict(color="#666666", linewidth=1.0),
            capprops=dict(color="#666666", linewidth=1.0),
        )

        # 2. Stripplot com Jitter customizado
        for m_idx, m_name in enumerate(models):
            sub = df[df["model_clean"] == m_name]
            for _, row in sub.iterrows():
                val = row[metric_col]
                if pd.isna(val):
                    continue
                d_name = row["dataset_clean"]
                opt_m = row["opt_metric_clean"]

                color = dataset_colors.get(d_name, "#333333")
                marker = opt_markers.get(opt_m, "o")

                # Jitter horizontal determinístico
                jitter = np.random.uniform(-0.16, 0.16)
                x_pos = m_idx + jitter

                ax.scatter(
                    x_pos, val,
                    color=color,
                    marker=marker,
                    s=65,
                    alpha=0.9,
                    edgecolors="none",
                    zorder=4,
                )

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=10.5)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel("Model", fontsize=11, labelpad=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5, color="#cccccc")
        ax.set_axisbelow(True)

    # 3. Legendas à direita (Dataset e Optimized Metric)
    # Legenda 1: Datasets
    dataset_handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=d,
                   markerfacecolor=dataset_colors.get(d, "#333333"), markersize=10)
        for d in datasets_present
    ]
    leg1 = fig.legend(
        handles=dataset_handles,
        title="Dataset",
        bbox_to_anchor=(1.01, 0.90),
        loc="upper left",
        frameon=False,
        fontsize=9.5,
        title_fontsize=10.5,
    )
    fig.add_artist(leg1)

    # Legenda 2: Optimized Metric
    opt_handles = [
        plt.Line2D([0], [0], marker=opt_markers.get(m, "o"), color="w", label=m,
                   markerfacecolor="#222222", markeredgecolor="#222222", markersize=8)
        for m in opt_metrics_present
    ]
    fig.legend(
        handles=opt_handles,
        title="Optimized Metric",
        bbox_to_anchor=(1.01, 0.52),
        loc="upper left",
        frameon=False,
        fontsize=9.5,
        title_fontsize=10.5,
    )

    # Caption abaixo da figura (estilo artigo)
    fig.text(
        0.5, -0.05,
        f"Figure 1. Overall model performance in terms of {ylabel_left.lower()} and {ylabel_right.lower()}.\n"
        "Colors represent datasets, and marker shapes indicate the optimization metric.",
        ha="center", fontsize=11, fontweight="bold"
    )

    plt.tight_layout()

    out_png = os.path.join(RESULTS_DIR, save_filename)
    out_pdf = os.path.join(RESULTS_DIR, save_filename.replace(".png", ".pdf"))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"  [FIG] Salvo: {out_png}")
    print(f"  [FIG] Salvo: {out_pdf}")
    return out_png


def generate_latex_table(df, save_filename="tabela_modelos_resumo.tex"):
    """
    Gera tabela acadêmica comparativa em formato LaTeX (booktabs).
    """
    # Agrupa por dataset e modelo calculando média e desvio
    summary = df.groupby(["dataset_clean", "model_clean"]).agg({
        "accuracy_mean": ["mean", "std"],
        "roc_auc_mean": ["mean", "std"],
        "pr_auc_mean": ["mean", "std"],
        "max_aaod_mean": ["mean", "std"],
        "sensitivity_gap_mean": ["mean", "std"],
    }).reset_index()

    lines = [
        "% Tabela Resumo: Desempenho e Equidade Interseccional por Dataset e Modelo",
        "% Pacotes recomendados no preâmbulo LaTeX: \\usepackage{booktabs}, \\usepackage{multirow}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\caption{Desempenho preditivo e disparidade interseccional (médias $\\pm$ desvios-padrão entre as métricas de otimização).}",
        "\\label{tab:modelos_resumo}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Modelo} & \\textbf{Acurácia} & \\textbf{ROC-AUC} & \\textbf{PR-AUC} & \\textbf{Max AAOD} $\\downarrow$ & \\textbf{Sens. Gap} $\\downarrow$ \\\\",
        "\\midrule",
    ]

    cur_ds = None
    for _, row in summary.iterrows():
        ds = row[("dataset_clean", "")]
        model = row[("model_clean", "")]

        acc_m, acc_s = row[("accuracy_mean", "mean")], row[("accuracy_mean", "std")]
        roc_m, roc_s = row[("roc_auc_mean", "mean")], row[("roc_auc_mean", "std")]
        pr_m, pr_s = row[("pr_auc_mean", "mean")], row[("pr_auc_mean", "std")]
        aaod_m, aaod_s = row[("max_aaod_mean", "mean")], row[("max_aaod_mean", "std")]
        gap_m, gap_s = row[("sensitivity_gap_mean", "mean")], row[("sensitivity_gap_mean", "std")]

        acc_str = f"{acc_m:.3f} $\\pm$ {acc_s:.3f}" if not pd.isna(acc_m) else "N/A"
        roc_str = f"{roc_m:.3f} $\\pm$ {roc_s:.3f}" if not pd.isna(roc_m) else "N/A"
        pr_str = f"{pr_m:.3f} $\\pm$ {pr_s:.3f}" if not pd.isna(pr_m) else "N/A"
        aaod_str = f"{aaod_m:.4f} $\\pm$ {aaod_s:.4f}" if not pd.isna(aaod_m) else "N/A"
        gap_str = f"{gap_m:.4f} $\\pm$ {gap_s:.4f}" if not pd.isna(gap_m) else "N/A"

        ds_prefix = ds if ds != cur_ds else ""
        if ds != cur_ds and cur_ds is not None:
            lines.append("\\midrule")
        cur_ds = ds

        lines.append(f"{ds_prefix} & {model} & {acc_str} & {roc_str} & {pr_str} & {aaod_str} & {gap_str} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\multicolumn{7}{l}{\\footnotesize $\\downarrow$ Indica métricas onde valores menores representam maior equidade interseccional.} \\\\",
        "\\end{tabular}",
        "\\end{table}",
    ])

    tex_content = "\n".join(lines)
    out_path = os.path.join(RESULTS_DIR, save_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    print(f"  [TEX] Salvo: {out_path}")
    return tex_content


def generate_markdown_table(df, save_filename="tabela_modelos_resumo.md"):
    """
    Gera tabela resumo em formato Markdown para visualização rápida.
    """
    summary = df.groupby(["dataset_clean", "model_clean"]).agg({
        "accuracy_mean": ["mean", "std"],
        "roc_auc_mean": ["mean", "std"],
        "pr_auc_mean": ["mean", "std"],
        "max_aaod_mean": ["mean", "std"],
        "sensitivity_gap_mean": ["mean", "std"],
    }).reset_index()

    headers = [
        "| Dataset | Modelo | Acurácia | ROC-AUC | PR-AUC | Max AAOD (↓) | Sens. Gap (↓) |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    rows = []
    for _, row in summary.iterrows():
        ds = row[("dataset_clean", "")]
        model = row[("model_clean", "")]
        acc_m, acc_s = row[("accuracy_mean", "mean")], row[("accuracy_mean", "std")]
        roc_m, roc_s = row[("roc_auc_mean", "mean")], row[("roc_auc_mean", "std")]
        pr_m, pr_s = row[("pr_auc_mean", "mean")], row[("pr_auc_mean", "std")]
        aaod_m, aaod_s = row[("max_aaod_mean", "mean")], row[("max_aaod_mean", "std")]
        gap_m, gap_s = row[("sensitivity_gap_mean", "mean")], row[("sensitivity_gap_mean", "std")]

        rows.append(
            f"| **{ds}** | {model} | {acc_m:.3f} ± {acc_s:.3f} | {roc_m:.3f} ± {roc_s:.3f} | "
            f"{pr_m:.3f} ± {pr_s:.3f} | {aaod_m:.4f} ± {aaod_s:.4f} | {gap_m:.4f} ± {gap_s:.4f} |"
        )

    md_content = "\n".join(headers + rows)
    out_path = os.path.join(RESULTS_DIR, save_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"  [MD] Salvo: {out_path}")
    return md_content


def print_executive_summary(df):
    """
    Imprime síntese executiva dos melhores modelos por critério.
    """
    out_buffer = []
    out_buffer.append("\n" + "=" * 70)
    out_buffer.append("SÍNTESE COMPARATIVA CROSS-DATASET DOS MODELOS")
    out_buffer.append("=" * 70)

    for ds in sorted(df["dataset_clean"].unique()):
        sub = df[df["dataset_clean"] == ds]
        best_acc = sub.loc[sub["accuracy_mean"].idxmax()]
        best_auc = sub.loc[sub["roc_auc_mean"].idxmax()] if not sub["roc_auc_mean"].isna().all() else None
        best_fair = sub.loc[sub["max_aaod_mean"].idxmin()] if not sub["max_aaod_mean"].isna().all() else None

        out_buffer.append(f"\n📁 Dataset: {ds} (N={len(sub)} configurações)")
        out_buffer.append(f"  🏆 Maior Acurácia : {best_acc['model_clean']} (opt={best_acc['opt_metric_clean']}) -> {best_acc['accuracy_mean']:.4f}")
        if best_auc is not None and not pd.isna(best_auc['roc_auc_mean']):
            out_buffer.append(f"  🏆 Maior ROC-AUC  : {best_auc['model_clean']} (opt={best_auc['opt_metric_clean']}) -> {best_auc['roc_auc_mean']:.4f}")
        if best_fair is not None and not pd.isna(best_fair['max_aaod_mean']):
            out_buffer.append(f"  ⚖️ Mais Equitativo : {best_fair['model_clean']} (opt={best_fair['opt_metric_clean']}) -> Max AAOD = {best_fair['max_aaod_mean']:.4f}")

    out_buffer.append("\n" + "=" * 70 + "\n")
    summary_text = "\n".join(out_buffer)
    sys.stdout.buffer.write(summary_text.encode("utf-8"))


def main():
    if not os.path.exists(ALL_AGG_PATH):
        print(f"[ERRO] Arquivo {ALL_AGG_PATH} não encontrado. Execute run_experiments.py primeiro.")
        return 1

    print("\n[INFO] Carregando resultados consolidados...")
    raw_df = pd.read_parquet(ALL_AGG_PATH)
    df = clean_dataframe(raw_df)

    print(f"[INFO] {len(df)} modelos carregados dos datasets: {df['dataset_clean'].unique().tolist()}")

    # 1. Gera Figura 1 (Acurácia vs Recall - réplica idêntica do artigo)
    print("\n[1/3] Gerando Figura 1 (Performance: Acurácia e Recall)...")
    plot_performance_distribution(
        df,
        metric_left="accuracy_mean",
        metric_right="recall_mean",
        title_left="Model Accuracy Distribution",
        title_right="Model Recall Distribution",
        ylabel_left="Accuracy",
        ylabel_right="Recall",
        save_filename="fig_overall_performance.png",
    )

    # 2. Gera Figura 2 (Trade-off: ROC-AUC vs Max AAOD)
    print("\n[2/3] Gerando Figura 2 (Trade-off: ROC-AUC e Max AAOD)...")
    plot_performance_distribution(
        df,
        metric_left="roc_auc_mean",
        metric_right="max_aaod_mean",
        title_left="Model ROC-AUC Distribution",
        title_right="Model Intersectional Injustice (Max AAOD)",
        ylabel_left="ROC-AUC",
        ylabel_right="Max AAOD",
        save_filename="fig_fairness_distribution.png",
    )

    # 3. Gera Tabelas LaTeX e Markdown
    print("\n[3/3] Exportando tabelas para LaTeX e Markdown...")
    generate_latex_table(df)
    generate_markdown_table(df)

    # Sumário executivo
    print_executive_summary(df)
    print("[DONE] Relatório consolidado gerado com sucesso em data/results/!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
