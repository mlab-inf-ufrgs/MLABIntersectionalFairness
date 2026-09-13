# pyrefly: ignore [missing-import]
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from utils.i18n import t
from scripts.generate_model_report import clean_dataframe, generate_latex_table

if "lang" not in st.session_state:
    st.session_state.lang = "PT"

st.set_page_config(
    page_title=t("models_page_title"),
    page_icon="🤖",
    layout="wide",
)

st.sidebar.selectbox("Idioma / Language", ["PT", "EN"], key="lang")

st.title(t("models_title"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
ALL_AGG_PATH = os.path.join(RESULTS_DIR, "all_results.parquet")
ALL_SUB_PATH = os.path.join(RESULTS_DIR, "all_subgroup_results.parquet")

# ---------------------------------------------------------------------------
# Bloco 0 — Proveniência e Transparência Metodológica
# ---------------------------------------------------------------------------
def run_experiments_ui(dry_run=True):
    from scripts.run_experiments import main as run_exp_main
    import sys
    
    msg = "Executando testes rápidos (dry-run)..." if dry_run else "Executando pipeline completo... Isso pode demorar vários minutos. Acompanhe o terminal."
    with st.spinner(msg):
        old_argv = sys.argv
        sys.argv = ['run_experiments.py']
        if dry_run:
            sys.argv.append('--dry-run')
        try:
            run_exp_main()
        except Exception as e:
            st.error(f"Erro ao executar: {e}")
        finally:
            sys.argv = old_argv
    st.rerun()

with st.expander(t("provenance_header"), expanded=False):
    st.markdown(t("provenance_desc"))

    if os.path.exists(ALL_AGG_PATH):
        _meta = pd.read_parquet(ALL_AGG_PATH, columns=["run_timestamp", "dry_run"]).iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric(t("provenance_run_ts"), str(_meta["run_timestamp"])[:19].replace("T", " ") + " UTC")
        col2.metric(t("provenance_outer_k"), "3")
        col3.metric(t("provenance_inner_k"), "3 × RandomizedSearchCV (n_iter=30)")
        if _meta.get("dry_run", False):
            st.warning(t("provenance_dryrun_warn"))
            
        st.markdown("---")
        st.write("**Opções de re-execução e download:**")
        c_btn1, c_btn2, c_btn3, c_btn4 = st.columns(4)
        with c_btn1:
            if st.button("🔄 Re-executar (Dry-run)"):
                run_experiments_ui(dry_run=True)
        with c_btn2:
            if st.button("🚀 Re-executar Completo"):
                run_experiments_ui(dry_run=False)
        with c_btn3:
            with open(ALL_AGG_PATH, "rb") as f:
                st.download_button(
                    label="📥 Dados Brutos Globais",
                    data=f,
                    file_name="all_results.parquet",
                    mime="application/octet-stream"
                )
        with c_btn4:
            with open(ALL_SUB_PATH, "rb") as f:
                st.download_button(
                    label="📥 Dados Brutos (Subgrupos)",
                    data=f,
                    file_name="all_subgroup_results.parquet",
                    mime="application/octet-stream"
                )
                
    else:
        st.warning(t("no_results_warn"))
        
        st.write("Você pode executar os experimentos diretamente por aqui:")
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("Executar Experimentos Agora (Dry-run)", type="primary"):
                run_experiments_ui(dry_run=True)
        with c_btn2:
            if st.button("Executar Experimentos Completos (Demorado)"):
                run_experiments_ui(dry_run=False)
                
        st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data
def load_results():
    agg = pd.read_parquet(ALL_AGG_PATH)
    sub = pd.read_parquet(ALL_SUB_PATH)
    return agg, sub

try:
    agg_df, sub_df = load_results()
except Exception as e:
    st.error(f"{t('load_error')} {e}")
    st.stop()

# Helper para renderizar a Figura estilo Artigo (Boxplot + Jitter)
def render_figure_distribution(df, metric_left, metric_right, label_left, label_right):
    models = [m for m in ["Gradient Boosting", "Random Forest"] if m in df["model_clean"].values]
    if not models:
        models = sorted(df["model_clean"].unique())

    dataset_colors = {
        "Adult": "#1f77b4",          # Azul
        "COMPAS": "#ff7f0e",         # Laranja
        "SINASC": "#2ca02c",         # Verde
        "Dropout": "#00cc44",        # Verde claro
        "Heart": "#d62728",          # Vermelho
        "Intersectional bias": "#9467bd", # Roxo
    }

    opt_markers = {
        "accuracy": "o",
        "mcc": "s",
        "pr_auc": "^",
        "precision": "D",
        "recall": "v",
        "roc_auc": "<",
        "specificity": ">",
    }

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#444444"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), dpi=200)

    panels = [
        (ax1, metric_left, f"Model {label_left} Distribution", label_left),
        (ax2, metric_right, f"Model {label_right} Distribution", label_right),
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
        box_data = [df[df["model_clean"] == m][metric_col].dropna() for m in models]
        ax.boxplot(
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

                jitter = np.random.uniform(-0.16, 0.16)
                ax.scatter(m_idx + jitter, val, color=color, marker=marker, s=75, alpha=0.9, edgecolors="none", zorder=4)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=10.5)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel("Model", fontsize=11, labelpad=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5, color="#cccccc")
        ax.set_axisbelow(True)

    dataset_handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=d,
                   markerfacecolor=dataset_colors.get(d, "#333333"), markersize=10)
        for d in datasets_present
    ]
    leg1 = fig.legend(handles=dataset_handles, title="Dataset", bbox_to_anchor=(1.01, 0.90), loc="upper left", frameon=False, fontsize=9.5, title_fontsize=10.5)
    fig.add_artist(leg1)

    opt_handles = [
        plt.Line2D([0], [0], marker=opt_markers.get(m, "o"), color="w", label=m,
                   markerfacecolor="#222222", markeredgecolor="#222222", markersize=8)
        for m in opt_metrics_present
    ]
    fig.legend(handles=opt_handles, title="Optimized Metric", bbox_to_anchor=(1.01, 0.52), loc="upper left", frameon=False, fontsize=9.5, title_fontsize=10.5)

    fig.text(
        0.5, -0.06,
        f"Figure 1. Overall model performance in terms of {label_left.lower()} and {label_right.lower()}.\n"
        "Colors represent datasets, and marker shapes indicate the optimization metric.",
        ha="center", fontsize=11, fontweight="bold"
    )

    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# ABAS PRINCIPAIS
# ---------------------------------------------------------------------------
tab_consolidated, tab_detailed = st.tabs([t("tab_consolidated_title"), t("tab_detailed_title")])

# ===========================================================================
# ABA 1: VISÃO CONSOLIDADA (CROSS-DATASET)
# ===========================================================================
with tab_consolidated:
    st.subheader(t("consolidated_header"))
    st.markdown(t("consolidated_desc"))

    clean_df = clean_dataframe(agg_df)

    # 1. Destaques Best-in-Class por Dataset
    st.markdown(f"#### {t('best_in_class_header')}")
    ds_list = sorted(clean_df["dataset_clean"].unique())
    cols_ds = st.columns(len(ds_list))

    for idx, ds_name in enumerate(ds_list):
        with cols_ds[idx]:
            sub_d = clean_df[clean_df["dataset_clean"] == ds_name]
            best_acc = sub_d.loc[sub_d["accuracy_mean"].idxmax()]
            best_fair = sub_d.loc[sub_d["max_aaod_mean"].idxmin()] if not sub_d["max_aaod_mean"].isna().all() else None

            st.markdown(f"**📁 {ds_name}**")
            st.metric("Acurácia Máx", f"{best_acc['accuracy_mean']:.3f}", f"{best_acc['model_clean']} ({best_acc['opt_metric_clean']})")
            if best_fair is not None and not pd.isna(best_fair['max_aaod_mean']):
                st.metric("Menor Max AAOD (↓)", f"{best_fair['max_aaod_mean']:.4f}", f"{best_fair['model_clean']} ({best_fair['opt_metric_clean']})")

    st.markdown("---")

    # 2. Figura Estilo Artigo (Boxplot + Jitter Interativo)
    st.markdown(f"#### {t('fig1_interactive_title')}")
    
    metric_options = {
        "accuracy_mean": "Accuracy",
        "recall_mean": "Recall",
        "precision_mean": "Precision",
        "roc_auc_mean": "ROC-AUC",
        "pr_auc_mean": "PR-AUC",
        "max_aaod_mean": "Max AAOD (↓ Viés)",
        "sensitivity_gap_mean": "Sensitivity Gap (↓ Viés)",
    }

    c_sel1, c_sel2 = st.columns(2)
    with c_sel1:
        sel_left_key = st.selectbox(
            t("fig1_select_left"),
            list(metric_options.keys()),
            index=0, # accuracy_mean
            format_func=lambda k: metric_options[k],
        )
    with c_sel2:
        sel_right_key = st.selectbox(
            t("fig1_select_right"),
            list(metric_options.keys()),
            index=1, # recall_mean
            format_func=lambda k: metric_options[k],
        )

    fig_interactive = render_figure_distribution(
        clean_df,
        metric_left=sel_left_key,
        metric_right=sel_right_key,
        label_left=metric_options[sel_left_key],
        label_right=metric_options[sel_right_key],
    )
    st.pyplot(fig_interactive, use_container_width=True)

    # Botão de Download da Figura salva em alta resolução
    fig_png_path = os.path.join(RESULTS_DIR, "fig_overall_performance.png")
    if os.path.exists(fig_png_path):
        with open(fig_png_path, "rb") as f_img:
            st.download_button(
                label=t("btn_download_highres"),
                data=f_img,
                file_name="fig_overall_performance.png",
                mime="image/png",
            )

    st.markdown("---")

    # 3. Tabela Comparativa Consolidada
    st.markdown(f"#### {t('summary_table_header')}")
    summary_table = clean_df.groupby(["dataset_clean", "model_clean"]).agg(
        Acurácia=("accuracy_mean", lambda x: f"{x.mean():.3f} ± {x.std():.3f}"),
        ROC_AUC=("roc_auc_mean", lambda x: f"{x.mean():.3f} ± {x.std():.3f}" if not x.isna().all() else "N/A"),
        PR_AUC=("pr_auc_mean", lambda x: f"{x.mean():.3f} ± {x.std():.3f}" if not x.isna().all() else "N/A"),
        Max_AAOD=("max_aaod_mean", lambda x: f"{x.mean():.4f} ± {x.std():.4f}" if not x.isna().all() else "N/A"),
        Sens_Gap=("sensitivity_gap_mean", lambda x: f"{x.mean():.4f} ± {x.std():.4f}" if not x.isna().all() else "N/A"),
    ).reset_index()

    summary_table.rename(columns={
        "dataset_clean": "Dataset",
        "model_clean": "Modelo",
        "ROC_AUC": "ROC-AUC",
        "PR_AUC": "PR-AUC",
        "Max_AAOD": "Max AAOD (↓)",
        "Sens_Gap": "Sens. Gap (↓)",
    }, inplace=True)

    st.dataframe(summary_table, use_container_width=True, hide_index=True)

    # 4. Código LaTeX para Overleaf/Artigo
    with st.expander(t("latex_code_header"), expanded=False):
        st.markdown(t("copy_latex_desc"))
        tex_path = os.path.join(RESULTS_DIR, "tabela_modelos_resumo.tex")
        if os.path.exists(tex_path):
            with open(tex_path, "r", encoding="utf-8") as f_tex:
                tex_code = f_tex.read()
        else:
            tex_code = generate_latex_table(clean_df)

        st.code(tex_code, language="latex")
        st.download_button(
            label="📥 Baixar tabela LaTeX (.tex)",
            data=tex_code,
            file_name="tabela_modelos_resumo.tex",
            mime="text/plain",
        )


# ===========================================================================
# ABA 2: AUDITORIA DETALHADA POR DATASET
# ===========================================================================
with tab_detailed:
    st.header(t("b1_models_header"))

    datasets_available = sorted(agg_df["dataset"].unique())
    attrs_available_map = {
        ds: sorted(agg_df[agg_df["dataset"] == ds]["attrs"].unique())
        for ds in datasets_available
    }

    col_ds, col_attrs = st.columns(2)
    with col_ds:
        sel_dataset = st.selectbox(t("select_dataset"), datasets_available, key="detailed_ds")
    with col_attrs:
        sel_attrs = st.selectbox(t("select_attrs_combo"), attrs_available_map[sel_dataset], key="detailed_attrs")

    # Filter
    agg_filt = agg_df[(agg_df["dataset"] == sel_dataset) & (agg_df["attrs"] == sel_attrs)].copy()
    sub_filt = sub_df[(sub_df["dataset"] == sel_dataset) & (sub_df["attrs"] == sel_attrs)].copy()

    models_avail = sorted(agg_filt["model"].unique())
    metrics_avail = sorted(agg_filt["opt_metric"].unique())

    col_m, col_om = st.columns(2)
    with col_m:
        sel_model = st.selectbox(t("select_model"), models_avail, key="detailed_model")
    with col_om:
        sel_opt = st.selectbox(t("select_opt_metric"), metrics_avail, key="detailed_opt")

    # Single combination
    single_row = agg_filt[
        (agg_filt["model"] == sel_model) & (agg_filt["opt_metric"] == sel_opt)
    ]

    if single_row.empty:
        st.warning(t("no_results_warn"))
        st.stop()

    r = single_row.iloc[0]

    st.subheader(t("global_metrics_header"))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{r['accuracy_mean']:.3f}", f"±{r['accuracy_std']:.3f}")
    c2.metric("Recall", f"{r['recall_mean']:.3f}", f"±{r['recall_std']:.3f}")
    c3.metric("Precision", f"{r['precision_mean']:.3f}", f"±{r['precision_std']:.3f}")
    c4.metric("ROC-AUC", f"{r['roc_auc_mean']:.3f}" if not np.isnan(r['roc_auc_mean']) else "N/A",
              f"±{r['roc_auc_std']:.3f}" if not np.isnan(r['roc_auc_std']) else None)
    c5.metric("PR-AUC", f"{r['pr_auc_mean']:.3f}" if not np.isnan(r['pr_auc_mean']) else "N/A",
              f"±{r['pr_auc_std']:.3f}" if not np.isnan(r['pr_auc_std']) else None)

    f1, f2 = st.columns(2)
    f1.metric(t("max_aaod_metric"), f"{r['max_aaod_mean']:.4f}", f"±{r['max_aaod_std']:.4f}")
    f2.metric(t("sens_gap_metric"), f"{r['sensitivity_gap_mean']:.4f}" if not np.isnan(r['sensitivity_gap_mean']) else "N/A",
              f"±{r['sensitivity_gap_std']:.4f}" if not np.isnan(r['sensitivity_gap_std']) else None)

    with st.expander(t("fairness_def_expander")):
        st.markdown(t("fairness_def_text"))

    st.divider()

    # ---------------------------------------------------------------------------
    # Bloco 2 — Trade-off Pareto (Performance × Injustiça Interseccional)
    # ---------------------------------------------------------------------------
    st.header(t("pareto_header"))
    st.markdown(t("pareto_desc"))

    pareto_data = agg_filt.copy()

    perf_metric = st.selectbox(
        t("pareto_xaxis_label"),
        ["accuracy_mean", "recall_mean", "precision_mean", "roc_auc_mean", "pr_auc_mean"],
        format_func=lambda x: x.replace("_mean", "").replace("_", " ").title(),
        key="detailed_perf_metric",
    )

    opt_region = pd.DataFrame({
        "x": [pareto_data[perf_metric].max()],
        "y": [pareto_data["max_aaod_mean"].min()],
        "label": [t("pareto_ideal_label")],
    })

    pareto_base = alt.Chart(pareto_data).encode(
        x=alt.X(f"{perf_metric}:Q",
                title=perf_metric.replace("_mean", "").replace("_", " ").title(),
                scale=alt.Scale(zero=False)),
        y=alt.Y("max_aaod_mean:Q", title=t("max_aaod_label"), scale=alt.Scale(zero=True)),
        color=alt.Color("model:N", legend=alt.Legend(title=t("model_legend"))),
        shape=alt.Shape("opt_metric:N", legend=alt.Legend(title=t("opt_metric_legend"))),
        tooltip=[
            alt.Tooltip("model:N", title=t("model_legend")),
            alt.Tooltip("opt_metric:N", title=t("opt_metric_legend")),
            alt.Tooltip(f"{perf_metric}:Q", format=".4f"),
            alt.Tooltip("max_aaod_mean:Q", format=".4f", title=t("max_aaod_label")),
            alt.Tooltip("max_aaod_std:Q", format=".4f", title="±AAOD"),
            alt.Tooltip("sensitivity_gap_mean:Q", format=".4f", title=t("sens_gap_metric")),
        ],
    )

    pareto_points = pareto_base.mark_point(size=150, filled=True, opacity=0.85)
    ideal_label = alt.Chart(opt_region).mark_text(
        align="right", baseline="top", dy=10, dx=-5, color="green", fontSize=11
    ).encode(
        x=alt.X("x:Q"),
        y=alt.Y("y:Q"),
        text="label:N",
    )

    pareto_chart = (pareto_points + ideal_label).properties(height=420)
    st.altair_chart(pareto_chart, use_container_width=True)
    st.caption(t("pareto_caption"))

    st.divider()

    # ---------------------------------------------------------------------------
    # Bloco 3 — Dumbbell Plot: DI Pós-treinamento por Subgrupo
    # ---------------------------------------------------------------------------
    st.header(t("dumbbell_header"))
    st.markdown(t("dumbbell_desc"))

    dumb_data = sub_filt[
        (sub_filt["model"] == sel_model) & (sub_filt["opt_metric"] == sel_opt)
    ].copy()

    if not dumb_data.empty:
        dumb_avg = dumb_data.groupby("subgroup").agg(
            post_di_mean=("post_di", "mean"),
            post_di_std=("post_di", "std"),
            tpr_mean=("tpr", "mean"),
            fpr_mean=("fpr", "mean"),
            n_mean=("n", "mean"),
        ).reset_index()

        dumb_avg["di_color"] = np.where(
            dumb_avg["post_di_mean"] < 0.8, "#d73027",
            np.where(dumb_avg["post_di_mean"] > 1.25, "#fc8d59", "#4575b4")
        )

        ref_line = alt.Chart(pd.DataFrame({"di": [1.0]})).mark_rule(
            color="gray", strokeDash=[6, 4], opacity=0.7
        ).encode(x="di:Q")

        thresh_80 = alt.Chart(pd.DataFrame({"di": [0.8]})).mark_rule(
            color="orange", strokeDash=[4, 4], opacity=0.6
        ).encode(x="di:Q")

        thresh_125 = alt.Chart(pd.DataFrame({"di": [1.25]})).mark_rule(
            color="orange", strokeDash=[4, 4], opacity=0.6
        ).encode(x="di:Q")

        dumb_base = alt.Chart(dumb_avg).encode(
            y=alt.Y("subgroup:N", title="", sort="-x"),
        )

        dumb_points = dumb_base.mark_point(size=120, filled=True).encode(
            x=alt.X("post_di_mean:Q", title=t("post_di_label"), scale=alt.Scale(zero=False)),
            color=alt.Color("di_color:N", scale=None),
            tooltip=[
                alt.Tooltip("subgroup:N", title=t("tbl_subgroup")),
                alt.Tooltip("post_di_mean:Q", format=".3f", title=t("post_di_label")),
                alt.Tooltip("post_di_std:Q", format=".3f", title="±DI"),
                alt.Tooltip("tpr_mean:Q", format=".3f", title="TPR"),
                alt.Tooltip("fpr_mean:Q", format=".3f", title="FPR"),
                alt.Tooltip("n_mean:Q", format=".0f", title="N (média)"),
            ],
        )

        dumb_errbar = dumb_base.mark_errorbar().encode(
            x=alt.X("post_di_mean:Q"),
            xError=alt.XError("post_di_std:Q"),
            y=alt.Y("subgroup:N"),
            color=alt.value("gray"),
        )

        dumbbell_chart = (ref_line + thresh_80 + thresh_125 + dumb_errbar + dumb_points).properties(
            height=max(250, len(dumb_avg) * 40)
        )

        st.altair_chart(dumbbell_chart, use_container_width=True)
        st.caption(t("dumbbell_caption"))

        with st.expander(t("dumbbell_table_expander")):
            st.dataframe(
                dumb_avg.style.format({
                    "post_di_mean": "{:.3f}",
                    "post_di_std": "{:.3f}",
                    "tpr_mean": "{:.3f}",
                    "fpr_mean": "{:.3f}",
                    "n_mean": "{:.0f}",
                }),
                use_container_width=True,
            )
    else:
        st.info(t("no_subgroup_data"))

    st.divider()

    # ---------------------------------------------------------------------------
    # Bloco 4 — Ranking de Métricas de Otimização (Heatmap AAOD por Subgrupo)
    # ---------------------------------------------------------------------------
    st.header(t("ranking_header"))
    st.markdown(t("ranking_desc"))

    sel_model_rank = st.selectbox(
        t("ranking_model_select"),
        models_avail,
        key="detailed_ranking_model",
    )

    rank_data = sub_filt[sub_filt["model"] == sel_model_rank].copy()

    if not rank_data.empty:
        rank_avg = rank_data.groupby(["opt_metric", "subgroup"]).agg(
            aaod_mean=("aaod", "mean")
        ).reset_index()

        metric_label_map = {
            "accuracy": "Accuracy",
            "recall": "Recall",
            "precision": "Precision",
            "roc_auc": "ROC-AUC",
            "average_precision": "PR-AUC",
        }
        rank_avg["opt_metric_label"] = rank_avg["opt_metric"].map(metric_label_map).fillna(rank_avg["opt_metric"])

        hm_base = alt.Chart(rank_avg).encode(
            x=alt.X("subgroup:N", title="", axis=alt.Axis(labelAngle=-30, labelLimit=200)),
            y=alt.Y("opt_metric_label:N", title=t("opt_metric_legend"),
                    sort=["Accuracy", "Recall", "Precision", "ROC-AUC", "PR-AUC"]),
        )

        hm_rect = hm_base.mark_rect().encode(
            color=alt.Color(
                "aaod_mean:Q",
                scale=alt.Scale(scheme="orangered", domain=[0, rank_avg["aaod_mean"].max()]),
                title=t("max_aaod_label"),
            ),
            tooltip=[
                alt.Tooltip("opt_metric_label:N", title=t("opt_metric_legend")),
                alt.Tooltip("subgroup:N", title=t("tbl_subgroup")),
                alt.Tooltip("aaod_mean:Q", format=".4f", title=t("max_aaod_label")),
            ],
        )

        hm_text = hm_base.mark_text(fontSize=11).encode(
            text=alt.Text("aaod_mean:Q", format=".3f"),
            color=alt.condition(
                alt.datum.aaod_mean > rank_avg["aaod_mean"].quantile(0.7),
                alt.value("white"),
                alt.value("black"),
            ),
        )

        ranking_chart = (hm_rect + hm_text).properties(
            height=max(200, len(rank_avg["opt_metric"].unique()) * 55)
        )

        st.altair_chart(ranking_chart, use_container_width=True)
        st.caption(t("ranking_caption"))

        rank_export = rank_avg[["opt_metric", "subgroup", "aaod_mean"]].copy()
        rank_export.insert(0, "dataset", sel_dataset)
        rank_export.insert(1, "attrs", sel_attrs)
        rank_export.insert(2, "model", sel_model_rank)

        st.download_button(
            label=t("export_ranking_csv"),
            data=rank_export.to_csv(index=False).encode("utf-8"),
            file_name=f"aaod_ranking_{sel_dataset}_{sel_model_rank}.csv",
            mime="text/csv",
        )
    else:
        st.info(t("no_subgroup_data"))
