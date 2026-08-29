# pyrefly: ignore [missing-import]
import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils.i18n import t

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

with st.expander(t("provenance_header"), expanded=True):
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
        st.write("**Opções de re-execução:**")
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("🔄 Re-executar (Dry-run)"):
                run_experiments_ui(dry_run=True)
        with c_btn2:
            if st.button("🚀 Re-executar Completo"):
                run_experiments_ui(dry_run=False)
                
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

# ---------------------------------------------------------------------------
# Bloco 1 — Seletor de Contexto
# ---------------------------------------------------------------------------
st.header(t("b1_models_header"))

datasets_available = sorted(agg_df["dataset"].unique())
attrs_available_map = {
    ds: sorted(agg_df[agg_df["dataset"] == ds]["attrs"].unique())
    for ds in datasets_available
}

col_ds, col_attrs = st.columns(2)
with col_ds:
    sel_dataset = st.selectbox(t("select_dataset"), datasets_available)
with col_attrs:
    sel_attrs = st.selectbox(t("select_attrs_combo"), attrs_available_map[sel_dataset])

# Filter
agg_filt = agg_df[(agg_df["dataset"] == sel_dataset) & (agg_df["attrs"] == sel_attrs)].copy()
sub_filt = sub_df[(sub_df["dataset"] == sel_dataset) & (sub_df["attrs"] == sel_attrs)].copy()

models_avail = sorted(agg_filt["model"].unique())
metrics_avail = sorted(agg_filt["opt_metric"].unique())

col_m, col_om = st.columns(2)
with col_m:
    sel_model = st.selectbox(t("select_model"), models_avail)
with col_om:
    sel_opt = st.selectbox(t("select_opt_metric"), metrics_avail)

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

# Allow user to choose the performance axis
perf_metric = st.selectbox(
    t("pareto_xaxis_label"),
    ["accuracy_mean", "recall_mean", "precision_mean", "roc_auc_mean", "pr_auc_mean"],
    format_func=lambda x: x.replace("_mean", "").replace("_", " ").title(),
)

# Optimal region annotation data
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
pareto_errbar = pareto_base.mark_errorband(extent="stdev").encode(
    x=alt.X(f"{perf_metric}:Q"),
    y=alt.Y("max_aaod_mean:Q"),
    yError="max_aaod_std:Q",
)

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
# Bloco 3 — Dumbbell Plot: DI Pré vs. Pós-treinamento por Subgrupo
# ---------------------------------------------------------------------------
st.header(t("dumbbell_header"))
st.markdown(t("dumbbell_desc"))

# Load pre-training DI from the Dados tab logic
# We use the subgroup_results which contains post_di, and we join with
# the pre-training DI computed from the raw data (stored in subgroup results
# as the dataset's data before training).
# The pre-training DI is the favorable_rate of the subgroup / ref_favorable_rate
# computed on the full dataset, not the test fold.
# For a clean scientific comparison we display post_di from the selected config.
dumb_data = sub_filt[
    (sub_filt["model"] == sel_model) & (sub_filt["opt_metric"] == sel_opt)
].copy()

if not dumb_data.empty:
    # Average across folds per subgroup
    dumb_avg = dumb_data.groupby("subgroup").agg(
        post_di_mean=("post_di", "mean"),
        post_di_std=("post_di", "std"),
        tpr_mean=("tpr", "mean"),
        fpr_mean=("fpr", "mean"),
        n_mean=("n", "mean"),
    ).reset_index()

    # Reference line at DI = 1.0 (perfect parity)
    ref_line = alt.Chart(pd.DataFrame({"di": [1.0]})).mark_rule(
        color="gray", strokeDash=[6, 4], opacity=0.7
    ).encode(x="di:Q")

    # Threshold lines at 0.8 and 1.25 (80% rule)
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
        color=alt.condition(
            alt.datum.post_di_mean < 0.8,
            alt.value("#d73027"),
            alt.condition(
                alt.datum.post_di_mean > 1.25,
                alt.value("#fc8d59"),
                alt.value("#4575b4"),
            ),
        ),
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
    key="ranking_model",
)

rank_data = sub_filt[sub_filt["model"] == sel_model_rank].copy()

if not rank_data.empty:
    # Average AAOD per (opt_metric, subgroup) across folds
    rank_avg = rank_data.groupby(["opt_metric", "subgroup"]).agg(
        aaod_mean=("aaod", "mean")
    ).reset_index()

    # Metric display labels
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

    # Export
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
