# pyrefly: ignore [missing-import]
"""
Página de leitura acompanhada da Seção 5.4 (Resultados) da Proposta de Qualificação.

Diferente da aba "Auditoria Detalhada" (genérica, com seletores livres), esta página
é fixada na configuração exata usada no texto para cada dataset (mesmo modelo, mesma
métrica de otimização), e reproduz as mesmas tabelas — mas calculadas ao vivo a partir
dos parquets em data/results/ e data/processed/, para nunca dessincronizar do texto.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from data_module import load_local_parquet
from utils.bias_metrics import calculate_dynamic_metrics, pairwise_gerrymandering_audit, calculate_cddl, intersectional_di_comparison, _format_ratio
from utils.report_tables import (
    pretrain_subgroup_table as _pretrain_subgroup_table, subgroup_pre_post_table,
    model_comparison_table, model_comparison_display, latex_model_comparison, latex_subgroup_pre_post,
    MODEL_LABELS,
)

if "lang" not in st.session_state:
    st.session_state.lang = "PT"

st.set_page_config(page_title="Principais Achados", page_icon="📄", layout="wide")
st.sidebar.selectbox("Idioma / Language", ["PT", "EN"], key="lang", disabled=True,
                      help="Esta página acompanha o texto da qualificação, escrito em PT-BR.")

# Paleta categórica validada (skill dataviz): slot 1 = azul, slot 2 = laranja.
COLOR_PRE = "#2a78d6"
COLOR_POST = "#eb6834"
COLOR_AAOD = "#2a78d6"
COLOR_GAP = "#eb6834"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")

st.title("📄 Principais Achados")
st.markdown(
    "Esta página reproduz, **com dados calculados ao vivo**, as tabelas e figuras da "
    "Seção 5.4 (*Resultados Experimentais*) do texto da qualificação — na mesma configuração "
    "(modelo, métrica de otimização) usada em cada trecho. Use-a para conferir rapidamente se "
    "os números do documento continuam batendo com o pipeline, e como referência visual de apoio "
    "à leitura para a banca."
)


@st.cache_data
def load_all_results():
    agg = pd.read_parquet(os.path.join(RESULTS_DIR, "all_results.parquet"))
    sub = pd.read_parquet(os.path.join(RESULTS_DIR, "all_subgroup_results.parquet"))
    return agg, sub


try:
    all_agg, all_sub = load_all_results()
except Exception as e:
    st.error(f"Não foi possível carregar data/results/all_results.parquet: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Configuração fixa por dataset — a mesma usada no texto da Seção 5.4
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    "Adult 🇺🇸": {
        "parquet": "adult_processed.parquet",
        "target_col": "income",
        "favorable_val": 1,
        "attrs": "sex & race",
        "group_cols": ["sex", "race"],
        "model": "GradientBoosting",
        "opt_metric": "average_precision",
        "opt_metric_label": "AUPRC",
        "rich_pretrain": True,
        "dynamic_attrs": ["sex", "race", "age_group", "education_group", "relationship"],
    },
    "COMPAS 🇺🇸": {
        "parquet": "compas_processed.parquet",
        "target_col": "two_year_recid",
        "favorable_val": 0,
        "attrs": "sex & race",
        "group_cols": ["sex", "race"],
        "model": "GradientBoosting",
        "opt_metric": "roc_auc",
        "opt_metric_label": "AUROC",
        "rich_pretrain": True,
        "dynamic_attrs": ["sex", "race", "age_group"],
    },
    "SINASC (DATASUS) 🇧🇷": {
        "parquet": "sinasc_processed.parquet",
        "target_col": "desfecho_nascimento",
        "favorable_val": "Normal",
        "attrs": "raca_cor_mae & idade_mae",
        "group_cols": ["raca_cor_mae", "idade_mae"],
        "model": "GradientBoosting",
        "opt_metric": "average_precision",
        "opt_metric_label": "AUPRC",
        "rich_pretrain": True,
        "dynamic_attrs": ["raca_cor_mae", "idade_mae", "escolaridade_mae", "uf"],
    },
}


@st.cache_data
def load_processed(parquet_name):
    return load_local_parquet(parquet_name)


def pretrain_subgroup_table(df, cfg):
    """Taxa favorável, Pre-DI e Pre-SPD por subgrupo do par auditado (rótulos da base completa)."""
    return _pretrain_subgroup_table(df, cfg["group_cols"], cfg["target_col"], cfg["favorable_val"])


def posttrain_subgroup_table(dataset_key, cfg, pre_df):
    """Post-DI, TPR, FPR e AAOD por subgrupo (média entre folds), com a taxa e o Pre-DI dos dados ao lado."""
    return subgroup_pre_post_table(all_sub, dataset_key, cfg["attrs"], cfg["model"], cfg["opt_metric"], pre_df)


def global_metrics_row(dataset_key, cfg):
    row = all_agg[
        (all_agg["dataset"] == dataset_key)
        & (all_agg["attrs"] == cfg["attrs"])
        & (all_agg["model"] == cfg["model"])
        & (all_agg["opt_metric"] == cfg["opt_metric"])
    ]
    return row.iloc[0] if not row.empty else None


def fairmlp_baseline_subgroup_table(dataset_key, cfg, pre_df):
    """Quebra por subgrupo do baseline de rede neural (MLP, λ=0,0 — sem mitigação), com o Pre-DI ao lado."""
    return subgroup_pre_post_table(all_sub, dataset_key, cfg["attrs"], "FairMLP", "lambda_0.0", pre_df)


POST_COLUMNS = {
    "subgroup": "Subgrupo", "n_mean": "N (teste/fold)", "data_rate": "Taxa nos dados", "pre_di": "Pre-DI",
    "pred_rate": "Taxa de predição", "post_di": "Post-DI", "tpr": "TPR", "fpr": "FPR", "aaod": "AAOD",
}


def show_pre_post_table(table):
    """Tabela por subgrupo com Pre-DI e Post-DI lado a lado; o subgrupo de maior AAOD em negrito."""
    show = table[list(POST_COLUMNS)].rename(columns=POST_COLUMNS)

    def _highlight_worst(row):
        is_worst = row["AAOD"] == show["AAOD"].max() and show["AAOD"].max() > 0
        return ["font-weight: bold" if is_worst else "" for _ in row]

    st.dataframe(
        show.style.format({c: "{:.3f}" for c in list(POST_COLUMNS.values())[2:]} | {"N (teste/fold)": "{:.0f}"})
        .apply(_highlight_worst, axis=1),
        width='stretch', hide_index=True,
    )


def lambda_sweep_table(dataset_key, cfg):
    sub = all_agg[
        (all_agg["dataset"] == dataset_key)
        & (all_agg["attrs"] == cfg["attrs"])
        & (all_agg["model"] == "FairMLP")
    ].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("lambda_fairness").reset_index(drop=True)
    sub["Configuração"] = sub["lambda_fairness"].apply(
        lambda l: "MLP sem mitigação (λ = 0,0)" if l == 0.0 else f"MLP (λ = {l:.1f})".replace(".", ",")
    )
    return sub


def render_pre_post_di_chart(pre_df, post_df, group_cols):
    """Gráfico de barras agrupadas: Pre-DI vs Post-DI por subgrupo (skill dataviz: 1 eixo, cor categórica fixa)."""
    pre_small = pre_df[["subgroup", "pre_di"]].rename(columns={"pre_di": "DI"})
    pre_small["Momento"] = "Pré-treino"
    post_small = post_df[["subgroup", "post_di"]].rename(columns={"post_di": "DI"})
    post_small["Momento"] = "Pós-treino"
    combined = pd.concat([pre_small, post_small], ignore_index=True)

    order = post_df.sort_values("post_di")["subgroup"].tolist()

    chart = alt.Chart(combined).mark_bar().encode(
        y=alt.Y("subgroup:N", title="", sort=order),
        x=alt.X("DI:Q", title="Disparate Impact (DI)", scale=alt.Scale(domain=[0, 1.05])),
        yOffset=alt.YOffset("Momento:N"),
        color=alt.Color(
            "Momento:N",
            scale=alt.Scale(domain=["Pré-treino", "Pós-treino"], range=[COLOR_PRE, COLOR_POST]),
            legend=alt.Legend(title=""),
        ),
        tooltip=[
            alt.Tooltip("subgroup:N", title="Subgrupo"),
            alt.Tooltip("Momento:N", title="Momento"),
            alt.Tooltip("DI:Q", format=".3f", title="Disparate Impact"),
        ],
    )
    rule = alt.Chart(pd.DataFrame({"x": [0.8]})).mark_rule(
        color="gray", strokeDash=[4, 4], opacity=0.7
    ).encode(x="x:Q")
    return (chart + rule).properties(height=max(160, 34 * len(order)))


def render_lambda_chart(sweep_df):
    """Linha Max AAOD / Sensitivity Gap x lambda — um único eixo (skill dataviz: nunca dois eixos y)."""
    long_df = pd.concat([
        sweep_df[["lambda_fairness", "max_aaod_mean"]].rename(columns={"max_aaod_mean": "valor"}).assign(métrica="Max AAOD"),
        sweep_df[["lambda_fairness", "sensitivity_gap_mean"]].rename(columns={"sensitivity_gap_mean": "valor"}).assign(métrica="Sensitivity Gap"),
    ], ignore_index=True)

    chart = alt.Chart(long_df).mark_line(point=alt.OverlayMarkDef(size=70)).encode(
        x=alt.X("lambda_fairness:Q", title="λ (penalização de equidade)"),
        y=alt.Y("valor:Q", title="Valor da métrica (↓ é melhor)", scale=alt.Scale(zero=True)),
        color=alt.Color(
            "métrica:N",
            scale=alt.Scale(domain=["Max AAOD", "Sensitivity Gap"], range=[COLOR_AAOD, COLOR_GAP]),
            legend=alt.Legend(title=""),
        ),
        tooltip=[
            alt.Tooltip("lambda_fairness:Q", title="λ"),
            alt.Tooltip("métrica:N", title="Métrica"),
            alt.Tooltip("valor:Q", format=".4f", title="Valor"),
        ],
    ).properties(height=300)
    return chart


COLOR_MODELS = {"Random Forest": "#2a78d6", "Gradient Boosting": "#eb6834", "MLP": "#1f9e74"}


def _di_rule():
    """Linha tracejada da Regra dos 80% (DI = 0,80)."""
    return alt.Chart(pd.DataFrame({"x": [0.8]})).mark_rule(
        color="gray", strokeDash=[4, 4], opacity=0.7
    ).encode(x="x:Q")


def render_attribute_di_chart(dyn_df):
    """DI favorável de cada atributo sensível isolado (pior vs. melhor categoria)."""
    chart = alt.Chart(dyn_df).mark_bar(color=COLOR_PRE).encode(
        y=alt.Y("Atributo:N", title="", sort="x"),
        x=alt.X("DI:Q", title="Disparate Impact (DI)", scale=alt.Scale(domain=[0, 1.05])),
        tooltip=[
            alt.Tooltip("Atributo:N"), alt.Tooltip("Privilegiado:N"), alt.Tooltip("Desprivilegiado:N"),
            alt.Tooltip("DI:Q", format=".3f"),
        ],
    )
    return (chart + _di_rule()).properties(height=max(120, 34 * len(dyn_df)))


def render_di_cmp_chart(di_cmp):
    """DI (Favorável) de cada recorte, do marginal ao interseccional; cor = nível do recorte."""
    data = di_cmp.dropna(subset=["DI (Favorável)"])
    levels = sorted(data["Nível"].unique(), key=lambda s: (s != "Marginal", s))
    palette = [COLOR_PRE, COLOR_POST, "#1f9e74", "#9b59b6", "#7f7f7f"]
    chart = alt.Chart(data).mark_bar().encode(
        y=alt.Y("Recorte:N", title="", sort="x", axis=alt.Axis(labelLimit=320)),
        x=alt.X("DI (Favorável):Q", title="DI (pior / melhor subgrupo viável)", scale=alt.Scale(domain=[0, 1.05])),
        color=alt.Color("Nível:N", scale=alt.Scale(domain=levels, range=palette[:len(levels)]),
                        legend=alt.Legend(title="", orient="bottom")),
        tooltip=[
            alt.Tooltip("Recorte:N"), alt.Tooltip("Nível:N"),
            alt.Tooltip("Pior Subgrupo:N"), alt.Tooltip("Taxa Fav. Pior:Q", format=".2%"),
            alt.Tooltip("Melhor Subgrupo:N"), alt.Tooltip("Taxa Fav. Melhor:Q", format=".2%"),
            alt.Tooltip("DI (Favorável):Q", format=".3f"),
        ],
    )
    return (chart + _di_rule()).properties(height=max(160, 20 * len(data)))


def render_gerry_chart(gerry):
    """Halteres: Gap Esperado (pior marginal) → Gap Real (interseccional), em p.p."""
    d = gerry.copy()
    d["esperado"] = d["Gap Esperado (Marginal Máx)"] * 100
    d["real"] = d["Gap Real (Interseccional)"] * 100
    d["oculto"] = d["Viés Oculto (Excedente)"] * 100
    order = d.sort_values("real", ascending=False)["Par de Intersecção"].tolist()
    y = alt.Y("Par de Intersecção:N", title="", sort=order, axis=alt.Axis(labelLimit=320))
    tooltip = [
        alt.Tooltip("Par de Intersecção:N", title="Par"),
        alt.Tooltip("esperado:Q", format=".2f", title="Gap Esperado (p.p.)"),
        alt.Tooltip("real:Q", format=".2f", title="Gap Real (p.p.)"),
        alt.Tooltip("oculto:Q", format="+.2f", title="Viés Oculto (p.p.)"),
    ]
    link = alt.Chart(d).mark_rule(color="#b0b0b0", strokeWidth=3).encode(y=y, x="esperado:Q", x2="real:Q")
    long_df = pd.concat([
        d.assign(valor=d["esperado"], Gap="Esperado (pior marginal)"),
        d.assign(valor=d["real"], Gap="Real (interseccional)"),
    ], ignore_index=True)
    points = alt.Chart(long_df).mark_circle(size=110, opacity=1).encode(
        y=y,
        x=alt.X("valor:Q", title="Gap de taxa favorável (p.p.)"),
        color=alt.Color("Gap:N", scale=alt.Scale(domain=["Esperado (pior marginal)", "Real (interseccional)"],
                                                 range=[COLOR_PRE, COLOR_POST]),
                        legend=alt.Legend(title="", orient="bottom")),
        tooltip=tooltip,
    )
    return (link + points).properties(height=max(160, 30 * len(d)))


def render_cddl_chart(cddl_df):
    """CDDL por atributo protegido × proxy; cor = atributo protegido."""
    d = cddl_df.copy()
    d["Combinação"] = d["Atributo Protegido (Vítima)"] + " | " + d["Variável Proxy (Condição)"]
    chart = alt.Chart(d).mark_bar().encode(
        y=alt.Y("Combinação:N", title="", sort="-x", axis=alt.Axis(labelLimit=320)),
        x=alt.X("CDDL:Q", title="CDDL"),
        color=alt.Color("Atributo Protegido (Vítima):N", legend=alt.Legend(title="Atributo protegido", orient="bottom")),
        tooltip=[
            alt.Tooltip("Atributo Protegido (Vítima):N", title="Atributo protegido"),
            alt.Tooltip("Grupo Desprivilegiado:N"),
            alt.Tooltip("Variável Proxy (Condição):N", title="Proxy"),
            alt.Tooltip("CDDL:Q", format=".4f"),
        ],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="gray").encode(x="x:Q")
    return (chart + zero).properties(height=max(160, 22 * len(d)))


def render_favorable_rate_chart(pre_df, overall_rate):
    """Taxa favorável por subgrupo do par auditado, com a taxa global como referência."""
    d = pre_df.copy()
    chart = alt.Chart(d).mark_bar(color=COLOR_PRE).encode(
        y=alt.Y("subgroup:N", title="", sort="-x"),
        x=alt.X("favorable_rate:Q", title="Taxa favorável nos dados", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            alt.Tooltip("subgroup:N", title="Subgrupo"), alt.Tooltip("n:Q", title="N", format=","),
            alt.Tooltip("favorable_rate:Q", format=".2%", title="Taxa favorável"),
            alt.Tooltip("pre_di:Q", format=".3f", title="Pre-DI"),
        ],
    )
    rule = alt.Chart(pd.DataFrame({"x": [overall_rate]})).mark_rule(
        color="gray", strokeDash=[4, 4], opacity=0.8
    ).encode(x="x:Q")
    return (chart + rule).properties(height=max(160, 30 * len(d)))


def render_tradeoff_chart(comp_df):
    """Dispersão desempenho × equidade: cada ponto é um modelo/configuração."""
    d = comp_df.dropna(subset=["accuracy_mean", "max_aaod_mean"]).copy()
    domain = [m for m in COLOR_MODELS if m in set(d["Modelo"])]
    return alt.Chart(d).mark_point(size=120, filled=True, opacity=0.85).encode(
        x=alt.X("max_aaod_mean:Q", title="Max AAOD (↓ mais equitativo)", scale=alt.Scale(zero=False)),
        y=alt.Y("accuracy_mean:Q", title="Acurácia (↑)", scale=alt.Scale(zero=False)),
        color=alt.Color("Modelo:N", scale=alt.Scale(domain=domain, range=[COLOR_MODELS[m] for m in domain]),
                        legend=alt.Legend(title="", orient="bottom")),
        shape=alt.Shape("Modelo:N", scale=alt.Scale(domain=domain), legend=None),
        tooltip=[
            alt.Tooltip("Modelo:N"), alt.Tooltip("Configuração:N"),
            alt.Tooltip("accuracy_mean:Q", format=".4f", title="Acurácia"),
            alt.Tooltip("max_aaod_mean:Q", format=".4f", title="Max AAOD"),
            alt.Tooltip("sensitivity_gap_mean:Q", format=".4f", title="Sens. Gap"),
        ],
    ).properties(height=340)


def render_tpr_fpr_chart(post_df):
    """TPR e FPR por subgrupo (média entre folds): a distância entre subgrupos é o que o AAOD resume."""
    long_df = pd.concat([
        post_df[["subgroup", "tpr", "aaod"]].rename(columns={"tpr": "valor"}).assign(Taxa="TPR"),
        post_df[["subgroup", "fpr", "aaod"]].rename(columns={"fpr": "valor"}).assign(Taxa="FPR"),
    ], ignore_index=True).dropna(subset=["valor"])
    order = post_df.sort_values("aaod", ascending=False)["subgroup"].tolist()
    return alt.Chart(long_df).mark_point(size=110, filled=True).encode(
        y=alt.Y("subgroup:N", title="", sort=order),
        x=alt.X("valor:Q", title="Taxa (média entre folds)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Taxa:N", scale=alt.Scale(domain=["TPR", "FPR"], range=[COLOR_PRE, COLOR_POST]),
                        legend=alt.Legend(title="", orient="bottom")),
        tooltip=[
            alt.Tooltip("subgroup:N", title="Subgrupo"), alt.Tooltip("Taxa:N"),
            alt.Tooltip("valor:Q", format=".3f", title="Valor"), alt.Tooltip("aaod:Q", format=".3f", title="AAOD"),
        ],
    ).properties(height=max(160, 30 * len(order)))


def render_lambda_perf_chart(sweep_df):
    """Desempenho preditivo × λ (eixo sem zero, para tornar visível o custo da penalização)."""
    cols = {"accuracy_mean": "Acurácia", "roc_auc_mean": "AUROC", "pr_auc_mean": "AUPRC"}
    long_df = pd.concat([
        sweep_df[["lambda_fairness", c]].rename(columns={c: "valor"}).assign(métrica=label)
        for c, label in cols.items() if c in sweep_df
    ], ignore_index=True).dropna(subset=["valor"])
    return alt.Chart(long_df).mark_line(point=alt.OverlayMarkDef(size=70)).encode(
        x=alt.X("lambda_fairness:Q", title="λ (penalização de equidade)"),
        y=alt.Y("valor:Q", title="Valor da métrica (↑ é melhor)", scale=alt.Scale(zero=False)),
        color=alt.Color("métrica:N", scale=alt.Scale(domain=list(cols.values()),
                                                     range=[COLOR_PRE, COLOR_POST, "#1f9e74"]),
                        legend=alt.Legend(title="")),
        tooltip=[
            alt.Tooltip("lambda_fairness:Q", title="λ"), alt.Tooltip("métrica:N", title="Métrica"),
            alt.Tooltip("valor:Q", format=".4f", title="Valor"),
        ],
    ).properties(height=300)


def fmt_pct(x):
    return f"{x*100:.2f}%".replace(".", ",")


def fmt3(x):
    return f"{x:.3f}".replace(".", ",")


# ---------------------------------------------------------------------------
# Render por dataset — cada seção (A, B, C, LaTeX) é um bloco expansível.
# O Streamlit não permite expansor dentro de expansor; por isso os blocos
# internos da Seção A são abas.
# ---------------------------------------------------------------------------
st.caption("Clique no título de cada seção para expandi-la ou recolhê-la.")

tabs = st.tabs(list(DATASET_CONFIG.keys()))

for tab, (dataset_key, cfg) in zip(tabs, DATASET_CONFIG.items()):
    with tab:
        df = load_processed(cfg["parquet"])
        pre_df = pretrain_subgroup_table(df, cfg)
        gm = global_metrics_row(dataset_key, cfg)
        post_df = posttrain_subgroup_table(dataset_key, cfg, pre_df)
        mlp_base_df = fairmlp_baseline_subgroup_table(dataset_key, cfg, pre_df)
        comp_df = model_comparison_table(all_agg, dataset_key, cfg["attrs"])
        sweep_df = lambda_sweep_table(dataset_key, cfg)

        # ------------------------------------------------------------------ A
        with st.expander("**A. Diagnóstico do Viés nos Dados Brutos (Pré-Treinamento)**", expanded=True):
            pair_label = " × ".join(cfg["group_cols"])
            if cfg.get("rich_pretrain"):
                tab_dyn, tab_di, tab_gerry, tab_cddl, tab_sub = st.tabs([
                    "Métricas por atributo", "DI marginal × interseccional", "Gerrymandering",
                    "CDDL", f"Subgrupos ({pair_label})",
                ])
            else:
                tab_sub = st.container()

            if cfg.get("rich_pretrain"):
                with tab_dyn:
                    rows = []
                    for attr in cfg["dynamic_attrs"]:
                        m = calculate_dynamic_metrics(df, attr, cfg["target_col"], cfg["favorable_val"])
                        rows.append({"Atributo": attr, "Privilegiado": m["priv"], "Desprivilegiado": m["unpriv"],
                                     "CI": m["CI"], "DI": m["DI"], "DI Adverso": m["DI_adv"], "KL": m["KL"], "KS": m["KS"]})
                    dyn_df = pd.DataFrame(rows)
                    st.markdown("**Métricas dinâmicas de viés por atributo sensível**")
                    st.dataframe(dyn_df, width='stretch', hide_index=True)
                    st.altair_chart(render_attribute_di_chart(dyn_df), width='stretch')
                    st.caption("DI = taxa favorável do grupo desprivilegiado / do privilegiado, para cada atributo "
                               "isolado. A linha tracejada marca o limiar de 0,80 da Regra dos 80%.")

                with tab_di:
                    di_cmp = intersectional_di_comparison(
                        df, cfg["dynamic_attrs"], cfg["target_col"], cfg["favorable_val"], min_n=100
                    )
                    if di_cmp.empty:
                        st.info("Não há subgrupos viáveis (N ≥ 100) com variação de desfecho para a comparação.")
                    else:
                        st.dataframe(
                            di_cmp.style.format({
                                "Taxa Fav. Pior": "{:.2%}", "Taxa Fav. Melhor": "{:.2%}",
                                "DI (Favorável)": "{:.3f}", "DI (Adverso)": _format_ratio,
                            }),
                            width='stretch', hide_index=True,
                        )
                        st.altair_chart(render_di_cmp_chart(di_cmp), width='stretch')
                    st.caption("Pior e melhor subgrupo viável (N ≥ 100) de cada recorte. DI (Favorável) = taxa favorável "
                               "pior / melhor; DI (Adverso) = taxa desfavorável pior / melhor. '∞' indica que o melhor "
                               "subgrupo não possui desfechos adversos. No gráfico, quanto mais à esquerda, maior a "
                               "disparidade; a linha tracejada marca o limiar de 0,80.")

                with tab_gerry:
                    gerry = pairwise_gerrymandering_audit(
                        df, cfg["dynamic_attrs"], cfg["target_col"], cfg["favorable_val"]
                    )
                    gerry = gerry.sort_values("Viés Oculto (Excedente)", ascending=False)
                    st.dataframe(
                        gerry.style.format({
                            "Gap Indiv. A": lambda v: f"{v * 100:.2f} p.p.",
                            "Gap Indiv. B": lambda v: f"{v * 100:.2f} p.p.",
                            "Gap Esperado (Marginal Máx)": lambda v: f"{v * 100:.2f} p.p.",
                            "Gap Real (Interseccional)": lambda v: f"{v * 100:.2f} p.p.",
                            "Viés Oculto (Excedente)": lambda v: f"{v * 100:+.2f} p.p.",
                        }),
                        width='stretch', hide_index=True,
                    )
                    st.altair_chart(render_gerry_chart(gerry), width='stretch')
                    st.caption("Gap = amplitude (maior − menor taxa favorável), em pontos percentuais. Viés Oculto "
                               "positivo indica que a interseção revela disparidade maior do que qualquer atributo "
                               "isolado; no gráfico, é o comprimento da barra cinza entre os dois pontos. O limiar de "
                               "0,05 p.p. é descritivo, não um teste de significância.")

                with tab_cddl:
                    cddl_parts = []
                    for prot_attr in cfg["dynamic_attrs"]:
                        dyn = calculate_dynamic_metrics(df, prot_attr, cfg["target_col"], cfg["favorable_val"])
                        proxies = [a for a in cfg["dynamic_attrs"] if a != prot_attr]
                        cddl_parts.append(calculate_cddl(
                            df, cfg["target_col"], cfg["favorable_val"],
                            prot_attr, dyn["priv"], dyn["unpriv"], proxies,
                        ))
                    cddl_df = pd.concat(cddl_parts, ignore_index=True).sort_values("CDDL", ascending=False)
                    st.dataframe(cddl_df.style.format({"CDDL": "{:.4f}"}), width='stretch', hide_index=True)
                    st.altair_chart(render_cddl_chart(cddl_df), width='stretch')
                    st.caption("CDDL compara o grupo desprivilegiado com o restante da população dentro de cada "
                               "estrato da variável proxy. Barras maiores indicam disparidade que persiste mesmo "
                               "condicionando pela proxy.")

            with tab_sub:
                st.markdown(f"**Disparidades de paridade estatística por subgrupo ({pair_label})**")
                show_pre = pre_df[["subgroup", "n", "favorable_rate", "pre_di", "pre_spd"]].rename(columns={
                    "subgroup": "Subgrupo", "n": "N", "favorable_rate": "Taxa Favorável",
                    "pre_di": "Pre-DI", "pre_spd": "Pre-SPD",
                })
                st.dataframe(
                    show_pre.style.format({"Taxa Favorável": "{:.4f}", "Pre-DI": "{:.3f}", "Pre-SPD": lambda v: f"{v * 100:+.2f} p.p."}),
                    width='stretch', hide_index=True,
                )
                overall_rate = float((df[cfg["target_col"]] == cfg["favorable_val"]).mean())
                st.altair_chart(render_favorable_rate_chart(pre_df, overall_rate), width='stretch')
                st.caption(f"A linha tracejada marca a taxa favorável da base inteira ({fmt_pct(overall_rate)}).")

        # ------------------------------------------------------------------ B
        with st.expander("**B. Efeitos do Treinamento nos Baselines Tradicionais**", expanded=False):
            if not comp_df.empty:
                st.markdown("**Comparação de desempenho e equidade entre todos os modelos testados**")
                st.dataframe(model_comparison_display(comp_df), width='stretch', hide_index=True)
                st.caption(
                    "Média ± desvio-padrão entre os 3 folds externos. Random Forest e Gradient Boosting aparecem em cada "
                    "métrica usada na seleção de hiperparâmetros; a MLP, em cada valor de λ. "
                    "↑: maior é melhor; ↓: menor é melhor."
                )
                st.markdown("**Trade-off entre desempenho e equidade**")
                st.altair_chart(render_tradeoff_chart(comp_df), width='stretch')
                st.caption("Cada ponto é um modelo/configuração da tabela acima (passe o mouse para ver qual). "
                           "O canto superior esquerdo concentra as configurações mais acuradas e mais equitativas.")

            if gm is None or post_df is None:
                st.info("Sem resultados pós-treino salvos para esta combinação ainda.")
            else:
                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Modelo — Config.", MODEL_LABELS.get(cfg["model"], cfg["model"]),
                          f"otim. por {cfg['opt_metric_label']}")
                c2.metric("Acurácia", f"{gm['accuracy_mean']:.3f}")
                c3.metric("AUROC", f"{gm['roc_auc_mean']:.3f}" if not pd.isna(gm['roc_auc_mean']) else "N/A")
                c4.metric("Max Intersectional AAOD", f"{gm['max_aaod_mean']:.4f}")

                st.markdown("**Disparate Impact antes e depois do treinamento e métricas de equidade por subgrupo** "
                            "(média entre folds externos)")
                show_pre_post_table(post_df)
                st.caption("Taxa nos dados e Pre-DI: rótulos da base completa. Demais colunas: predições nos folds de "
                           "teste. Em negrito, o subgrupo com maior AAOD.")

                col_di, col_rates = st.columns(2)
                with col_di:
                    st.markdown("**Amplificação do viés: DI antes vs. depois do treinamento**")
                    st.altair_chart(render_pre_post_di_chart(pre_df, post_df, cfg["group_cols"]), width='stretch')
                    st.caption(
                        "A linha tracejada marca o limiar de $0,80$ da Regra dos 80%. Barras de Pós-treino à esquerda "
                        "da barra de Pré-treino indicam amplificação do viés pelo modelo."
                    )
                with col_rates:
                    st.markdown("**Taxas de verdadeiros e falsos positivos por subgrupo**")
                    st.altair_chart(render_tpr_fpr_chart(post_df), width='stretch')
                    st.caption("Subgrupos ordenados do maior para o menor AAOD. Quanto mais distantes os pontos de "
                               "um subgrupo em relação aos demais, maior a disparidade de erro que o AAOD resume.")

        # ------------------------------------------------------------------ C
        with st.expander("**C. Mitigação por Rede Neural Artificial (MLP: Variação de λ)**", expanded=False):
            if mlp_base_df is not None:
                st.markdown("**Baseline de rede neural sem mitigação (MLP, λ = 0,0) — quebra por subgrupo**")
                show_pre_post_table(mlp_base_df)
                st.caption(
                    "Referência para comparar com o melhor baseline de árvore (Seção B) — antes de qualquer "
                    "penalização de equidade ($\\lambda = 0$)."
                )
                st.altair_chart(render_pre_post_di_chart(pre_df, mlp_base_df, cfg["group_cols"]), width='stretch')
                st.caption("DI antes do treinamento (dados) e depois (predições da MLP sem mitigação). A linha "
                           "tracejada marca o limiar de $0,80$.")

            if sweep_df is None:
                st.info("Sem resultados de lambda sweep salvos para esta combinação ainda.")
            else:
                def _mean_std(mean_col, std_col, fmt="{:.4f}"):
                    return sweep_df.apply(
                        lambda r: f"{fmt.format(r[mean_col])} ± {fmt.format(r[std_col])}"
                        if not pd.isna(r.get(std_col, float('nan'))) else fmt.format(r[mean_col]),
                        axis=1,
                    )

                st.divider()
                show_sweep = pd.DataFrame({
                    "Configuração": sweep_df["Configuração"],
                    "Acurácia": _mean_std("accuracy_mean", "accuracy_std"),
                    "AUROC": _mean_std("roc_auc_mean", "roc_auc_std"),
                    "AUPRC": _mean_std("pr_auc_mean", "pr_auc_std"),
                    "Max AAOD": _mean_std("max_aaod_mean", "max_aaod_std"),
                    "Sensitivity Gap": _mean_std("sensitivity_gap_mean", "sensitivity_gap_std"),
                })
                st.dataframe(show_sweep, width='stretch', hide_index=True)
                st.caption("Valores no formato média ± desvio-padrão entre folds/execuções.")

                col_fair, col_perf = st.columns(2)
                with col_fair:
                    st.markdown("**Equidade × λ**")
                    st.altair_chart(render_lambda_chart(sweep_df), width='stretch')
                with col_perf:
                    st.markdown("**Desempenho preditivo × λ**")
                    st.altair_chart(render_lambda_perf_chart(sweep_df), width='stretch')
                st.caption(
                    "Uma curva de equidade decrescente com λ indicaria mitigação monotônica; o gráfico de desempenho "
                    "mostra o custo correspondente (eixo sem zero, para tornar visíveis variações pequenas). Platôs "
                    "indicam que a penalização não altera as predições: no COMPAS, porque as poucas combinações de "
                    "entrada só mudam de classe quando cruzam o limiar de decisão; no SINASC, porque o modelo prediz "
                    "a classe favorável para todas as instâncias."
                )

        # ------------------------------------------------------------- LaTeX
        with st.expander("📄 **Código LaTeX das tabelas deste dataset (para a proposta)**", expanded=False):
            st.markdown("Formato do texto da tese: vírgula decimal e linhas `\\hline`. Requer os pacotes "
                        "`multirow`, `pdflscape` e `float`.")
            tex_blocks = []
            if not comp_df.empty:
                tex_blocks.append(("Comparação entre todos os modelos",
                                   latex_model_comparison(comp_df, dataset_key, cfg["attrs"])))
            if post_df is not None:
                tex_blocks.append((f"Subgrupos — {MODEL_LABELS.get(cfg['model'], cfg['model'])} "
                                   f"otimizado por {cfg['opt_metric_label']}",
                                   latex_subgroup_pre_post(post_df, dataset_key, cfg["model"], cfg["opt_metric"])))
            if mlp_base_df is not None:
                tex_blocks.append(("Subgrupos — MLP sem mitigação (λ = 0,0)",
                                   latex_subgroup_pre_post(mlp_base_df, dataset_key, "FairMLP", "lambda_0.0", 0.0)))
            for title, tex in tex_blocks:
                st.markdown(f"**{title}**")
                st.code(tex, language="latex")
            if tex_blocks:
                st.download_button(
                    "📥 Baixar todas as tabelas (.tex)",
                    data="\n\n".join(tex for _, tex in tex_blocks),
                    file_name=f"{dataset_key.split()[0].lower()}_tabelas_resultados.tex",
                    mime="text/plain",
                    key=f"dl_tex_{dataset_key}",
                )
