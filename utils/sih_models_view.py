"""Visão dos resultados do SIH 2024 (etapa 1 da tese) na página de Modelos.

Separada do código da qualificação de propósito: os blocos da página original supõem uma
linha por configuração e os benchmarks da qualificação; aqui há 5 sementes por configuração,
17 capítulos e métricas de justiça recalculadas por scripts/recompute_fairness.py (TPR/FPR
indefinidas, quando o subgrupo não tem uma das classes na dobra, ficam NaN em vez de 0).
"""
import altair as alt
import pandas as pd
import streamlit as st

from utils.thesis_results import load_sih_results, sih_model_subgroup_n

# Paleta categórica de referência (slots 1-3, validada para daltonismo): um tom por modelo.
MODEL_COLORS = {"Random Forest": "#2a78d6", "XGBoost": "#eb6834", "FairMLP": "#1baf7a"}

METRICS = {
    "roc_auc_mean": ("AUROC", "AUROC"),
    "pr_auc_mean": ("AUPRC (alta)", "AUPRC (discharge)"),
    "recall_desfav_mean": ("Recall do óbito", "Death recall"),
    "precision_desfav_mean": ("Precisão do óbito", "Death precision"),
    "max_aaod_mean": ("Max AAOD (↓)", "Max AAOD (↓)"),
    "sensitivity_gap_mean": ("Gap de sensibilidade (↓)", "Sensitivity gap (↓)"),
}


def _L(pt, en):
    return pt if st.session_state.get("lang", "PT") == "PT" else en


def _m(key):
    return METRICS[key][0] if st.session_state.get("lang", "PT") == "PT" else METRICS[key][1]


@st.cache_data(show_spinner=False)
def _load():
    agg, sub = load_sih_results()
    agg["modelo"] = agg["model"].map({"RandomForest": "Random Forest"}).fillna(agg["model"])
    # Uma linha por configuração e capítulo: média e desvio entre as 5 sementes.
    keys = ["capitulo_n", "dataset", "model", "modelo", "opt_metric", "lambda_fairness", "config"]
    num = list(METRICS)
    by_seed = agg.groupby(keys, dropna=False)[num]
    cfg = by_seed.mean().join(by_seed.std().add_suffix("_sd")).reset_index()
    cfg["n_seeds"] = by_seed.size().values
    return agg, sub, cfg


@st.cache_data(show_spinner=False)
def _chapter_names():
    try:
        from data_module.loaders_datasus import sih_groups
        g = sih_groups(2024)
        return {int(r.capitulo_n): str(r.capitulo_cid) for r in g.itertuples()}
    except Exception:
        return {}


def undefined_rate_notice():
    st.info(_L(
        "**Subgrupos sem óbitos (ou sem altas) numa dobra de teste** têm FPR (ou TPR) indefinida "
        "nessa dobra. Nesses casos o AAOD do subgrupo não é calculado naquela dobra, em vez de "
        "assumir taxa 0, o que gerava um AAOD artificial de 0,5. O grupo de referência e o gap de "
        "sensibilidade consideram apenas subgrupos com pelo menos 30 pessoas na dobra.",
        "**Subgroups with no deaths (or no discharges) in a test fold** have an undefined FPR (or "
        "TPR) in that fold. There the subgroup's AAOD is not computed for that fold, instead of "
        "assuming a rate of 0, which produced an artificial AAOD of 0.5. The reference group and the "
        "sensitivity gap consider only subgroups with at least 30 people in the fold.",
    ), icon="ℹ️")


def render_sih_view():
    agg, sub, cfg = _load()
    names = _chapter_names()
    chap_label = lambda c: names.get(c, f"Cap. {c}")

    st.subheader(_L("SIH 2024 — etapa 1: modelos de referência",
                    "SIH 2024 — stage 1: baseline models"))
    st.markdown(_L(
        "Óbito hospitalar por capítulo da CID-10 (17 capítulos). Random Forest e XGBoost com 5 "
        "métricas de otimização; FairMLP com penalização de justiça λ fixa. Validação cruzada "
        "aninhada (3 × 3) repetida com 5 sementes. Subgrupos: raça/cor × sexo.",
        "In-hospital death by ICD-10 chapter (17 chapters). Random Forest and XGBoost with 5 "
        "optimisation metrics; FairMLP with a fixed fairness penalty λ. Nested cross-validation "
        "(3 × 3) repeated with 5 seeds. Subgroups: race/colour × sex.",
    ))
    undefined_rate_notice()

    tab_over, tab_chap = st.tabs([_L("Visão geral (17 capítulos)", "Overview (17 chapters)"),
                                  _L("Por capítulo", "By chapter")])

    # ------------------------------------------------------------------ visão geral
    with tab_over:
        metric = st.selectbox(_L("Métrica", "Metric"), list(METRICS), format_func=_m,
                              index=list(METRICS).index("max_aaod_mean"), key="sih_over_metric")
        hm = cfg.assign(capitulo=cfg["capitulo_n"].map(chap_label))
        cfg_order = (hm[["model", "opt_metric", "lambda_fairness", "config"]].drop_duplicates()
                     .sort_values(["model", "lambda_fairness", "opt_metric"])["config"].tolist())
        chap_order = [chap_label(c) for c in sorted(hm["capitulo_n"].unique())]
        lower_better = metric in ("max_aaod_mean", "sensitivity_gap_mean")
        base = alt.Chart(hm).encode(
            x=alt.X("config:N", sort=cfg_order, title=None, axis=alt.Axis(labelAngle=-40, labelLimit=180)),
            y=alt.Y("capitulo:N", sort=chap_order, title=None, axis=alt.Axis(labelLimit=320)),
        )
        rect = base.mark_rect(stroke="white", strokeWidth=1).encode(
            color=alt.Color(f"{metric}:Q", title=_m(metric),
                            scale=alt.Scale(scheme="blues", reverse=not lower_better)),
            tooltip=[alt.Tooltip("capitulo:N", title=_L("Capítulo", "Chapter")),
                     alt.Tooltip("config:N", title=_L("Configuração", "Configuration")),
                     alt.Tooltip(f"{metric}:Q", format=".3f", title=_m(metric)),
                     alt.Tooltip(f"{metric}_sd:Q", format=".3f", title=_L("DP entre sementes", "SD across seeds"))],
        )
        st.altair_chart(rect.properties(height=len(chap_order) * 26), width="stretch")
        st.caption(_L(
            "Cada célula é a média das 5 sementes. Cor mais escura = melhor "
            f"({'menor' if lower_better else 'maior'} valor).",
            f"Each cell is the mean over 5 seeds. Darker = better ({'lower' if lower_better else 'higher'} value)."))

        st.markdown(_L("**Média dos 17 capítulos por configuração**",
                       "**Mean over the 17 chapters by configuration**"))
        overall = (cfg.groupby("config")[list(METRICS)].mean().reindex(cfg_order).reset_index()
                   .rename(columns={"config": _L("Configuração", "Configuration"),
                                    **{k: _m(k) for k in METRICS}}))
        st.dataframe(overall.style.format({_m(k): "{:.3f}" for k in METRICS}),
                     hide_index=True, width="stretch")
        st.caption(_L(
            "A classe favorável é a alta; o óbito (≈5% das internações) é a classe rara. Com limiar "
            "0,5, os modelos predizem alta para quase todos: o recall do óbito fica baixo e um Max AAOD "
            "baixo pode refletir predição quase uniforme, não um modelo mais justo. Leia as métricas de "
            "justiça junto com o recall do óbito.",
            "The favourable class is discharge; death (≈5% of admissions) is the rare class. With a 0.5 "
            "threshold, models predict discharge for almost everyone: death recall is low and a low Max "
            "AAOD may reflect near-uniform prediction rather than a fairer model. Read fairness metrics "
            "together with death recall."))

    # ------------------------------------------------------------------ por capítulo
    with tab_chap:
        chaps = sorted(cfg["capitulo_n"].unique())
        cap = st.selectbox(_L("Capítulo da CID-10", "ICD-10 chapter"), chaps, format_func=chap_label,
                           key="sih_chapter")
        c = cfg[cfg["capitulo_n"] == cap].copy()

        # Tabela de configurações
        tbl = c.sort_values(["model", "lambda_fairness", "opt_metric"])
        show = pd.DataFrame({_L("Configuração", "Configuration"): tbl["config"]})
        for k in METRICS:
            show[_m(k)] = [f"{m:.3f} ± {s:.3f}" if pd.notna(m) else "—"
                           for m, s in zip(tbl[k], tbl[f"{k}_sd"])]
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption(_L("Média ± desvio padrão entre as 5 sementes.", "Mean ± SD across the 5 seeds."))

        # Trade-off desempenho × justiça
        st.markdown(_L("**Desempenho × justiça**", "**Performance × fairness**"))
        perf = st.selectbox(_L("Eixo de desempenho", "Performance axis"),
                            ["roc_auc_mean", "recall_desfav_mean", "pr_auc_mean"], format_func=_m,
                            key="sih_perf_axis")
        sc = alt.Chart(c).mark_point(filled=True, size=110, opacity=0.9,
                                     stroke="white", strokeWidth=2).encode(
            x=alt.X(f"{perf}:Q", title=_m(perf), scale=alt.Scale(zero=False)),
            y=alt.Y("max_aaod_mean:Q", title=_m("max_aaod_mean"), scale=alt.Scale(zero=True)),
            color=alt.Color("modelo:N", title=_L("Modelo", "Model"),
                            scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values()))),
            tooltip=[alt.Tooltip("config:N", title=_L("Configuração", "Configuration")),
                     alt.Tooltip(f"{perf}:Q", format=".3f", title=_m(perf)),
                     alt.Tooltip("max_aaod_mean:Q", format=".3f", title=_m("max_aaod_mean")),
                     alt.Tooltip("recall_desfav_mean:Q", format=".3f", title=_m("recall_desfav_mean"))],
        )
        st.altair_chart(sc.properties(height=380), width="stretch")

        # Varredura de λ da FairMLP: duas métricas, dois gráficos (sem eixo duplo)
        mlp = c[c["model"] == "FairMLP"].sort_values("lambda_fairness")
        if not mlp.empty:
            st.markdown(_L("**FairMLP: efeito de λ**", "**FairMLP: effect of λ**"))
            cols = st.columns(2)
            for col, k in zip(cols, ["max_aaod_mean", "recall_desfav_mean"]):
                band = alt.Chart(mlp).mark_area(opacity=0.18, color=MODEL_COLORS["FairMLP"]).encode(
                    x=alt.X("lambda_fairness:Q", title="λ"),
                    y=alt.Y("lo:Q", title=_m(k)), y2="hi:Q",
                ).transform_calculate(lo=f"datum['{k}'] - datum['{k}_sd']", hi=f"datum['{k}'] + datum['{k}_sd']")
                line = alt.Chart(mlp).mark_line(point=alt.OverlayMarkDef(size=70, filled=True),
                                                strokeWidth=2, color=MODEL_COLORS["FairMLP"]).encode(
                    x="lambda_fairness:Q", y=alt.Y(f"{k}:Q", title=_m(k)),
                    tooltip=[alt.Tooltip("lambda_fairness:Q", title="λ"),
                             alt.Tooltip(f"{k}:Q", format=".3f", title=_m(k)),
                             alt.Tooltip(f"{k}_sd:Q", format=".3f", title="± DP")],
                )
                col.altair_chart((band + line).properties(height=260, title=_m(k)), width="stretch")
            st.caption(_L("Faixa = ± 1 desvio padrão entre sementes.", "Band = ± 1 SD across seeds."))

        # AAOD por subgrupo
        st.markdown(_L("**AAOD por subgrupo**", "**AAOD by subgroup**"))
        nmod = sih_model_subgroup_n(sub, cap)
        s = sub[sub["capitulo_n"] == cap]
        sg = s.groupby(["config", "subgroup"])["aaod_n"].mean().reset_index()
        sg = sg.merge(nmod, on="subgroup")
        sg["rotulo"] = [f"{g} (N={n:,})".replace(",", ".") for g, n in zip(sg["subgroup"], sg["n_modelo"])]
        cfg_order_c = tbl["config"].tolist()
        sub_order = sg.drop_duplicates("rotulo").sort_values("n_modelo", ascending=False)["rotulo"].tolist()
        base = alt.Chart(sg).encode(
            x=alt.X("rotulo:N", sort=sub_order, title=None, axis=alt.Axis(labelAngle=-35, labelLimit=220)),
            y=alt.Y("config:N", sort=cfg_order_c, title=None),
        )
        rect = base.mark_rect(stroke="white", strokeWidth=1).encode(
            color=alt.condition("isValid(datum.aaod_n)",
                                alt.Color("aaod_n:Q", title="AAOD", scale=alt.Scale(scheme="orangered")),
                                alt.value("#e8e8e4")),
            tooltip=[alt.Tooltip("subgroup:N", title=_L("Subgrupo", "Subgroup")),
                     alt.Tooltip("n_modelo:Q", title=_L("N na amostra", "N in sample"), format=","),
                     alt.Tooltip("config:N", title=_L("Configuração", "Configuration")),
                     alt.Tooltip("aaod_n:Q", format=".3f", title="AAOD")],
        )
        st.altair_chart(rect.properties(height=len(cfg_order_c) * 24), width="stretch")
        st.caption(_L(
            "N = tamanho do subgrupo na amostra de modelagem do capítulo. AAOD médio entre dobras e "
            "sementes em que ele é definido; célula cinza = indefinido em todas (subgrupo sem óbitos "
            "no teste). Subgrupos pequenos têm AAOD mais instável.",
            "N = subgroup size in the chapter's modelling sample. AAOD averaged over the folds and "
            "seeds where it is defined; grey cell = undefined in all of them (no deaths in the test "
            "set). Small subgroups have less stable AAOD."))
