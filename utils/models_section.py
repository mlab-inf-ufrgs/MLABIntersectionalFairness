"""Seção "5. Modelos" da aba Dados: aparece só para as bases que já têm resultados processados
(utils/thesis_results.results_key). Desenho combinado em 22/11/2026:

  - Boxplots, um por métrica de desempenho, para o modelo escolhido. Cada ponto é uma semente;
    a forma indica a métrica de otimização (árvores) e o tamanho do círculo, o λ (FairMLP).
  - Seletores de modelo, métrica de desempenho, métrica de equidade e atributos sensíveis.
    Com 1 atributo: dispersão 2D (equidade × desempenho). Com 2: dispersão 3D (equidade em cada
    atributo × desempenho). Uma tabela traz todos os valores, inclusive o interseccional.

As mesmas formas valem no 2D (Altair) e no 3D (Plotly), que só aceita círculo, quadrado,
losango, cruz e x, nas versões cheia e vazada; por isso não há triângulos.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from utils.thesis_results import OPT_LABELS, load_dataset_results

MODEL_COLORS = {"Random Forest": "#2a78d6", "XGBoost": "#eb6834", "FairMLP": "#1baf7a"}

# Formas por métrica de otimização: (Altair, Plotly 3D). Caminhos SVG no quadrado [-1, 1].
_X_PATH = "M-1,-0.6L-0.6,-1L0,-0.4L0.6,-1L1,-0.6L0.4,0L1,0.6L0.6,1L0,0.4L-0.6,1L-1,0.6L-0.4,0Z"
_SQ_OPEN = "M-1,-1H1V1H-1Z M-0.55,-0.55V0.55H0.55V-0.55Z"
OPT_SHAPES = {
    "Accuracy": ("diamond", "diamond"),
    "Recall": ("square", "square"),
    "Precision": ("cross", "cross"),
    "AUROC": (_X_PATH, "x"),
    "AUPRC": (_SQ_OPEN, "square-open"),
}
LAMBDA_SIZES_2D = {0.0: 25, 0.5: 70, 1.0: 140, 2.0: 250, 4.0: 420}
LAMBDA_SIZES_3D = {0.0: 4, 0.5: 6, 1.0: 8, 2.0: 11, 4.0: 14}

PERF = {
    "roc_auc_mean": ("AUROC", "AUROC"),
    "pr_auc_mean": ("AUPRC (alta)", "AUPRC (discharge)"),
    "accuracy_mean": ("Acurácia", "Accuracy"),
    "recall_mean": ("Recall (alta)", "Recall (discharge)"),
    "precision_mean": ("Precisão (alta)", "Precision (discharge)"),
    "recall_desfav_mean": ("Recall do óbito", "Death recall"),
    "precision_desfav_mean": ("Precisão do óbito", "Death precision"),
}
FAIR = {
    "max_aaod": ("Max AAOD (↓)", "Max AAOD (↓)"),
    "sensitivity_gap": ("Gap de sensibilidade (↓)", "Sensitivity gap (↓)"),
}
ATTR_LABELS = {"raca_cor": ("Raça/cor", "Race/colour"), "sexo": ("Sexo", "Sex"),
               "faixa_etaria": ("Faixa etária", "Age group")}


def _pt():
    return st.session_state.get("lang", "PT") == "PT"


def _L(pt, en):
    return pt if _pt() else en


def _lbl(d, k):
    return d[k][0] if _pt() else d[k][1]


def _prepare(agg):
    a = agg.copy()
    a["modelo"] = a["model"].map({"RandomForest": "Random Forest"}).fillna(a["model"])
    a["otimizacao"] = np.where(a["model"] == "FairMLP", "λ",
                               a["opt_metric"].map(OPT_LABELS).fillna(a["opt_metric"]))
    a["lam"] = a["lambda_fairness"]
    return a


def _shape_scale():
    return alt.Scale(domain=list(OPT_SHAPES), range=[v[0] for v in OPT_SHAPES.values()])


def _color_scale(models):
    dom = [m for m in MODEL_COLORS if m in models]
    return alt.Scale(domain=dom, range=[MODEL_COLORS[m] for m in dom])


def render_models_section(key, attrs_available, uf_filtered=False):
    agg, sub = load_dataset_results(key)
    if agg is None:
        return
    a = _prepare(agg)
    attr_cols = [c for c in attrs_available
                 if f"max_aaod_{c}_mean" in a.columns]

    st.header(_L("5. Modelos", "5. Models"))
    st.markdown(_L(
        "Etapa 1 da tese: Random Forest e XGBoost (5 métricas de otimização) e FairMLP com "
        "penalização λ fixa, em validação cruzada aninhada (3 × 3) repetida com 5 sementes, "
        "sobre a amostra de modelagem do capítulo (até 200 mil internações). Equidade medida "
        "nos subgrupos raça/cor × sexo.",
        "Thesis stage 1: Random Forest and XGBoost (5 optimisation metrics) and FairMLP with a "
        "fixed penalty λ, in nested cross-validation (3 × 3) repeated with 5 seeds, on the "
        "chapter's modelling sample (up to 200k admissions). Fairness measured on race/colour × "
        "sex subgroups."))
    if uf_filtered:
        st.caption(_L("Os modelos foram treinados com o Brasil inteiro; o filtro de UF acima não se aplica aqui.",
                      "Models were trained on the whole country; the state filter above does not apply here."))
    st.caption(_L(
        "Subgrupos sem óbitos numa dobra de teste têm FPR indefinida nessa dobra e ficam fora do "
        "AAOD dela (em vez de entrar com taxa 0).",
        "Subgroups with no deaths in a test fold have an undefined FPR there and are left out of "
        "that fold's AAOD (instead of counting as rate 0)."))

    models = [m for m in MODEL_COLORS if m in a["modelo"].unique()]
    all_lbl = _L("Todos", "All")

    # ---------------------------------------------------------------- boxplots
    with st.expander(_L("Desempenho por métrica", "Performance by metric"), expanded=True):
        sel_box = st.selectbox(_L("Selecione o modelo", "Select the model"), [all_lbl] + models,
                               key=f"ms_box_model_{key}")
        b = a if sel_box == all_lbl else a[a["modelo"] == sel_box]
        mods_b = [m for m in models if m in b["modelo"].unique()]
        seeds = sorted(b["seed"].unique())
        long = b.melt(id_vars=["modelo", "otimizacao", "lam", "seed", "config"],
                      value_vars=list(PERF), var_name="metrica", value_name="valor").dropna(subset=["valor"])
        long["metrica"] = long["metrica"].map(lambda k: _lbl(PERF, k))
        long["semente"] = long["seed"].astype(str)
        # Posições horizontais calculadas aqui (eixo x numérico): métrica i ocupa [i-0.4, i+0.4],
        # dividida numa faixa por modelo; os pontos de cada semente se espalham dentro da faixa
        # do seu modelo. Assim caixas e pontos ficam sempre alinhados.
        order = [_lbl(PERF, k) for k in PERF if _lbl(PERF, k) in set(long["metrica"])]
        w = 0.8 / len(mods_b)
        m_idx = {m: i for i, m in enumerate(order)}
        mod_idx = {m: j for j, m in enumerate(mods_b)}
        seed_idx = {s: k for k, s in enumerate(seeds)}
        step = w * 0.75 / max(1, len(seeds))
        long["centro"] = [m_idx[me] - 0.4 + w * (mod_idx[mo] + 0.5)
                          for me, mo in zip(long["metrica"], long["modelo"])]
        long["xpos"] = long["centro"] + [(seed_idx[s] - (len(seeds) - 1) / 2) * step for s in long["seed"]]

        st_ = (long.groupby(["metrica", "modelo", "centro"])["valor"]
               .quantile([0, 0.25, 0.5, 0.75, 1]).unstack().reset_index())
        st_.columns = ["metrica", "modelo", "centro", "vmin", "q1", "med", "q3", "vmax"]
        st_["x0"], st_["x1"] = st_["centro"] - w * 0.42, st_["centro"] + w * 0.42

        lo = np.floor(long["valor"].min() * 20) / 20
        labels = "[" + ",".join(f"'{o}'" for o in order) + "]"
        x_scale = alt.Scale(domain=[-0.5, len(order) - 0.5], nice=False, zero=False)
        x_axis = alt.Axis(values=list(range(len(order))), labelExpr=f"{labels}[datum.value]",
                          title=None, grid=False, ticks=False, labelAngle=0, labelFontWeight="bold",
                          labelLimit=150, labelOverlap=False, domain=False)
        y_scale = alt.Scale(domain=[lo, 1.0])
        tip_box = [alt.Tooltip("metrica:N", title=_L("Métrica", "Metric")),
                   alt.Tooltip("modelo:N", title=_L("Modelo", "Model")),
                   alt.Tooltip("med:Q", format=".3f", title=_L("Mediana", "Median")),
                   alt.Tooltip("q1:Q", format=".3f", title="Q1"),
                   alt.Tooltip("q3:Q", format=".3f", title="Q3"),
                   alt.Tooltip("vmin:Q", format=".3f", title=_L("Mínimo", "Min")),
                   alt.Tooltip("vmax:Q", format=".3f", title=_L("Máximo", "Max"))]
        boxes = alt.Chart(st_)
        whisker = boxes.mark_rule(color="#a8a7a0").encode(
            x=alt.X("centro:Q", scale=x_scale, axis=x_axis),
            y=alt.Y("vmin:Q", scale=y_scale, title=_L("Valor", "Value")), y2="vmax:Q")
        body = boxes.mark_rect(color="#c9c8c2", opacity=0.45, cornerRadius=2).encode(
            x=alt.X("x0:Q", scale=x_scale, axis=x_axis), x2="x1:Q",
            y=alt.Y("q1:Q", scale=y_scale), y2="q3:Q", tooltip=tip_box)
        median = boxes.mark_rule(color="#52514e", strokeWidth=2).encode(
            x=alt.X("x0:Q", scale=x_scale, axis=x_axis), x2="x1:Q",
            y=alt.Y("med:Q", scale=y_scale), tooltip=tip_box)
        # Separadores discretos entre métricas
        seps = alt.Chart(pd.DataFrame({"s": [i + 0.5 for i in range(len(order) - 1)]})).mark_rule(
            color="#e4e3df").encode(x=alt.X("s:Q", scale=x_scale, axis=x_axis))

        tip = [alt.Tooltip("metrica:N", title=_L("Métrica", "Metric")),
               alt.Tooltip("config:N", title=_L("Configuração", "Configuration")),
               alt.Tooltip("semente:N", title=_L("Semente", "Seed")),
               alt.Tooltip("valor:Q", format=".3f", title=_L("Valor", "Value"))]
        color = alt.Color("modelo:N", scale=_color_scale(models), title=_L("Modelo", "Model"))
        pts = alt.Chart(long)
        xp = alt.X("xpos:Q", scale=x_scale, axis=x_axis)
        pts_tree = pts.mark_point(filled=True, size=40, opacity=0.85).encode(
            x=xp, y=alt.Y("valor:Q", scale=y_scale), color=color,
            shape=alt.Shape("otimizacao:N", scale=_shape_scale(),
                            title=_L("Métrica de otimização", "Optimisation metric")),
            tooltip=tip,
        ).transform_filter("datum.otimizacao != 'λ'")
        pts_mlp = pts.mark_circle(opacity=0.6, stroke="white", strokeWidth=0.5).encode(
            x=xp, y=alt.Y("valor:Q", scale=y_scale), color=color,
            size=alt.Size("lam:Q", title="λ (FairMLP)", scale=alt.Scale(domain=[0, 4], range=[12, 160]),
                          legend=alt.Legend(values=list(LAMBDA_SIZES_2D), format="~g",
                                            symbolFillColor="#1baf7a", symbolStrokeWidth=0)),
            tooltip=tip,
        ).transform_filter("datum.otimizacao == 'λ'")
        chart = alt.layer(seps, whisker, body, median, pts_tree, pts_mlp).properties(height=430)
        st.altair_chart(chart, width="stretch")
        st.caption(_L(
            "Todas as métricas na mesma escala. Em cada métrica, uma caixa por modelo (mínimo, quartis "
            "e mediana); cada ponto é uma semente (média das 3 dobras externas).",
            "All metrics on the same scale. Within each metric, one box per model (min, quartiles and "
            "median); each point is a seed (mean of the 3 outer folds)."))

    # ---------------------------------------------------------------- dispersão
    with st.expander(_L("Desempenho × equidade", "Performance × fairness"), expanded=True):

        # ------------------------------------------------------------ dispersão
        c1, c2, c3, c4 = st.columns(4)
        sel_m = c1.selectbox(_L("Modelo", "Model"), [all_lbl] + models, key=f"ms_sc_model_{key}")
        perf = c2.selectbox(_L("Métrica de desempenho", "Performance metric"), list(PERF),
                            format_func=lambda k: _lbl(PERF, k), key=f"ms_sc_perf_{key}")
        fair = c3.selectbox(_L("Métrica de equidade", "Fairness metric"), list(FAIR),
                            format_func=lambda k: _lbl(FAIR, k), key=f"ms_sc_fair_{key}")
        sel_attrs = c4.multiselect(_L("Atributos sensíveis", "Sensitive attributes"), attr_cols,
                                   default=attr_cols, format_func=lambda c: _lbl(ATTR_LABELS, c)
                                   if c in ATTR_LABELS else c, key=f"ms_sc_attrs_{key}")

        d = a if sel_m == all_lbl else a[a["modelo"] == sel_m]
        cols_needed = [perf, f"{fair}_mean"] + [f"{fair}_{c}_mean" for c in attr_cols]
        cfg = (d.groupby(["modelo", "otimizacao", "lam", "config"], dropna=False)[cols_needed]
               .agg(["mean", "std"]))
        cfg.columns = [f"{c}|{s}" for c, s in cfg.columns]
        cfg = cfg.reset_index()
        for c in cols_needed:
            cfg[c] = cfg[f"{c}|mean"]

        if not sel_attrs:
            st.warning(_L("Selecione ao menos um atributo sensível.", "Select at least one sensitive attribute."))
        elif len(sel_attrs) == 1:
            fx = f"{fair}_{sel_attrs[0]}_mean"
            xt = f"{_lbl(FAIR, fair)} — {_lbl(ATTR_LABELS, sel_attrs[0])}"
            enc = dict(
                x=alt.X(f"{fx}:Q", title=xt, scale=alt.Scale(zero=True)),
                y=alt.Y(f"{perf}:Q", title=_lbl(PERF, perf), scale=alt.Scale(zero=False)),
                color=alt.Color("modelo:N", scale=_color_scale(models), title=_L("Modelo", "Model")),
                tooltip=[alt.Tooltip("config:N", title=_L("Configuração", "Configuration")),
                         alt.Tooltip(f"{fx}:Q", format=".3f", title=xt),
                         alt.Tooltip(f"{perf}:Q", format=".3f", title=_lbl(PERF, perf))],
            )
            base = alt.Chart(cfg[["modelo", "otimizacao", "lam", "config"] + cols_needed])
            tree = base.mark_point(filled=True, size=120, opacity=0.9).encode(
                shape=alt.Shape("otimizacao:N", scale=_shape_scale(),
                                title=_L("Métrica de otimização", "Optimisation metric")), **enc
            ).transform_filter("datum.otimizacao != 'λ'")
            mlp = base.mark_circle(opacity=0.6, stroke="white", strokeWidth=1.5).encode(
                size=alt.Size("lam:Q", title="λ (FairMLP)",
                              scale=alt.Scale(domain=[0, 4], range=[25, 420]),
                              legend=alt.Legend(values=list(LAMBDA_SIZES_2D), format="~g", symbolFillColor="#1baf7a", symbolStrokeWidth=0)),
                **enc
            ).transform_filter("datum.otimizacao == 'λ'")
            pdf = cfg[["config", fx, perf]].dropna()
            pdf = pdf.assign(pareto=_pareto(pdf, [fx], perf))
            mx, my = pdf[fx].median(), pdf[perf].median()
            top = pdf[perf].max() + 0.05 * (pdf[perf].max() - pdf[perf].min() or 0.01)
            region = alt.Chart(pd.DataFrame({"x0": [0.0], "x1": [mx], "y0": [my], "y1": [top]})).mark_rect(
                color=REGION_COLOR, opacity=0.12).encode(x="x0:Q", x2="x1:Q", y="y0:Q", y2="y1:Q")
            region_lbl = alt.Chart(pd.DataFrame({"x": [0.0], "y": [top],
                                                 "t": [_L("↖ região desejada", "↖ desired region")]})).mark_text(
                align="left", baseline="top", dx=4, dy=4, color=REGION_COLOR, fontWeight="bold").encode(
                x="x:Q", y="y:Q", text="t:N")
            front = pdf[pdf["pareto"]].sort_values(fx)
            front_line = alt.Chart(front).mark_line(interpolate="step-after", strokeDash=[5, 4],
                                                    color="#52514e", strokeWidth=1.5).encode(
                x=f"{fx}:Q", y=f"{perf}:Q")
            front_ring = alt.Chart(front).mark_point(size=380, color="#52514e", strokeWidth=1.5,
                                                     filled=False).encode(
                x=f"{fx}:Q", y=f"{perf}:Q",
                tooltip=[alt.Tooltip("config:N", title=_L("Fronteira de Pareto", "Pareto front"))])
            st.altair_chart((region + region_lbl + front_line + tree + mlp + front_ring).properties(height=420),
                            width="stretch")
        else:
            _scatter_3d(cfg, perf, fair, sel_attrs[:2], models)
            if len(sel_attrs) > 2:
                st.caption(_L("O gráfico usa os dois primeiros atributos; a tabela abaixo traz todos.",
                              "The chart uses the first two attributes; the table below has all of them."))
        st.caption(_L("Cada ponto é uma configuração (média das 5 sementes). Forma = métrica de "
                      "otimização; tamanho do círculo = λ da FairMLP. Região sombreada = melhor que a "
                      "mediana das configurações exibidas em todos os eixos (equidade menor, desempenho "
                      "maior). Pontos circulados = fronteira de Pareto: nenhuma outra configuração é "
                      "melhor em todos os eixos ao mesmo tempo.",
                      "Each point is a configuration (mean of 5 seeds). Shape = optimisation metric; "
                      "circle size = FairMLP λ. Shaded region = better than the median of the displayed "
                      "configurations on every axis (lower unfairness, higher performance). Circled "
                      "points = Pareto front: no other configuration is better on all axes at once."))

        # ------------------------------------------------------------ tabela
        tcols = [(perf, _lbl(PERF, perf)), (f"{fair}_mean", f"{_lbl(FAIR, fair)} — " + _L("interseccional", "intersectional"))]
        tcols += [(f"{fair}_{c}_mean", f"{_lbl(FAIR, fair)} — {_lbl(ATTR_LABELS, c)}") for c in attr_cols]
        table = pd.DataFrame({_L("Configuração", "Configuration"): cfg["config"]})
        for c, lbl in tcols:
            table[lbl] = [f"{m:.3f} ± {s:.3f}" if pd.notna(m) else "—"
                          for m, s in zip(cfg[f"{c}|mean"], cfg[f"{c}|std"])]
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(_L("Média ± desvio padrão entre as 5 sementes. Interseccional = subgrupos raça/cor × sexo; "
                      "por atributo = grupos de um atributo só, somando os subgrupos.",
                      "Mean ± SD across the 5 seeds. Intersectional = race/colour × sex subgroups; per "
                      "attribute = groups of a single attribute, pooling the subgroups."))


REGION_COLOR = "#0ca30c"


def _pareto(df, minimize, maximize):
    """Máscara das linhas não dominadas: nenhuma outra é <= em todas as colunas a minimizar,
    >= na coluna a maximizar e estritamente melhor em ao menos uma."""
    v = np.column_stack([df[c].to_numpy() for c in minimize] + [-df[maximize].to_numpy()])
    keep = np.ones(len(v), bool)
    for i in range(len(v)):
        others = np.delete(v, i, axis=0)
        keep[i] = not np.any(np.all(others <= v[i], axis=1) & np.any(others < v[i], axis=1))
    return keep


def _scatter_3d(cfg, perf, fair, attrs, models):
    import plotly.graph_objects as go

    fx, fy = f"{fair}_{attrs[0]}_mean", f"{fair}_{attrs[1]}_mean"
    tx = f"{_lbl(FAIR, fair)} — {_lbl(ATTR_LABELS, attrs[0])}"
    ty = f"{_lbl(FAIR, fair)} — {_lbl(ATTR_LABELS, attrs[1])}"
    tz = _lbl(PERF, perf)
    fig = go.Figure()

    # Região desejada: caixa translúcida melhor que a mediana nos três eixos.
    pdf = cfg[["config", fx, fy, perf]].dropna()
    if len(pdf):
        mx, my, mz = pdf[fx].median(), pdf[fy].median(), pdf[perf].median()
        zt = pdf[perf].max()
        bx, by, bz = [0, mx, mx, 0, 0, mx, mx, 0], [0, 0, my, my, 0, 0, my, my], [mz] * 4 + [zt] * 4
        fig.add_trace(go.Mesh3d(
            x=bx, y=by, z=bz, alphahull=0, color=REGION_COLOR, opacity=0.12, hoverinfo="skip",
            name=_L("Região desejada", "Desired region"), showlegend=True))
        front = pdf[_pareto(pdf, [fx, fy], perf)]
        fig.add_trace(go.Scatter3d(
            x=front[fx], y=front[fy], z=front[perf], mode="markers",
            name=_L("Fronteira de Pareto", "Pareto front"),
            marker=dict(symbol="circle-open", size=16, color="#52514e", line=dict(width=2)),
            text=front["config"], hovertemplate="%{text}<extra>Pareto</extra>"))
    for (modelo, otim), g in cfg.groupby(["modelo", "otimizacao"], sort=False):
        if otim == "λ":
            symbol, size = "circle", [LAMBDA_SIZES_3D.get(l, 8) for l in g["lam"]]
            name = f"{modelo} · λ"
        else:
            symbol, size, name = OPT_SHAPES.get(otim, ("", "circle"))[1], 7, f"{modelo} · {otim}"
        fig.add_trace(go.Scatter3d(
            x=g[fx], y=g[fy], z=g[perf], mode="markers", name=name,
            marker=dict(symbol=symbol, size=size, color=MODEL_COLORS.get(modelo, "#52514e"),
                        opacity=0.85, line=dict(width=0)),
            text=g["config"],
            hovertemplate=f"%{{text}}<br>{tx}: %{{x:.3f}}<br>{ty}: %{{y:.3f}}<br>{tz}: %{{z:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        height=600, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(xaxis_title=tx, yaxis_title=ty, zaxis_title=tz,
                   camera=dict(eye=dict(x=1.7, y=1.7, z=0.9))),
        legend=dict(title=_L("Modelo · otimização", "Model · optimisation"), itemsizing="constant"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(_L("Gráfico 3D: arraste para girar, role para aproximar. Na FairMLP, o tamanho do "
                  "círculo cresce com λ.",
                  "3D chart: drag to rotate, scroll to zoom. For FairMLP, circle size grows with λ."))
