# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from data_module import DATASETS
from utils.bias_metrics import intersectional_audit_metrics, calculate_base_metrics, calculate_cramer_v, pairwise_gerrymandering_audit, calculate_cddl, calculate_dynamic_metrics
from utils.i18n import t

if "lang" not in st.session_state:
    st.session_state.lang = "PT"

st.set_page_config(page_title=t("data_page_title"), page_icon="📊", layout="wide")

st.sidebar.selectbox("Idioma / Language", ["PT", "EN"], key="lang")
lang_suffix = f"_{st.session_state.lang.lower()}"

st.title(t("data_title"))

# ---------------------------------------------------------
# Seletor cascata: País/Região → Dataset
# ---------------------------------------------------------
COUNTRY_ORDER = ['🇧🇷 Brasil', '🇺🇸 Estados Unidos', '🇵🇹 Portugal', '🧪 Sem nacionalidade (simulado)',
                 '🇧🇷 Brazil', '🇺🇸 United States', '🇵🇹 Portugal', '🧪 No nationality (simulated)']

countries = sorted(
    set(info[f'country{lang_suffix}'] for info in DATASETS.values()),
    key=lambda x: COUNTRY_ORDER.index(x) if x in COUNTRY_ORDER else 99
)

selected_country = st.selectbox(t("select_country"), countries)
filtered_datasets = {k: v for k, v in DATASETS.items() if v[f'country{lang_suffix}'] == selected_country}

# Monta nomes de exibição com ícone de domínio
display_map = {f"{v['icon']} {k}": k for k, v in filtered_datasets.items()}
selected_display = st.selectbox(t("select_dataset"), list(display_map.keys()))
dataset_name = display_map[selected_display]
dataset_info = filtered_datasets[dataset_name]

@st.cache_data
def load_data(name, uf=None):
    if DATASETS[name].get('supports_uf', False):
        return DATASETS[name]['loader'](uf)
    return DATASETS[name]['loader']()

uf_selected = ['Todos']
if dataset_info.get('supports_uf', False):
    ufs_options = ['Todos', 'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO']
    uf_selected = st.multiselect(t("select_state"), ufs_options, default=['Todos'])

with st.spinner(t("loading")):
    df = load_data(dataset_name, uf=uf_selected)

target_col = dataset_info['target']
favorable_val = dataset_info['favorable_val']
protected_attrs = dataset_info['protected_attributes']
proxy_attrs = dataset_info.get('proxy_attributes', [])
all_attrs = protected_attrs + proxy_attrs

# Card de metadados do dataset
with st.container(border=True):
    st.markdown(f"{t('domain_label')} {dataset_info.get(f'domain{lang_suffix}', '—')}")
    st.markdown(dataset_info.get(f'description{lang_suffix}', ''))
    link = dataset_info.get('link', '')
    if link:
        st.markdown(f"{t('original_source')}({link})")
    
    orig_n = dataset_info.get('original_n')
    orig_n_str = f"{orig_n:,}".replace(',', '.') if orig_n else dataset_info.get('n_approx', '—')
    processed_n_str = f"{len(df):,}".replace(',', '.')
    
    base_metrics = calculate_base_metrics(df, dataset_info)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("n_original"), orig_n_str)
    c2.metric(t("n_processed"), processed_n_str)
    c3.metric(t("year"), dataset_info.get('year', '') or '—')
    c4.metric(
        t("class_dist"), 
        base_metrics['Distribuição de Classes (%)'], 
        help=t("class_dist_help")
    )
st.divider()


# ---------------------------------------------------------
# Bloco 1: Distribuição Geral
# ---------------------------------------------------------
st.header(t("b1_header"))
st.markdown(f"{t('target')} `{target_col}` — {dataset_info.get(f'target_label{lang_suffix}', '')}")
st.markdown(f"{t('favorable_class')} `{favorable_val}` — {dataset_info.get(f'favorable_label{lang_suffix}', '')}")

# Cria uma cópia apenas para a visualização
df_viz = df.copy()
target_mapping = dataset_info.get('target_mapping')
if target_mapping:
    df_viz[target_col] = df_viz[target_col].map(target_mapping)

target_counts = df_viz[target_col].value_counts().reset_index()
target_counts.columns = [target_col, 'count']

# Filtro
selected_classes = st.multiselect(
    t("filter_classes"), 
    options=target_counts[target_col].unique(), 
    default=target_counts[target_col].unique()
)

filtered_counts = target_counts[target_counts[target_col].isin(selected_classes)]

base_general = alt.Chart(filtered_counts).encode(
    x=alt.X(f"{target_col}:N", title=t("class_label")),
    y=alt.Y("count:Q", title=t("freq_label")),
    color=f"{target_col}:N",
    tooltip=[target_col, alt.Tooltip("count:Q", title="N")]
).properties(height=300)

chart_general = base_general.mark_bar() + base_general.mark_text(dy=-5).encode(text='count:Q')

st.altair_chart(chart_general, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Bloco 2: Viés Unidimensional
# ---------------------------------------------------------
st.header(t("b2_header"))
st.markdown(t("b2_desc"))
st.markdown(f"{t('protected_label')} {', '.join(protected_attrs)}")
if proxy_attrs:
    st.markdown(f"{t('proxies_label')} {', '.join(proxy_attrs)}")

view_mode = st.radio(
    t("view"), 
    [t("view_by_attr"), t("view_agg")], 
    horizontal=True,
    label_visibility="collapsed"
)

# Mapeamento para tradução de rótulos nos gráficos e tabelas
translation_dict = {
    'White': 'Branco', 'Black': 'Negro', 'Asian-Pac-Islander': 'Asiático/Pacífico', 'Amer-Indian-Eskimo': 'Indígena/Esquimó', 'Other': 'Outro',
    'Caucasian': 'Caucasiano', 'African-American': 'Afro-americano', 'Hispanic': 'Hispânico', 'Native American': 'Nativo Americano', 'Asian': 'Asiático',
    'Female': 'Mulher', 'Male': 'Homem',
    'Wife': 'Esposa', 'Own-child': 'Filho(a)', 'Husband': 'Marido', 'Not-in-family': 'Não-familiar', 'Other-relative': 'Outro parente', 'Unmarried': 'Solteiro(a)',
    'Young': 'Jovem', 'Middle-aged': 'Meia-idade', 'Senior': 'Idoso', 'Less than 25': 'Menor que 25', 'Greater than 45': 'Maior que 45', 'Adult': 'Adulto',
    'Schooling': 'Ed. Básica', 'Associate/College': 'Ensino Sup./Téc.', 'Bachelors': 'Bacharelado', 'Masters/Doctorate/Prof': 'Pós-graduação',
    'Stable': 'Estável', 'Unstable': 'Instável', 'Yes': 'Sim', 'No': 'Não'
}

df_mapped = df.copy()
if st.session_state.lang == "PT":
    for col in all_attrs:
        if col in df_mapped.columns:
            # Cast to object to avoid TypeError if the column is categorical
            df_mapped[col] = df_mapped[col].astype(object).replace(translation_dict)

global_favorable_rate = (df[target_col] == favorable_val).mean()

if view_mode == t("view_agg"):
    marginal_frames = []
    for attr in all_attrs:
        attr_grouped = df_mapped.groupby(attr)[target_col].apply(
            lambda x: pd.Series({t("favorable_rate"): (x == favorable_val).mean(), "N": len(x)})
        ).unstack().reset_index()
        attr_grouped.rename(columns={attr: t('subgroup')}, inplace=True)
        attr_grouped['Atributo'] = attr
        marginal_frames.append(attr_grouped)

    marginal_df = pd.concat(marginal_frames, ignore_index=True)
    marginal_df[t('subgroup')] = marginal_df[t('subgroup')].astype(str)

    base_chart = alt.Chart(marginal_df).encode(
        x=alt.X(f"{t('subgroup')}:N", title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f"{t('favorable_rate')}:Q", title=t("favorable_rate"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Atributo:N", legend=None),
        tooltip=['Atributo', t('subgroup'), alt.Tooltip("N:Q", title="N"), alt.Tooltip(f"{t('favorable_rate')}:Q", format=".1%")]
    )

    bar = base_chart.mark_bar(opacity=0.9)

    text = base_chart.mark_text(dy=-5, fontSize=10).encode(
        y=alt.Y(f"{t('favorable_rate')}:Q"),
        text=alt.Text(f"{t('favorable_rate')}:Q", format=".1%")
    )

    rule = alt.Chart(pd.DataFrame({'mean': [global_favorable_rate]})).mark_rule(
        color='red', strokeDash=[5, 5]
    ).encode(y='mean:Q')

    layered = (bar + text + rule)

    faceted_chart = layered.facet(
        column=alt.Column("Atributo:N", title=None, header=alt.Header(labelOrient='bottom', titleOrient='bottom', labelFontWeight='bold'))
    ).resolve_scale(x='independent')

    st.altair_chart(faceted_chart, use_container_width=False)
    st.caption(f"{t('global_mean_legend')} ({global_favorable_rate:.1%}).")

else:
    uni_attr = st.selectbox(t("select_attr"), all_attrs)
    
    uni_grouped = df_mapped.groupby(uni_attr)[target_col].apply(
        lambda x: pd.Series({t("favorable_rate"): (x == favorable_val).mean(), "N": len(x)})
    ).unstack().reset_index()

    base_chart = alt.Chart(uni_grouped).encode(x=alt.X(f"{uni_attr}:N", title=t("subgroup"), axis=alt.Axis(labelAngle=0)))
    bar = base_chart.mark_bar(opacity=0.9).encode(
        y=alt.Y(f"{t('favorable_rate')}:Q", title=t("favorable_rate"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(f"{uni_attr}:N", legend=None),
        tooltip=[alt.Tooltip(f"{uni_attr}:N", title=t("subgroup")), alt.Tooltip("N:Q", title="N"), alt.Tooltip(f"{t('favorable_rate')}:Q", format=".1%")]
    )
    text = base_chart.mark_text(dy=-5, fontSize=12).encode(
        y=alt.Y(f"{t('favorable_rate')}:Q"),
        text=alt.Text(f"{t('favorable_rate')}:Q", format=".1%")
    )
    rule = alt.Chart(pd.DataFrame({'mean': [global_favorable_rate]})).mark_rule(
        color='red', strokeDash=[5, 5]
    ).encode(y='mean:Q')
    
    rule_label = alt.Chart(pd.DataFrame({'mean': [global_favorable_rate]})).mark_text(
        align='left', baseline='bottom', dy=-5, dx=5, color='red', text=f"{t('global_mean')}: {global_favorable_rate:.1%}"
    ).encode(y='mean:Q', x=alt.value(0))

    st.altair_chart((bar + text + rule + rule_label).properties(height=350), use_container_width=True)
    
    dyn_metrics = calculate_dynamic_metrics(df_mapped, uni_attr, target_col, favorable_val)
    
    if dyn_metrics['priv'] is not None:
        st.markdown(f"{t('fairness_metrics_for')} `{uni_attr}`**")
        st.markdown(t("groups_identified").format(dyn_metrics['priv'], dyn_metrics['unpriv']), unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("ci_metric"), dyn_metrics['CI'])
        c2.metric(t("di_metric"), dyn_metrics['DI'])
        c3.metric(t("kl_metric"), dyn_metrics['KL'])
        c4.metric(t("ks_metric"), dyn_metrics['KS'])
        
        with st.expander(t("understand_metrics")):
            st.markdown(t("metrics_explanation"))

st.divider()

# ---------------------------------------------------------
# Bloco 3: Viés Interseccional
# ---------------------------------------------------------
st.header(t("b3_header"))
st.markdown(t("b3_desc"))

selected_attrs = st.multiselect(t("select_multi_attr"), all_attrs, default=all_attrs[:2])

if len(selected_attrs) < 2:
    st.warning(t("warn_multi_attr"))
else:
    tab1, tab2 = st.tabs([t("tab_subgroups"), t("tab_pairs")])
    
    with tab1:
        if len(selected_attrs) == 2:
            st.subheader(t("viz_intersec_2"))
            
            inter_grouped = df_mapped.groupby(selected_attrs)[target_col].apply(
                lambda x: pd.Series({t("favorable_rate"): (x == favorable_val).mean(), "N": len(x)})
            ).unstack().reset_index()
            
            # Marcar inviáveis (N < 100)
            inter_grouped['Viável'] = inter_grouped['N'] >= 100
            inter_grouped['Global Mean'] = global_favorable_rate
            
            base_bar_chart = alt.Chart(inter_grouped).encode(
                x=alt.X(f"{selected_attrs[1]}:N", title=selected_attrs[1]),
                y=alt.Y(f"{t('favorable_rate')}:Q", title=t("favorable_rate"), scale=alt.Scale(domain=[0, 1])),
                color=f"{selected_attrs[0]}:N",
                opacity=alt.condition(alt.datum.Viável, alt.value(1.0), alt.value(0.3)),
                tooltip=[selected_attrs[0], selected_attrs[1], alt.Tooltip("N:Q", title="N"), alt.Tooltip(f"{t('favorable_rate')}:Q", format=".1%")]
            ).properties(width=150, height=350)
            
            base_bar = base_bar_chart.mark_bar()
            text = base_bar_chart.mark_text(dy=-5).encode(text='N:Q')
            
            rule_inter = alt.Chart(inter_grouped).mark_rule(color='red', strokeDash=[5, 5]).encode(y='Global Mean:Q')
            
            layered_chart = (base_bar + text + rule_inter).facet(column=f"{selected_attrs[0]}:N")
            
            st.altair_chart(layered_chart, use_container_width=False)
            st.caption(t("caption_translucent"))
        
        else:
            st.subheader(t("audit_gerry_3"))
            
            with st.spinner(t("calc_gerry")):
                audit_df = intersectional_audit_metrics(df_mapped, selected_attrs, target_col, favorable_val)
            
            # Rename columns to match language
            audit_df = audit_df.rename(columns={
                'Subgrupo': t('tbl_subgroup'),
                'Taxa Favorável': t('tbl_fav_rate'),
                'Gap Real (Interseccional)': t('tbl_real_gap'),
                'Gap Esperado (Marginal Máx)': t('tbl_exp_gap'),
                'Viés Oculto (Excedente)': t('tbl_hidden_bias'),
                'Score de Prioridade': t('tbl_prio_score'),
                'DI Pré-treinamento': t('tbl_pre_di'),
                'Veredito da Auditoria': t('tbl_audit_verdict')
            })

            # Translate verdicts if EN
            if st.session_state.lang == "EN":
                verdict_map = {
                    "Inviável (N<100)": "Inviable (N<100)",
                    "Alto Viés Oculto": "High Hidden Bias",
                    "GERRYMANDERING": "GERRYMANDERING",
                    "Viés Marginal Predominante": "Predominant Marginal Bias",
                    "Ok": "Ok"
                }
                audit_df[t('tbl_audit_verdict')] = audit_df[t('tbl_audit_verdict')].replace(verdict_map)

            # Estilização condicional
            def color_verdict(val):
                color = 'green' if val == 'Ok' else 'orange' if 'Inviável' in val or 'Inviable' in val else 'red'
                return f'color: {color}'
                
            st.dataframe(
                audit_df.style.map(color_verdict, subset=[t('tbl_audit_verdict')])
                      .format({t('tbl_fav_rate'): '{:.2%}', t('tbl_real_gap'): '{:.4f}', 
                               t('tbl_exp_gap'): '{:.4f}', t('tbl_hidden_bias'): '{:.4f}',
                               t('tbl_prio_score'): '{:.2f}', t('tbl_pre_di'): '{:.4f}'}),
                use_container_width=True
            )
            
            st.caption(t("caption_inviable"))
            
            # Exportação Long CSV
            long_csv = audit_df.melt(id_vars=[t('tbl_subgroup'), 'N'], 
                                     value_vars=[t('tbl_fav_rate'), t('tbl_real_gap'), t('tbl_exp_gap'), t('tbl_hidden_bias'), t('tbl_prio_score'), t('tbl_pre_di')],
                                     var_name=t('tbl_metric'), value_name=t('tbl_value'))
            long_csv.insert(0, 'Dataset', dataset_name)
            
            st.download_button(
                label=t("export_csv"),
                data=long_csv.to_csv(index=False).encode('utf-8'),
                file_name=f"{dataset_name}_intersectional_audit_long.csv",
                mime="text/csv"
            )
            
        st.divider()
        st.subheader(t("cddl_header"))
        if not proxy_attrs:
            st.info(t("no_proxy"))
        else:
            primary_protected = dataset_info.get('primary_protected')
            priv_group = dataset_info.get('privileged_group')
            unpriv_group = dataset_info.get('unprivileged_group')
            
            if primary_protected and priv_group and unpriv_group and primary_protected in df.columns:
                with st.spinner(t("calc_cddl")):
                    cddl_df = calculate_cddl(df, target_col, favorable_val, primary_protected, priv_group, unpriv_group, proxy_attrs)
                    if st.session_state.lang == "PT":
                        cddl_df['Proxy (Estrato)'] = cddl_df['Proxy (Estrato)'].replace(translation_dict)
                
                st.dataframe(
                    cddl_df.style.format({'CDDL': '{:.4f}'}),
                    use_container_width=True
                )
                st.caption(t("caption_cddl"))
            else:
                st.warning(t("warn_metadata"))

    with tab2:
        st.subheader(t("audit_pairs"))
        st.markdown(t("audit_pairs_desc"))
        
        with st.spinner(t("scanning_pairs")):
            pair_df = pairwise_gerrymandering_audit(df_mapped, selected_attrs, target_col, favorable_val)

            # Rename columns
            pair_df = pair_df.rename(columns={
                'Par de Intersecção': t('tbl_pair'),
                'Gap Indiv. A': t('tbl_gap_a'),
                'Gap Indiv. B': t('tbl_gap_b'),
                'Gap Esperado (Marginal Máx)': t('tbl_exp_gap'),
                'Gap Real (Interseccional)': t('tbl_real_gap'),
                'Viés Oculto (Excedente)': t('tbl_hidden_bias'),
                'Veredito da Auditoria': t('tbl_audit_verdict')
            })

            # Translate verdicts if EN
            if st.session_state.lang == "EN":
                verdict_map = {
                    "Inviável (N<100)": "Inviable (N<100)",
                    "Alto Viés Oculto": "High Hidden Bias",
                    "GERRYMANDERING": "GERRYMANDERING",
                    "Viés Marginal Predominante": "Predominant Marginal Bias",
                    "Ok": "Ok"
                }
                pair_df[t('tbl_audit_verdict')] = pair_df[t('tbl_audit_verdict')].replace(verdict_map)
            
        def color_pair_verdict(val):
            return 'background-color: #ffcccc; color: #cc0000; font-weight: bold;' if 'GERRYMANDERING' in val else ''
            
        st.dataframe(
            pair_df.style.map(color_pair_verdict, subset=[t('tbl_audit_verdict')])
                  .format({
                      t('tbl_gap_a'): '{:.2%}',
                      t('tbl_gap_b'): '{:.2%}',
                      t('tbl_exp_gap'): '{:.2%}',
                      t('tbl_real_gap'): '{:.2%}',
                      t('tbl_hidden_bias'): '{:+.2%}'
                  }),
            use_container_width=True
        )
        
        st.markdown(f"**{dataset_name} — {t('gap_audit_title')}**")
        
        melted_pairs = pair_df.melt(
            id_vars=[t('tbl_pair')],
            value_vars=[t('tbl_exp_gap'), t('tbl_real_gap')],
            var_name=t('tbl_gap_type'),
            value_name=t('tbl_gap_amplitude')
        )
        
        base = alt.Chart(melted_pairs).encode(
            x=alt.X(f"{t('tbl_pair')}:N", title='', axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(f"{t('tbl_gap_amplitude')}:Q", title=t('tbl_gap_amplitude'), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(f"{t('tbl_gap_type')}:N", 
                scale=alt.Scale(range=['#4c6b8b', '#002f6c']),
                legend=alt.Legend(title="", orient="top-left")
            )
        )
        
        bars = base.mark_bar().encode(xOffset=alt.XOffset(f"{t('tbl_gap_type')}:N"))
        
        text = base.mark_text(
            align='center',
            baseline='bottom',
            dy=-5,
            fontSize=10
        ).encode(
            xOffset=alt.XOffset(f"{t('tbl_gap_type')}:N"),
            text=alt.Text(f"{t('tbl_gap_amplitude')}:Q", format='.1%')
        ).transform_filter(
            alt.datum[t('tbl_gap_type')] == t('tbl_real_gap')
        )
        
        st.altair_chart((bars + text).properties(height=400), use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Bloco 4: Correlação Categórica (Risco de Proxies)
# ---------------------------------------------------------
st.header(t("b4_header"))
st.markdown(t("b4_desc"))

vars_to_correlate = [target_col] + all_attrs

if len(vars_to_correlate) > 1:
    with st.spinner(t("calc_cramer")):
        matrix = []
        for var1 in vars_to_correlate:
            row = []
            for var2 in vars_to_correlate:
                if var1 == var2:
                    row.append(1.0)
                else:
                    row.append(calculate_cramer_v(df, var1, var2))
            matrix.append(row)
            
        corr_df = pd.DataFrame(matrix, index=vars_to_correlate, columns=vars_to_correlate)
        
        # Format for Altair
        corr_melt = corr_df.reset_index().melt(id_vars='index')
        corr_melt.columns = ['Var1', 'Var2', 'Cramer_V']
        
        base_hm = alt.Chart(corr_melt).encode(
            x=alt.X('Var1:N', title=''),
            y=alt.Y('Var2:N', title='')
        )
        
        hm = base_hm.mark_rect().encode(
            color=alt.Color('Cramer_V:Q', scale=alt.Scale(scheme='blues', domain=[0, 1]), title=t("cramer_v")),
            tooltip=['Var1', 'Var2', alt.Tooltip('Cramer_V:Q', format=".2f")]
        )
        
        text_hm = base_hm.mark_text(baseline='middle').encode(
            text=alt.Text('Cramer_V:Q', format=".2f"),
            color=alt.condition(
                alt.datum.Cramer_V > 0.5,
                alt.value('white'),
                alt.value('black')
            )
        )
        
        heatmap = (hm + text_hm).properties(height=max(400, len(vars_to_correlate)*40))
        st.altair_chart(heatmap, use_container_width=True)
else:
    st.info(t("insufficient_attrs"))
