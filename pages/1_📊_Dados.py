# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from data_module import DATASETS
from utils.bias_metrics import intersectional_audit_metrics, calculate_base_metrics, calculate_cramer_v, pairwise_gerrymandering_audit, calculate_cddl, calculate_dynamic_metrics

st.set_page_config(page_title="Dados (EDA)", page_icon="📊", layout="wide")

st.title("Análise Exploratória e Viés Social Pré-Treino")

# ---------------------------------------------------------
# Seletor cascata: País/Região → Dataset
# ---------------------------------------------------------
COUNTRY_ORDER = ['🇧🇷 Brasil', '🇺🇸 Estados Unidos', '🇵🇹 Portugal', '🧪 Sem nacionalidade (simulado)']

countries = sorted(
    set(info['country'] for info in DATASETS.values()),
    key=lambda x: COUNTRY_ORDER.index(x) if x in COUNTRY_ORDER else 99
)

selected_country = st.selectbox("Selecione o país/região:", countries)
filtered_datasets = {k: v for k, v in DATASETS.items() if v['country'] == selected_country}

# Monta nomes de exibição com ícone de domínio
display_map = {f"{v['icon']} {k}": k for k, v in filtered_datasets.items()}
selected_display = st.selectbox("Selecione o Dataset:", list(display_map.keys()))
dataset_name = display_map[selected_display]
dataset_info = filtered_datasets[dataset_name]

@st.cache_data
def load_data(name, uf=None):
    if uf:
        return DATASETS[name]['loader'](uf)
    return DATASETS[name]['loader']()

uf_selected = None
if dataset_info.get('supports_uf', False):
    uf_selected = st.selectbox("Selecione o Estado (UF):", ['SP', 'RS', 'RJ', 'MG', 'BA', 'PE', 'DF'])

with st.spinner("Carregando e processando dados..."):
    df = load_data(dataset_name, uf=uf_selected)

target_col = dataset_info['target']
favorable_val = dataset_info['favorable_val']
protected_attrs = dataset_info['protected_attributes']
proxy_attrs = dataset_info.get('proxy_attributes', [])
all_attrs = protected_attrs + proxy_attrs

# Card de metadados do dataset
with st.container(border=True):
    st.markdown(f"**Domínio:** {dataset_info.get('domain', '—')}")
    st.markdown(dataset_info.get('description', ''))
    link = dataset_info.get('link', '')
    if link:
        st.markdown(f"🔗 [Fonte original]({link})")
    
    orig_n = dataset_info.get('original_n')
    orig_n_str = f"{orig_n:,}".replace(',', '.') if orig_n else dataset_info.get('n_approx', '—')
    processed_n_str = f"{len(df):,}".replace(',', '.')
    
    base_metrics = calculate_base_metrics(df, dataset_info)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N Original (Bruto)", orig_n_str)
    c2.metric("N Pós-processamento", processed_n_str)
    c3.metric("Ano", dataset_info.get('year', '') or '—')
    c4.metric(
        "Dist. de Classes (Alvo)", 
        base_metrics['Distribuição de Classes (%)'], 
        help="Proporção global entre a classe majoritária e minoritária da variável-alvo. Mede o quão desbalanceados estão os desfechos em toda a base de dados."
    )
st.divider()


# ---------------------------------------------------------
# Bloco 1: Distribuição Geral
# ---------------------------------------------------------
st.header("1. Distribuição Geral do Alvo")
st.markdown(f"**Alvo:** `{target_col}` — {dataset_info.get('target_label', '')}")
st.markdown(f"**Classe favorável:** `{favorable_val}` — {dataset_info.get('favorable_label', '')}")

# Cria uma cópia apenas para a visualização
df_viz = df.copy()
target_mapping = dataset_info.get('target_mapping')
if target_mapping:
    df_viz[target_col] = df_viz[target_col].map(target_mapping)

target_counts = df_viz[target_col].value_counts().reset_index()
target_counts.columns = [target_col, 'count']

# Filtro
selected_classes = st.multiselect(
    "Filtrar classes:", 
    options=target_counts[target_col].unique(), 
    default=target_counts[target_col].unique()
)

filtered_counts = target_counts[target_counts[target_col].isin(selected_classes)]

base_general = alt.Chart(filtered_counts).encode(
    x=alt.X(f"{target_col}:N", title="Classe"),
    y=alt.Y("count:Q", title="Frequência"),
    color=f"{target_col}:N",
    tooltip=[target_col, alt.Tooltip("count:Q", title="N")]
).properties(height=300)

chart_general = base_general.mark_bar() + base_general.mark_text(dy=-5).encode(text='count:Q')

st.altair_chart(chart_general, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Bloco 2: Viés Unidimensional
# ---------------------------------------------------------
# ---------------------------------------------------------
st.header("2. Viés Unidimensional")
st.markdown("Distribuição dos resultados favoráveis por subgrupo demográfico")
st.markdown(f"**Atributos protegidos (Demográficos):** {', '.join(protected_attrs)}")
if proxy_attrs:
    st.markdown(f"**Proxies (Socioeconômicos/Comportamentais):** {', '.join(proxy_attrs)}")

view_mode = st.radio(
    "Visão", 
    ["Visão por atributo", "Visão agregada"], 
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
for col in all_attrs:
    if col in df_mapped.columns:
        df_mapped[col] = df_mapped[col].replace(translation_dict)

global_favorable_rate = (df[target_col] == favorable_val).mean()

if view_mode == "Visão agregada":
    marginal_frames = []
    for attr in all_attrs:
        attr_grouped = df_mapped.groupby(attr)[target_col].apply(
            lambda x: pd.Series({"Taxa Favorável": (x == favorable_val).mean(), "N": len(x)})
        ).unstack().reset_index()
        attr_grouped.rename(columns={attr: 'Subgrupo'}, inplace=True)
        attr_grouped['Atributo'] = attr
        marginal_frames.append(attr_grouped)

    marginal_df = pd.concat(marginal_frames, ignore_index=True)
    marginal_df['Subgrupo'] = marginal_df['Subgrupo'].astype(str)

    base_chart = alt.Chart(marginal_df).encode(
        x=alt.X("Subgrupo:N", title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("Taxa Favorável:Q", title="Taxa Favorável", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Atributo:N", legend=None),
        tooltip=['Atributo', 'Subgrupo', alt.Tooltip("N:Q", title="N"), alt.Tooltip("Taxa Favorável:Q", format=".1%")]
    )

    bar = base_chart.mark_bar(opacity=0.9)

    text = base_chart.mark_text(dy=-5, fontSize=10).encode(
        y=alt.Y("Taxa Favorável:Q"),
        text=alt.Text('Taxa Favorável:Q', format=".1%")
    )

    rule = alt.Chart(pd.DataFrame({'mean': [global_favorable_rate]})).mark_rule(
        color='red', strokeDash=[5, 5]
    ).encode(y='mean:Q')

    layered = (bar + text + rule)

    faceted_chart = layered.facet(
        column=alt.Column("Atributo:N", title=None, header=alt.Header(labelOrient='bottom', titleOrient='bottom', labelFontWeight='bold'))
    ).resolve_scale(x='independent')

    st.altair_chart(faceted_chart, use_container_width=False)
    st.caption(f"A linha pontilhada vermelha representa a Média Global ({global_favorable_rate:.1%}).")

else:
    uni_attr = st.selectbox("Selecione o atributo para inspecionar:", all_attrs)
    
    uni_grouped = df_mapped.groupby(uni_attr)[target_col].apply(
        lambda x: pd.Series({"Taxa Favorável": (x == favorable_val).mean(), "N": len(x)})
    ).unstack().reset_index()

    base_chart = alt.Chart(uni_grouped).encode(x=alt.X(f"{uni_attr}:N", title="Subgrupo", axis=alt.Axis(labelAngle=0)))
    bar = base_chart.mark_bar(opacity=0.9).encode(
        y=alt.Y("Taxa Favorável:Q", title="Taxa Favorável", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(f"{uni_attr}:N", legend=None),
        tooltip=[alt.Tooltip(f"{uni_attr}:N", title="Subgrupo"), alt.Tooltip("N:Q", title="N"), alt.Tooltip("Taxa Favorável:Q", format=".1%")]
    )
    text = base_chart.mark_text(dy=-5, fontSize=12).encode(
        y=alt.Y("Taxa Favorável:Q"),
        text=alt.Text('Taxa Favorável:Q', format=".1%")
    )
    rule = alt.Chart(pd.DataFrame({'mean': [global_favorable_rate]})).mark_rule(
        color='red', strokeDash=[5, 5]
    ).encode(y='mean:Q')
    
    rule_label = alt.Chart(pd.DataFrame({'mean': [global_favorable_rate]})).mark_text(
        align='left', baseline='bottom', dy=-5, dx=5, color='red', text=f"Média Global: {global_favorable_rate:.1%}"
    ).encode(y='mean:Q', x=alt.value(0))

    st.altair_chart((bar + text + rule + rule_label).properties(height=350), use_container_width=True)
    
    dyn_metrics = calculate_dynamic_metrics(df_mapped, uni_attr, target_col, favorable_val)
    
    if dyn_metrics['priv'] is not None:
        st.markdown(f"**Métricas de Equidade para `{uni_attr}`**")
        st.markdown(f"Grupos identificados automaticamente para cálculo:<br>**Privilegiado** = `{dyn_metrics['priv']}` (maior taxa de sucesso) | **Desprivilegiado** = `{dyn_metrics['unpriv']}` (menor taxa de sucesso).", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Desbalanceamento de Classe (CI)", dyn_metrics['CI'])
        c2.metric("DI Pré-treino", dyn_metrics['DI'])
        c3.metric("Divergência KL", dyn_metrics['KL'])
        c4.metric("Estatística KS", dyn_metrics['KS'])
        
        with st.expander("ℹ️ Entenda as Métricas"):
            st.markdown("""
            * **Desbalanceamento de Classe (CI)**: Mede o desbalanceamento demográfico entre os dois grupos extremos listados acima. Varia de -1 (todas as amostras no grupo desprivilegiado) a 1 (todas no privilegiado). Ideal: 0.
            * **DI Pré-treino (Disparate Impact)**: Razão entre a chance do grupo desprivilegiado pertencer à classe alvo favorável e a do grupo privilegiado. Valores < 0.8 ou > 1.2 indicam disparidade (Regra dos 80%).
            * **Divergência KL**: Mede a divergência (distância) entre as distribuições de probabilidade de resultados do grupo privilegiado e desprivilegiado.
            * **Estatística KS (Kolmogorov-Smirnov)**: Mede a distância máxima entre as distribuições acumuladas dos dois grupos. Valores altos indicam desigualdade na distribuição.
            """)

st.divider()

# ---------------------------------------------------------
# Bloco 3: Viés Interseccional
# ---------------------------------------------------------
st.header("3. Viés Interseccional")
st.markdown("Selecione múltiplos atributos para análise de interseccionalidade e Justice Gerrymandering.")

selected_attrs = st.multiselect("Selecione 2 ou mais atributos:", all_attrs, default=all_attrs[:2])

if len(selected_attrs) < 2:
    st.warning("Selecione pelo menos 2 atributos para análise interseccional.")
else:
    tab1, tab2 = st.tabs(["Auditoria de Subgrupos", "Auditoria em Pares (Global)"])
    
    with tab1:
        if len(selected_attrs) == 2:
            st.subheader("Visualização Interseccional (2 Atributos)")
            
            inter_grouped = df_mapped.groupby(selected_attrs)[target_col].apply(
                lambda x: pd.Series({"Taxa Favorável": (x == favorable_val).mean(), "N": len(x)})
            ).unstack().reset_index()
            
            # Marcar inviáveis (N < 100)
            inter_grouped['Viável'] = inter_grouped['N'] >= 100
            inter_grouped['Global Mean'] = global_favorable_rate
            
            base_bar_chart = alt.Chart(inter_grouped).encode(
                x=alt.X(f"{selected_attrs[1]}:N", title=selected_attrs[1]),
                y=alt.Y("Taxa Favorável:Q", title="Taxa Favorável", scale=alt.Scale(domain=[0, 1])),
                color=f"{selected_attrs[0]}:N",
                opacity=alt.condition(alt.datum.Viável, alt.value(1.0), alt.value(0.3)),
                tooltip=[selected_attrs[0], selected_attrs[1], alt.Tooltip("N:Q", title="N"), alt.Tooltip("Taxa Favorável:Q", format=".1%")]
            ).properties(width=150, height=350)
            
            base_bar = base_bar_chart.mark_bar()
            text = base_bar_chart.mark_text(dy=-5).encode(text='N:Q')
            
            rule_inter = alt.Chart(inter_grouped).mark_rule(color='red', strokeDash=[5, 5]).encode(y='Global Mean:Q')
            
            layered_chart = (base_bar + text + rule_inter).facet(column=f"{selected_attrs[0]}:N")
            
            st.altair_chart(layered_chart, use_container_width=False)
            st.caption("Barras translúcidas indicam N < 100 (subgrupo inviável estatisticamente). Linha vermelha = Média Global.")
        
        else:
            st.subheader("Auditoria de Justice Gerrymandering (3+ Atributos)")
            
            with st.spinner("Calculando métricas de Gerrymandering..."):
                audit_df = intersectional_audit_metrics(df_mapped, selected_attrs, target_col, favorable_val)
            
            # Estilização condicional
            def color_verdict(val):
                color = 'green' if val == 'Ok' else 'orange' if 'Inviável' in val else 'red'
                return f'color: {color}'
                
            st.dataframe(
                audit_df.style.map(color_verdict, subset=['Veredito da Auditoria'])
                      .format({'Taxa Favorável': '{:.2%}', 'Gap Real (Interseccional)': '{:.4f}', 
                               'Gap Esperado (Marginal Máx)': '{:.4f}', 'Viés Oculto (Excedente)': '{:.4f}',
                               'Score de Prioridade': '{:.2f}', 'DI Pré-treinamento': '{:.4f}'}),
                use_container_width=True
            )
            
            st.caption("Subgrupos com N < 100 são marcados como 'Inviável'. DI pré-treinamento indica Pre-training Disparate Impact contra a média global.")
            
            # Exportação Long CSV
            long_csv = audit_df.melt(id_vars=['Subgrupo', 'N'], 
                                     value_vars=['Taxa Favorável', 'Gap Real (Interseccional)', 'Gap Esperado (Marginal Máx)', 'Viés Oculto (Excedente)', 'Score de Prioridade', 'DI Pré-treinamento'],
                                     var_name='Métrica', value_name='Valor')
            long_csv.insert(0, 'Dataset', dataset_name)
            
            st.download_button(
                label="📥 Exportar Dados (CSV Longo Consolidado)",
                data=long_csv.to_csv(index=False).encode('utf-8'),
                file_name=f"{dataset_name}_intersectional_audit_long.csv",
                mime="text/csv"
            )
            
        st.divider()
        st.subheader("Disparidade Demográfica Condicional (CDDL)")
        if not proxy_attrs:
            st.info("Nenhum proxy definido")
        else:
            primary_protected = dataset_info.get('primary_protected')
            priv_group = dataset_info.get('privileged_group')
            unpriv_group = dataset_info.get('unprivileged_group')
            
            if primary_protected and priv_group and unpriv_group and primary_protected in df.columns:
                with st.spinner("Calculando CDDL..."):
                    cddl_df = calculate_cddl(df, target_col, favorable_val, primary_protected, priv_group, unpriv_group, proxy_attrs)
                    cddl_df['Proxy (Estrato)'] = cddl_df['Proxy (Estrato)'].replace(translation_dict)
                
                st.dataframe(
                    cddl_df.style.format({'CDDL': '{:.4f}'}),
                    use_container_width=True
                )
                st.caption("Valores positivos indicam que a classe desprivilegiada sofre mais resultados desfavoráveis (e menos favoráveis) condicionado aos estratos do proxy.")
            else:
                st.warning("Metadados incompletos para definir grupo privilegiado/desprivilegiado principal.")

    with tab2:
        st.subheader("Auditoria de Gerrymandering em Pares (Global)")
        st.markdown("Esta visão varre **todos os pares possíveis** entre os atributos selecionados para encontrar a disparidade máxima na margem versus na interseção.")
        
        with st.spinner("Varrendo pares..."):
            pair_df = pairwise_gerrymandering_audit(df_mapped, selected_attrs, target_col, favorable_val)
            
        def color_pair_verdict(val):
            return 'background-color: #ffcccc; color: #cc0000; font-weight: bold;' if 'GERRYMANDERING' in val else ''
            
        st.dataframe(
            pair_df.style.map(color_pair_verdict, subset=['Veredito da Auditoria'])
                  .format({
                      'Gap Indiv. A': '{:.2%}',
                      'Gap Indiv. B': '{:.2%}',
                      'Gap Esperado (Marginal Máx)': '{:.2%}',
                      'Gap Real (Interseccional)': '{:.2%}',
                      'Viés Oculto (Excedente)': '{:+.2%}'
                  }),
            use_container_width=True
        )
        
        st.markdown(f"**{dataset_name} — Auditoria de Gap: Disparidade Marginal vs. Interseccional**")
        
        melted_pairs = pair_df.melt(
            id_vars=['Par de Intersecção'],
            value_vars=['Gap Esperado (Marginal Máx)', 'Gap Real (Interseccional)'],
            var_name='Tipo de Gap',
            value_name='Amplitude do Gap (%)'
        )
        
        base = alt.Chart(melted_pairs).encode(
            x=alt.X('Par de Intersecção:N', title='', axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('Amplitude do Gap (%):Q', title='Amplitude do Gap (%)', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('Tipo de Gap:N', 
                scale=alt.Scale(range=['#4c6b8b', '#002f6c']),
                legend=alt.Legend(title="", orient="top-left")
            )
        )
        
        bars = base.mark_bar(xOffset=alt.XOffset("Tipo de Gap:N"))
        
        text = base.mark_text(
            align='center',
            baseline='bottom',
            dy=-5,
            fontSize=10
        ).encode(
            xOffset=alt.XOffset("Tipo de Gap:N"),
            text=alt.Text('Amplitude do Gap (%):Q', format='.1%')
        ).transform_filter(
            alt.datum['Tipo de Gap'] == 'Gap Real (Interseccional)'
        )
        
        st.altair_chart((bars + text).properties(height=400), use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Bloco 4: Correlação Categórica (Risco de Proxies)
# ---------------------------------------------------------
st.header("4. Matriz de Correlação (Risco de Proxies)")
st.markdown("O **V de Cramér** mede a associação estatística entre variáveis categóricas (0 = sem associação, 1 = associação perfeita). Valores altos entre um *proxy* socioeconômico e um atributo protegido indicam alto risco de que modelos descubram a classe sensível indiretamente (*Redlining*).")

vars_to_correlate = [target_col] + all_attrs

if len(vars_to_correlate) > 1:
    with st.spinner("Calculando V de Cramér..."):
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
            color=alt.Color('Cramer_V:Q', scale=alt.Scale(scheme='blues', domain=[0, 1]), title="V de Cramér"),
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
    st.info("Atributos insuficientes para gerar a matriz de correlação.")
