import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from data_module import DATASETS
from utils.bias_metrics import intersectional_audit_metrics

st.set_page_config(page_title="Dados (EDA)", page_icon="📊", layout="wide")

st.title("Análise Exploratória e Viés Social Pré-Treino")

# ---------------------------------------------------------
# Seletor de Dataset
# ---------------------------------------------------------
dataset_name = st.selectbox("Selecione o Dataset:", list(DATASETS.keys()))
dataset_info = DATASETS[dataset_name]

if 'description' in dataset_info:
    st.info(dataset_info['description'])

@st.cache_data
def load_data(name):
    return DATASETS[name]['loader']()

with st.spinner("Carregando e processando dados..."):
    df = load_data(dataset_name)

target_col = dataset_info['target']
favorable_val = dataset_info['favorable_val']
protected_attrs = dataset_info['protected_attributes']

st.markdown(f"**Alvo ({target_col}):** A classe favorável é `{favorable_val}`. **N Total:** {len(df)}")

st.divider()

# ---------------------------------------------------------
# Bloco 1: Distribuição Geral
# ---------------------------------------------------------
st.header("1. Distribuição Geral do Alvo")
target_counts = df[target_col].value_counts().reset_index()
target_counts.columns = [target_col, 'count']

# Filtro
selected_classes = st.multiselect(
    f"Filtrar classes de {target_col}:", 
    options=target_counts[target_col].unique(), 
    default=target_counts[target_col].unique()
)

filtered_counts = target_counts[target_counts[target_col].isin(selected_classes)]

chart_general = alt.Chart(filtered_counts).mark_bar().encode(
    x=alt.X(f"{target_col}:N", title="Classe"),
    y=alt.Y("count:Q", title="Frequência"),
    color=f"{target_col}:N",
    tooltip=[target_col, 'count']
).properties(height=300)

st.altair_chart(chart_general, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Bloco 2: Viés Unidimensional
# ---------------------------------------------------------
st.header("2. Viés Unidimensional")
st.markdown("Taxa de resultado favorável por subgrupo marginal.")

uni_attr = st.radio("Selecione 1 atributo sensível:", protected_attrs, horizontal=True)

global_favorable_rate = (df[target_col] == favorable_val).mean()

uni_grouped = df.groupby(uni_attr)[target_col].apply(
    lambda x: (x == favorable_val).mean()
).reset_index()
uni_grouped.columns = [uni_attr, 'Taxa Favorável']

# Linha de média global
base_chart = alt.Chart(uni_grouped).encode(x=alt.X(f"{uni_attr}:N", title=uni_attr))
bar = base_chart.mark_bar(opacity=0.8).encode(
    y=alt.Y("Taxa Favorável:Q", title="Taxa Favorável", scale=alt.Scale(domain=[0, 1])),
    color=f"{uni_attr}:N",
    tooltip=[uni_attr, alt.Tooltip("Taxa Favorável:Q", format=".1%")]
)
rule = alt.Chart(pd.DataFrame({'mean': [global_favorable_rate]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='mean:Q')

st.altair_chart((bar + rule).properties(height=350), use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Bloco 3: Viés Interseccional
# ---------------------------------------------------------
st.header("3. Viés Interseccional")
st.markdown("Selecione múltiplos atributos para análise de interseccionalidade e Justice Gerrymandering.")

selected_attrs = st.multiselect("Selecione 2 ou mais atributos:", protected_attrs, default=protected_attrs[:2])

if len(selected_attrs) < 2:
    st.warning("Selecione pelo menos 2 atributos para análise interseccional.")
elif len(selected_attrs) == 2:
    st.subheader("Visualização Interseccional (2 Atributos)")
    
    inter_grouped = df.groupby(selected_attrs)[target_col].apply(
        lambda x: pd.Series({"Taxa Favorável": (x == favorable_val).mean(), "N": len(x)})
    ).unstack().reset_index()
    
    # Marcar inviáveis (N < 100)
    inter_grouped['Viável'] = inter_grouped['N'] >= 100
    
    bar_inter = alt.Chart(inter_grouped).mark_bar().encode(
        x=alt.X(f"{selected_attrs[1]}:N", title=selected_attrs[1]),
        y=alt.Y("Taxa Favorável:Q", title="Taxa Favorável", scale=alt.Scale(domain=[0, 1])),
        color=f"{selected_attrs[0]}:N",
        column=f"{selected_attrs[0]}:N",
        opacity=alt.condition(alt.datum.Viável, alt.value(1.0), alt.value(0.3)),
        tooltip=[selected_attrs[0], selected_attrs[1], 'N', alt.Tooltip("Taxa Favorável:Q", format=".1%")]
    ).properties(width=150, height=350)
    
    rule_inter = alt.Chart(pd.DataFrame({'mean': [global_favorable_rate]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='mean:Q')
    
    st.altair_chart((bar_inter + rule_inter), use_container_width=False)
    st.caption("Barras translúcidas indicam N < 100 (subgrupo inviável estatisticamente). Linha vermelha = Média Global.")

else:
    st.subheader("Auditoria de Justice Gerrymandering (3+ Atributos)")
    
    with st.spinner("Calculando métricas de Gerrymandering..."):
        audit_df = intersectional_audit_metrics(df, selected_attrs, target_col, favorable_val)
    
    # Estilização condicional
    def color_verdict(val):
        color = 'green' if val == 'Ok' else 'orange' if 'Inviable' in val else 'red'
        return f'color: {color}'
        
    st.dataframe(
        audit_df.style.applymap(color_verdict, subset=['Audit Veredict'])
              .format({'Favorable Rate': '{:.2%}', 'Real Gap (Intersectional)': '{:.4f}', 
                       'Expected Gap (Max Marginal)': '{:.4f}', 'Hidden Bias (Surplus)': '{:.4f}',
                       'Priority Score': '{:.2f}', 'Pre-training DI': '{:.4f}'}),
        use_container_width=True
    )
    
    st.caption("Subgrupos com N < 100 são marcados como 'Inviable'. DI pré-treinamento indica Pre-training Disparate Impact contra a média global.")
    
    # Exportação Long CSV
    long_csv = audit_df.melt(id_vars=['Subgroup', 'N'], 
                             value_vars=['Favorable Rate', 'Real Gap (Intersectional)', 'Expected Gap (Max Marginal)', 'Hidden Bias (Surplus)', 'Priority Score', 'Pre-training DI'],
                             var_name='Metrica', value_name='Valor')
    long_csv.insert(0, 'Dataset', dataset_name)
    
    st.download_button(
        label="📥 Exportar Dados (CSV Longo Consolidado)",
        data=long_csv.to_csv(index=False).encode('utf-8'),
        file_name=f"{dataset_name}_intersectional_audit_long.csv",
        mime="text/csv"
    )
