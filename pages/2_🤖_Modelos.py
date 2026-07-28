import streamlit as st

st.set_page_config(page_title="Modelos (Em Breve)", page_icon="🤖")

st.title("Modelos Pós-Treinamento")

st.info("""
**Aba em desenvolvimento.**

Nesta etapa futura, adicionaremos os resultados de modelos (RF, GBM, MLP) treinados sobre os datasets selecionados.
As métricas de auditoria incluirão:
- Sensitivity Gap
- Average Absolute Odds Difference (AAOD)
- Disparate Impact (Pós-treinamento)

Por enquanto, utilize a aba **Dados** para análise exploratória e diagnóstico de viés inerente (pré-treinamento).
""")
