import streamlit as st

st.set_page_config(
    page_title="Diagnóstico de Viés Interseccional",
    page_icon="⚖️",
    layout="wide"
)

st.title("Análise de impacto cumulativo do viés social")

st.markdown("""
**Navegação:**
- **Dados**: Diagnóstico pré-treinamento e Análise Exploratória (EDA). Avalia o viés inerente aos dados e auditoria de Gerrymandering.
- **Modelos**: (Em Breve) Avaliação de disparidades e trade-offs de justiça algorítmica pós-treinamento.

Selecione a aba desejada no menu lateral.
""")
