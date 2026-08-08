import streamlit as st
from utils.i18n import t

if "lang" not in st.session_state:
    st.session_state.lang = "PT"

st.set_page_config(
    page_title=t("models_page_title"), 
    page_icon="🤖"
)

st.sidebar.selectbox("Idioma / Language", ["PT", "EN"], key="lang")

st.title(t("models_title"))

st.info(t("models_info"))
