import streamlit as st
from utils.i18n import t

if "lang" not in st.session_state:
    st.session_state.lang = "PT"

st.set_page_config(
    page_title=t("home_page_title"),
    page_icon="⚖️",
    layout="wide"
)

st.sidebar.selectbox("Idioma / Language", ["PT", "EN"], key="lang")

st.title(t("home_title"))

st.markdown(t("home_nav"))
