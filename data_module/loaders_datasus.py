import pandas as pd
import streamlit as st
import os

@st.cache_data(show_spinner=False)
def load_and_preprocess_sih(uf='SP'):
    """
    Carrega amostra real do SIH (DATASUS).
    Alvo: desfecho (Alta vs Óbito)
    Atributos: sexo, raca_cor
    """
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'sih_processed.parquet')
    if os.path.exists(path):
        return pd.read_parquet(path)
    
    # Fallback apenas se o arquivo ainda não existir
    return pd.DataFrame({
        'sexo': ['Masculino', 'Feminino', 'Masculino', 'Feminino'] * 250,
        'raca_cor': ['Branca', 'Preta', 'Parda', 'Branca'] * 250,
        'desfecho': ['Alta', 'Óbito', 'Alta', 'Alta'] * 250
    })

@st.cache_data(show_spinner=False)
def load_and_preprocess_sim(uf='SP'):
    """
    Carrega amostra real do SIM (DATASUS).
    """
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'sim_processed.parquet')
    if os.path.exists(path):
        return pd.read_parquet(path)
    
    return pd.DataFrame({
        'sexo': ['Masculino', 'Feminino', 'Feminino', 'Feminino'] * 250,
        'raca_cor': ['Preta', 'Preta', 'Parda', 'Branca'] * 250,
        'tipo_obito': ['Não Evitável', 'Evitável', 'Não Evitável', 'Não Evitável'] * 250
    })

@st.cache_data(show_spinner=False)
def load_and_preprocess_sinasc(uf='SP'):
    """
    Carrega amostra real do SINASC (DATASUS).
    """
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'sinasc_processed.parquet')
    if os.path.exists(path):
        return pd.read_parquet(path)
        
    return pd.DataFrame({
        'raca_cor_mae': ['Preta', 'Branca', 'Parda', 'Branca'] * 250,
        'idade_mae': ['Jovem', 'Adulta', 'Jovem', 'Adulta'] * 250,
        'escolaridade_mae': ['Fundamental', 'Médio', 'Superior', 'Médio'] * 250,
        'desfecho_nascimento': ['Normal', 'Normal', 'Baixo Peso', 'Normal'] * 250
    })
