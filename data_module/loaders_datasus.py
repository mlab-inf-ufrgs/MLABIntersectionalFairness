import pandas as pd
import streamlit as st
import os

@st.cache_data(show_spinner=False)
def load_and_preprocess_sih(ufs=['Todos']):
    """
    Carrega amostra real do SIH (DATASUS).
    Alvo: desfecho (Alta vs Óbito)
    Atributos: sexo, raca_cor
    """
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'sih_processed.parquet')
    if os.path.exists(path):
        df = pd.read_parquet(path)
        if 'Todos' not in ufs and len(ufs) > 0:
            df = df[df['uf'].isin(ufs)]
        return df
    
    # Fallback explícito para evitar confusão com dados reais
    st.error("⚠️ Base SIH real não encontrada! Execute os scripts de coleta para baixar os dados.")
    return pd.DataFrame(columns=['sexo', 'raca_cor', 'desfecho', 'uf'])

@st.cache_data(show_spinner=False)
def load_and_preprocess_sim(ufs=['Todos']):
    """
    Carrega amostra real do SIM (DATASUS).
    """
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'sim_processed.parquet')
    if os.path.exists(path):
        df = pd.read_parquet(path)
        if 'Todos' not in ufs and len(ufs) > 0:
            df = df[df['uf'].isin(ufs)]
        return df
    
    # Fallback explícito para evitar confusão com dados reais
    st.error("⚠️ Base SIM real não encontrada! Execute os scripts de coleta para baixar os dados.")
    return pd.DataFrame(columns=['sexo', 'raca_cor', 'tipo_obito', 'uf'])

@st.cache_data(show_spinner=False)
def load_and_preprocess_sinasc(ufs=['Todos']):
    """
    Carrega amostra real do SINASC (DATASUS).
    """
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'sinasc_processed.parquet')
    if os.path.exists(path):
        df = pd.read_parquet(path)
        if 'Todos' not in ufs and len(ufs) > 0:
            df = df[df['uf'].isin(ufs)]
        return df
        
    # Fallback explícito para evitar confusão com dados reais
    st.error("⚠️ Base SINASC real não encontrada! Execute os scripts de coleta para baixar os dados.")
    return pd.DataFrame(columns=['raca_cor_mae', 'idade_mae', 'escolaridade_mae', 'desfecho_nascimento', 'uf'])

@st.cache_data(show_spinner=False)
def load_and_preprocess_sinan(ufs=['Todos']):
    """
    Carrega amostra real do SINAN (DATASUS).
    """
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'sinan_processed.parquet')
    if os.path.exists(path):
        df = pd.read_parquet(path)
        if 'Todos' not in ufs and len(ufs) > 0:
            df = df[df['uf'].isin(ufs)]
        return df
        
    # Fallback explícito para evitar confusão com dados reais
    st.error("⚠️ Base SINAN real não encontrada! Execute os scripts de coleta para baixar os dados.")
    return pd.DataFrame(columns=['sexo', 'raca_cor', 'escolaridade', 'evolucao_caso', 'uf'])
