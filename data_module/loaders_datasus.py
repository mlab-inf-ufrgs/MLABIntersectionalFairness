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
    
    # Fallback apenas se o arquivo ainda não existir
    return pd.DataFrame({
        'sexo': ['Masculino', 'Feminino', 'Masculino', 'Feminino'] * 250,
        'raca_cor': ['Branca', 'Preta', 'Parda', 'Branca'] * 250,
        'desfecho': ['Alta', 'Óbito', 'Alta', 'Alta'] * 250
    })

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
    
    return pd.DataFrame({
        'sexo': ['Masculino', 'Feminino', 'Feminino', 'Feminino'] * 250,
        'raca_cor': ['Preta', 'Preta', 'Parda', 'Branca'] * 250,
        'tipo_obito': ['Não Evitável', 'Evitável', 'Não Evitável', 'Não Evitável'] * 250
    })

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
        
    return pd.DataFrame({
        'raca_cor_mae': ['Preta', 'Branca', 'Parda', 'Branca'] * 250,
        'idade_mae': ['Jovem', 'Adulta', 'Jovem', 'Adulta'] * 250,
        'escolaridade_mae': ['Fundamental', 'Médio', 'Superior', 'Médio'] * 250,
        'desfecho_nascimento': ['Normal', 'Normal', 'Baixo Peso', 'Normal'] * 250
    })

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
        
    return pd.DataFrame({
        'sexo': ['Masculino', 'Feminino', 'Masculino', 'Feminino'] * 250,
        'raca_cor': ['Preta', 'Branca', 'Parda', 'Branca'] * 250,
        'escolaridade': ['Fundamental', 'Médio', 'Superior', 'Médio'] * 250,
        'evolucao_caso': ['Cura', 'Óbito', 'Cura', 'Cura'] * 250
    })
