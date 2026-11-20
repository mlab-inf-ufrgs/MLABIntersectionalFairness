import pandas as pd
import streamlit as st
import os

SIH_YEARS = [2024]
_SIH_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'thesis', 'sih_v2')
_SIH_PANEL_COLS = ['uf', 'sexo', 'raca_cor', 'faixa_etaria', 'desfecho']


@st.cache_data(show_spinner=False)
def sih_groups(year=2024):
    """Capítulos da CID-10 disponíveis no ano: DataFrame com capitulo_n, capitulo_cid, aihs_no_ano, mantidos."""
    path = os.path.join(_SIH_DIR, str(year), f'sih_v2_{year}_resumo.parquet')
    if not os.path.exists(path):
        return pd.DataFrame(columns=['capitulo_n', 'capitulo_cid', 'aihs_no_ano', 'mantidos', 'obito_pct'])
    resumo = pd.read_parquet(path)
    # 'incluido' = False para capítulos sem internações ou retirados da análise (ex.: XV, óbito materno raro).
    keep = resumo['incluido'] if 'incluido' in resumo else resumo['mantidos'] > 0
    return resumo[keep].reset_index(drop=True)


def load_and_preprocess_sih(ufs=['Todos'], year=2024, group=1):
    """
    Carrega as internações do SIH v2 (FTP do DATASUS) de um ano e de um capítulo da CID-10,
    um arquivo por capítulo em data/processed/thesis/sih_v2/{ano}/ (scripts/build_thesis_datasets.py).

    O ano inteiro tem ~11 milhões de internações; carregar um capítulo por vez, só com as colunas
    do painel e como category, mantém a memória dentro do limite do Streamlit Cloud.
    O arquivo da qualificação foi arquivado, sem alteração, em data/old/processed/sih_processed.parquet.
    """
    path = os.path.join(_SIH_DIR, str(int(year)), f'sih_v2_{int(year)}_cap{int(group):02d}.parquet')
    if os.path.exists(path):
        filters = [('uf', 'in', list(ufs))] if ufs and 'Todos' not in ufs else None
        return pd.read_parquet(path, columns=_SIH_PANEL_COLS, filters=filters)
    
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

SINASC_YEARS = [2022, 2023, 2024]
# Só as colunas que o painel usa: o arquivo do ano tem 27, mas o parquet lê apenas estas.
_SINASC_PANEL_COLS = ['uf', 'raca_cor_mae', 'idade_mae', 'escolaridade_mae', 'desfecho_nascimento']


def load_and_preprocess_sinasc(ufs=['Todos'], year=2022):
    """
    Carrega um ano do SINASC v2 (Portal de Dados Abertos do SUS), um arquivo por ano em
    data/processed/thesis/sinasc_v2/ (gerados por scripts/build_thesis_datasets.py).

    Para caber na memória do Streamlit Cloud: lê só as colunas do painel, filtra a UF na
    própria leitura do parquet e mantém as colunas como category (≈12 MB por ano).
    O arquivo da qualificação (data/processed/sinasc_processed.parquet) não é alterado.
    """
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'thesis', 'sinasc_v2',
                        f'sinasc_v2_{int(year)}.parquet')
    if os.path.exists(path):
        filters = [('uf', 'in', list(ufs))] if ufs and 'Todos' not in ufs else None
        return pd.read_parquet(path, columns=_SINASC_PANEL_COLS, filters=filters)
        
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
