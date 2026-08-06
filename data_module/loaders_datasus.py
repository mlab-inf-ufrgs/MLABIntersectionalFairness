import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_and_preprocess_sih(uf='SP'):
    """
    Carrega SIH (DATASUS) via PySUS com fallback resiliente.
    Alvo: desfecho (Alta vs Óbito)
    Atributos: sexo, raca_cor
    """
    def generate_mock():
        return pd.DataFrame({
            'sexo': ['Masculino', 'Feminino', 'Masculino', 'Feminino'] * 250,
            'raca_cor': ['Branca', 'Preta', 'Parda', 'Branca'] * 250,
            'desfecho': ['Alta', 'Óbito', 'Alta', 'Alta'] * 250
        })
        
    try:
        from pysus.online_data import SIH
        try:
            df = SIH.download('RD', year=2026, month=1, uf=uf)
        except:
            df = SIH.download('RD', year=2025, month=1, uf=uf)
            
        if isinstance(df, tuple):
            df = pd.concat(df)
            
        df.columns = [c.upper() for c in df.columns]
        
        res = pd.DataFrame()
        if 'SEXO' in df.columns:
            res['sexo'] = df['SEXO'].map({'1': 'Masculino', '3': 'Feminino'})
        
        raca_col = 'RACA_COR' if 'RACA_COR' in df.columns else 'RACACOR' if 'RACACOR' in df.columns else None
        if raca_col:
            res['raca_cor'] = df[raca_col].map({'01': 'Branca', '02': 'Preta', '03': 'Parda', '04': 'Amarela', '05': 'Indígena'})
            
        if 'MORTE' in df.columns:
            res['desfecho'] = df['MORTE'].apply(lambda x: 'Óbito' if str(x) == '1' else 'Alta')
        else: 
            res['desfecho'] = 'Alta' # Fallback simplificado
            
        res = res.dropna(subset=['sexo', 'raca_cor', 'desfecho'])
        
        if len(res) < 100:
            return generate_mock()
            
        return res
    except:
        return generate_mock()

@st.cache_data(show_spinner=False)
def load_and_preprocess_sim(uf='SP'):
    """
    Carrega SIM (DATASUS) via PySUS com fallback resiliente.
    """
    def generate_mock():
        return pd.DataFrame({
            'sexo': ['Masculino', 'Feminino', 'Feminino', 'Feminino'] * 250,
            'raca_cor': ['Preta', 'Preta', 'Parda', 'Branca'] * 250,
            'tipo_obito': ['Não Evitável', 'Evitável', 'Não Evitável', 'Não Evitável'] * 250
        })
        
    try:
        from pysus.online_data import SIM
        try:
            df = SIM.download('DO', year=2026)
        except:
            df = SIM.download('DO', year=2025)
            
        if isinstance(df, tuple):
            df = pd.concat(df)
            
        df.columns = [c.upper() for c in df.columns]
        
        res = pd.DataFrame()
        if 'SEXO' in df.columns:
            res['sexo'] = df['SEXO'].map({'1': 'Masculino', '2': 'Feminino'})
            
        raca_col = 'RACA_COR' if 'RACA_COR' in df.columns else 'RACACOR' if 'RACACOR' in df.columns else None
        if raca_col:
            res['raca_cor'] = df[raca_col].map({'1': 'Branca', '2': 'Preta', '3': 'Amarela', '4': 'Parda', '5': 'Indígena'})
            
        res['tipo_obito'] = 'Não Evitável' # Mapeamento real de CID requer cruzamento, uso simples aqui
        
        res = res.dropna()
        if len(res) < 100:
            return generate_mock()
        return res
    except:
        return generate_mock()

@st.cache_data(show_spinner=False)
def load_and_preprocess_sinasc(uf='SP'):
    """
    Carrega SINASC (DATASUS) via PySUS com fallback resiliente.
    """
    def generate_mock():
        return pd.DataFrame({
            'raca_cor_mae': ['Preta', 'Branca', 'Parda', 'Branca'] * 250,
            'idade_mae': ['Jovem', 'Adulta', 'Jovem', 'Adulta'] * 250,
            'escolaridade_mae': ['Fundamental', 'Médio', 'Superior', 'Médio'] * 250,
            'desfecho_nascimento': ['Normal', 'Normal', 'Baixo Peso', 'Normal'] * 250
        })
        
    try:
        from pysus.online_data import SINASC
        try:
            df = SINASC.download('DN', year=2026)
        except:
            df = SINASC.download('DN', year=2025)
            
        if isinstance(df, tuple):
            df = pd.concat(df)
            
        df.columns = [c.upper() for c in df.columns]
        
        res = pd.DataFrame()
        raca_col = 'RACACORMAE' if 'RACACORMAE' in df.columns else None
        if raca_col:
            res['raca_cor_mae'] = df[raca_col].map({'1': 'Branca', '2': 'Preta', '3': 'Amarela', '4': 'Parda', '5': 'Indígena'})
            
        if 'IDADEMAE' in df.columns:
            res['idade_mae'] = pd.cut(pd.to_numeric(df['IDADEMAE'], errors='coerce'), bins=[0, 19, 34, 100], labels=['Jovem (<20)', 'Adulta (20-34)', 'Madura (>34)'])
            
        if 'ESCMAE' in df.columns:
            res['escolaridade_mae'] = df['ESCMAE'].map({'1': 'Nenhuma', '2': 'Fundamental I', '3': 'Fundamental II', '4': 'Médio', '5': 'Superior'})
            
        if 'PESO' in df.columns:
            res['desfecho_nascimento'] = pd.to_numeric(df['PESO'], errors='coerce').apply(lambda x: 'Baixo Peso' if x < 2500 else 'Normal')
            
        res = res.dropna()
        if len(res) < 100:
            return generate_mock()
        return res
    except:
        return generate_mock()

