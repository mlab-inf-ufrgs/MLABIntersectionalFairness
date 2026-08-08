import pandas as pd
import os
from pysus import sih, sim, sinasc

def fetch_sih():
    print("Baixando SIH (SP, Janeiro de 2023)...")
    try:
        paths = sih(state='SP', year=2023, month=1, group='RD')
        df = pd.read_parquet(paths)
        df.columns = [c.upper() for c in df.columns]
        
        res = pd.DataFrame()
        if 'SEXO' in df.columns:
            res['sexo'] = df['SEXO'].map({'1': 'Masculino', '3': 'Feminino'})
        else:
            res['sexo'] = 'Ignorado'
        
        raca_col = 'RACA_COR' if 'RACA_COR' in df.columns else 'RACACOR' if 'RACACOR' in df.columns else None
        if raca_col:
            res['raca_cor'] = df[raca_col].map({'01': 'Branca', '02': 'Preta', '03': 'Parda', '04': 'Amarela', '05': 'Indígena'})
        else:
            res['raca_cor'] = 'Ignorado'
            
        if 'MORTE' in df.columns:
            res['desfecho'] = df['MORTE'].apply(lambda x: 'Óbito' if str(x) == '1' else 'Alta')
        else: 
            res['desfecho'] = 'Alta'
            
        res = res.dropna(subset=['sexo', 'raca_cor', 'desfecho'])
        
        os.makedirs('data/processed', exist_ok=True)
        res.to_parquet('data/processed/sih_processed.parquet')
        print(f"SIH salvo com {len(res)} registros.")
    except Exception as e:
        print(f"Erro no SIH: {e}")

def fetch_sim():
    print("Baixando SIM (SP, 2022)...")
    try:
        paths = sim(state='SP', year=2022, group='DO')
        df = pd.read_parquet(paths)
        df.columns = [c.upper() for c in df.columns]
        
        res = pd.DataFrame()
        if 'SEXO' in df.columns:
            res['sexo'] = df['SEXO'].map({'1': 'Masculino', '2': 'Feminino'})
        else:
            res['sexo'] = 'Ignorado'
            
        raca_col = 'RACA_COR' if 'RACA_COR' in df.columns else 'RACACOR' if 'RACACOR' in df.columns else None
        if raca_col:
            res['raca_cor'] = df[raca_col].map({'1': 'Branca', '2': 'Preta', '3': 'Amarela', '4': 'Parda', '5': 'Indígena'})
        else:
            res['raca_cor'] = 'Ignorado'
            
        res['tipo_obito'] = 'Não Evitável' 
        
        res = res.dropna(subset=['sexo', 'raca_cor'])
        os.makedirs('data/processed', exist_ok=True)
        res.to_parquet('data/processed/sim_processed.parquet')
        print(f"SIM salvo com {len(res)} registros.")
    except Exception as e:
        print(f"Erro no SIM: {e}")

def fetch_sinasc():
    print("Baixando SINASC (SP, 2022)...")
    try:
        paths = sinasc(state='SP', year=2022, group='DN')
        df = pd.read_parquet(paths)
        df.columns = [c.upper() for c in df.columns]
        
        res = pd.DataFrame()
        raca_col = 'RACACORMAE' if 'RACACORMAE' in df.columns else None
        if raca_col:
            res['raca_cor_mae'] = df[raca_col].map({'1': 'Branca', '2': 'Preta', '3': 'Amarela', '4': 'Parda', '5': 'Indígena'})
        else:
            res['raca_cor_mae'] = 'Ignorado'
            
        if 'IDADEMAE' in df.columns:
            res['idade_mae'] = pd.cut(pd.to_numeric(df['IDADEMAE'], errors='coerce'), bins=[0, 19, 34, 100], labels=['Jovem (<20)', 'Adulta (20-34)', 'Madura (>34)'])
        else:
            res['idade_mae'] = 'Adulta (20-34)'
            
        if 'ESCMAE' in df.columns:
            res['escolaridade_mae'] = df['ESCMAE'].map({'1': 'Nenhuma', '2': 'Fundamental I', '3': 'Fundamental II', '4': 'Médio', '5': 'Superior'})
        else:
            res['escolaridade_mae'] = 'Ignorado'
            
        if 'PESO' in df.columns:
            res['desfecho_nascimento'] = pd.to_numeric(df['PESO'], errors='coerce').apply(lambda x: 'Baixo Peso' if x < 2500 else 'Normal')
        else:
            res['desfecho_nascimento'] = 'Normal'
            
        res = res.dropna()
        os.makedirs('data/processed', exist_ok=True)
        res.to_parquet('data/processed/sinasc_processed.parquet')
        print(f"SINASC salvo com {len(res)} registros.")
    except Exception as e:
        print(f"Erro no SINASC: {e}")

if __name__ == '__main__':
    fetch_sih()
    fetch_sim()
    fetch_sinasc()
