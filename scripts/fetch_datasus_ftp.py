import pandas as pd
import os
from ftplib import FTP
import pyreaddbc
from dbfread import DBF

def download_file(ftp, ftp_dir, filename, out_dir='data/raw/'):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    print(f"Baixando {filename} do FTP DATASUS...")
    ftp.cwd('/' + ftp_dir)
    with open(out_path, 'wb') as f:
        ftp.retrbinary(f'RETR {filename}', f.write)
    return out_path

def convert_dbc_to_df(dbc_path):
    print(f"Convertendo {dbc_path} para DBF e carregando no Pandas...")
    dbf_path = dbc_path.replace('.dbc', '.dbf')
    pyreaddbc.dbc2dbf(dbc_path, dbf_path)
    
    table = DBF(dbf_path, load=True, encoding='latin-1')
    df = pd.DataFrame(iter(table))
    
    os.remove(dbc_path)
    os.remove(dbf_path)
    return df

def process_sih(df):
    print("Processando SIH...")
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

def process_sim(df):
    print("Processando SIM...")
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

def process_sinasc(df):
    print("Processando SINASC...")
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

if __name__ == '__main__':
    ftp = FTP('ftp.datasus.gov.br')
    ftp.login()
    
    # SIH
    sih_dbc = download_file(ftp, 'dissemin/publicos/SIHSUS/200801_/Dados', 'RDSP2301.dbc')
    df_sih = convert_dbc_to_df(sih_dbc)
    process_sih(df_sih)
    
    # SIM
    sim_dbc = download_file(ftp, 'dissemin/publicos/SIM/CID10/DORES', 'DOSP2022.dbc')
    df_sim = convert_dbc_to_df(sim_dbc)
    process_sim(df_sim)
    
    # SINASC
    sinasc_dbc = download_file(ftp, 'dissemin/publicos/SINASC/1996_/Dados/DNRES', 'DNSP2022.dbc')
    df_sinasc = convert_dbc_to_df(sinasc_dbc)
    process_sinasc(df_sinasc)
    
    ftp.quit()
    print("Sucesso! Todos os datasets foram salvos em data/processed/")
