import pandas as pd
import os
from ftplib import FTP
import pyreaddbc
from dbfread import DBF
import itertools

def process_sinan_chunk(df):
    df.columns = [c.upper() for c in df.columns]
    res = pd.DataFrame()
    if 'CS_SEXO' in df.columns:
        res['sexo'] = df['CS_SEXO'].map({'M': 'Masculino', 'F': 'Feminino'})
    else:
        res['sexo'] = 'Ignorado'
        
    raca_col = 'CS_RACA' if 'CS_RACA' in df.columns else None
    if raca_col:
        res['raca_cor'] = df[raca_col].map({'1': 'Branca', '2': 'Preta', '3': 'Amarela', '4': 'Parda', '5': 'Indígena'})
    else:
        res['raca_cor'] = 'Ignorado'
        
    escolaridade_col = 'CS_ESCOL_N' if 'CS_ESCOL_N' in df.columns else None
    if escolaridade_col:
        res['escolaridade'] = df[escolaridade_col].map({
            '00': 'Nenhuma', '01': 'Fundamental I', '02': 'Fundamental II', 
            '03': 'Médio', '04': 'Superior', '05': 'Não se aplica'
        })
    else:
        res['escolaridade'] = 'Ignorado'
        
    if 'SG_UF_NOT' in df.columns:
        res['uf'] = df['SG_UF_NOT'].astype(str).apply(lambda x: x.zfill(2))
        # Map IBGE codes to UF abbreviations (since SG_UF_NOT is usually numeric IBGE code)
        uf_map = {
            '11': 'RO', '12': 'AC', '13': 'AM', '14': 'RR', '15': 'PA', '16': 'AP', '17': 'TO',
            '21': 'MA', '22': 'PI', '23': 'CE', '24': 'RN', '25': 'PB', '26': 'PE', '27': 'AL', '28': 'SE', '29': 'BA',
            '31': 'MG', '32': 'ES', '33': 'RJ', '35': 'SP',
            '41': 'PR', '42': 'SC', '43': 'RS',
            '50': 'MS', '51': 'MT', '52': 'GO', '53': 'DF'
        }
        res['uf'] = res['uf'].map(uf_map).fillna('Ignorado')
    else:
        res['uf'] = 'Ignorado'
        
    if 'EVOLUCAO' in df.columns:
        # 1-Cura, 2-Óbito por dengue, 3-Óbito por outras causas, 9-Ignorado
        res['evolucao_caso'] = df['EVOLUCAO'].map({'1': 'Cura', '2': 'Óbito', '3': 'Óbito', '9': 'Ignorado'})
    else: 
        res['evolucao_caso'] = 'Ignorado'
        
    res = res.dropna(subset=['sexo', 'raca_cor', 'evolucao_caso'])
    res = res[res['evolucao_caso'] != 'Ignorado']
    return res

def fetch_and_process_sinan(disease_prefix='DENG', year='22'):
    ftp = FTP('ftp.datasus.gov.br')
    ftp.login()
    filename = f"{disease_prefix}BR{year}.dbc"
    print(f"Baixando {filename} do FTP DATASUS...")
    
    out_dir = 'data/raw/'
    os.makedirs(out_dir, exist_ok=True)
    dbc_path = os.path.join(out_dir, filename)
    
    ftp.cwd('/dissemin/publicos/SINAN/DADOS/FINAIS')
    with open(dbc_path, 'wb') as f:
        ftp.retrbinary(f'RETR {filename}', f.write)
    ftp.quit()
    
    print(f"Convertendo {dbc_path} para DBF...")
    dbf_path = dbc_path.replace('.dbc', '.dbf')
    pyreaddbc.dbc2dbf(dbc_path, dbf_path)
    
    print("Processando dados (chunked)...")
    table = DBF(dbf_path, load=False, encoding='latin-1')
    
    chunk_size = 5000
    iterator = iter(table)
    chunks = []
    columns_to_keep = {'CS_SEXO', 'CS_RACA', 'CS_ESCOL_N', 'SG_UF_NOT', 'EVOLUCAO'}
    
    while True:
        chunk_raw = list(itertools.islice(iterator, chunk_size))
        if not chunk_raw:
            break
        
        chunk_filtered = [{k: v for k, v in row.items() if k in columns_to_keep} for row in chunk_raw]
        df_chunk = pd.DataFrame(chunk_filtered)
        
        processed_chunk = process_sinan_chunk(df_chunk)
        chunks.append(processed_chunk)
        print(f"Processado chunk de {len(chunk_raw)} registros -> Filtrado: {len(processed_chunk)}")
        
    if chunks:
        final_df = pd.concat(chunks, ignore_index=True)
        os.makedirs('data/processed', exist_ok=True)
        out_parquet = 'data/processed/sinan_processed.parquet'
        final_df.to_parquet(out_parquet)
        print(f"SINAN Salvo com {len(final_df)} registros em {out_parquet}")
    
    os.remove(dbc_path)
    os.remove(dbf_path)

if __name__ == '__main__':
    fetch_and_process_sinan()
