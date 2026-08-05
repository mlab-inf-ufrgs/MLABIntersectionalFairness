import os
import pandas as pd

# ==============================================================================
# Este script gera os arquivos .parquet localmente na pasta data/processed/
# Após gerar, faça o upload manual para o bucket: mlab-intersectionalfairness
# ==============================================================================

# Como este script será rodado na raiz do projeto, precisamos garantir que
# os módulos locais possam ser importados.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_module.adult import load_and_preprocess_adult
from data_module.compas import load_and_preprocess_compas
from data_module.dropout import load_and_preprocess_dropout
from data_module.intersectional_bias import load_and_preprocess_intersectional_bias
from data_module.loaders_folktables import load_and_preprocess_folktables_income, load_and_preprocess_folktables_coverage
# Obs: DATASUS/CadÚnico usam dados mockados, então não vamos fazer upload deles por enquanto,
# até que você integre os dados reais via PySUS.

DATASETS_TO_UPLOAD = {
    'adult_processed.parquet': load_and_preprocess_adult,
    'compas_processed.parquet': load_and_preprocess_compas,
    'dropout_processed.parquet': load_and_preprocess_dropout,
    'intersectional_bias_processed.parquet': load_and_preprocess_intersectional_bias,
    'acsincome_processed.parquet': load_and_preprocess_folktables_income,
    'acspubliccoverage_processed.parquet': load_and_preprocess_folktables_coverage,
}

def save_dataframe_locally(df, file_name):
    """Salva o DataFrame como Parquet em uma pasta local."""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, file_name)
    print(f"-> Salvando {file_name} em {file_path}...")
    
    # Salva o dataframe como parquet
    df.to_parquet(file_path, index=False, engine='pyarrow')
    
    # Pega o tamanho do arquivo gerado
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"   [OK] Arquivo gerado! Tamanho: {file_size_mb:.2f} MB")

def main():
    print("Iniciando ETL para gerar arquivos Parquet locais...")
    
    for file_name, loader_func in DATASETS_TO_UPLOAD.items():
        print(f"\nBaixando e pré-processando: {file_name}...")
        try:
            # Roda a função de loader
            df = loader_func()
            # Salva localmente
            save_dataframe_locally(df, file_name)
        except Exception as e:
            print(f"   [ERRO] Falha ao processar {file_name}: {e}")
            
    print("\nTodos os arquivos foram gerados com sucesso na pasta 'data/processed/'!")
    print("Por favor, faça o upload manual desses arquivos para o bucket no GCP pelo navegador.")

if __name__ == "__main__":
    main()
