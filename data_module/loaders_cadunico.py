import pandas as pd

def load_and_preprocess_cadunico():
    """
    Simula/Carrega e pré-processa a amostra SAGI/MDS (2012-2016) do CadÚnico.
    Alvo: pobreza_extrema (Renda per capita <= R$ 77,00 = 1, > R$ 77,00 = 0)
    Atributos: raca_cor, sexo, escolaridade
    """
    # Em um cenário real, carregar os dados CSV/Parquet fornecidos
    print("Aviso: Carregando amostra local do CadÚnico...")
    
    # MOCK data return for structural execution
    # Renda simulada
    df = pd.DataFrame({
        'raca_cor': ['Branca', 'Preta', 'Parda', 'Branca'] * 250,
        'sexo': ['M', 'F', 'M', 'F'] * 250,
        'escolaridade': ['Fundamental', 'Médio', 'Sem Instrução', 'Superior'] * 250,
        'renda_per_capita': [150.0, 50.0, 70.0, 500.0] * 250
    })
    
    # Aplica o limiar oficial do MDS de R$ 77,00 para pobreza extrema no período 2014-2016
    LIMIAR_POBREZA_EXTREMA = 77.00
    df['pobreza_extrema'] = df['renda_per_capita'].apply(lambda x: 1 if x <= LIMIAR_POBREZA_EXTREMA else 0)
    
    df = df.drop(columns=['renda_per_capita'])
    df = df.dropna(subset=['raca_cor', 'sexo', 'escolaridade', 'pobreza_extrema'])
    
    return df
