import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data(show_spinner=False)
def load_and_preprocess_cadunico():
    """
    Simula e pré-processa a amostra SAGI/MDS do CadÚnico.
    Utiliza matemática estatística para refletir as distribuições demográficas
    reais do IPEA (viés socioeconômico racial) como *fallback* na ausência 
    de pacote nativo sem chaves de API pagas do GCP.
    
    Alvo: pobreza_extrema (Renda per capita <= R$ 77,00 = 1, > R$ 77,00 = 0)
    Atributos: raca_cor, sexo, escolaridade
    """
    np.random.seed(42) # Consistência na banca
    
    n_samples = 4000
    
    # Sexo no CadÚnico (chefes de família costumam ser a imensa maioria mulheres)
    # Ref: SAGI aponta ~85% a 90% das responsáveis familiares sendo mulheres
    sexo = np.random.choice(['Feminino', 'Masculino'], size=n_samples, p=[0.88, 0.12])
    
    # Raça/Cor no CadÚnico 
    # Ref: Maioria expressiva de Pretos e Pardos no CadÚnico (aprox 75%)
    raca_cor = np.random.choice(['Parda', 'Preta', 'Branca', 'Amarela', 'Indígena'], size=n_samples, p=[0.55, 0.20, 0.22, 0.01, 0.02])
    
    # Escolaridade 
    escolaridade = np.random.choice(
        ['Sem Instrução', 'Fundamental', 'Médio', 'Superior'], 
        size=n_samples, p=[0.30, 0.45, 0.23, 0.02]
    )
    
    # Renda per capita e Pobreza Extrema (Limiar clássico de R$ 77.00 na época do estudo base)
    renda = np.zeros(n_samples)
    
    # APLICANDO VIÉS ESTATÍSTICO REAL:
    # A probabilidade de cair na pobreza extrema é muito maior para pessoas Negras e com Baixa Escolaridade
    for i in range(n_samples):
        base_income = np.random.normal(loc=120, scale=40)
        
        # Penalidades históricas refletidas em dados estruturais
        if raca_cor[i] in ['Preta', 'Parda']:
            base_income -= 40
        if raca_cor[i] == 'Branca':
            base_income += 50
            
        if escolaridade[i] == 'Sem Instrução':
            base_income -= 30
        if escolaridade[i] == 'Superior':
            base_income += 250
            
        if sexo[i] == 'Feminino': # Chefes de família mono-parentais (mulheres) sofrem mais pressão de renda per capita
            base_income -= 15
            
        renda[i] = max(0, base_income) # Renda não pode ser negativa
        
    df = pd.DataFrame({
        'raca_cor': raca_cor,
        'sexo': sexo,
        'escolaridade': escolaridade,
        'renda_per_capita': renda
    })
    
    LIMIAR_POBREZA_EXTREMA = 77.00
    df['pobreza_extrema'] = df['renda_per_capita'].apply(lambda x: 1 if x <= LIMIAR_POBREZA_EXTREMA else 0)
    
    df = df.drop(columns=['renda_per_capita'])
    
    # Mapeando target para a lógica de favorable_val no painel (que geralmente é o não ser pobre ou a classe de predição do assistente)
    # Para o painel de EDA de bias, 1 costuma ser "vantagem" ou a classe primária. Deixaremos 1 como Extrema Pobreza por ser o target principal do MDS, mas 'favorable_val' será 0 (Fora da Extrema Pobreza).
    
    return df

