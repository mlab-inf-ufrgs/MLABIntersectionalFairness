import pandas as pd
import streamlit as st

@st.cache_data(show_spinner="Baixando/Processando dados do ACS (Folktables)... Pode levar um minuto no primeiro acesso.")
def load_and_preprocess_folktables_income():
    """
    Carrega e pré-processa o dataset ACSIncome do pacote folktables.
    Substitui o antigo Adult dataset (Censo 1994) por dados contemporâneos (ACS).
    Para manter a performance do painel, baixamos uma amostra grande (ex: California) 
    ou o país inteiro se preferível, mas 'CA' já tem ~400k linhas o que é enorme.
    Vamos usar 'CA' por padrão para evitar travamentos em máquinas locais.
    """
    try:
        from folktables import ACSDataSource, ACSIncome
    except ImportError:
        st.error("Pacote 'folktables' não está instalado. Rode 'pip install folktables'.")
        return pd.DataFrame()

    # Baixa dados de 2018 (padrão robusto e amplamente usado em papers) 
    # Usando CA (California) para manter uma amostra grande (~370k) mas performática.
    # Se quiser o país todo, states=None (pode demorar bastante a baixar e consumir RAM).
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    
    # df_to_pandas retorna features, target e group
    features, label, _ = ACSIncome.df_to_pandas(acs_data)
    
    # Junta features e label
    df = features.copy()
    df['PINCP'] = label['PINCP'].astype(int) # Alvo: 1 se renda > 50k, 0 caso contrário
    
    # --- PRÉ-PROCESSAMENTO: Mapeamento de Códigos para Strings Legíveis ---
    
    # 1. SEX (1: Male, 2: Female)
    df['SEX'] = df['SEX'].map({1.0: 'Male', 2.0: 'Female'})
    
    # 2. RAC1P (Raça - Padrão folktables ACS)
    # 1: White, 2: Black, 6: Asian, 8: Other, 9: Two or more
    race_map = {
        1.0: 'White',
        2.0: 'Black',
        3.0: 'American Indian',
        4.0: 'Alaska Native',
        5.0: 'American Indian and/or Alaska Native',
        6.0: 'Asian',
        7.0: 'Native Hawaiian and Other Pacific Islander',
        8.0: 'Some Other Race',
        9.0: 'Two or More Races'
    }
    df['RAC1P'] = df['RAC1P'].map(race_map)
    # Filtra para manter grupos mais representativos para a análise simplificada (opcional, mas recomendado)
    # Vamos manter White, Black, Asian e Hispanic (que vem do PUMA, mas vamos focar em Race)
    
    # 3. AGEP (Idade contínua) -> Discretizando como no Adult
    bins = [18, 35, 50, 65, 100]
    labels = ['Young', 'Middle-aged', 'Senior', 'Elderly']
    df['AGEP_Group'] = pd.cut(df['AGEP'], bins=bins, labels=labels)
    df = df.drop(columns=['AGEP'])
    
    # 4. SCHL (Escolaridade)
    # 1-15: Schooling, 16: High School, 17-20: College, 21: Bachelors, 22-24: Graduate
    def map_education(x):
        if x <= 15: return 'Schooling'
        elif x == 16: return 'High School'
        elif 17 <= x <= 20: return 'Associate/College'
        elif x == 21: return 'Bachelors'
        elif x >= 22: return 'Graduate Degree'
        return 'Unknown'
        
    df['SCHL'] = df['SCHL'].apply(map_education)
    
    # Drop rows with NaNs in key protected attributes
    df = df.dropna(subset=['SEX', 'RAC1P', 'AGEP_Group', 'SCHL', 'PINCP'])
    
    return df


@st.cache_data(show_spinner="Baixando/Processando dados do ACS Public Coverage...")
def load_and_preprocess_folktables_coverage():
    """
    Carrega o dataset ACSPublicCoverage do pacote folktables.
    Prevê se a pessoa possui seguro de saúde público.
    """
    try:
        from folktables import ACSDataSource, ACSPublicCoverage
    except ImportError:
        st.error("Pacote 'folktables' não está instalado.")
        return pd.DataFrame()

    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    
    features, label, _ = ACSPublicCoverage.df_to_pandas(acs_data)
    
    df = features.copy()
    df['PUBCOV'] = label['PUBCOV'].astype(int) # Alvo: 1 se tem cobertura, 0 se não tem
    
    # Pré-processamento
    df['SEX'] = df['SEX'].map({1.0: 'Male', 2.0: 'Female'})
    
    race_map = {
        1.0: 'White', 2.0: 'Black', 3.0: 'Am. Indian', 6.0: 'Asian',
        8.0: 'Other', 9.0: 'Two or More'
    }
    df['RAC1P'] = df['RAC1P'].map(race_map).fillna('Other')
    
    bins = [-1, 18, 35, 50, 65, 120]
    labels = ['Minor', 'Young', 'Middle-aged', 'Senior', 'Elderly']
    df['AGEP_Group'] = pd.cut(df['AGEP'], bins=bins, labels=labels)
    df = df.drop(columns=['AGEP'])
    
    df = df.dropna(subset=['SEX', 'RAC1P', 'AGEP_Group', 'PUBCOV'])
    
    return df
