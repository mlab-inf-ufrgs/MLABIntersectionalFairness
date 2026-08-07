import pandas as pd
import urllib.request
import os

def load_and_preprocess_compas():
    """
    Loads and preprocesses the COMPAS dataset.
    Target: 'two_year_recid' (binary: 1=recidivate, 0=did not)
    
    Privileged Groups:
        - sex: Female
        - race: Caucasian
        - age_group: Senior
        
    Note: priors_count_group and c_charge_degree are features but NEVER protected axes.
    """
    # Download if not exists
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    local_path = "compas-scores-two-years.csv"
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(url, local_path)
        
    df = pd.read_csv(local_path)
    
    # 1. Target Binarization
    # two_year_recid is already 0 or 1
    
    # 2. Filtering & Binning
    # sex: Male/Female
    if 'sex' in df.columns:
        df['sex'] = df['sex'].map({'Male': 'Masculino', 'Female': 'Feminino'})
    
    # race: Caucasian, African-American, Hispanic
    df = df[df['race'].isin(['Caucasian', 'African-American', 'Hispanic'])].copy()
    df['race'] = df['race'].map({'Caucasian': 'Branco', 'African-American': 'Negro', 'Hispanic': 'Hispânico'})
    
    # age_group: discretize 19-65
    df = df[(df['age'] >= 19) & (df['age'] <= 65)].copy()
    bins = [18, 35, 50, 65]
    labels = ['Jovem', 'Meia-idade', 'Sênior']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
    df = df.drop(columns=['age'])
    
    # priors_count: mantido como variável contínua original
    
    # c_charge_degree: Felony/Misdemeanor
    df = df[df['c_charge_degree'].isin(['F', 'M'])].copy()
    df['c_charge_degree'] = df['c_charge_degree'].map({'F': 'Crime grave', 'M': 'Contravenção'})
    
    # 3. Keep only relevant features and target to prevent data leakage
    features_to_keep = ['two_year_recid', 'sex', 'race', 'age_group', 'priors_count', 'c_charge_degree']
    df = df[features_to_keep].dropna().reset_index(drop=True)
    
    return df
