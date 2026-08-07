import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_and_preprocess_adult():
    """
    Loads and preprocesses the Adult dataset for the intersectional fairness experiment.
    
    Target: 'income' (binary: >50K=1, <=50K=0)
    Privileged Groups:
        - sex: Male
        - race: White
        - age_group: Senior
        - education_group: Graduate Degree
        - relationship: Husband
    """
    # Fetch dataset
    adult = fetch_ucirepo(id=2) 
    
    # data (as pandas dataframes) 
    X = adult.data.features 
    y = adult.data.targets 
    
    df = pd.concat([X, y], axis=1)
    
    # 1. Target Binarization
    # UCI repo sometimes has '>50K.' and '<=50K.'
    df['income'] = df['income'].str.replace('.', '', regex=False)
    df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    
    # 2. Filtering & Binning
    # sex: Male/Female
    if 'sex' in df.columns:
        df['sex'] = df['sex'].map({'Male': 'Masculino', 'Female': 'Feminino'})
    
    # race: keep only White/Black
    df = df[df['race'].isin(['White', 'Black'])].copy()
    df['race'] = df['race'].map({'White': 'Branco', 'Black': 'Negro'})
    
    # age_group: discretize age, filter 19-65
    df = df[(df['age'] >= 19) & (df['age'] <= 65)].copy()
    bins = [18, 35, 50, 65]
    labels = ['Jovem', 'Meia-idade', 'Sênior']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
    df = df.drop(columns=['age'])
    
    # education_group: regroup
    edu_map = {
        'Bachelors': 'Bacharelado',
        'Some-college': 'Superior Incompleto',
        'Assoc-acdm': 'Superior Incompleto',
        'Assoc-voc': 'Superior Incompleto',
        'Prof-school': 'Pós-graduação',
        'Masters': 'Pós-graduação',
        'Doctorate': 'Pós-graduação',
        '11th': 'Ensino Básico/Médio',
        '9th': 'Ensino Básico/Médio',
        '7th-8th': 'Ensino Básico/Médio',
        '12th': 'Ensino Básico/Médio',
        '1st-4th': 'Ensino Básico/Médio',
        '10th': 'Ensino Básico/Médio',
        '5th-6th': 'Ensino Básico/Médio',
        'Preschool': 'Ensino Básico/Médio',
        'HS-grad': 'Ensino Básico/Médio'
    }
    df['education_group'] = df['education'].map(edu_map)
    df = df.drop(columns=['education', 'education-num'])
    
    # relationship: keep original, remove rare
    rare_rels = [] # if any to remove
    df = df[~df['relationship'].isin(rare_rels)].copy()
    
    # 3. Drop specified columns
    cols_to_drop = ['fnlwgt', 'marital-status', 'native-country']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # Clean up NaNs
    df = df.dropna().reset_index(drop=True)
    
    return df
