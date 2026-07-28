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
    # race: keep only White/Black
    df = df[df['race'].isin(['White', 'Black'])].copy()
    
    # age_group: discretize age, filter 19-65
    df = df[(df['age'] >= 19) & (df['age'] <= 65)].copy()
    bins = [18, 35, 50, 65]
    labels = ['Young', 'Middle-aged', 'Senior']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
    df = df.drop(columns=['age'])
    
    # education_group: regroup
    edu_map = {
        'Bachelors': 'Bachelors',
        'Some-college': 'Associate/College',
        'Assoc-acdm': 'Associate/College',
        'Assoc-voc': 'Associate/College',
        'Prof-school': 'Graduate Degree',
        'Masters': 'Graduate Degree',
        'Doctorate': 'Graduate Degree',
        '11th': 'Schooling',
        '9th': 'Schooling',
        '7th-8th': 'Schooling',
        '12th': 'Schooling',
        '1st-4th': 'Schooling',
        '10th': 'Schooling',
        '5th-6th': 'Schooling',
        'Preschool': 'Schooling',
        'HS-grad': 'Schooling'
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
