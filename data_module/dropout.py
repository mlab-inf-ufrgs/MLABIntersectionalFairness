import pandas as pd
from ucimlrepo import fetch_ucirepo
import numpy as np

def load_and_preprocess_dropout():
    """
    Loads and preprocesses the Dropout dataset (UCI ID 697).
    Target: 'Target' (Dropout=0, Graduate/Enrolled=1)
    
    Privileged Groups:
        - Gender: Male 
          ** NOTE: NOT A BUG. Female retention is naturally higher here. 
          ** DI > 1 for Female vs Male is expected and intended for this experiment.
        - Age_Group: <=20
        - Mother_Qualification_Group: Higher Education
        - Attendance: Daytime (Daytime/Evening attendance)
        - Debtor: No
    """
    # Fetch dataset
    dropout = fetch_ucirepo(id=697) 
    X = dropout.data.features 
    y = dropout.data.targets 
    
    df = pd.concat([X, y], axis=1)
    
    # Clean hidden tabs in column names if any
    df.columns = df.columns.str.replace('\t', '').str.strip()
    
    # 1. Target Binarization
    df['Target'] = df['Target'].apply(lambda x: 0 if x == 'Dropout' else 1)
    
    # 2. Filtering & Binning
    # Gender is 1/0, map to Male/Female (Usually 1 is Male, 0 is Female in this dataset)
    df['Gender'] = df['Gender'].map({1: 'Male', 0: 'Female'})
    
    # Age_Group: <=20, 21-25, >25 (filter > 50 as outliers)
    df = df[df['Age at enrollment'] <= 50].copy()
    bins = [0, 20, 25, 50]
    labels = ['<=20', '21-25', '>25']
    df['Age_Group'] = pd.cut(df['Age at enrollment'], bins=bins, labels=labels)
    
    # Qualifications mapping
    def map_qual(x):
        # Simplistic mapping based on general structure of UCI dataset
        # 1: Secondary, 2-6: Higher, else: Basic/Other
        if x == 1:
            return 'Secondary Education'
        elif 2 <= x <= 6:
            return 'Higher Education'
        else:
            return 'Basic/Other'
            
    df['Mother_Qualification_Group'] = df['Mother\'s qualification'].apply(map_qual)
    df['Father_Qualification_Group'] = df['Father\'s qualification'].apply(map_qual)
    
    # Nacionality (1 is usually Portuguese)
    df['Nacionality_Group'] = df['Nacionality'].apply(lambda x: 'Portuguese' if x == 1 else 'International')
    
    # Attendance (1 is Daytime, 0 is Evening)
    if 'Daytime/evening attendance\t' in df.columns:
        att_col = 'Daytime/evening attendance\t'
    else:
        att_col = 'Daytime/evening attendance'
    df['Attendance'] = df[att_col].map({1: 'Daytime', 0: 'Evening'})
    
    # Debtor
    df['Debtor'] = df['Debtor'].map({1: 'Yes', 0: 'No'})
    
    # Scholarship holder
    df['Scholarship holder'] = df['Scholarship holder'].map({1: 'Yes', 0: 'No'})
    
    # Displaced
    df['Displaced'] = df['Displaced'].map({1: 'Yes', 0: 'No'})
    
    # Drop unused continuous or raw columns to avoid redundancy
    cols_to_drop = ['Age at enrollment', 'Mother\'s qualification', 'Father\'s qualification', 'Nacionality', att_col]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    df = df.dropna().reset_index(drop=True)
    
    return df
