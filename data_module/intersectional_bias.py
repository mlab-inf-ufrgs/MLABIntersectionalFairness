from sklearn.datasets import fetch_openml
import pandas as pd
import streamlit as st

@st.cache_data
def load_and_preprocess_intersectional_bias():
    """
    Loads the Intersectional Bias TRAINING dataset (OpenML ID 44202, v.2).
    10.000 instâncias sintéticas de saúde mental (esquizofrenia/depressão).
    Target: 'diagnosis' (favorable=0, sem diagnóstico de doença)
    
    Privileged Groups:
        - race: White
        - sex: Male
    """
    data = fetch_openml(data_id=44202, as_frame=True, parser='auto')
    df = data.frame
    
    df.columns = df.columns.str.lower()
    
    if 'diagnosis' in df.columns:
        df['diagnosis'] = df['diagnosis'].astype(int)
        
    return df

