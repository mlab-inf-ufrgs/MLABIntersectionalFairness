from sklearn.datasets import fetch_openml
import pandas as pd

def load_and_preprocess_intersectional_bias():
    """
    Loads the Intersectional Bias dataset (OpenML ID 44203).
    Target: 'diagnosis' (favorable=0)
    
    Privileged Groups:
        - race: White
        - sex: Female
        
    Dataset is simulated and needs no cleaning.
    """
    # Fetch dataset
    data = fetch_openml(data_id=44203, as_frame=True, parser='auto')
    df = data.frame
    
    # Target is diagnosis (favorable = 0)
    # Just ensure it's integer for consistency
    df.columns = df.columns.str.lower()
    
    if 'diagnosis' in df.columns:
        df['diagnosis'] = df['diagnosis'].astype(int)
        
    return df
