from pysus.online_data.SINAN import download
import pandas as pd
import os

try:
    paths = download(disease='DENG', year=2022)
    print("Paths:", paths)
    df = pd.read_parquet(paths[0])
    print(df.columns)
    print(df.head())
except Exception as e:
    print(e)
