import pandas as pd
try:
    from pysus.online_data import SIH, SIM, SINASC
except ImportError:
    print("PySUS not installed")
    import sys
    sys.exit(1)

try:
    print("Tentando baixar SIH (SP, 2023, mes 1)...")
    df_sih = SIH.download('RD', year=2023, month=1, uf='SP')
    print("SIH Colunas:", df_sih.columns[:20] if not isinstance(df_sih, tuple) else df_sih[0].columns[:20])
except Exception as e:
    print("Erro SIH:", e)
