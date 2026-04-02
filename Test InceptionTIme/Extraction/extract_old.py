import pandas as pd
import numpy as np
import os

def extract_data_survie(file_path):
    if not os.path.exists(file_path):
            print("le Fichier n'existe pas")
            return None

    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path, engine='pyarrow')
    else:
        df = pd.read_csv(file_path)

    df['delta_hour'] = pd.to_numeric(df['delta_hour'], errors='coerce')
    df['pam'] = pd.to_numeric(df['pam'], errors='coerce')
                                                
    print(f"{len(df)} lignes chargées.")
    return df