import polars as pl
import numpy as np
import os
 
def extract_data_survie(file_path):
    if not os.path.exists(file_path):
        print("le Fichier n'existe pas")
        return None
 
    if file_path.endswith('.parquet'):
        df = pl.read_parquet(file_path)
    else:
        df = pl.read_csv(file_path)
    df = df.with_columns([
        pl.col("delta_hour").cast(pl.Float64, strict=False),
        pl.col("pam").cast(pl.Float64, strict=False),
        # Je corrige ici le mauvais stockage de tracheo et ecmo_type en str et non en bool
        # Pour l'instant la base est propre donc on a que des "true" et "false" mais ici
        # cela fonctionnerait même si on avait "FaLsE" ou "True"
        pl.col("tracheo").str.to_lowercase().str.strip_chars().eq("true").alias("tracheo"),
        pl.col("ecmo_type").str.to_lowercase().str.strip_chars().eq("true").alias("ecmo_type"),
    ])
 
    print(f"{df.height} lignes chargées.")
    return df