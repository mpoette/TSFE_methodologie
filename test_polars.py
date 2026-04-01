import polars as pl
res = pl.read_parquet("Datasets/clean_full_static_ano.parquet")
print(res.head())

res2 = pl.read_parquet("Datasets/df_with_calculated_features.parquet")
print(res2.head())

res3 = pl.read_parquet("Datasets/temporal_tailored_imputation.parquet")
print(res2.head())