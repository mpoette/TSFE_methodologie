import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import matplotlib.pyplot as plt
    import utilitaries.marimo_utils as mo_utils
    import utilitaries.extract_data_utils as extract

    return extract, pl, plt


@app.cell
def _(extract, pl):
    _path = '../Datasets/clean_full_static_ano.parquet'
    df_static = pl.read_parquet(_path)
    df_static = df_static.with_columns(pl.col(extract.ID_COL).cast(pl.Int32))
    df_static
    return (df_static,)


@app.cell
def _(df_static, pl):
    df_static_bis = df_static.filter(pl.col("adm_unit").is_in(["RANGUEIL DECHO. REA.","NEURO-CHIR REA", "PURPAN DECHO. REA.", "RANGUEIL REA. POLY.", "PURPAN REA. POLY."	]))
    return (df_static_bis,)


@app.cell
def _(df_static_bis, pl):
    df_static_0 = df_static_bis.with_columns(pl.when(pl.col('deces_datediff_days').is_between(-1, 0)).then(0).otherwise(pl.col('deces_datediff_days')).alias('deces_datediff_days')).filter((pl.col('deces_datediff_days') >= 0) | pl.col('deces_datediff_days').is_null())
    return (df_static_0,)


@app.cell
def _(extract):
    _path = '../Datasets/df_with_calculated_features.parquet'
    df_test = extract.extract_data_survie(_path)
    return (df_test,)


@app.cell
def _(df_static_0, df_test, extract):
    df_test_1 = df_test.join(df_static_0[[extract.ID_COL, 'age', 'deces_datediff_days', "hx_respi_chronique"]], on=extract.ID_COL, how='inner')
    return (df_test_1,)


@app.cell
def _(df_test_1, pl):
    marge = 0
    df_test_2 = df_test_1.with_columns([
            pl.when(pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 24)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_lt_24h"),

            pl.when(pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 24 + marge)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_lt_24h_EXTENDED"),

            pl.when(pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 672)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_lt_28d"),


            pl.when(pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 672 + marge)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_lt_28d_EXTENDED"),

           pl.when(pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 168)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_lt_7d"),

            pl.when(pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 168 + marge)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_lt_7d_EXTENDED"),

            pl.when(pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 2190)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_lt_3m"),

            pl.when(pl.col("deces_datediff_days") * 24 <= pl.col("delta_hour") + 2190 + marge)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_lt_3m_EXTENDED"),
            pl.when(pl.col("deces_datediff_days") * 24 > pl.col("delta_hour") + 2190)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_gt_3m"),

            pl.when(pl.col("deces_datediff_days") * 24 > pl.col("delta_hour") + 2190 + marge)
              .then(1)
              .otherwise(0)
              .alias("isDeceased_gt_3m_EXTENDED"),
        ])
    return (df_test_2,)


@app.cell
def _(df_test_2):
    df_test_2["delta_hour"].sort()
    return


@app.cell
def _():
    target_col = "isDeceased_lt_28d"
    return (target_col,)


@app.cell
def _(df_test_2, extract, pl, target_col):
    df_clean_raw, _ = extract.prepare_base_data(df_test_2, target_col = target_col, other_cols=[], used_distribution="flexible")
    df_clean_raw = df_clean_raw.with_columns(pl.col("heure_calibree").cast(pl.Int64))
    return (df_clean_raw,)


@app.cell
def _(df_clean_raw, extract):
    df_clean_final = extract.finalize_data(df_clean_raw, df_clean_raw, strict_mode = False)
    return (df_clean_final,)


@app.cell
def _(df_clean_final):
    df_clean_final.describe()
    return


@app.cell
def _(df_clean_final, target_col):
    df_clean_final[target_col].describe()
    return


@app.cell
def _(df_clean_final, extract, pl):
    # D'abord, on met tous les patients sur le même temps pour pouvoir les merge
    df_modif = df_clean_final.with_columns(
        pl.arange(0,pl.len()).over(extract.ID_COL).alias("hour_local")
    )
    return (df_modif,)


@app.cell
def _(df_modif):
    df_modif["hour_local"].describe()
    return


@app.cell
def _(df_modif):
    df_modif.select("hour_local", "heure_calibree")
    return


@app.cell
def _(df_modif):
    df_modif.describe()
    return


@app.cell
def _(df_modif, pl, target_col):
    df_modif_agg = (
        df_modif
        .group_by("hour_local")
        .agg([
            pl.col(target_col).sum().alias("deaths"),
            (1 - pl.col(target_col).cast(pl.Int8)).sum().alias("alive"),
            pl.len().alias("n_patients")
        ])
        .sort("hour_local")
    )
    return (df_modif_agg,)


@app.cell
def _(df_modif_agg, plt, target_col):
    plt.plot(df_modif_agg["hour_local"], df_modif_agg["deaths"], label = target_col)
    plt.plot(df_modif_agg["hour_local"], df_modif_agg["alive"], label = "Alive or dead after")
    plt.xlim(0, 128)
    # plt.ylim(0, 10000)
    plt.yscale("log")
    plt.legend()
    plt.show()
    return


@app.cell
def _(df_modif, pl):
    df_modif.filter(pl.col("hour_local") == 0)
    return


@app.cell
def _(df_static_0):
    df_static_0["encounterId"].unique()
    return


@app.cell
def _(df_test):
    df_test["encounterId"].unique()
    return


@app.cell
def _(df_clean_raw):
    df_clean_raw["encounterId"].unique()
    return


@app.cell
def _(df_clean_raw, pl):
    df_clean_raw.filter(pl.col("heure_calibree") == 0).describe()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
