import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from dataclasses import dataclass
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import time
    import matplotlib.pyplot as plt
    import seaborn as sns
    import optuna
    import math
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        roc_auc_score,
        roc_curve,
        brier_score_loss
    )
    from pathlib import Path
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    from Extraction import extract
    from Transformation_Pretraitement import preprocessing_polars
    from inceptionTimeModified import (
        evaluate_on_test,
        load_model_from_checkpoint,
        predict_proba,
        train_inception_time,
    )
    import utils_inception as ui

    return extract, mo, pl, plt, preprocessing_polars, ui


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Là, je fais un curseur initialisé avec la valeur 12 qui est le max_hour soit le temps que je veux sanctuariser avant la sortie du patient
    """)
    return


@app.cell
def _(mo):
    get_max_hour, set_max_hour = mo.state(12)
    gap_start, gap_end = 0, 12
    return gap_end, gap_start, get_max_hour, set_max_hour


@app.cell
def _(gap_end, gap_start, get_max_hour, mo, set_max_hour):
    slider_max_hour = mo.ui.slider(gap_start, gap_end, value=get_max_hour(), on_change=set_max_hour)
    return (slider_max_hour,)


@app.cell
def _(gap_end, gap_start, get_max_hour, mo, set_max_hour):
    number_max_hour = mo.ui.number(gap_start, gap_end, value=get_max_hour(), on_change=set_max_hour)
    return (number_max_hour,)


@app.cell
def _(pl, ui):
    _path = '../Datasets/clean_full_static_ano.parquet'
    df_static = pl.read_parquet(_path)
    df_static = df_static.with_columns(pl.col(ui.patient_col).cast(pl.Int32))
    # df_static = df_static.filter(pl.col("adm_unit").is_in(["RANGUEIL DECHO. REA.","NEURO-CHIR REA", "PURPAN DECHO. REA.", "RANGUEIL REA. POLY.", "PURPAN REA. POLY."	]))
    df_static
    return (df_static,)


@app.cell
def _(df_static, pl):
    df_static_0 = df_static.with_columns(pl.when(pl.col('deces_datediff_days').is_between(-1, 0)).then(0).otherwise(pl.col('deces_datediff_days')).alias('deces_datediff_days')).filter((pl.col('deces_datediff_days') >= 0) | pl.col('deces_datediff_days').is_null())
    return (df_static_0,)


@app.cell
def _(df_static_0, pl):
    df_static_2 = df_static_0.with_columns(pl.when(pl.col('deces_datediff_days').is_between(-1, 0)).then(0).otherwise(pl.col('deces_datediff_days')).alias('deces_datediff_days'))
    return (df_static_2,)


@app.cell
def _(df_static_2):
    df_static_2['isDeceased'].describe()
    return


@app.cell
def _(extract):
    _path = '../Datasets/df_with_calculated_features.parquet'
    df_test = extract.extract_data_survie(_path)
    return (df_test,)


@app.cell
def _(df_static_2, df_test, ui):
    df_test_1 = df_test.join(df_static_2, on=ui.patient_col, how='left')
    return (df_test_1,)


@app.cell
def _(df_test_1, pl):
    df_test_1.filter(pl.col("deces_datediff_days").is_not_null()).select("deces_datediff_days", "delta_hour")
    return


@app.cell
def _(df_test_1, pl):
    marge = 12
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
    ])
    return (df_test_2,)


@app.cell
def _(df_test_2):
    df_test_2.describe()
    return


@app.cell
def _(mo, number_max_hour, slider_max_hour):
    # slider and number are synchronized to have the same value (try it!)
    mo.vstack([
        mo.md("Durée sanctuarisée avant la sortie du patient"),
        slider_max_hour,
        number_max_hour])
    return


@app.cell
def _(df_test_2, get_max_hour, preprocessing_polars):
    max_hour = get_max_hour()
    df_clean = preprocessing_polars.prepare_data(df_test_2, hour_offset = 0, random = True, max_hour = 0, used_distribution="flexible", target_col = "isDeceased_lt_24h_EXTENDED")
    return (df_clean,)


@app.cell
def _(df_clean, pl, ui):
    print("nombre d'enregistrement de patients morts moins de 24 heures après la fin de la fenêtre", df_clean.filter(pl.col('isDeceased_lt_24h_EXTENDED') == 1).select(pl.col(ui.patient_col).n_unique()).item())
    return


@app.cell
def _(df_test_2, pl, plt):
    cols_flags = [
        "isDeceased_lt_24h",
        "isDeceased_lt_28d",
        "isDeceased_lt_3m",
    ]

    # Calcul des limites (on vire les 1% les plus extrêmes du passé et du futur)
    stats = df_test_2.select([
        pl.col("delta_hour").quantile(0.01).alias("q01"),
        pl.col("delta_hour").quantile(0.99).alias("q99")
    ]).row(0)

    q01 = max(stats[0], -2000) 
    q99 = stats[1]

    print(f"Fenêtre d'analyse : de {q01}h à {q99}h")

    # 2) Filtrage et agrégation directe
    dist = (
        df_test_2
        .with_columns(pl.col("delta_hour").round(0).cast(pl.Int64))
        .drop_nulls(subset=["delta_hour"])
        .filter(pl.col("delta_hour").is_between(q01, q99))
        .group_by("delta_hour")
        .agg([pl.col(c).sum() for c in cols_flags])
        .sort("delta_hour")
    )

    pdf = dist.to_pandas()

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 12))
    fig.suptitle("Distribution des décès (Outliers exclus)", fontsize=16)

    axes[0].bar(pdf["delta_hour"], pdf["isDeceased_lt_24h"], width=1.0, color='tab:blue')
    axes[0].set_title("< 24h")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(pdf["delta_hour"], pdf["isDeceased_lt_28d"], width=1.0, color='tab:orange')
    axes[1].set_title("< 28j")
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(pdf["delta_hour"], pdf["isDeceased_lt_3m"], width=1.0, color='tab:green')
    axes[2].set_title("< 3 mois")
    axes[2].set_xlabel("delta_hour")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(config_sidebar, mo):
    mo.sidebar(
    mo.vstack([
        mo.md(config_sidebar),
    
        mo.md("</div>")]),
    width = "550px")
    return


if __name__ == "__main__":
    app.run()
