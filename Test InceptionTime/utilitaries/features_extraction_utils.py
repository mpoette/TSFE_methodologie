# TSFEL
from sklearn.feature_selection import VarianceThreshold
import tsfel
import polars as pl
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import warnings
def extract_tsfel_per_patient(df, patient_col, time_col, feature_cols,
    target_col):
    # Juste pour avoir un tqdm propre
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Precision loss occurred in moment calculation.*",
            category=RuntimeWarning,
        )
        results = []
        # Tri global (on sait jamais)
        df = df.sort([patient_col, time_col])
        cfg = tsfel.get_features_by_domain()
        # On enlève les features spectrales car avec 24 points elles renvoient des erreurs
        cfg.pop("spectral", None)
        groups = df.partition_by(patient_col, maintain_order=True)

        for g in tqdm(groups, desc = "Extraction TSFEL par patient", unit = "patient") :
            patient_id = g[patient_col][0] # On prend la première ligne vu qu'elles sont toutes pareil
            X_pl = g.select(feature_cols)
            # On transforme en pandas parce que TSFEL est capricieux
            X_pd = X_pl.to_pandas()
            X_pd = X_pd.apply(pd.to_numeric, errors="coerce")

            feats_pd = tsfel.time_series_features_extractor(cfg, X_pd, fs = 1, verbose = 0)
            feats_pd[target_col] = g[target_col][-1]
            feats_pd[patient_col] = patient_id

            results.append(pl.from_pandas(feats_pd))

        if not results:
            raise ValueError("Aucun groupe traité !")

        return pl.concat(results, how = "diagonal")


# Fonction reprise de mon stage de M1 (adaptée quand même ^^')
def filtrage_corr_var(Dataset_train, Dataset_test, patient_col, target_col):
    intruder = [patient_col, target_col]

    train_meta = Dataset_train.select(intruder).to_pandas().reset_index(drop=True)
    test_meta = Dataset_test.select(intruder).to_pandas().reset_index(drop=True)

    train_X = Dataset_train.drop(intruder).to_pandas()
    test_X = Dataset_test.drop(intruder).to_pandas()

    corr_features, train_X_corr = tsfel.correlated_features(
        train_X,
        drop_correlated=True
    )

    test_X_corr = test_X.loc[:, train_X_corr.columns]

    selector = VarianceThreshold()
    train_arr = selector.fit_transform(train_X_corr)
    test_arr = selector.transform(test_X_corr)

    support = selector.get_support()
    variableList = train_X_corr.columns[support]

    train_X_final = pd.DataFrame(train_arr, columns=variableList)
    test_X_final = pd.DataFrame(test_arr, columns=variableList)

    train_final = pd.concat([train_meta, train_X_final], axis=1)
    test_final = pd.concat([test_meta, test_X_final], axis=1)

    print("Après corrélation :", train_X_corr.shape)
    print("Après variance :", train_X_final.shape)
    print("Variables restantes :", len(variableList))

    return pl.from_pandas(train_final), pl.from_pandas(test_final), list(variableList)
    
    

    