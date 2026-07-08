# TSFEL
from sklearn.feature_selection import VarianceThreshold
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import tsfel
import polars as pl
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import warnings
import matplotlib.pyplot as plt
import re
from joblib import Parallel, delayed

import warnings
# Monkey patch pour contourner les erreurs d'histogramme
import tsfel.feature_extraction.features as tsfel_feats
tsfel_feats.hist_mode = lambda signal, nbins=10: 0.0
tsfel_feats.hist_entropy = lambda signal, nbins=10: 0.0

with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Precision loss occurred in moment calculation.*",
            category=RuntimeWarning,
        )
        
def _process_single_patient(g, cfg, feature_cols, patient_col, target_col):
    """Fonction atomique exécutée en parallèle pour un patient donné."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Precision loss occurred in moment calculation.*",
            category=RuntimeWarning,
        )

        patient_id = g[patient_col][0]
        X_pl = g.select(feature_cols)
        
        # Conversion Pandas obligatoire pour TSFEL
        X_pd = X_pl.to_pandas()
        X_pd = X_pd.apply(pd.to_numeric, errors="coerce")

        # Extraction locale
        feats_pd = tsfel.time_series_features_extractor(cfg, X_pd, fs=1, verbose=0)
        
        # Post-processing et typage
        feats_pd[target_col] = g[target_col][-1]
        feats_pd[patient_col] = patient_id
        cols_to_cast = [c for c in feats_pd.columns if c not in [patient_col, target_col]]
        feats_pd[cols_to_cast] = feats_pd[cols_to_cast].astype(float)
        
        return pl.from_pandas(feats_pd)


def extract_tsfel_per_patient(df, patient_col, time_col, feature_cols, target_col):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Precision loss occurred in moment calculation.*",
                category=RuntimeWarning,
            )
        # 1. Tri et préparation de la config TSFEL
        df = df.sort([patient_col, time_col])
        cfg = tsfel.get_features_by_domain()
        cfg.pop("spectral", None)  # On retire le domaine spectral obsolète ici
        
        # 2. Découpage en groupes par patient
        groups = df.partition_by(patient_col, maintain_order=True)

        print(f"Lancement du calcul parallèle sur {len(groups)} patients...")
        
        results = Parallel(n_jobs=-1)(
            delayed(_process_single_patient)(g, cfg, feature_cols, patient_col, target_col)
            for g in tqdm(groups, desc="Extraction TSFEL parallèle", unit="patient")
        )

        if not results:
            raise ValueError("Aucun groupe traité !")

        # 4. Reconstitution du DataFrame final
        return pl.concat(results, how="vertical")


# Fonction reprise de mon stage de M1 (adaptée quand même ^^')
def filtrage_corr_var(Dataset_train, Dataset_test, patient_col, target_col):
    intruder = [patient_col, target_col]

    train_meta = Dataset_train.select(intruder).to_pandas().reset_index(drop=True)
    test_meta = Dataset_test.select(intruder).to_pandas().reset_index(drop=True)

    train_X = Dataset_train.drop(intruder).to_pandas()

    # On s'assure que l'ordre est toujours le même
    train_X = train_X.reindex(sorted(train_X.columns), axis=1)
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

def filtrage_boruta(Dataset_train, Dataset_test, patient_col, target_col, max_iter=100, seed = 42):
    # 1. Séparation propre des features (X) et nettoyage des Inf/NaN en Polars
    # (On remplace les valeurs infinies par du Null, puis on remplit par la médiane du Train)
    # On fait ça parce que TSFEL pour générer des valeurs infinies ou des null
    features_to_keep = sorted([c for c in Dataset_train.columns if c not in [patient_col, target_col]])
    X_train_pl = Dataset_train.select(features_to_keep).with_columns(pl.all().replace([np.inf, -np.inf], None))
    X_train_clean = X_train_pl.with_columns(pl.all().fill_null(pl.all().median()))
    X_test_pl = Dataset_test.select(features_to_keep).with_columns(pl.all().replace([np.inf, -np.inf], None))
    medians_dict = X_train_pl.median().to_dicts()[0]
    X_test_clean = X_test_pl.with_columns([pl.col(col).fill_null(medians_dict[col]) for col in X_test_pl.columns])

    # 2. Entraînement de Boruta sur les tableaux NumPy sous-jacents
    # On fixe la profondeur maximale de la forêt pour éviter l'overfeating
    rf = RandomForestClassifier(n_jobs=-1, max_depth=5, class_weight='balanced', random_state=seed)
    # alpha : 1 - pvalues => pvalue à 0.95 ce qui est raisonable
    # perc : dans un 1V1, il faut que la feature gagne dans 100% du temps si perc = 100. 
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose = 2, alpha = 0.05, perc = 100, max_iter=max_iter, random_state=seed)
    
    feat_selector.fit(X_train_clean.to_numpy(), Dataset_train[target_col].to_numpy())

    # 3. Extraction des variables validées
    variableList = [col for col, keep in zip(X_train_clean.columns, feat_selector.support_) if keep]

    # 4. Reconstruction des datasets finaux
    intruders = [patient_col, target_col]
    train_final = Dataset_train.select(intruders).with_columns(X_train_clean.select(variableList))
    test_final = Dataset_test.select(intruders).with_columns(X_test_clean.select(variableList))

    print(f"Boruta terminé : {len(variableList)} variables conservées.")
    return train_final, test_final, variableList

def generer_suffixes_tsfel():
    tsfel_feats = tsfel.get_features_by_domain()
    suffixes = []
    for d in tsfel_feats.keys():
        for feature in tsfel_feats[d].keys():
            suffixes.append(feature)
    # Tri du plus long au plus court pour éviter les faux positifs à la découpe
    suffixes.sort(key=len, reverse=True)
    return suffixes

def extraire_racine(name, suffixes):
    for suffixe in suffixes:
        pattern = rf"_{suffixe}(_\d+)?$"
        
        # Ajout de flags=re.IGNORECASE ici pour chercher sans se soucier des majuscules/minuscules
        if re.search(pattern, name, flags=re.IGNORECASE):
            racine = re.split(pattern, name, flags=re.IGNORECASE)[0]
            return racine
    