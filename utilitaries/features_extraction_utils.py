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
import shap
import matplotlib.pyplot as plt
import re
def extract_tsfel_per_patient(df, patient_col, time_col, feature_cols,
    target_col):
    # --- LE MONKEY PATCH DE LA DERNIÈRE CHANCE ---
    # On force TSFEL à remplacer sa fonction hist_mode qui bugue par un truc inoffensif
    import tsfel.feature_extraction.features as tsfel_feats
    tsfel_feats.hist_mode = lambda signal, nbins=10: 0.0
    tsfel_feats.hist_entropy = lambda signal, nbins=10: 0.0
    # ---------------------------------------------
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
        # On enlève les features spectrales car avec 24 points elles renvoient des erreurs
        cfg = tsfel.get_features_by_domain()
        cfg.pop("spectral", None)          
        
        # Sécurisation du domaine statistique pour les séries courtes de 24 points
        # if "statistical" in cfg:
        #     # On crée une copie des clés pour pouvoir itérer sans casser le dictionnaire
        #     stat_keys = list(cfg["statistical"].keys())
        #     for k in stat_keys:
        #         if "hist" in k:  # Supprime 'hist_mode', 'hist_entropy', etc.
        #             cfg["statistical"].pop(k, None)
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
            cols_to_cast = [c for c in feats_pd.columns if c not in [patient_col, target_col]]
            feats_pd[cols_to_cast] = feats_pd[cols_to_cast].astype(float)
            results.append(pl.from_pandas(feats_pd))

        if not results:
            raise ValueError("Aucun groupe traité !")

        return pl.concat(results, how = "vertical")


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
    X_train_pl = Dataset_train.drop([patient_col, target_col]).with_columns(pl.all().replace([np.inf, -np.inf], None))
    X_train_clean = X_train_pl.with_columns(pl.all().fill_null(pl.all().median()))
    X_test_pl = Dataset_test.drop([patient_col, target_col]).with_columns(pl.all().replace([np.inf, -np.inf], None))
    medians_dict = {col: val for col, val in zip(X_train_clean.columns, X_train_pl.median().row(0))}
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
    
def mesureImportance_tsfel(model, X_train, varnames, top_n=20, class_labels=None, folder = "", savefig = True, transparent = True, seed = 42):
    """
    Analyse et visualise l'importance des features (MDI, SHAP) pour n'importe quel 
    ensemble de features TSFEL (ex: après filtrage Boruta).
    
    X_train : numpy.ndarray ou pandas.DataFrame (les features déjà filtrées)
    varnames : liste ou array des noms de ces features
    """
    np.random.seed(seed)
    X_train = X_train.to_pandas()
    X_arr = X_train.values if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
    varnames = list(varnames)
    
    # On ajuste top_n si on a moins de features que prévu
    k = int(min(top_n, len(varnames)))

    # ----- 1) Importances "forêt" -----
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_varnames = np.array(varnames)[sorted_idx]
    sorted_importances = importances[sorted_idx]
    plt.figure(figsize=(12, 4))
    plt.bar(range(k), sorted_importances[:k])
    plt.xticks(range(k), sorted_varnames[:k], rotation=90)
    plt.ylabel("Importance (forêt)")
    plt.tight_layout()
    if savefig:
        plt.savefig(f"{folder}feature_importance.pdf", bbox_inches="tight", transparent=transparent)
    plt.show()

    # Pareil mais cumulé : 
    suffixes = generer_suffixes_tsfel()
    racines_varnames = [extraire_racine(name, suffixes) for name in varnames]
    df_mdi = pd.DataFrame({
        'Feature_Globale': racines_varnames,
        'Importance': importances
    })
    # On additionne les importances des sous-features appartenant à la même feature globale
    df_mdi_agg = df_mdi.groupby('Feature_Globale').sum().sort_values(by='Importance', ascending=False)
    k_agg = int(min(top_n, len(df_mdi_agg)))
    
    plt.figure(figsize=(12, 4))
    plt.bar(range(k_agg), df_mdi_agg['Importance'].head(k_agg))
    plt.xticks(range(k_agg), df_mdi_agg.index[:k_agg], rotation=90)
    plt.ylabel("Importance Globale Cumulative (forêt)")
    plt.title("Top Importance Globale des Features (MDI)")
    plt.tight_layout()
    if savefig:
        plt.savefig(f"{folder}global_feature_importance_mdi.pdf", bbox_inches="tight", transparent=transparent)
    plt.show()
    # ----- 2) SHAP -----
    X_pure_numpy = np.array(X_arr, dtype = np.float32)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_pure_numpy)

    # Gestion de la structure des shap_values selon la version de SHAP / type de modèle
    if isinstance(shap_values, list):
        shap_arr = np.stack(shap_values, axis=-1)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_arr = shap_values
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        shap_arr = shap_values[:, :, None]
    else:
        raise ValueError(f"Format SHAP inattendu: type={type(shap_values)}, shape={getattr(shap_values, 'shape', None)}")

    n_samples, n_features, n_classes = shap_arr.shape

    # ----- 3) Agrégations SHAP -----
    mean_abs_by_class = np.abs(shap_arr).mean(axis=0)  # (n_features, n_classes)
    feature_sum = mean_abs_by_class.sum(axis=1)
    
    # On ajuste le top index au nombre réel de features disponibles
    top_idx = np.argsort(feature_sum)[::-1][:min(k, n_features)]
    plot_data = mean_abs_by_class[top_idx]
    plot_labels = [varnames[i] for i in top_idx]

    # Noms de classes dynamiques
    if class_labels is not None:
        class_names = list(class_labels)
    else:
        class_names = [str(i) for i in range(n_classes)]
        
    if len(class_names) != n_classes:
        class_names = [class_names[i] if i < len(class_names) else f"Classe {i}" for i in range(n_classes)]

    # ----- 4) Barres empilées -----
    fig, ax = plt.subplots(figsize=(16, 8))
    left = np.zeros(len(top_idx))
    for c_id in range(n_classes):
        ax.barh(
            y=np.arange(len(top_idx)),
            width=plot_data[:, c_id],
            left=left,
            label=class_names[c_id]
        )
        left += plot_data[:, c_id]
        
    ax.set_yticks(np.arange(len(top_idx)))
    ax.set_yticklabels(plot_labels)
    ax.invert_yaxis()
    ax.set_xlabel("mean(|SHAP value|) (average impact per class)")
    ax.legend(title="Classes", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    if savefig:
        plt.savefig(f"{folder}shap_importance.pdf", bbox_inches="tight", transparent=transparent)
    plt.show()

    # ----- 5) Summary plot par classe -----
    for class_id, class_name in enumerate(class_names):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_arr[..., class_id],
            X_arr,
            feature_names=varnames,
            show=False
        )
        plt.title(f"SHAP Value Impact for {class_name}")
        plt.tight_layout()
        if savefig:
            plt.savefig(f"{folder}shapValues_{class_name}.pdf", bbox_inches="tight", transparent=transparent)
        plt.show()

    # ----- 6) Découpage par classe prédite -----
    y_pred = model.predict(X_arr)
    unique_classes = np.unique(y_pred)
    X_by_class = {cls: X_arr[y_pred == cls] for cls in unique_classes}
    
    # ----- 7) Importance globale agrégée
    df_shap_dict = {'Feature_Globale': racines_varnames}
    for c_id in range(n_classes):
        df_shap_dict[f'SHAP_class_{c_id}'] = mean_abs_by_class[:, c_id]
    
    # Fusionner et grouper par feature globale en sommant
    df_shap_agg = pd.DataFrame(df_shap_dict).groupby('Feature_Globale').sum()
    
    # Calcul de la somme toutes classes confondues pour trier le Top N global
    df_shap_agg['Total_Impact'] = df_shap_agg.sum(axis=1)
    df_shap_agg = df_shap_agg.sort_values(by='Total_Impact', ascending=False).drop(columns=['Total_Impact'])
    
    k_shap_agg = int(min(top_n, len(df_shap_agg)))
    df_shap_plot = df_shap_agg.head(k_shap_agg)
    
    # Graphique SHAP agrégé
    fig, ax = plt.subplots(figsize=(16, 8))
    left_agg = np.zeros(k_shap_agg)
    
    for c_id in range(n_classes):
        ax.barh(
            y=np.arange(k_shap_agg),
            width=df_shap_plot[f'SHAP_class_{c_id}'].values,
            left=left_agg,
            label=class_names[c_id]
        )
        left_agg += df_shap_plot[f'SHAP_class_{c_id}'].values
        
    ax.set_yticks(np.arange(k_shap_agg))
    ax.set_yticklabels(df_shap_plot.index)
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative mean(|SHAP value|) (average impact per class)")
    ax.set_title("Top Importance Globale des Features (SHAP)")
    ax.legend(title="Classes", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    if savefig:
        plt.savefig(f"{folder}global_shap_importance.pdf", bbox_inches="tight", transparent=transparent)
    plt.show()

    # =====================================================================
    # BLOC : GENERATION DU GRAPHIQUE SHAP SUMMARY PLOT (ROSE/BLEU) CUMULÉ
    # =====================================================================

    # 1. Extraction des racines uniques via les suffixes TSFEL
    suffixes = generer_suffixes_tsfel()
    racines_varnames = [extraire_racine(name, suffixes) for name in varnames]

    # 2. Agrégation de la matrice X par racine (en transposant pour éviter les warnings Pandas)
    # On prend la moyenne des sous-features pour conserver une notion de valeur "Basse" ou "Haute"
    df_X = pd.DataFrame(X_arr, columns=varnames)
    X_global_df = df_X.T.groupby(racines_varnames).mean().T

    # Liste finale des variables fusionnées (triées automatiquement par le groupby)
    liste_racines = list(X_global_df.columns)
    X_global_arr = X_global_df.values

    # 3. Agrégation des Shapley Values pour la classe de ton choix (ex: classe 0)
    class_id = 0  
    shap_classe_pure = shap_arr[..., class_id] # Forme (n_samples, n_features)

    # Initialisation de la matrice SHAP globale : (n_samples, n_racines)
    shap_global_arr = np.zeros((n_samples, len(liste_racines)), dtype=np.float32)

    # Remplissage par la somme des contributions SHAP de chaque sous-feature
    for idx, racine in enumerate(liste_racines):
        indices_sous_features = [i for i, r in enumerate(racines_varnames) if r == racine]
        shap_global_arr[:, idx] = shap_classe_pure[:, indices_sous_features].sum(axis=1)

    # 4. Affichage du graphique SHAP Summary Plot (Bleu / Rose) fusionné
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_global_arr, 
        X_global_arr, 
        feature_names=liste_racines, 
        show=False
    )
    plt.title(f"SHAP Value Impact (Global Fused) - {class_names[class_id]}")
    plt.tight_layout()

    if savefig:
        plt.savefig(f"{folder}global_fused_shap_summary_{class_names[class_id]}.pdf", bbox_inches="tight", transparent=transparent)
    plt.show()
    return {
        "X_by_class": X_by_class,
        "top_feat": plot_labels,
        "top_global_feat_mdi" : list(df_mdi_agg.index[:k_agg]),
        "top_global_feat_shap" : list(df_shap_plot.index)
    }
    