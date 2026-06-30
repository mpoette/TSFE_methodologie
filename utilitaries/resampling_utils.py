import polars as pl
import numpy as np
import os
import json

def resample_icu_stays(df: pl.DataFrame, target_length: int, features: list[str], variables_json) -> pl.DataFrame:
    """
    Redimensionne les séries temporelles de longueur variable vers une longueur fixe.
    """
    # 1. Sauvegarder la durée originale et gérer le min/max par patient
    df = df.with_columns(
        min_h=pl.col("delta_hour").min().over("encounterId"),
        max_h=pl.col("delta_hour").max().over("encounterId")
    ).with_columns(
        duree_reelle_sejour=pl.col("max_h")
    )
    
    # 2. Créer un temps normalisé (avec sécurité si max == min)
    df = df.with_columns(
        t_norm=pl.when(pl.col("max_h") == pl.col("min_h"))
        .then(pl.lit(0.0))
        .otherwise((pl.col("delta_hour") - pl.col("min_h")) / (pl.col("max_h") - pl.col("min_h")))
    )

    # 3. Agrégation sur les VRAIS doublons de temps s'il y en a à la même minute
    expressions_agreg = []
    for f in features:
        type_feat = variables_json[f]["type_feat"]
        if type_feat == "num_features":
            expressions_agreg.append(pl.col(f).median().alias(f))
        elif type_feat == "sum_features":
            expressions_agreg.append(pl.col(f).sum().alias(f))
        elif type_feat == "str_features":
            expressions_agreg.append(pl.col(f).mode().first().alias(f))
        else:
            raise ValueError(f"Le type de feature proposé ({type_feat}) n'est pas encore supporté")

    df_clinique = (
        df.group_by(["encounterId", "t_norm", "duree_reelle_sejour"])
        .agg(expressions_agreg)
        .with_columns(pl.lit(True).alias("est_origine"))
    )

    # 4. Créer la "grille cible" parfaite
    patients_uniques = df_clinique.select(["encounterId", "duree_reelle_sejour"]).unique()
    grille_t = np.linspace(0, 1, target_length)
 
    grille_cible = patients_uniques.join(
        pl.DataFrame({"t_norm": grille_t}), how="cross"
    ).with_columns(
        pl.lit(False).alias("est_origine"),
        # Génération propre de l'heure virtuelle par patient
        heure_virtuelle=pl.int_range(0, target_length).over("encounterId")
    )

    # 5. Fusion, Tri et Interpolation Native Polars
    df_resampled = (
        pl.concat([df_clinique, grille_cible], how="diagonal")
        # On trie par patient (croissant), t_norm (croissant), et est_origine (décroissant : True puis False)
        .sort(
            ["encounterId", "t_norm", "est_origine"], 
            descending=[False, False, True]
        )
        .with_columns(
            pl.col(features).interpolate().over("encounterId")
        )
        # Remplissage des extrêmes au cas où la grille déborde des mesures réelles
        .with_columns(
            pl.col(features).forward_fill().backward_fill().over("encounterId")
        )
        # On ne garde que la grille virtuelle
        .filter(~pl.col("est_origine"))
        # Tri final propre
        .sort(["encounterId", "heure_virtuelle"])
        .select(["encounterId", "heure_virtuelle", "duree_reelle_sejour"] + features)
    )
    
    return df_resampled
