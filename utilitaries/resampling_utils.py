import polars as pl
import numpy as np
import os
import json

def resample_icu_stays_origine(df: pl.DataFrame, target_length: int, features: list[str], variables_json, help_json = None) -> pl.DataFrame:
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

    # 3. Séparation des features par type pour l'interpolation future
    num_features = []
    cat_features = [] # string ou cumul qui ne s'interpolent pas linéairement
    expressions_agreg = []

    # 3. Agrégation sur les VRAIS doublons de temps s'il y en a à la même minute
    for f in features:
        try:
            type_feat = variables_json[f]["type_feat"]
        except KeyError:
            print("la variable n'existe pas dans les variables dynamiques brutes")
            print("Tentative d'extraction du type des features parentes")
            try:
                # Je tente de prendre le premier élément de la liste des variables parentes
                # de la feature construite (qui aura le bon type)
                type_feat = variables_json[help_json[f][0]]["type_feat"]
            except:
                raise KeyError(f"La variable {f} n'a toujours pas été trouvée !")
        if type_feat == "num_features":
            expressions_agreg.append(pl.col(f).median().alias(f))
            num_features.append(f)
        elif type_feat == "sum_features":
            expressions_agreg.append(pl.col(f).sum().alias(f))
            cat_features.append(f)
        elif type_feat == "str_features":
            expressions_agreg.append(pl.col(f).mode().first().alias(f))
            cat_features.append(f)
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
    df_grille = pl.DataFrame({
        "t_norm": grille_t,
        "heure_virtuelle": np.arange(target_length) # Aligné parfaitement avec grille_t
    })
    grille_cible = patients_uniques.join(df_grille, how="cross").with_columns(
        pl.lit(False).alias("est_origine")
    )
    
    # 5. Fusion et Tri global en amont
    df_merged = (
        pl.concat([df_clinique, grille_cible], how="diagonal")
        .sort(["encounterId", "t_norm", "est_origine"], descending=[False, False, True])
    )
    # 6. Interpolation différenciée (Numérique vs Catégoriel)
    # On applique les transformations par patient
    expressions_interp = []
    
    if num_features:
        expressions_interp.extend([
            pl.col(f).interpolate().forward_fill().backward_fill().over("encounterId") for f in num_features
        ])
    if cat_features:
        expressions_interp.extend([
            pl.col(f).forward_fill().backward_fill().over("encounterId") for f in cat_features
        ])

    df_resampled = (
        df_merged.with_columns(expressions_interp)
        # On ne garde que la grille virtuelle
        .filter(~pl.col("est_origine"))
        # Tri final et sélection
        .sort(["encounterId", "heure_virtuelle"])
        .select(["encounterId", "heure_virtuelle", "duree_reelle_sejour"] + features)
    )
    
    return df_resampled



def resample_icu_stays(df: pl.DataFrame, target_length: int, features: list[str], variables_json, help_json = None) -> pl.DataFrame:
    """
    Redimensionne les séries temporelles de longueur variable vers une longueur fixe de manière optimisée en RAM.
    """
    # 1. Sauvegarder la durée originale et gérer le min/max par patient
    df = df.with_columns(
        min_h=pl.col("delta_hour").min().over("encounterId"),
        max_h=pl.col("delta_hour").max().over("encounterId")
    ).with_columns(
        duree_reelle_sejour=pl.col("max_h")
    )
    
    # 2. Créer un temps normalisé
    df = df.with_columns(
        t_norm=pl.when(pl.col("max_h") == pl.col("min_h"))
        .then(pl.lit(0.0))
        .otherwise((pl.col("delta_hour") - pl.col("min_h")) / (pl.col("max_h") - pl.col("min_h")))
    )

    # 3. Séparation et préparation des expressions d'agrégation
    num_features = []
    cat_features = []
    expressions_agreg = []

    for f in features:
        try:
            type_feat = variables_json[f]["type_feat"]
        except KeyError:
            try:
                type_feat = variables_json[help_json[f][0]]["type_feat"]
            except:
                raise KeyError(f"La variable {f} n'a toujours pas été trouvée !")
        
        if type_feat == "num_features":
            expressions_agreg.append(pl.col(f).median().alias(f))
            num_features.append(f)
        elif type_feat == "sum_features":
            expressions_agreg.append(pl.col(f).sum().alias(f))
            cat_features.append(f)
        elif type_feat == "str_features":
            expressions_agreg.append(pl.col(f).mode().first().alias(f))
            cat_features.append(f)
        else:
            raise ValueError(f"Le type de feature proposé ({type_feat}) n'est pas encore supporté")

    # Agrégation et pré-tri pour économiser la mémoire plus tard
    df_clinique = (
        df.group_by(["encounterId", "t_norm", "duree_reelle_sejour"])
        .agg(expressions_agreg)
        .with_columns(
            pl.lit(True).alias("est_origine"),
            pl.lit(None, dtype=pl.Int64).alias("heure_virtuelle") # Aligner le schéma à l'avance
        )
    )

    # 4. Créer la "grille cible" parfaite (Triée par défaut)
    patients_uniques = df_clinique.select(["encounterId", "duree_reelle_sejour"]).unique().sort("encounterId")
    grille_t = np.linspace(0, 1, target_length)
    
    df_grille = pl.DataFrame({
        "t_norm": grille_t,
        "heure_virtuelle": np.arange(target_length, dtype=np.int64)
    })
    
    # Construction de la grille avec colonnes de features initialisées à Null pour éviter le diagonal concat
    grille_cible = (
        patients_uniques.join(df_grille, how="cross")
        .with_columns(
            pl.lit(False).alias("est_origine"),
            *[pl.lit(None, dtype=df_clinique.schema[f]).alias(f) for f in features]
        )
    )
    
    # 5. Fusion verticale (beaucoup plus rapide que diagonale) et tri localisé
    # Sélectionner les colonnes dans le même ordre exact pour autoriser le how="vertical"
    colonnes_ordre = ["encounterId", "t_norm", "duree_reelle_sejour", "est_origine", "heure_virtuelle"] + features
    
    df_merged = pl.concat([
        df_clinique.select(colonnes_ordre),
        grille_cible.select(colonnes_ordre)
    ], how="vertical").sort(["encounterId", "t_norm", "est_origine"], descending=[False, False, True])

    # 6. Interpolation via Group_by + Explode (Game changer pour la RAM)
    expressions_interp = []
    if num_features:
        expressions_interp.extend([
            pl.col(f).interpolate().forward_fill().backward_fill() for f in num_features
        ])
    if cat_features:
        expressions_interp.extend([
            pl.col(f).forward_fill().backward_fill() for f in cat_features
        ])

    df_resampled = (
        df_merged.group_by("encounterId", maintain_order=True)
        .agg([
            pl.col("heure_virtuelle"),
            pl.col("duree_reelle_sejour"),
            pl.col("est_origine"),
            *expressions_interp
        ])
        .explode(pl.all().exclude("encounterId"))
        # On filtre la grille et on nettoie
        .filter(~pl.col("est_origine"))
        .select([
            pl.col("encounterId"),
            pl.col("heure_virtuelle").alias("delta_hour"),
            pl.col("duree_reelle_sejour")
        ] + features)
    )
    
    return df_resampled