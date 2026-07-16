from pathlib import Path
from typing import Any

import json
import polars as pl

import utilitaries.extract_data_utils as extract
import utilitaries.resampling_utils as resampling
import utilitaries.timestamp_sampling_utils as tsu

def _compute_median_target_length(
    df: pl.DataFrame,
    patient_col: str = "encounterId",
    time_col: str = "delta_hour",
) -> int:
    """
    Calcule la médiane du dernier delta_hour par séjour.

    Comme delta_hour commence à 1 et qu'il existe exactement une mesure
    par heure, max(delta_hour) correspond au nombre de points du séjour.
    """

    required_columns = {patient_col, time_col}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Colonnes manquantes pour calculer target_length : "
            f"{sorted(missing_columns)}"
        )

    target_length = (
        df
        .group_by(patient_col)
        .agg(
            pl.col(time_col)
            .max()
            .alias("stay_duration")
        )
        .select(
            pl.col("stay_duration").median()
        )
        .item()
    )

    if target_length is None:
        raise ValueError(
            "Impossible de calculer target_length."
        )

    target_length = int(round(target_length))

    if target_length < 1:
        raise ValueError(
            f"target_length invalide : {target_length}"
        )

    return target_length


def _load_json(path: str | Path) -> dict:
    """Charge un fichier JSON avec un message d'erreur explicite."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Le fichier JSON n'existe pas : {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
    
def prepare_dataset_from_config(
    df_merged: pl.DataFrame,
    config_mode: Any,
    target_col: str,
    patient_col: str,
    targets: list[str],
    seed: int,
    dynamic_features_path: str | Path = "../utilitaries/dynamic_features.json",
    calculated_features_path: str | Path = (
        "../utilitaries/calculated_features_to_original_features.json"
    ),
    show_fig: bool = True,
) -> tuple[pl.DataFrame, list[str], int | None]:
    """
    Nettoie et prépare les données selon le mode configuré.

    Modes pris en charge
    --------------------
    windows :
        - 24h_alea_lomax_prio24h_no-fill
        - autres modes gérés par extract.prepare_data

    resampling :
        - resampling_x_points
        - resampling_x_points_alea_lomax_prio_24h_no-fill

    Retours
    -------
    df_clean :
        DataFrame préparé.

    features_list :
        Liste des variables dynamiques utilisées pour le resampling.
        Liste vide pour le mode windows.

    target_length :
        Nombre de points utilisé pour le resampling.
        None pour le mode windows.
    """

    features_list: list[str] = []
    target_length: int | None = None

    # Nettoyage commun à tous les modes
    df_clean = extract.remove_null_values(df_merged)

    if df_clean.is_empty():
        raise ValueError("Le DataFrame est vide après remove_null_values.")

    # Calcul des timestamps uniquement pour les modes Lomax
    df_timestamp: pl.DataFrame | None = None

    if "lomax" in config_mode.name:
        df_timestamp = tsu.build_sampling_dataset(
            df_clean,
            sanctuary_hours=6,
            min_observation_hours=24,
            lomax_alpha=4.3085,
            lomax_lambda=1161.9368,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Mode fenêtres
    # ------------------------------------------------------------------
    if config_mode.mode == "windows":
        if config_mode.name == "24h_alea_lomax_prio24h_no-fill":
            if df_timestamp is None:
                raise RuntimeError(
                    "df_timestamp n'a pas été calculé pour le mode Lomax."
                )

            df_clean = (
                df_clean
                .join(
                    df_timestamp.select(
                        [
                            patient_col,
                            pl.col("delta_hour").alias("windows_end"),
                        ]
                    ),
                    on=patient_col,
                    how="inner",
                )
                .filter(
                    (pl.col("delta_hour") <= pl.col("windows_end"))
                    & (
                        pl.col("delta_hour")
                        > pl.col("windows_end") - 24
                    )
                )
            )

        else:
            df_clean = extract.prepare_data(
                df_clean,
                hour_offset=config_mode.hour_offset,
                random=config_mode.random,
                max_hour=6,
                used_distribution=config_mode.used_distribution,
                strict_mode=config_mode.strict_mode,
                target_col=target_col,
                other_cols=targets,
                show_fig=show_fig,
            )

        return df_clean, features_list, target_length

    # ------------------------------------------------------------------
    # Mode resampling
    # ------------------------------------------------------------------
    if config_mode.mode == "resampling":
        variables_json = _load_json(dynamic_features_path)
        calc_json = _load_json(calculated_features_path)

        correct_features = (
            set(variables_json)
            | {
                column
                for column in df_merged.columns
                if column in calc_json
            }
        )

        features_list = [
            column
            for column in df_merged.columns
            if column not in {target_col, patient_col}
            and column in correct_features
        ]

        if not features_list:
            raise ValueError(
                "Aucune variable dynamique valide n'a été trouvée "
                "pour le resampling."
            )

        print(f"Variables dynamiques retenues : {features_list}")

        columns_to_exclude = list(
            dict.fromkeys(
                features_list
                + [
                    "delta_hour",
                    "t_norm",
                    "min_h",
                    "max_h",
                    "duree_reelle_sejour",
                    "observed_duration",
                    "real_time_hours",
                    "windows_end",
                ]
            )
        )

        # --------------------------------------------------------------
        # Resampling du séjour disponible
        # --------------------------------------------------------------
        if config_mode.name == "resampling_x_points":
            target_length = _compute_median_target_length(
                df_clean,
                patient_col=patient_col,
                time_col="delta_hour",
            )

            df_clean = resampling.resample_icu_stays(
                df=df_clean,
                target_length=target_length,
                features=features_list,
                variables_json=variables_json,
                help_json=calc_json,
            )

            if "duree_reelle_sejour" not in df_clean.columns:
                raise ValueError(
                    "resample_icu_stays n'a pas produit la colonne "
                    "'duree_reelle_sejour'."
                )

            # Hypothèse : après resampling, delta_hour varie de 1
            # à target_length.
            #
            # Le premier point correspond à l'heure 1 et le dernier
            # à duree_reelle_sejour.
            if target_length == 1:
                df_clean = df_clean.with_columns(
                    pl.lit(1.0).alias("real_time_hours")
                )
            else:
                df_clean = df_clean.with_columns(
                    (
                        1
                        + (
                            pl.col("duree_reelle_sejour") - 1
                        )
                        * (
                            pl.col("delta_hour") - 1
                        )
                        / (target_length - 1)
                    ).alias("real_time_hours")
                )

            # La durée complète pourrait provoquer une fuite
            # d'information.
            df_clean = df_clean.drop("duree_reelle_sejour")

        # --------------------------------------------------------------
        # Lomax, puis resampling de la partie observée
        # --------------------------------------------------------------
        elif (
            config_mode.name
            == "resampling_x_points_alea_lomax_prio_24h_no-fill"
        ):
            if df_timestamp is None:
                raise RuntimeError(
                    "df_timestamp n'a pas été calculé pour le mode Lomax."
                )

            df_clean = (
                df_clean
                .join(
                    df_timestamp.select(
                        [
                            patient_col,
                            pl.col("delta_hour").alias("windows_end"),
                        ]
                    ),
                    on=patient_col,
                    how="inner",
                )
                .filter(
                    pl.col("delta_hour") <= pl.col("windows_end")
                )
            )

            if df_clean.is_empty():
                raise ValueError(
                    "Le filtrage Lomax a supprimé toutes les observations."
                )

            target_length = _compute_median_target_length(
                df_clean,
                patient_col=patient_col,
                time_col="delta_hour",
            )

            df_clean = resampling.resample_icu_stays(
                df=df_clean,
                target_length=target_length,
                features=features_list,
                variables_json=variables_json,
                help_json=calc_json,
            )

            if "duree_reelle_sejour" not in df_clean.columns:
                raise ValueError(
                    "resample_icu_stays n'a pas produit la colonne "
                    "'duree_reelle_sejour'."
                )

            # Ici, il s'agit de la durée disponible au moment
            # de la prédiction, et non de la durée finale du séjour.
            df_clean = df_clean.rename(
                {
                    "duree_reelle_sejour": "observed_duration",
                }
            )

        else:
            raise ValueError(
                "Version de resampling non implémentée : "
                f"{config_mode.name}"
            )

        # Ajout des variables statiques
        static_columns = [
            column
            for column in df_merged.columns
            if column not in columns_to_exclude
        ]

        if patient_col not in static_columns:
            static_columns.append(patient_col)

        df_static_patient = (
            df_merged
            .select(static_columns)
            .unique(subset=[patient_col], keep="first")
        )

        df_clean = df_clean.join(
            df_static_patient,
            on=patient_col,
            how="inner",
        )

        return df_clean, features_list, target_length

    raise ValueError(
        f"Mode non implémenté : {config_mode.mode}"
    )



