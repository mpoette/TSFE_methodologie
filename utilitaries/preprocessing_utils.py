import os

import joblib
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

import utilitaries.extract_data_utils as edu
import utilitaries.extract_data_utils as extract
import utilitaries.features_extraction_utils as extract_feat


def scaling(
    df_train: pl.DataFrame,
    *dfs_to_transform: pl.DataFrame,
):
    """Scale continuous numerical features using a StandardScaler.

    Boolean columns and numerical columns containing only binary values
    (0, 1, or null) are excluded from scaling.

    The scaler is fitted exclusively on the training set and then applied
    to every additional dataset without refitting.

    Args:
        df_train:
            Training DataFrame used to fit the scaler.
        *dfs_to_transform:
            Validation, holdout, or other DataFrames to transform using the
            scaler fitted on the training data.

    Returns:
        A tuple containing the scaled training DataFrame followed by all
        transformed DataFrames in their original argument order.

    Raises:
        ValueError:
            If one of the additional DataFrames is missing a numerical column
            required by the scaler fitted on the training set.
    """
    # Identify numerical columns while excluding the patient identifier.
    all_num_cols = df_train.select(
        pl.col(pl.NUMERIC_DTYPES).exclude(edu.ID_COL)
    ).columns

    # Exclude columns containing only 0, 1, or null values.
    num_cols_to_scale = []

    for col in all_num_cols:
        unique_vals = (
            df_train
            .select(pl.col(col).unique())
            .to_series()
            .to_list()
        )

        is_binary = all(
            value in [0, 1, None]
            for value in unique_vals
        )

        if not is_binary:
            num_cols_to_scale.append(col)

    # Return the original DataFrames when no continuous column is available.
    if not num_cols_to_scale:
        return (df_train, *dfs_to_transform)

    # Ensure every dataset can be transformed with the training feature set.
    for dataframe_index, dataframe in enumerate(dfs_to_transform):
        missing_columns = sorted(
            set(num_cols_to_scale) - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                f"DataFrame {dataframe_index + 1} is missing columns required "
                f"by the training scaler: {missing_columns}"
            )

    # Fit the scaler exclusively on the training set.
    scaler = StandardScaler()

    train_scaled_values = scaler.fit_transform(
        df_train
        .select(num_cols_to_scale)
        .to_pandas()
    )

    # Replace the original training columns with their scaled values.
    scaled_train = df_train.with_columns(
        [
            pl.Series(
                name,
                train_scaled_values[:, index],
            )
            for index, name in enumerate(num_cols_to_scale)
        ]
    )

    scaled_others = []

    # Apply the fitted scaler to validation, holdout, and any other dataset.
    for dataframe in dfs_to_transform:
        transformed_values = scaler.transform(
            dataframe
            .select(num_cols_to_scale)
            .to_pandas()
        )

        scaled_dataframe = dataframe.with_columns(
            [
                pl.Series(
                    name,
                    transformed_values[:, index],
                )
                for index, name in enumerate(num_cols_to_scale)
            ]
        )

        scaled_others.append(scaled_dataframe)

    return (scaled_train, *scaled_others)


def build_sequences(
    df,
    patient_col,
    target_col,
    expected_length,
    keep_features,
):
    """Convert patient observations into fixed-length 3D sequences.

    Each patient group must contain exactly ``expected_length`` rows.
    Feature columns are sorted alphabetically to guarantee a stable order.

    Args:
        df:
            Input patient DataFrame.
        patient_col:
            Name of the patient identifier column.
        target_col:
            Name of the target column.
        expected_length:
            Expected number of time points per patient.
        keep_features:
            Feature columns to include in each sequence.

    Returns:
        A tuple containing:

        - A 3D feature array with shape
          ``(n_patients, expected_length, n_features)``.
        - A 1D array containing one target value per patient.

    Raises:
        ValueError:
            If a patient group does not contain the expected number of rows.
    """
    # Guarantee a stable patient-group order.
    df = df.sort(patient_col)

    # Guarantee a stable alphabetical feature order.
    keep_features = sorted(list(keep_features))

    X_list = []
    y_list = []

    for subdf in df.partition_by(
        patient_col,
        maintain_order=True,
    ):
        if subdf.height != expected_length:
            raise ValueError(
                f"Patient group {subdf[patient_col][0]} does not contain "
                f"{expected_length} rows."
            )

        X_list.append(
            subdf
            .select(keep_features)
            .to_numpy()
        )

        # Keep one patient-level label per sequence.
        y_list.append(
            subdf[target_col][0]
        )

    X_3d = np.stack(X_list).astype(
        np.float32
    )

    y_1d = np.array(y_list).astype(
        np.int64
    )

    return X_3d, y_1d


def downsample_train_patients(
    train_df: pl.DataFrame,
    patient_col: str,
    col_24h: str = "isDeceased_lt_24h",
    col_28d: str = "isDeceased_lt_28d",
    seed: int = 42,
) -> pl.DataFrame:
    """Downsample patient stays while preserving all rows per patient.

    Patients are divided into three mutually exclusive groups:

    - Death within 24 hours.
    - Death between 24 hours and 28 days.
    - Survivors beyond 28 days.

    All patients who died within 24 hours are retained. Patients from the
    other groups are sampled so that their combined count matches the number
    of patients who died within 24 hours.

    Args:
        train_df:
            Training DataFrame containing repeated rows per patient.
        patient_col:
            Name of the patient identifier column.
        col_24h:
            Column indicating death within 24 hours.
        col_28d:
            Column indicating death within 28 days.
        seed:
            Random seed used during patient sampling.

    Returns:
        The downsampled training DataFrame with all rows retained for each
        selected patient.

    Raises:
        ValueError:
            If no patient belongs to the 24-hour mortality group or if the
            remaining groups do not contain enough patients.
    """
    # Build one row per patient with patient-level target labels.
    # Mortality columns are assumed to remain constant across patient rows.
    patient_labels = (
        train_df
        .group_by(patient_col)
        .agg(
            [
                pl.last(col_24h).alias(col_24h),
                pl.last(col_28d).alias(col_28d),
            ]
        )
        # Sort patients to guarantee deterministic sampling order.
        .sort(patient_col)
        .with_columns(
            [
                pl.when(
                    pl.col(col_24h) == 1
                )
                .then(
                    pl.lit("lt_24h")
                )
                .when(
                    (pl.col(col_24h) == 0)
                    & (pl.col(col_28d) == 1)
                )
                .then(
                    pl.lit("lt_28d_only")
                )
                .otherwise(
                    pl.lit("healthy")
                )
                .alias("group")
            ]
        )
    )

    # Split patient identifiers by mortality group.
    ids_24h = (
        patient_labels
        .filter(
            pl.col("group") == "lt_24h"
        )
        .select(patient_col)
    )

    ids_28d = (
        patient_labels
        .filter(
            pl.col("group") == "lt_28d_only"
        )
        .select(patient_col)
    )

    ids_healthy = (
        patient_labels
        .filter(
            pl.col("group") == "healthy"
        )
        .select(patient_col)
    )

    # Count patients in each group.
    n_24h = ids_24h.height
    n_28d = ids_28d.height
    n_healthy = ids_healthy.height

    if n_24h == 0:
        raise ValueError(
            "No patient belongs to the lt_24h group."
        )

    # The sampled 28-day-only and healthy groups must match the size of the
    # 24-hour mortality group.
    if n_28d + n_healthy < n_24h:
        raise ValueError(
            "Not enough patients in the lt_28d_only and healthy groups "
            f"to match lt_24h: {n_28d} + {n_healthy} < {n_24h}"
        )

    n_28d_sample = min(
        n_28d,
        n_24h // 2,
    )

    n_healthy_sample = (
        n_24h - n_28d_sample
    )

    # Complete the sample with healthy patients when the 28-day-only group
    # does not contain enough patients.
    if n_28d_sample > n_28d:
        missing = (
            n_28d_sample - n_28d
        )

        n_28d_sample = n_28d

        n_healthy_sample = (
            n_healthy_sample + missing
        )

    # Sample patient identifiers from the non-24-hour groups.
    sampled_28d = ids_28d.sample(
        n=n_28d_sample,
        with_replacement=False,
        shuffle=True,
        seed=seed,
    )

    sampled_healthy = ids_healthy.sample(
        n=n_healthy_sample,
        with_replacement=False,
        shuffle=True,
        seed=seed,
    )

    # Retain all 24-hour mortality patients and sampled patients from the
    # remaining groups.
    selected_ids = pl.concat(
        [
            ids_24h,
            sampled_28d,
            sampled_healthy,
        ]
    )

    # Join selected identifiers back to the complete training dataset so that
    # all rows belonging to each selected patient are preserved.
    train_down = (
        train_df
        .join(
            selected_ids,
            on=patient_col,
            how="inner",
        )
        .sort(patient_col)
    )

    return train_down


def equilibrer_dataset_tabulaire(
    df: pl.DataFrame,
    id_col: str,
    target_col: str,
    method: str,
    seed: int = 42,
) -> pl.DataFrame:
    """Balance a tabular Polars DataFrame.

    Depending on ``method``, the function either leaves the dataset unchanged,
    applies the custom patient-level downsampling strategy, or uses an
    imbalanced-learn random sampler.

    Args:
        df:
            Input DataFrame.
        id_col:
            Name of the patient identifier column.
        target_col:
            Name of the binary target column.
        method:
            Balancing method identifier.
        seed:
            Random seed used during sampling.

    Returns:
        A tuple containing:

        - The balanced DataFrame.
        - A dictionary containing class counts before and after sampling,
          or ``None`` when no balancing is applied.

    Raises:
        ValueError:
            If the requested balancing method is unsupported.
    """
    # Count classes before sampling.
    n_sains_avant = (
        df
        .filter(
            pl.col(target_col) == 0
        )
        .height
    )

    n_malades_avant = (
        df
        .filter(
            pl.col(target_col) == 1
        )
        .height
    )

    if method == "":
        # No balancing is applied here.
        return df, None

    elif method == "downsampling_homemade":
        df_balanced = downsample_train_patients(
            df,
            patient_col=id_col,
            seed=seed,
        )

        stats = {
            "n_sains_avant": n_sains_avant,
            "n_malades_avant": n_malades_avant,
            "n_sains_apres": (
                df_balanced
                .filter(
                    pl.col(target_col) == 0
                )
                .height
            ),
            "n_malades_apres": (
                df_balanced
                .filter(
                    pl.col(target_col) == 1
                )
                .height
            ),
        }

        return df_balanced, stats

    # Convert to Pandas for imbalanced-learn.
    X_p = (
        df
        .select(
            pl.all().exclude(target_col)
        )
        .to_pandas()
    )

    y_p = (
        df
        .select(target_col)
        .to_pandas()
        .values
        .ravel()
    )

    if method == "downsampling_50-50":
        from imblearn.under_sampling import RandomUnderSampler

        rs = RandomUnderSampler(
            random_state=seed
        )

    elif method == "upsampling_50-50":
        from imblearn.over_sampling import RandomOverSampler

        rs = RandomOverSampler(
            random_state=seed
        )

    else:
        raise ValueError(
            f"Balancing method {method!r} is not supported."
        )

    X_res, y_res = rs.fit_resample(
        X_p,
        y_p,
    )

    # Convert the resampled dataset back to Polars.
    df_balanced = (
        pl.from_pandas(X_res)
        .with_columns(
            pl.Series(
                target_col,
                y_res,
            )
        )
    )

    # Count classes after sampling.
    stats = {
        "n_sains_avant": n_sains_avant,
        "n_malades_avant": n_malades_avant,
        "n_sains_apres": np.sum(
            y_res == 0
        ),
        "n_malades_apres": np.sum(
            y_res == 1
        ),
    }

    return df_balanced, stats


def collect_boruta_features_for_fold(
    fold_idx,
    train_idx,
    test_idx,
    X,
    y,
    groups,
    seed,
    **kwargs,
):
    """Run Boruta for one fold and return the selected feature list.

    This helper is used in the first phase of cross-fold Boruta selection:
    each fold runs correlation/variance filtering followed by Boruta, and the
    resulting feature list is collected. After all folds have been processed,
    the frequency of each feature across folds is analysed to determine a
    unified feature set.

    Args:
        fold_idx:
            Zero-based fold index.
        train_idx:
            Training indices for the current fold.
        test_idx:
            Validation indices for the current fold.
        X:
            Dataset used to recover patient identifiers for the split.
        y:
            Target array associated with the complete dataset.
        groups:
            Group array associated with the complete dataset.
        seed:
            Random seed used for feature selection.
        **kwargs:
            Pipeline configuration, including ``corr_threshold``,
            ``static_feats``, and ``keep_static``.

    Returns:
        The list of feature names selected by Boruta for this fold (after
        correlation/variance filtering). Returns an empty list when Boruta
        is disabled.
    """
    patient_col = kwargs["patient_col"]
    target_col = kwargs["target_col"]
    train_init_tsfel = kwargs["train_init"]
    static_feats = kwargs.get("static_feats", [])
    keep_static = kwargs.get("keep_static", True)
    boruta_filter = kwargs.get("boruta_filter", False)

    # Retrieve patient identifiers for the current fold.
    train_patients = (
        X[train_idx]
        .select(patient_col)
        .unique()
    )

    test_patients = (
        X[test_idx]
        .select(patient_col)
        .unique()
    )

    train_fold_tsfel = (
        train_init_tsfel
        .join(train_patients, on=patient_col, how="inner")
        .sort(patient_col)
    )

    test_fold_tsfel = (
        train_init_tsfel
        .join(test_patients, on=patient_col, how="inner")
        .sort(patient_col)
    )

    # Step 1: Correlation and variance filtering.
    corr_threshold = kwargs.get("corr_threshold", 0.95)
    train_clean, test_clean, _ = extract_feat.filtrage_corr_var(
        train_fold_tsfel,
        test_fold_tsfel,
        patient_col,
        target_col,
        corr_threshold=corr_threshold,
        static_features=static_feats,
        keep_static=keep_static,
    )

    if not boruta_filter:
        # No Boruta: return all features after corr/var filtering.
        feature_names = (
            train_clean
            .select(pl.exclude(patient_col, target_col))
            .columns
        )
        return list(feature_names)

    # Step 2: Boruta selection.
    _, _, boruta_features = extract_feat.filtrage_boruta(
        train_clean,
        test_clean,
        patient_col,
        target_col,
        max_iter=100,
        seed=seed,
        static_features=static_feats,
        keep_static=keep_static,
    )

    return list(boruta_features)


def process_tsfel_fold(
    fold_idx,
    train_idx,
    test_idx,
    X,
    y,
    groups,
    seed,
    **kwargs,
):
    """Prepare one cross-validation fold for TSFEL-based models.

    The pipeline performs patient selection, correlation and variance
    filtering, optional Boruta feature selection, class balancing, integrity
    checks, and feature scaling. The independent holdout is transformed with
    the exact feature set and scaler learned from the current training fold.

    When ``boruta_crossfold_features`` is provided in kwargs, the Boruta
    selection is skipped in favour of the unified cross-fold feature set.

    Args:
        fold_idx:
            Zero-based fold index.
        train_idx:
            Training indices for the current fold.
        test_idx:
            Validation indices for the current fold.
        X:
            Dataset used to recover patient identifiers for the split.
        y:
            Target array associated with the complete dataset.
        groups:
            Group array associated with the complete dataset.
        seed:
            Random seed used for feature selection and balancing.
        **kwargs:
            Additional pipeline configuration values, including the raw
            independent holdout under ``holdout_init``.

    Returns:
        A tuple containing training features, validation features, holdout
        features, training labels, validation labels, holdout labels, training
        groups, holdout patient identifiers, fold feature names, and balancing
        statistics.
    """
    # Extract the fold configuration.
    patient_col = kwargs["patient_col"]
    target_col = kwargs["target_col"]
    train_init_tsfel = kwargs["train_init"]
    holdout_init_tsfel = kwargs["holdout_init"]
    boruta_filter = kwargs["boruta_filter"]
    exp = kwargs["exp"]
    balance_method = kwargs["balance_method"]
    final_features = kwargs["final_features"]
    static_feats = kwargs.get("static_feats", [])
    keep_static = kwargs.get("keep_static", True)
    boruta_crossfold_features = kwargs.get("boruta_crossfold_features")

    # Retrieve patient identifiers associated with the current split.
    train_patients = (
        X[train_idx]
        .select(patient_col)
        .unique()
    )

    test_patients = (
        X[test_idx]
        .select(patient_col)
        .unique()
    )

    # Filter the precomputed TSFEL DataFrame for the current fold.
    train_fold_tsfel = (
        train_init_tsfel
        .join(train_patients, on=patient_col, how="inner")
        .sort(patient_col)
    )

    test_fold_tsfel = (
        train_init_tsfel
        .join(test_patients, on=patient_col, how="inner")
        .sort(patient_col)
    )

    filename_train_boruta = exp.get_tsfel_boruta(
            "train",
            fold_idx,
        )
    filename_test_boruta = exp.get_tsfel_boruta(
        "test",
        fold_idx,
    )

    # Apply correlation and variance filtering, then optional Boruta selection.
    # The correlation threshold is computed once before the fold loop in the
    # pipeline and passed via kwargs["corr_threshold"].
    if not (
            os.path.exists(filename_train_boruta)
            and os.path.exists(filename_test_boruta)
        ):

        # Remove correlated and zero-variance features.
        corr_threshold = kwargs.get("corr_threshold", 0.95)
        train_clean, test_clean, keepVariableList_1 = (
            extract_feat.filtrage_corr_var(
                train_fold_tsfel,
                test_fold_tsfel,
                patient_col,
                target_col,
                corr_threshold=corr_threshold,
                static_features=static_feats,
                keep_static=keep_static,
            )
        )

    # Optionally apply Boruta feature selection.
    if boruta_filter:
        if boruta_crossfold_features is not None:
            # Use the unified cross-fold Boruta feature set instead of
            # running per-fold Boruta. Simply filter the already cleaned
            # data to keep only the features that passed the frequency
            # threshold across all folds.
            available_features = set(train_clean.columns) - {
                patient_col, target_col
            }
            crossfold_available = [
                f for f in boruta_crossfold_features
                if f in available_features
            ]
            print(
                f"[BORUTA CROSS-FOLD] Fold {fold_idx}: "
                f"using {len(crossfold_available)} features from "
                f"unified cross-fold set "
                f"(out of {len(boruta_crossfold_features)} total)."
            )
            train_clean = train_clean.select(
                [patient_col, target_col, *crossfold_available]
            )
            test_clean = test_clean.select(
                [patient_col, target_col, *crossfold_available]
            )
            keepVariableList_2 = crossfold_available
        elif (
            os.path.exists(filename_train_boruta)
            and os.path.exists(filename_test_boruta)
        ):
            print(
                "Reading existing Boruta files for "
                f"fold {fold_idx} (seed {seed})."
            )
            train_clean = pl.read_parquet(filename_train_boruta)
            test_clean = pl.read_parquet(filename_test_boruta)
        else:
            train_clean, test_clean, keepVariableList_2 = (
                extract_feat.filtrage_boruta(
                    train_clean,
                    test_clean,
                    patient_col,
                    target_col,
                    max_iter=100,
                    seed=seed,
                    static_features=static_feats,
                    keep_static=keep_static,
                )
            )

            train_clean.write_parquet(filename_train_boruta)
            test_clean.write_parquet(filename_test_boruta)

            print(
                "Saving Boruta results for "
                f"fold {fold_idx} (seed {seed})."
            )

            parent_folder = filename_train_boruta.parent
            np.save(
                parent_folder / f"keepVariableList_1_fold_{fold_idx}.npy",
                keepVariableList_1,
            )
            np.save(
                parent_folder / f"keepVariableList_2_fold_{fold_idx}.npy",
                keepVariableList_2,
            )

    # Sort the fold data before alignment and balancing checks.
    train_clean = train_clean.sort(patient_col)
    test_clean = test_clean.sort(patient_col)

    # Keep the exact final feature set selected for the current fold.
    final_feature_names = (
        train_clean
        .select(pl.exclude(patient_col, target_col))
        .columns
    )

    if not final_feature_names:
        raise ValueError(
            f"No TSFEL feature remains after filtering in fold {fold_idx}."
        )

    # Verify that the independent holdout contains every selected feature.
    missing_holdout_columns = sorted(
        set(final_feature_names) - set(holdout_init_tsfel.columns)
    )
    if missing_holdout_columns:
        raise ValueError(
            "The TSFEL holdout is missing fold-selected features: "
            f"{missing_holdout_columns}"
        )

    # Select and order the holdout using the fold-specific feature set.
    holdout_clean = (
        holdout_init_tsfel
        .select([patient_col, target_col, *final_feature_names])
        .sort(patient_col)
    )

    # Sanitize holdout: replace NaN and infinite values with medians computed from
    # the training fold. The holdout dataset bypasses feature filtering (which
    # performs median imputation on train/test), so non-finite values must be
    # handled here to prevent models that do not natively support missing values
    # (e.g., SVC, LogisticRegression) from crashing at prediction time.
    _holdout_impute = {
        col: train_clean[col].median()
        for col in final_feature_names
    }

    for _col, _med in _holdout_impute.items():
        # Fallback to 0.0 if the training median itself is undefined or NaN.
        med_val = 0.0 if (_med is None or np.isnan(_med)) else _med

        holdout_clean = holdout_clean.with_columns(
            pl.col(_col)
            .replace_infinite(None)  # Convert +/-inf to null
            .fill_nan(None)          # Convert NaN to null
            .fill_null(med_val)      # Impute all missing values using the train median
            .alias(_col)
        )

    if holdout_clean.height != holdout_clean[patient_col].n_unique():
        raise ValueError(
            "The TSFEL holdout must contain exactly one row per patient."
        )

    # Preserve one identifier and one target value per holdout patient.
    holdout_patient_ids = (
        holdout_clean[patient_col]
        .to_numpy()
        .reshape(-1)
    )
    y_holdout = (
        holdout_clean[target_col]
        .to_numpy()
        .reshape(-1)
    )

    # Apply class balancing only to the training fold.
    train_clean, stats = equilibrer_dataset_tabulaire(
        train_clean,
        patient_col,
        target_col,
        method=balance_method,
        seed=seed,
    )

    # Upsampling necessarily duplicates patient identifiers.
    if balance_method != "upsampling_50-50":
        assert train_clean.height == train_clean[patient_col].n_unique(), (
            "Train TSFEL alignment error in "
            f"fold {fold_idx}"
        )
        assert test_clean.height == test_clean[patient_col].n_unique(), (
            "Validation TSFEL alignment error in "
            f"fold {fold_idx}"
        )

    # Extract labels and training patient groups before removing metadata.
    y_train_fold = train_clean[target_col].to_numpy().reshape(-1)
    y_test_fold = test_clean[target_col].to_numpy().reshape(-1)
    groups_fold = train_clean[patient_col].to_numpy().reshape(-1)

    # Retain only model features in the same order for all three datasets.
    train_features = train_clean.select(final_feature_names)
    test_features = test_clean.select(final_feature_names)
    holdout_features = holdout_clean.select(final_feature_names)

    # Fit scaling on the training fold and transform validation and holdout.
    (
        X_train_fold,
        X_test_fold,
        X_holdout_fold,
    ) = scaling(
        train_features,
        test_features,
        holdout_features,
    )

    return (
        X_train_fold,
        X_test_fold,
        X_holdout_fold,
        y_train_fold,
        y_test_fold,
        y_holdout,
        groups_fold,
        holdout_patient_ids,
        final_feature_names,
        stats,
    )



def process_time_fold(
    fold_idx,
    train_idx,
    test_idx,
    seed,
    **kwargs,
):
    """Prepare one cross-validation fold for time-series models.

    The pipeline extracts fold-specific patient data, optionally balances the
    training set, scales numerical features, builds fixed-length 3D sequences,
    replaces remaining missing values, and saves the resulting arrays. The
    independent holdout is transformed using the scaler fitted exclusively on
    the current training fold.

    Args:
        fold_idx:
            Zero-based fold index.
        train_idx:
            Training indices for the current fold.
        test_idx:
            Validation indices for the current fold.
        seed:
            Random seed used during balancing.
        **kwargs:
            Additional pipeline configuration values, including the raw
            independent holdout under ``holdout_init``.

    Returns:
        A tuple containing training sequences, validation sequences, holdout
        sequences, training labels, validation labels, holdout labels, training
        groups, holdout patient identifiers, ordered feature names, and
        balancing statistics.
    """
    patient_col = kwargs["patient_col"]
    time_col = kwargs["time_col"]
    target_col = kwargs["target_col"]
    train_init_df = kwargs["train_init"]
    holdout_init_df = kwargs["holdout_init"]
    final_features = kwargs["final_features"]
    balance_method = kwargs["balance_method"]
    expected_length = kwargs["expected_length"]
    exp = kwargs["exp"]

    print(
        f"\n[TIME FOLD {fold_idx + 1}/5] === process_time_fold START ===",
        flush=True,
    )
    print(
        f"[TIME FOLD {fold_idx + 1}] balance_method={balance_method!r}, "
        f"expected_length={expected_length}, "
        f"final_features count={len(final_features)}",
        flush=True,
    )
    print(
        f"[TIME FOLD {fold_idx + 1}] train_init_df rows={len(train_init_df)}, "
        f"holdout_init_df rows={len(holdout_init_df)}",
        flush=True,
    )
    print(
        f"[TIME FOLD {fold_idx + 1}] train_idx length={len(train_idx)}, "
        f"test_idx length={len(test_idx)}",
        flush=True,
    )

    # Extract and sort train, validation, and holdout observations.
    train_df = train_init_df[train_idx].sort([patient_col, time_col])
    test_df = train_init_df[test_idx].sort([patient_col, time_col])
    holdout_df = holdout_init_df.sort([patient_col, time_col])

    print(
        f"[TIME FOLD {fold_idx + 1}] After split -> train_df rows={len(train_df)}, "
        f"test_df rows={len(test_df)}, holdout_df rows={len(holdout_df)}",
        flush=True,
    )
    print(
        f"[TIME FOLD {fold_idx + 1}] After split -> train unique patients="
        f"{train_df[patient_col].n_unique()}, "
        f"test unique patients={test_df[patient_col].n_unique()}, "
        f"holdout unique patients={holdout_df[patient_col].n_unique()}",
        flush=True,
    )

    # Apply custom patient-level balancing before sequence construction.
    print(
        f"[TIME FOLD {fold_idx + 1}] Entering balancing step with method={balance_method!r}",
        flush=True,
    )
    if balance_method in ["downsampling_homemade", ""]:
        train_df, _ = equilibrer_dataset_tabulaire(
            train_df,
            patient_col,
            target_col,
            method=balance_method,
            seed=seed,
        )
        print(
            f"[TIME FOLD {fold_idx + 1}] After balancing -> train_df rows={len(train_df)}, "
            f"unique patients={train_df[patient_col].n_unique()}",
            flush=True,
        )
    else:
        print(
            f"[TIME FOLD {fold_idx + 1}] Skipping early balancing (method={balance_method!r}), "
            f"will apply imbalanced-learn after build_sequences.",
            flush=True,
        )

    # Fit scaling on training rows and transform validation and holdout rows.
    print(
        f"[TIME FOLD {fold_idx + 1}] Entering scaling step...",
        flush=True,
    )
    train_df, test_df, holdout_df = scaling(
        train_df,
        test_df,
        holdout_df,
    )
    print(
        f"[TIME FOLD {fold_idx + 1}] After scaling -> train_df rows={len(train_df)}, "
        f"test_df rows={len(test_df)}, holdout_df rows={len(holdout_df)}",
        flush=True,
    )

    # Guarantee the same alphabetical feature order for every sequence.
    ordered_feature_names = sorted(list(final_features))
    print(
        f"[TIME FOLD {fold_idx + 1}] Ordered feature names ({len(ordered_feature_names)}): "
        f"{ordered_feature_names}",
        flush=True,
    )

    # Convert patient observations into fixed-length 3D sequences.
    print(
        f"[TIME FOLD {fold_idx + 1}] Entering build_sequences for TRAIN (patients={train_df[patient_col].n_unique()})...",
        flush=True,
    )
    X_train_fold, y_train_fold = build_sequences(
        train_df,
        patient_col,
        target_col,
        expected_length,
        ordered_feature_names,
    )
    print(
        f"[TIME FOLD {fold_idx + 1}] build_sequences TRAIN DONE -> X_train shape={X_train_fold.shape}, "
        f"y_train positives={int(y_train_fold.sum())}/{len(y_train_fold)}",
        flush=True,
    )

    print(
        f"[TIME FOLD {fold_idx + 1}] Entering build_sequences for VALIDATION (patients={test_df[patient_col].n_unique()})...",
        flush=True,
    )
    X_test_fold, y_test_fold = build_sequences(
        test_df,
        patient_col,
        target_col,
        expected_length,
        ordered_feature_names,
    )
    print(
        f"[TIME FOLD {fold_idx + 1}] build_sequences VALIDATION DONE -> X_test shape={X_test_fold.shape}, "
        f"y_test positives={int(y_test_fold.sum())}/{len(y_test_fold)}",
        flush=True,
    )

    print(
        f"[TIME FOLD {fold_idx + 1}] Entering build_sequences for HOLDOUT (patients={holdout_df[patient_col].n_unique()})...",
        flush=True,
    )
    X_holdout_fold, y_holdout = build_sequences(
        holdout_df,
        patient_col,
        target_col,
        expected_length,
        ordered_feature_names,
    )
    print(
        f"[TIME FOLD {fold_idx + 1}] build_sequences HOLDOUT DONE -> X_holdout shape={X_holdout_fold.shape}, "
        f"y_holdout positives={int(y_holdout.sum())}/{len(y_holdout)}",
        flush=True,
    )

    # Preserve the patient order used by the sequence builder.
    patients_time_fold = (
        train_df[patient_col]
        .unique()
        .sort()
        .to_numpy()
        .reshape(-1)
    )
    holdout_patient_ids = (
        holdout_df[patient_col]
        .unique()
        .sort()
        .to_numpy()
        .reshape(-1)
    )

    # Apply imbalanced-learn sampling using patient-sequence indices.
    if balance_method not in ["downsampling_homemade", ""]:
        n_sains_avant = int(np.sum(y_train_fold == 0))
        n_malades_avant = int(np.sum(y_train_fold == 1))

        n_samples, n_timesteps, n_feats = X_train_fold.shape
        X_train_fold_2d = X_train_fold.reshape(
            n_samples,
            n_timesteps * n_feats,
        )

        if balance_method == "downsampling_50-50":
            from imblearn.under_sampling import RandomUnderSampler
            rs = RandomUnderSampler(random_state=seed)
        elif balance_method == "upsampling_50-50":
            from imblearn.over_sampling import RandomOverSampler
            rs = RandomOverSampler(random_state=seed)
        else:
            raise ValueError(
                f"Balancing method {balance_method!r} is not implemented."
            )

        indices_arr = np.arange(n_samples).reshape(-1, 1)
        indices_resampled, y_train_fold = rs.fit_resample(
            indices_arr,
            y_train_fold,
        )
        indices_resampled = indices_resampled.flatten()

        X_train_fold = (
            X_train_fold_2d[indices_resampled]
            .reshape(-1, n_timesteps, n_feats)
        )
        groups_fold = patients_time_fold[indices_resampled]

        stats = {
            "n_sains_avant": n_sains_avant,
            "n_malades_avant": n_malades_avant,
            "n_sains_apres": int(np.sum(y_train_fold == 0)),
            "n_malades_apres": int(np.sum(y_train_fold == 1)),
        }
    else:
        stats = None
        groups_fold = patients_time_fold

    # Count and replace remaining NaN values before training and inference.
    print(
        "Total number of NaN values in the training fold: "
        f"{np.isnan(X_train_fold).sum()}"
    )

    X_train_fold = np.nan_to_num(X_train_fold, nan=0.0)
    X_test_fold = np.nan_to_num(X_test_fold, nan=0.0)
    X_holdout_fold = np.nan_to_num(X_holdout_fold, nan=0.0)

    # Save the fold-specific train and validation arrays as before.
    np.save(
        exp.get_time_path(mode="train", fold_idx=fold_idx),
        X_train_fold,
    )
    np.save(
        exp.get_time_path(mode="test", fold_idx=fold_idx),
        X_test_fold,
    )

    return (
        X_train_fold,
        X_test_fold,
        X_holdout_fold,
        y_train_fold,
        y_test_fold,
        y_holdout,
        groups_fold,
        holdout_patient_ids,
        ordered_feature_names,
        stats,
    )


