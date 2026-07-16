from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from utilitaries.models.inceptionTimeModified import (
    evaluate_on_test,
    load_model_from_checkpoint,
    predict_proba,
)
from utilitaries.models.lstmTimeModified import (
    evaluate_lstm_on_test,
    load_lstm_from_checkpoint,
    predict_proba_lstm,
)
from utilitaries.training_utils import get_root_estimator


def align_tsfel_features(
    X_train: pl.DataFrame,
    X_test: pl.DataFrame,
    root_model: Any,
) -> tuple[
    pl.DataFrame | np.ndarray | Any,
    pl.DataFrame | np.ndarray | Any,
]:
    """Align TSFEL feature columns with those expected by a fitted model.

    The function retrieves the feature names stored in the underlying model,
    adds missing columns with a default value of zero, and reorders the input
    datasets to match the training schema.

    When the model does not expose feature names, or when XGBoost uses generic
    feature names such as ``f0``, ``f1``, and so on, the input DataFrames are
    converted directly to NumPy arrays.

    XGBoost models using explicit feature names receive Pandas DataFrames,
    while other estimators retain Polars DataFrames.

    Args:
        X_train:
            Training feature matrix.
        X_test:
            Test feature matrix.
        root_model:
            Underlying fitted estimator, potentially extracted from a
            calibration or pipeline wrapper.

    Returns:
        A tuple containing the aligned training and test feature matrices.

        Depending on the estimator, the returned objects may be Polars
        DataFrames, Pandas DataFrames, or NumPy arrays.

    Raises:
        ValueError:
            If the test dataset is missing columns that are present in the
            training dataset but not handled by the expected model schema.
    """
    expected_features: list[str] | None = None

    if hasattr(root_model, "feature_names_in_"):
        expected_features = list(
            root_model.feature_names_in_
        )

    elif hasattr(root_model, "get_booster"):
        booster = root_model.get_booster()
        expected_features = booster.feature_names

    # Fall back to raw NumPy matrices when the estimator does not expose
    # explicit feature names.
    if expected_features is None:
        print(
            "No feature names were found in the root estimator. "
            "Using raw NumPy matrices."
        )
        return (
            X_train.to_numpy(),
            X_test.to_numpy(),
        )

    # XGBoost may store generic names such as f0, f1, and f2 when it was
    # originally trained from a NumPy array.
    uses_generic_xgboost_names = (
        bool(expected_features)
        and expected_features[0].startswith("f")
        and expected_features[0][1:].isdigit()
    )

    if uses_generic_xgboost_names:
        print(
            "XGBoost uses generic feature indices. "
            "Using raw NumPy matrices."
        )
        return (
            X_train.to_numpy(),
            X_test.to_numpy(),
        )

    # Ensure that every feature expected by the estimator exists in both
    # datasets, then enforce the exact training column order.
    missing_train_columns = [
        column
        for column in expected_features
        if column not in X_train.columns
    ]

    missing_test_columns = [
        column
        for column in expected_features
        if column not in X_test.columns
    ]

    if missing_train_columns:
        print(
            f"Adding {len(missing_train_columns)} missing training "
            "features with value 0.0."
        )

        X_train = X_train.with_columns(
            [
                pl.lit(0.0).alias(column)
                for column in missing_train_columns
            ]
        )

    if missing_test_columns:
        print(
            f"Adding {len(missing_test_columns)} missing test "
            "features with value 0.0."
        )

        X_test = X_test.with_columns(
            [
                pl.lit(0.0).alias(column)
                for column in missing_test_columns
            ]
        )

    X_train_final = X_train.select(
        expected_features
    )
    X_test_final = X_test.select(
        expected_features
    )

    # XGBoost generally preserves explicit feature-name validation more
    # reliably when it receives Pandas DataFrames.
    if "XGB" in type(root_model).__name__:
        X_train_final = X_train_final.to_pandas()
        X_test_final = X_test_final.to_pandas()

    return X_train_final, X_test_final


def evaluate_inception_fold(
    X_test: np.ndarray,
    y_test: np.ndarray,
    loaded_model: str | Path,
) -> dict[str, Any]:
    """Evaluate an InceptionTime model on one test fold.

    The checkpoint is evaluated using the project evaluation utility, then
    reloaded to generate both uncalibrated and temperature-calibrated
    probabilities.

    Args:
        X_test:
            Test time-series data.
        y_test:
            Ground-truth test labels.
        loaded_model:
            Path to the saved model checkpoint.

    Returns:
        A dictionary containing the AUC, Brier score, test labels,
        uncalibrated probabilities, and calibrated probabilities.
    """
    auc, brier, _ = evaluate_on_test(
        X_test,
        y_test,
        loaded_model,
    )

    model, _, temperature = load_model_from_checkpoint(
        loaded_model
    )

    # T=1.0 leaves the logits unchanged and therefore produces
    # uncalibrated probabilities.
    uncalibrated_probabilities = predict_proba(
        model,
        X_test,
        T=1.0,
    )

    # Apply the temperature learned during calibration.
    calibrated_probabilities = predict_proba(
        model,
        X_test,
        T=temperature,
    )

    return {
        "auc": auc,
        "brier": brier,
        "y_test": y_test,
        "probas_uncalib": uncalibrated_probabilities,
        "probas_calib": calibrated_probabilities,
    }


def evaluate_lstm_fold(
    fold_idx: int,
    X_test: np.ndarray,
    y_test: np.ndarray,
    loaded_model: str | Path,
) -> dict[str, Any]:
    """Evaluate an LSTM model on one test fold.

    The checkpoint is evaluated using the project evaluation utility, then
    reloaded to generate both uncalibrated and temperature-calibrated
    probabilities.

    Args:
        fold_idx:
            Zero-based fold index.
        X_test:
            Test time-series data.
        y_test:
            Ground-truth test labels.
        loaded_model:
            Path to the saved LSTM checkpoint.

    Returns:
        A dictionary containing the AUC, Brier score, test labels,
        uncalibrated probabilities, and calibrated probabilities.
    """
    print(
        f"\n[DEBUG EVAL - Fold {fold_idx + 1}] "
        f"X_test shape: {X_test.shape}"
    )

    auc, brier, _ = evaluate_lstm_on_test(
        X_test,
        y_test,
        loaded_model,
    )

    model, _, temperature = load_lstm_from_checkpoint(
        loaded_model
    )

    # T=1.0 leaves the logits unchanged and therefore produces
    # uncalibrated probabilities.
    uncalibrated_probabilities = predict_proba_lstm(
        model,
        X_test,
        T=1.0,
    )

    # Apply the temperature learned during calibration.
    calibrated_probabilities = predict_proba_lstm(
        model,
        X_test,
        T=temperature,
    )

    return {
        "auc": auc,
        "brier": brier,
        "y_test": y_test,
        "probas_uncalib": uncalibrated_probabilities,
        "probas_calib": calibrated_probabilities,
    }


def evaluate_tsfel_fold(
    fold_idx: int,
    X_train: pl.DataFrame,
    X_test: pl.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    loaded_model: str | Path,
    is_calibrated: bool,
) -> dict[str, Any]:
    """Evaluate a TSFEL-based estimator on one cross-validation fold.

    The serialized estimator is loaded from disk. Its underlying root
    estimator is then extracted so that TSFEL feature columns can be aligned
    with the schema expected during training.

    For calibrated models, probabilities from the calibration wrapper are
    returned as calibrated probabilities, while probabilities from the root
    estimator are returned as uncalibrated probabilities.

    Args:
        fold_idx:
            Zero-based fold index.
        X_train:
            Training TSFEL feature matrix.
        X_test:
            Test TSFEL feature matrix.
        y_train:
            Ground-truth training labels.
        y_test:
            Ground-truth test labels.
        loaded_model:
            Path to the serialized estimator.
        is_calibrated:
            Whether the loaded estimator is a calibrated wrapper.

    Returns:
        A dictionary containing training and test accuracy, ground-truth test
        labels, predicted classes, calibrated probabilities, and uncalibrated
        probabilities.

    Raises:
        AttributeError:
            If the loaded estimator or root estimator does not implement the
            required prediction methods.
        ValueError:
            If feature alignment fails.
    """
    classifier = joblib.load(
        loaded_model
    )

    root_model = get_root_estimator(
        classifier
    )

    # Align and reorder the TSFEL features according to the schema used when
    # the estimator was fitted.
    X_train_final, X_test_final = align_tsfel_features(
        X_train,
        X_test,
        root_model,
    )

    # Compute class predictions and accuracy scores.
    y_pred_test = classifier.predict(
        X_test_final
    )

    train_score = classifier.score(
        X_train_final,
        y_train,
    )

    test_score = classifier.score(
        X_test_final,
        y_test,
    )

    # Some root estimators, particularly XGBoost models fitted from NumPy
    # arrays, expect raw matrices rather than labeled DataFrames.
    X_test_numpy = (
        X_test_final.to_numpy()
        if hasattr(X_test_final, "to_numpy")
        else X_test_final
    )

    if is_calibrated:
        print(
            "    [INFO] Calibrated estimator detected on disk."
        )

        calibrated_probabilities = (
            classifier.predict_proba(
                X_test_final
            )[:, 1]
        )

        if "XGB" in type(root_model).__name__:
            uncalibrated_probabilities = (
                root_model.predict_proba(
                    X_test_numpy
                )[:, 1]
            )
        else:
            uncalibrated_probabilities = (
                root_model.predict_proba(
                    X_test_final
                )[:, 1]
            )

    else:
        print(
            "    [INFO] Uncalibrated estimator detected on disk."
        )

        raw_probabilities = (
            classifier.predict_proba(
                X_test_final
            )[:, 1]
        )

        uncalibrated_probabilities = raw_probabilities
        calibrated_probabilities = raw_probabilities

    print(
        f"Fold {fold_idx + 1} test accuracy: "
        f"{test_score:.4f}"
    )

    return {
        "train_score": train_score,
        "test_score": test_score,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "probas_uncalib": uncalibrated_probabilities,
        "probas_calib": calibrated_probabilities,
    }