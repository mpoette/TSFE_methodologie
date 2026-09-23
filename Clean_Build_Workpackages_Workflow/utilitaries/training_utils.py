"""Model training and calibration utilities for ICU prediction pipelines.

Provides functions to train neural-network and classical ML models,
apply temperature scaling and Platt calibration, handle class weighting,
and manage cross-validation folds with early stopping and checkpointing.
"""

import logging
import numpy as np
import polars as pl
import torch

logger = logging.getLogger(__name__)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from utilitaries.models.inceptionTimeModified import (
    predict_proba,
    train_inception_time,
)
from utilitaries.models.lstmTimeModified import (
    predict_proba_lstm,
    train_lstm_model,
)
from utilitaries.models.vanillaTransformerModified import (
    predict_proba as predict_proba_vt,
    train_vanilla_transformer,
)


# ============================================================================
# CALIBRATION WRAPPERS
# ============================================================================

class TemperatureScaledEstimator:
    """Wrap an estimator with a PyTorch temperature calibrator.

    The wrapper exposes a Scikit-Learn-compatible interface while applying
    temperature scaling to the positive-class logits returned by the
    underlying estimator.

    The calibrator is always placed in evaluation mode during inference.

    Args:
        estimator:
            Fitted estimator exposing a ``predict_proba`` method.
        calibrator:
            Fitted PyTorch temperature calibrator.
    """

    def __init__(self, estimator, calibrator):
        """Initialize the temperature-scaled estimator.

        Args:
            estimator:
                Fitted base estimator.
            calibrator:
                Fitted PyTorch temperature calibrator.
        """
        self.estimator = estimator
        self.calibrator = calibrator

        # Keep the calibrator in evaluation mode during inference.
        self.calibrator.eval()

        # Copy the class labels exposed by the wrapped estimator.
        if hasattr(estimator, "classes_"):
            self.classes_ = estimator.classes_
        else:
            self.classes_ = np.array([0, 1])

    def __getattr__(self, name):
        """Delegate unknown public attributes to the wrapped estimator.

        Args:
            name:
                Attribute name.

        Returns:
            The corresponding attribute from the wrapped estimator.

        Raises:
            AttributeError:
                If a private attribute is requested or the wrapped estimator
                has not been initialized yet.
        """
        if name.startswith("_"):
            raise AttributeError(
                f"Private or special attribute {name!r} is not handled "
                "by the wrapper."
            )

        if "estimator" not in self.__dict__:
            raise AttributeError(
                "The base estimator has not been initialized yet."
            )

        return getattr(self.estimator, name)

    def predict_proba(self, X):
        """Return temperature-calibrated class probabilities.

        Args:
            X:
                Input feature matrix.

        Returns:
            A two-column array containing class-0 and class-1 probabilities.
        """
        # Retrieve raw probabilities from the underlying estimator.
        probas = self.estimator.predict_proba(X)

        # Clip probabilities to avoid undefined logarithms.
        eps = 1e-7
        probas = np.clip(
            probas,
            eps,
            1 - eps,
        )

        # Convert positive-class probabilities to logits.
        p1 = probas[:, 1]
        logits = np.log(
            p1 / (1 - p1)
        )

        # Apply the learned temperature.
        logits_tensor = torch.tensor(
            logits,
            dtype=torch.float32,
        )

        with torch.no_grad():
            calibrated_logits = (
                self.calibrator(logits_tensor)
                .cpu()
                .numpy()
            )

        # Convert calibrated logits back to probabilities.
        calib_p1 = 1 / (
            1 + np.exp(-calibrated_logits)
        )
        calib_p0 = 1 - calib_p1

        return np.vstack(
            [
                calib_p0,
                calib_p1,
            ]
        ).T

    def predict(self, X):
        """Predict binary class labels using a 0.5 threshold.

        Args:
            X:
                Input feature matrix.

        Returns:
            Predicted binary labels.
        """
        return (
            self.predict_proba(X)[:, 1]
            >= 0.5
        ).astype(int)

    def score(self, X, y):
        """Return classification accuracy.

        Args:
            X:
                Input feature matrix.
            y:
                Ground-truth labels.

        Returns:
            Mean classification accuracy.
        """
        predictions = self.predict(X)
        y_array = np.asarray(y)

        return np.mean(
            predictions == y_array
        )


class PriorCorrectionWrapper:
    """Wrap an estimator with class-prior probability correction.

    Args:
        base_estimator:
            Fitted estimator exposing ``predict_proba``.
        beta:
            Prior-correction factor.
    """

    def __init__(self, base_estimator, beta):
        """Initialize the prior-correction wrapper.

        Args:
            base_estimator:
                Fitted base estimator.
            beta:
                Prior-correction factor.
        """
        self.base_estimator = base_estimator
        self.beta = beta

        self.classes_ = getattr(
            base_estimator,
            "classes_",
            np.array([0, 1]),
        )

    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped estimator.

        Args:
            name:
                Attribute name.

        Returns:
            The corresponding attribute from the wrapped estimator.

        Raises:
            AttributeError:
                If a special attribute is requested or the base estimator has
                not yet been restored during unpickling.
        """
        # Prevent delegation of Python special methods during pickling and
        # unpickling.
        if (
            name.startswith("__")
            and name.endswith("__")
        ):
            raise AttributeError(name)

        # The wrapped estimator may not yet exist while the object is being
        # deserialized.
        if "base_estimator" not in self.__dict__:
            raise AttributeError(
                "The 'base_estimator' attribute has not been initialized yet."
            )

        return getattr(
            self.base_estimator,
            name,
        )

    def predict_proba(self, X):
        """Return prior-corrected class probabilities.

        Args:
            X:
                Input feature matrix.

        Returns:
            A two-column array containing corrected class probabilities.
        """
        raw_probas = (
            self.base_estimator
            .predict_proba(X)
        )

        p = raw_probas[:, 1]

        p_calibrated = p / (
            p
            + (
                (1 - p)
                / self.beta
            )
        )

        calibrated_probas = np.zeros_like(
            raw_probas
        )

        calibrated_probas[:, 0] = (
            1 - p_calibrated
        )

        calibrated_probas[:, 1] = (
            p_calibrated
        )

        return calibrated_probas

    def predict(self, X):
        """Predict binary class labels using corrected probabilities.

        Args:
            X:
                Input feature matrix.

        Returns:
            Predicted binary labels.
        """
        return (
            self.predict_proba(X)[:, 1]
            >= 0.5
        ).astype(int)


# ============================================================================
# PRIOR CORRECTION
# ============================================================================

def apply_prior_calibration(model, beta):
    """Wrap a fitted model with prior-probability correction.

    Existing known wrappers are removed before applying the new correction.

    Args:
        model:
            Fitted estimator or supported estimator wrapper.
        beta:
            Prior-correction factor.

    Returns:
        A ``PriorCorrectionWrapper`` around the underlying raw estimator.
    """
    
    # Unwrap an existing prior-correction wrapper.
    class_name = type(model).__name__

    if class_name == "PriorCorrectionWrapper":
        model = model.base_estimator

    # Unwrap estimator-based calibration wrappers.
    elif class_name in ["TemperatureScaledEstimator", "FrozenEstimator"]:
        model = model.estimator

    return PriorCorrectionWrapper(
        base_estimator=model,
        beta=beta,
    )


# ============================================================================
# LEARNING-CURVE SAMPLING
# ============================================================================

def get_learning_curve_chunk(
    X_train,
    y_train,
    groups,
    p,
    is_dl_model,
    seed,
):
    """Select a subset of the training data for one learning-curve level.

    Deep-learning datasets are sampled directly by row index. Classical
    machine-learning datasets are sampled at patient level using ``groups``.

    Args:
        X_train:
            Complete training feature matrix.
        y_train:
            Complete training labels.
        groups:
            Patient or group identifiers.
        p:
            Fraction of the training data to retain.
        is_dl_model:
            Whether the dataset belongs to a deep-learning model.
        seed:
            Random seed used for shuffling.

    Returns:
        A tuple containing the selected features, selected labels, and the
        group-selection mask. The mask is ``None`` for deep-learning models.
    """
    total_samples = (
        X_train.shape[0]
        if hasattr(X_train, "shape")
        else len(X_train)
    )

    size_chunk = int(
        p * total_samples
    )

    if is_dl_model:
        indices = np.arange(
            total_samples
        )

        np.random.default_rng(
            seed=seed
        ).shuffle(indices)

        selected = indices[
            :size_chunk
        ]

        return (
            X_train[selected],
            np.asarray(y_train)[selected],
            None,
        )

    else:
        unique_patients = np.unique(
            groups
        )

        np.random.default_rng(
            seed=seed
        ).shuffle(unique_patients)

        n_patients = int(
            p * len(unique_patients)
        )

        selected = unique_patients[
            :n_patients
        ]

        mask = np.isin(
            groups,
            selected,
        )

        X_chunk = (
            X_train.filter(
                pl.Series(mask)
            )
            if hasattr(X_train, "filter")
            else X_train[mask]
        )

        X_chunk_np = (
            X_chunk.to_numpy()
            if hasattr(X_chunk, "to_numpy")
            else np.asarray(X_chunk)
        )

        return (
            X_chunk_np,
            np.asarray(y_train)[mask],
            mask,
        )


# ============================================================================
# MODEL TRAINING DISPATCH
# ============================================================================

def fit_model_by_name(
    model_name,
    X_train,
    y_train,
    X_val,
    y_val,
    seed,
    is_final_palier,
    save_path=None,
    lasso_args=None,
    **parameters,
):
    """Train the selected model and compute train and validation ROC-AUC.

    Supported models include InceptionTime, LSTM, random forests, XGBoost,
    SVC, and L1-regularized logistic regression.

    Args:
        model_name:
            Internal model identifier.
        X_train:
            Training feature matrix.
        y_train:
            Training labels.
        X_val:
            Validation feature matrix.
        y_val:
            Validation labels.
        seed:
            Random seed used during training.
        is_final_palier:
            Whether the current learning-curve level is the final one.
        save_path:
            Optional path used to save the best deep-learning checkpoint.
        lasso_args:
            Optional arguments required by the Lasso logistic-regression
            workflow.
        **parameters:
            Additional model-specific hyperparameters.

    Returns:
        A tuple containing the fitted model, training ROC-AUC, and validation
        ROC-AUC.

    Raises:
        ValueError:
            If ``model_name`` is unsupported.
    """
    if model_name == "InceptionTimeModified":
        model, T, _, _ = train_inception_time(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            save_best_path=save_path,
            seed=seed,
            **parameters,
        )

        return (
            model,
            roc_auc_score(
                y_train,
                predict_proba(
                    model,
                    X_train,
                    T=T,
                ),
            ),
            roc_auc_score(
                y_val,
                predict_proba(
                    model,
                    X_val,
                    T=T,
                ),
            ),
        )

    elif model_name == "LstmTimeModified":
        model, T, _, _ = train_lstm_model(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            save_best_path=save_path,
            seed=seed,
            **parameters,
        )

        return (
            model,
            roc_auc_score(
                y_train,
                predict_proba_lstm(
                    model,
                    X_train,
                    T=T,
                ),
            ),
            roc_auc_score(
                y_val,
                predict_proba_lstm(
                    model,
                    X_val,
                    T=T,
                ),
            ),
        )

    elif model_name == "VanillaTransformerModified":
        model, T, _, _ = train_vanilla_transformer(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            save_best_path=save_path,
            seed=seed,
            **parameters,
        )

        return (
            model,
            roc_auc_score(
                y_train,
                predict_proba_vt(
                    model,
                    X_train,
                    T=T,
                ),
            ),
            roc_auc_score(
                y_val,
                predict_proba_vt(
                    model,
                    X_val,
                    T=T,
                ),
            ),
        )

    # Classical machine-learning models using TSFEL features.
    elif model_name == "RandomForest TSFEL":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            **parameters,
        )

    elif model_name == "RandomForest Imbalanced TSFEL":
        from imblearn.ensemble import BalancedRandomForestClassifier

        clf = BalancedRandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            **parameters,
        )

    elif model_name == "XGBoost TSFEL":
        from xgboost import XGBClassifier

        ratio = (
            np.sum(y_train == 0)
            / np.sum(y_train == 1)
            if np.sum(y_train == 1) > 0
            else 1.0
        )

        clf = XGBClassifier(
            scale_pos_weight=ratio,
            random_state=seed,
            eval_metric="logloss",
            missing=np.nan,
            n_jobs=-1,
            **parameters,
        )

    elif model_name == "SVC TSFEL":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import SVC

        base_svc = SVC(
            random_state=seed,
            **parameters,
        )
        clf = CalibratedClassifierCV(estimator=base_svc, ensemble=False)

    elif model_name == "Logistic Regression Lasso TSFEL":
        if is_final_palier and lasso_args:
            if hasattr(
                lasso_args["X_raw"],
                "write_parquet",
            ):
                lasso_args["X_raw"].write_parquet(
                    lasso_args["file_X"]
                )
            else:
                pl.DataFrame(
                    lasso_args["X_raw"]
                ).write_parquet(
                    lasso_args["file_X"]
                )

            np.save(
                lasso_args["file_y"],
                lasso_args["y_raw"],
            )

        groups_lasso = np.asarray(
            lasso_args["groups_mask"]
        ).reshape(-1)

        assert len(groups_lasso) == len(X_train), (
            "Lasso groups and training rows are misaligned: "
            f"{len(groups_lasso)} groups for {len(X_train)} rows."
        )

        assert np.unique(groups_lasso).size >= 3, (
            "StratifiedGroupKFold requires at least 3 distinct groups, "
            f"but received {np.unique(groups_lasso).size}."
        )

        logger.debug(
            "LASSO INNER CV: rows=%d, groups=%d, unique_groups=%d",
            len(X_train),
            len(groups_lasso),
            np.unique(groups_lasso).size,
        )

        inner_cv = StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=seed,
        )

        lr = LogisticRegression(
            l1_ratio=1.0,
            solver="saga",
            max_iter=10000,
            random_state=seed,
        )

        lasso_cv = GridSearchCV(
            estimator=lr,
            param_grid={
                "C": np.logspace(
                    -4,
                    4,
                    10,
                )
            },
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1,
        )

        lasso_cv.fit(
            X_train,
            y_train,
            groups=groups_lasso,
        )

        # Always return the already-fitted best estimator.
        clf = lasso_cv.best_estimator_
    
    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    # Fit the selected classical model and compute ROC-AUC scores.
    if model_name != "Logistic Regression Lasso TSFEL":
        clf.fit(
            X_train,
            (
                y_train
                if "XGB" not in model_name
                else y_train.astype(int)
            ),
        )

    train_score = roc_auc_score(
        y_train,
        clf.predict_proba(
            X_train
        )[:, 1],
    )

    val_score = roc_auc_score(
        y_val,
        clf.predict_proba(
            X_val
        )[:, 1],
    )

    return (
        clf,
        train_score,
        val_score,
    )


def dl_validation_auc(model_name, model, X_val, y_val):
    """Compute the ROC-AUC of a fitted deep-learning model on a given set.

    Used for the learning curve when early stopping and temperature scaling
    were fitted on the held-out calibration split instead of the CV
    validation fold.

    Args:
        model_name:
            Internal deep-learning model identifier.
        model:
            Fitted PyTorch model returned by ``fit_model_by_name``.
        X_val:
            Validation tensor of shape (N, T, F).
        y_val:
            Validation labels.

    Returns:
        The ROC-AUC on ``(X_val, y_val)``.
    """
    predictors = {
        "InceptionTimeModified": predict_proba,
        "LstmTimeModified": predict_proba_lstm,
        "VanillaTransformerModified": predict_proba_vt,
    }
    temperature = float(getattr(model, "temperature_", 1.0))
    probas = predictors[model_name](model, np.asarray(X_val), T=temperature)
    return roc_auc_score(np.asarray(y_val).reshape(-1), np.asarray(probas).reshape(-1))


# ============================================================================
# POST-TRAINING CALIBRATION
# ============================================================================

def apply_model_calibration(
    model,
    X_calib,
    y_calib,
    method_calib,
):
    """Calibrate an already fitted root estimator.

    Supported calibration methods are Platt scaling and temperature scaling.

    Args:
        model:
            Fitted estimator or supported wrapper.
        X_calib:
            Calibration feature matrix.
        y_calib:
            Calibration labels.
        method_calib:
            Calibration method identifier.

    Returns:
        The calibrated estimator.

    Raises:
        ValueError:
            If the requested calibration method is unsupported.
    """
    base_model = get_root_estimator(
        model
    )

    if hasattr(
        X_calib,
        "to_numpy",
    ):
        X_calib = X_calib.to_numpy()

    else:
        X_calib = np.asarray(
            X_calib
        )

    if method_calib == "platt":
        calibrated_clf = CalibratedClassifierCV(
            estimator=FrozenEstimator(
                base_model
            ),
            method="sigmoid",
        )

        calibrated_clf.fit(
            X_calib,
            y_calib,
        )

        return calibrated_clf

    if method_calib == "temperature_scaling":
        from utilitaries.models.inceptionTimeModified import (
            TemperatureCalibrator,
        )

        probas = np.clip(
            base_model.predict_proba(
                X_calib
            )[:, 1],
            1e-7,
            1 - 1e-7,
        )

        logits = np.log(
            probas / (1 - probas)
        )

        calibrator = TemperatureCalibrator(
            init_T=1.0
        )

        calibrator.fit(
            torch.tensor(
                logits,
                dtype=torch.float32,
            ),
            torch.tensor(
                y_calib,
                dtype=torch.float32,
            ),
            max_iter=200,
        )

        return TemperatureScaledEstimator(
            base_model,
            calibrator,
        )

    raise ValueError(
        f"Calibration method {method_calib!r} is not supported."
    )


# ============================================================================
# ESTIMATOR INTROSPECTION
# ============================================================================

def get_root_estimator(estimator_to_unwrap):
    """Recursively extract the root estimator from known wrappers.

    Supported wrappers include ``CalibratedClassifierCV``,
    ``FrozenEstimator``, ``TemperatureScaledEstimator``, and
    ``PriorCorrectionWrapper``.

    Args:
        estimator_to_unwrap:
            Estimator or supported wrapper.

    Returns:
        The underlying root estimator.

    Raises:
        RuntimeError:
            If a ``CalibratedClassifierCV`` instance has not been fitted.
    """
    class_name = type(
        estimator_to_unwrap
    ).__name__

    if class_name == "CalibratedClassifierCV":
        calibrated_classifiers = getattr(
            estimator_to_unwrap,
            "calibrated_classifiers_",
            None,
        )

        if not calibrated_classifiers:
            raise RuntimeError(
                "CalibratedClassifierCV has not been fitted."
            )

        return get_root_estimator(
            calibrated_classifiers[
                0
            ].estimator
        )

    if class_name == "FrozenEstimator":
        return get_root_estimator(
            estimator_to_unwrap.estimator
        )

    if class_name == "TemperatureScaledEstimator":
        return get_root_estimator(
            estimator_to_unwrap.estimator
        )

    if class_name == "PriorCorrectionWrapper":
        return get_root_estimator(
            estimator_to_unwrap.base_estimator
        )

    return estimator_to_unwrap