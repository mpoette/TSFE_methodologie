import numpy as np
import polars as pl
import torch

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV

from utilitaries.models.inceptionTimeModified import (
    predict_proba,
    train_inception_time,
)
from utilitaries.models.lstmTimeModified import (
    predict_proba_lstm,
    train_lstm_model,
)

class TemperatureScaledEstimator:
    """
    Encapsule un modèle Scikit-Learn/XGBoost et un TemperatureCalibrator PyTorch.
    Version blindée contre le NotFittedError de Scikit-Learn.
    """
    def __init__(self, estimator, calibrator):
        self.estimator = estimator
        self.calibrator = calibrator
        self.calibrator.eval() # Toujours en mode eval pour l'inférence

        # Copie des classes
        if hasattr(estimator, 'classes_'):
            self.classes_ = estimator.classes_
        else:
            self.classes_ = np.array([0, 1])

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"Attribut privé ou magique '{name}' non géré par le wrapper.")

        if 'estimator' not in self.__dict__:
            raise AttributeError("L'estimateur de base n'est pas encore initialisé.")

        return getattr(self.estimator, name)

    def predict_proba(self, X):
        # 1. Récupérer les probabilités brutes (on s'assure d'appeler le vrai sous-modèle)
        probas = self.estimator.predict_proba(X)

        # Clip pour éviter les log(0) fatals
        eps = 1e-7
        probas = np.clip(probas, eps, 1 - eps)

        # 2. Extraire la proba de la classe positive et convertir en logits
        p1 = probas[:, 1]
        logits = np.log(p1 / (1 - p1))

        # 3. Appliquer la température via le calibrateur
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        with torch.no_grad():
            calibrated_logits = self.calibrator(logits_tensor).cpu().numpy()

        # 4. Reconvertir en probabilités via la fonction sigmoïde
        calib_p1 = 1 / (1 + np.exp(-calibrated_logits))
        calib_p0 = 1 - calib_p1

        return np.vstack([calib_p0, calib_p1]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        predictions = self.predict(X)
        y_array = np.asarray(y)
        return np.mean(predictions == y_array)

class PriorCorrectionWrapper:
    def __init__(self, base_estimator, beta):
        self.base_estimator = base_estimator
        self.beta = beta
        self.classes_ = getattr(base_estimator, "classes_", np.array([0, 1]))

    def __getattr__(self, name):
        # 1. SÉCURITÉ PICKLE : On bloque immédiatement les méthodes magiques/privées
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
            
        # 2. SÉCURITÉ UNPICKLING : Si base_estimator n'est pas encore chargé en mémoire
        if "base_estimator" not in self.__dict__:
            raise AttributeError("L'attribut 'base_estimator' n'est pas encore initialisé.")
            
        return getattr(self.base_estimator, name)

    def predict_proba(self, X):
        raw_probas = self.base_estimator.predict_proba(X)
        p = raw_probas[:, 1]
        p_calibrated = p / (p + ((1 - p) / self.beta))
        
        calibrated_probas = np.zeros_like(raw_probas)
        calibrated_probas[:, 0] = 1 - p_calibrated
        calibrated_probas[:, 1] = p_calibrated
        return calibrated_probas

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def apply_prior_calibration(model, beta):
    """Fonction dédiée (déplacée hors de la classe) pour encapsuler le modèle."""
    # Si le modèle est déjà encapsulé, on extrait le modèle brut d'abord
    if hasattr(model, "base_estimator"):
        model = model.base_estimator
    elif hasattr(model, "estimator"):
        model = model.estimator
        
    return PriorCorrectionWrapper(base_estimator=model, beta=beta)


def get_learning_curve_chunk(X_train, y_train, groups, p, is_dl_model, seed):
    """Découpe les données pour un palier donné (%)."""
    total_samples = X_train.shape[0] if hasattr(X_train, "shape") else len(X_train)
    size_chunk = int(p * total_samples)

    if is_dl_model:
        indices = np.arange(total_samples)
        np.random.default_rng(seed=seed).shuffle(indices)
        selected = indices[:size_chunk]
        return X_train[selected], np.asarray(y_train)[selected], None
    else:
        unique_patients = np.unique(groups)
        np.random.default_rng(seed=seed).shuffle(unique_patients)
        n_patients = int(p * len(unique_patients))
        selected = unique_patients[:n_patients]

        mask = np.isin(groups, selected)
        X_chunk = X_train.filter(pl.Series(mask)) if hasattr(X_train, "filter") else X_train[mask]

        X_chunk_np = X_chunk.to_numpy() if hasattr(X_chunk, "to_numpy") else np.asarray(X_chunk)
        return X_chunk_np, np.asarray(y_train)[mask], mask


def fit_model_by_name(model_name, X_train, y_train, X_val, y_val, seed, is_final_palier, save_path=None, lasso_args=None, **parameters):
    """Entraîne le modèle sélectionné et retourne le modèle ainsi que ses scores ROC-AUC (Train, Val)."""

    if model_name == "InceptionTimeModified":
        model, T, _, _ = train_inception_time(X_train, y_train, X_val=X_val, y_val=y_val, save_best_path=save_path, seed=seed, **parameters)
        return model, roc_auc_score(y_train, predict_proba(model, X_train, T=T)), roc_auc_score(y_val, predict_proba(model, X_val, T=T))

    elif model_name == "LstmTimeModified":
        model, T, _, _ = train_lstm_model(X_train, y_train, X_val=X_val, y_val=y_val, save_best_path=save_path, seed=seed, **parameters)
        return model, roc_auc_score(y_train, predict_proba_lstm(model, X_train, T=T)), roc_auc_score(y_val, predict_proba_lstm(model, X_val, T=T))

    # --- Modèles de Machine Learning (TSFEL) ---
    elif model_name == "RandomForest TSFEL":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=seed, n_jobs=-1, **parameters)

    elif model_name == "RandomForest Imbalanced TSFEL":
        from imblearn.ensemble import BalancedRandomForestClassifier
        clf = BalancedRandomForestClassifier(random_state=seed, n_jobs=-1, **parameters)

    elif model_name == "XGBoost TSFEL":
        from xgboost import XGBClassifier
        ratio = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1.0
        clf = XGBClassifier(scale_pos_weight=ratio, random_state=seed, eval_metric="logloss", missing=np.nan, n_jobs=-1, **parameters)

    elif model_name == "SVC TSFEL":
        from sklearn.svm import SVC
        clf = SVC(kernel="rbf", random_state=seed, probability=True, **parameters)

    elif model_name == "Logistic Regression Lasso TSFEL":
        if is_final_palier and lasso_args:
            if hasattr(lasso_args['X_raw'], "write_parquet"): lasso_args['X_raw'].write_parquet(lasso_args['file_X'])
            else: pl.DataFrame(X_train).write_parquet(lasso_args['file_X'])
            np.save(lasso_args['file_y'], lasso_args['y_raw'])

        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)

        # Pour la régression logistique, on passe les paramètres (comme max_iter) au solver de base
        lr_args = {"l1_ratio": 1.0, "solver": "saga", "max_iter": 10000, "random_state": seed}
        lr = LogisticRegression(**lr_args)
        lasso_cv = GridSearchCV(estimator=lr, param_grid={'C': np.logspace(-4, 4, 10)}, cv=inner_cv, scoring='roc_auc', n_jobs=-1)

        lasso_cv.fit(X_train, y_train, groups=lasso_args['groups_mask'])
        clf = lasso_cv.best_estimator_ if is_final_palier else lasso_cv

    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    # Fit standard et calcul du score pour le ML classique
    clf.fit(X_train, y_train if "XGB" not in model_name else y_train.astype(int))
    train_score = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    val_score = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    return clf, train_score, val_score


def apply_model_calibration(model, X_calib, y_calib, method_calib, seed):
    """Applique Platt Scaling ou Temperature Scaling sur un modèle de base."""
    # Formatage des sets de calibration
    if hasattr(model, "base_estimator"):
        model = model.base_estimator
    elif hasattr(model, "estimator"):
        model = model.estimator

        
    if "XGB" in type(model).__name__ or hasattr(X_calib, "to_numpy"):
        X_calib = X_calib.to_numpy() if hasattr(X_calib, "to_numpy") else np.asarray(X_calib)
        y_calib = np.asarray(y_calib).astype(int)

    if method_calib == "platt":
        frozen_model = FrozenEstimator(model)
        calibrated_clf = CalibratedClassifierCV(estimator=frozen_model, method="sigmoid")
        calibrated_clf.fit(X_calib, y_calib)
        return calibrated_clf

    elif method_calib == "temperature_scaling":
        from utilitaries.models.inceptionTimeModified import TemperatureCalibrator
        probas = np.clip(model.predict_proba(X_calib)[:, 1], 1e-7, 1 - 1e-7)
        logits = np.log(probas / (1 - probas))

        calibrator = TemperatureCalibrator(init_T=1.0)
        calibrator.fit(torch.tensor(logits, dtype=torch.float32), torch.tensor(y_calib, dtype=torch.float32), max_iter=200)
        return TemperatureScaledEstimator(model, calibrator)

    raise ValueError(f"Calibration {method_calib} non gérée.")



       