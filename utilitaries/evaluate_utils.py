import joblib

from utilitaries.models.inceptionTimeModified import (
    evaluate_on_test,
    load_model_from_checkpoint,
    predict_proba,
    train_inception_time,
)
from utilitaries.models.lstmTimeModified import (
    evaluate_lstm_on_test,
    load_lstm_from_checkpoint,
    predict_proba_lstm,
    train_lstm_model,
)

from utilitaries.training_utils import get_root_estimator

def align_tsfel_features(X_train, X_test, root_model):
    """Aligne les colonnes de X_train/X_test avec les features attendues par le modèle racine."""

    expected_features = None
    if hasattr(root_model, "feature_names_in_"):
        expected_features = list(root_model.feature_names_in_)
    elif hasattr(root_model, "get_booster"):
        expected_features = root_model.get_booster().feature_names

    # Cas sans features ou indices génériques XGBoost -> NumPy brut
    if expected_features is None:
        print("Aucun nom de feature trouvé dans le modèle racine. Passage en matrices NumPy brutes.")
        return X_train.to_numpy(), X_test.to_numpy()
        
    if expected_features and expected_features[0].startswith('f') and expected_features[0][1:].isdigit():
        print("XGBoost utilise des indices génériques. Utilisation des matrices brutes.")
        return X_train.to_numpy(), X_test.to_numpy()

    # Gestion des colonnes manquantes et ordonnancement (Polars)
    missing_cols = [c for c in expected_features if c not in X_train.columns]
    if missing_cols:
        print(f"Ajout de {len(missing_cols)} colonnes manquantes (0.0)")
        padding_expr = [pl.lit(0.0).alias(c) for c in missing_cols]
        X_train_final = X_train.with_columns(padding_expr).select(expected_features)
        X_test_final = X_test.with_columns(padding_expr).select(expected_features)
    else:
        X_train_final = X_train.select(expected_features)
        X_test_final = X_test.select(expected_features)

    # Conversion spécifique pour XGBoost
    if "XGB" in type(root_model).__name__:
        X_train_final = X_train_final.to_pandas()
        X_test_final = X_test_final.to_pandas()

    return X_train_final, X_test_final

def evaluate_inception_fold(X_test, y_test, loaded_model):
    auc, brier, T_1 = evaluate_on_test(X_test, y_test, loaded_model)
    model_1, _, T_1 = load_model_from_checkpoint(loaded_model)
    probas_fold = predict_proba(model_1, X_test, T=T_1)
    
    return {
        "auc": auc, "brier": brier, "y_test": y_test,
        "probas_uncalib": probas_fold, "probas_calib": probas_fold
    }


def evaluate_lstm_fold(fold_idx, X_test, y_test, loaded_model):
    print(f"\n[DEBUG EVAL - Fold {fold_idx + 1}] Shape de X_test_final: {X_test.shape}")
    auc, brier, T_1 = evaluate_lstm_on_test(X_test, y_test, loaded_model)
    model_1, _, T_1 = load_lstm_from_checkpoint(loaded_model)
    probas_fold = predict_proba_lstm(model_1, X_test, T=T_1)
    
    return {
        "auc": auc, "brier": brier, "y_test": y_test,
        "probas_uncalib": probas_fold, "probas_calib": probas_fold
    }


def evaluate_tsfel_fold(fold_idx, X_train, X_test, y_train, y_test, loaded_model, is_calibrated):
    clf = joblib.load(loaded_model)
    root_model = get_root_estimator(clf)
    
    # Alignement des colonnes
    X_train_final, X_test_final = align_tsfel_features(X_train, X_test, root_model)
    
    # Prédictions et scores
    y_pred_nb_test = clf.predict(X_test_final)
    train_score = clf.score(X_train_final, y_train)
    test_score = clf.score(X_test_final, y_test)
    
    # Extraction des probabilités (avec gestion XGBoost / Sklearn)
    X_test_final_numpy = X_test_final.to_numpy() if hasattr(X_test_final, "to_numpy") else X_test_final
    
    if is_calibrated:
        print("    [INFO] Objet de calibration détecté sur le disque.")
        prob_calib_fold = clf.predict_proba(X_test_final)[:, 1]
        if "XGB" in type(root_model).__name__:
            prob_uncalib_fold = root_model.predict_proba(X_test_final_numpy)[:, 1]
        else:
            prob_uncalib_fold = root_model.predict_proba(X_test_final)[:, 1]
    else:
        print("    [INFO] Modèle brut détecté (non calibré sur le disque).")
        prob_fold_brut = clf.predict_proba(X_test_final)[:, 1]
        prob_uncalib_fold = prob_fold_brut
        prob_calib_fold = prob_fold_brut

    print(f"Le score (Accuracy) sur le fold {fold_idx + 1} est : {test_score:.4f}")
    
    return {
        "train_score": train_score, "test_score": test_score,
        "y_test": y_test, "y_pred_test": y_pred_nb_test,
        "probas_uncalib": prob_uncalib_fold, "probas_calib": prob_calib_fold
    }