# train.py
import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, classification_report
)

from xgboost import XGBClassifier


# ---------------------------- Utils ----------------------------
def set_seeds(seed: int = 42):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def compute_business_cost(y_true, y_pred, fn_cost=10, fp_cost=1):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn * fn_cost + fp * fp_cost


def find_best_threshold(y_true, y_probas, thresholds=np.linspace(0.1, 0.9, 81), fn_cost=10, fp_cost=1):
    best_t, best_cost = 0.5, float("inf")
    for t in thresholds:
        y_hat = (y_probas >= t).astype(int)
        c = compute_business_cost(y_true, y_hat, fn_cost, fp_cost)
        if c < best_cost:
            best_t, best_cost = t, c
    return best_t, best_cost


def default_param_space(neg_pos_ratio=None):
    space = {
        "n_estimators": ("int", 80, 200),
        "max_depth": ("int", 3, 6),
        "learning_rate": ("float", 0.01, 0.2),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
        "gamma": ("float", 0.0, 5.0),
        "reg_lambda": ("float", 0.001, 10.0),
        "reg_alpha": ("float", 0.001, 10.0),
    }
    if neg_pos_ratio is not None and neg_pos_ratio > 0:
        space["scale_pos_weight"] = ("float", 1.0, neg_pos_ratio)
    return space


# ---------------------------- Training ----------------------------
def run_training(
    data_dir: Path,
    models_dir: Path,
    test_size: float = 0.2,
    cv_splits: int = 3,
    n_trials: int = 30,
    random_state: int = 42,
    fn_cost: int = 10,
    fp_cost: int = 1,
):

    # 0) Seeds
    set_seeds(random_state)

    # 1) Load data
    train_path = data_dir / "train_final.csv"
    assert train_path.exists(), f"Dataset introuvable: {train_path}"
    df = pd.read_csv(train_path)
    assert "TARGET" in df.columns, "La colonne TARGET est absente de train_final.csv"

    X = df.drop(columns=["TARGET"])
    y = df["TARGET"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # stats class imbalance
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    neg_pos_ratio = neg / max(1, pos)

    # 2) Optuna (seedé) pour hyperparams
    param_space = default_param_space(neg_pos_ratio=neg_pos_ratio)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        params = {
            k: (trial.suggest_int(k, *v[1:]) if v[0] == "int"
                else trial.suggest_float(k, *v[1:]) if v[0] == "float"
                else trial.suggest_categorical(k, v))
            for k, v in param_space.items()
        }

        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
            **params
        )

        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        y_cv_proba = cross_val_predict(
            model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]

        # coût à seuil neutre 0.5 pour comparer équitablement les params
        y_cv_pred = (y_cv_proba >= 0.5).astype(int)
        return compute_business_cost(y_train, y_cv_pred, fn_cost=fn_cost, fp_cost=fp_cost)

    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params

    # 3) Fit final sur TRAIN avec meilleurs hyperparams
    best_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=random_state,
        **best_params
    )
    best_model.fit(X_train, y_train)

    # 4) Choix du seuil via CV (sur TRAIN), puis évaluation sur TEST
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    y_cv_proba = cross_val_predict(
        best_model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    best_threshold, _ = find_best_threshold(y_train, y_cv_proba, fn_cost=fn_cost, fp_cost=fp_cost)

    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= best_threshold).astype(int)
    business_cost = compute_business_cost(y_test, y_test_pred, fn_cost=fn_cost, fp_cost=fp_cost)

    # 5) Affichage
    print(f"\nMeilleurs hyperparamètres: {best_params}")
    print(f"Seuil choisi (depuis CV sur train): {best_threshold:.4f}")
    print(f"Coût métier (TEST): {business_cost}")
    print("\nClassification report (TEST):\n")
    print(classification_report(y_test, y_test_pred, digits=3))
    print(f"ROC AUC (TEST): {roc_auc_score(y_test, y_test_proba):.3f}")

    # 6) Persistance des artefacts
    models_dir.mkdir(exist_ok=True, parents=True)

    joblib.dump(best_model, models_dir / "best_model.pkl")

    with open(models_dir / "features.txt", "w", encoding="utf-8") as f:
        for col in X_train.columns:
            f.write(col + "\n")

    with open(models_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": float(best_threshold)}, f)

    with open(models_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    metrics = {
        "business_cost": float(business_cost),
        "f1": float(f1_score(y_test, y_test_pred)),
        "precision": float(precision_score(y_test, y_test_pred)),
        "recall": float(recall_score(y_test, y_test_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_test_proba)),
        "accuracy": float(accuracy_score(y_test, y_test_pred))
    }
    with open(models_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Artefacts sauvegardés dans:", models_dir.resolve())


# ---------------------------- CLI ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train XGBoost credit scoring model with Optuna and save artifacts.")
    p.add_argument("--data-dir", type=str, default="datasets", help="Dossier contenant train_final.csv")
    p.add_argument("--models-dir", type=str, default="models", help="Dossier de sortie des artefacts")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-splits", type=int, default=3)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--fn-cost", type=int, default=10, help="Coût d'un faux négatif")
    p.add_argument("--fp-cost", type=int, default=1, help="Coût d'un faux positif")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        data_dir=Path(args.data_dir),
        models_dir=Path(args.models_dir),
        test_size=args.test_size,
        cv_splits=args.cv_splits,
        n_trials=args.n_trials,
        random_state=args.random_state,
        fn_cost=args.fn_cost,
        fp_cost=args.fp_cost,
    )
