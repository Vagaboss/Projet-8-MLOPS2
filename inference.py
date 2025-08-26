import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np


# Dossiers par défaut (relatifs à la racine du projet)
MODELS_DIR = Path("models")
DATA_DIR = Path("datasets")


def load_artifacts(models_dir: Path):
    """Charge le modèle, la liste de features et le seuil métier."""
    model_path = models_dir / "best_model.pkl"
    feats_path = models_dir / "features.txt"
    thr_path = models_dir / "threshold.json"

    assert model_path.exists(), f"Modèle introuvable: {model_path}"
    assert feats_path.exists(), f"Features introuvables: {feats_path}"
    assert thr_path.exists(), f"Seuil introuvable: {thr_path}"

    model = joblib.load(model_path)

    with open(feats_path, "r", encoding="utf-8") as f:
        features = [line.strip() for line in f if line.strip()]

    with open(thr_path, "r", encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])

    return model, features, threshold


def run_inference(input_csv: Path, output_csv: Path, models_dir: Path):
    """Lit input_csv, applique le modèle et écrit output_csv (score + prediction)."""
    # 1) Artefacts
    model, features, threshold = load_artifacts(models_dir)

    # 2) Données
    assert input_csv.exists(), f"Fichier d'entrée introuvable: {input_csv}"
    df = pd.read_csv(input_csv)

    # 3) Préparation X (on ignore TARGET si présent)
    X = df.drop(columns=["TARGET"], errors="ignore").copy()

    # 4) Vérification/alignement des features
    missing = [c for c in features if c not in X.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans l'entrée: {missing}")
    X = X[features]  # ordre identique à l'entraînement

    # 5) Prédictions
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        # fallback si le modèle ne fournit pas predict_proba
        raw = model.predict(X)
        proba = np.asarray(raw).reshape(-1)

    pred = (proba >= threshold).astype(int)

    # 6) Sortie
    out = df.copy()
    out["score"] = proba
    out["prediction"] = pred

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"✅ Prédictions écrites dans: {output_csv.resolve()}")
    print(out.head())


def parse_args():
    p = argparse.ArgumentParser(description="Batch inference for credit scoring model.")
    p.add_argument("--input", type=str, default=str(DATA_DIR / "test_final.csv"),
                   help="Chemin du CSV d'entrée (sans TARGET).")
    p.add_argument("--output", type=str, default=str(DATA_DIR / "predictions_test_final.csv"),
                   help="Chemin du CSV de sortie.")
    p.add_argument("--models-dir", type=str, default=str(MODELS_DIR),
                   help="Dossier contenant best_model.pkl, features.txt, threshold.json.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        models_dir=Path(args.models_dir),
    )
