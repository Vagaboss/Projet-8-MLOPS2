# tests/test_inference.py
from pathlib import Path
import json
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

# Ajouter la racine du projet au sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_artifacts_exist():
    models = Path("models")
    assert (models / "best_model.pkl").exists(), "best_model.pkl manquant"
    assert (models / "features.txt").exists(), "features.txt manquant"
    assert (models / "threshold.json").exists(), "threshold.json manquant"

def test_inference_smoke(tmp_path):
    # Charger artefacts existants
    models = Path("models")
    feats = [l.strip() for l in open(models / "features.txt", encoding="utf-8")]
    assert len(feats) > 0, "features.txt vide ?"

    # Construire un mini input CSV avec les mêmes colonnes
    # 1) si datasets/test_final.csv existe, on prend ses 5 premières lignes
    ds = Path("datasets")
    src = ds / "test_final.csv"
    if src.exists():
        df = pd.read_csv(src).head(5)
        # garder seulement les features (ignorer TARGET si présent)
        df = df.reindex(columns=[*feats], fill_value=0)
    else:
        # 2) sinon, fabriquer un DataFrame synthétique
        # (valeurs 0 -> le modèle doit quand même produire une proba)
        rows = 5
        data = {c: np.zeros(rows, dtype=float) for c in feats}
        df = pd.DataFrame(data)

    input_csv = tmp_path / "mini_input.csv"
    output_csv = tmp_path / "mini_output.csv"
    df.to_csv(input_csv, index=False)

    # Appeler la fonction d'inférence
    from inference import run_inference
    run_inference(input_csv=input_csv, output_csv=output_csv, models_dir=models)

    assert output_csv.exists(), "Le CSV de sortie n'a pas été créé"
    out = pd.read_csv(output_csv)
    assert "score" in out.columns and "prediction" in out.columns, \
        "Colonnes 'score' et/ou 'prediction' manquantes dans la sortie"
    assert len(out) == len(df), "La sortie ne correspond pas au nombre d’entrées"

#Test de robustesse



def test_inference_with_missing_values(tmp_path):
    models = Path("models")
    feats = [l.strip() for l in open(models / "features.txt", encoding="utf-8")]

    df = pd.DataFrame({col: np.zeros(5, dtype=float) for col in feats})
    df.iloc[0, 0] = np.nan  # valeur manquante

    input_csv = tmp_path / "input_missing.csv"
    output_csv = tmp_path / "output_missing.csv"
    df.to_csv(input_csv, index=False)

    from inference import run_inference
    run_inference(input_csv=input_csv, output_csv=output_csv, models_dir=models)

    assert output_csv.exists()
    out = pd.read_csv(output_csv)
    assert "score" in out.columns and "prediction" in out.columns


def test_inference_with_extreme_values(tmp_path):
    models = Path("models")
    feats = [l.strip() for l in open(models / "features.txt", encoding="utf-8")]

    df = pd.DataFrame({col: np.zeros(5, dtype=float) for col in feats})
    df.iloc[0, 1] = 1e12  # valeur extrême

    input_csv = tmp_path / "input_extreme.csv"
    output_csv = tmp_path / "output_extreme.csv"
    df.to_csv(input_csv, index=False)

    from inference import run_inference
    run_inference(input_csv=input_csv, output_csv=output_csv, models_dir=models)

    assert output_csv.exists()
    out = pd.read_csv(output_csv)
    assert "score" in out.columns and "prediction" in out.columns



def test_inference_with_invalid_types(tmp_path):
    models = Path("models")
    feats = [l.strip() for l in open(models / "features.txt", encoding="utf-8")]

    df = pd.DataFrame({col: np.zeros(5, dtype=float) for col in feats})
    df.iloc[0, 2] = "invalid_string"  # mauvais type de données

    input_csv = tmp_path / "input_invalid.csv"
    output_csv = tmp_path / "output_invalid.csv"
    df.to_csv(input_csv, index=False)

    from inference import run_inference
    with pytest.raises(Exception):  # on s'attend à une erreur
        run_inference(input_csv=input_csv, output_csv=output_csv, models_dir=models)
