# tests/test_train.py
from pathlib import Path
import numpy as np
import pandas as pd

from pathlib import Path
import sys

# Ajouter la racine du projet au PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import run_training

def test_run_training_smoke(tmp_path):
    # Créer un petit dataset synthétique compatible binaire
    # 200 lignes, 6 features numériques, TARGET équilibrée
    rows = 200
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "f1": rng.normal(0, 1, rows),
        "f2": rng.normal(0, 1, rows),
        "f3": rng.normal(0, 1, rows),
        "f4": rng.normal(0, 1, rows),
        "f5": rng.normal(0, 1, rows),
        "f6": rng.integers(0, 5, rows).astype(float),
    })
    # cible corrélée un peu à f1 pour que le modèle apprenne quelque chose
    y = (X["f1"] + rng.normal(0, 0.5, rows) > 0).astype(int)
    df = X.copy()
    df["TARGET"] = y

    data_dir = tmp_path / "datasets"
    models_dir = tmp_path / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_csv = data_dir / "train_final.csv"
    df.to_csv(train_csv, index=False)

    # Importer et lancer l'entraînement avec peu d'itérations (rapide)
    from train import run_training
    run_training(
        data_dir=data_dir,
        models_dir=models_dir,
        test_size=0.2,
        cv_splits=2,
        n_trials=1,          # une seule tentative Optuna -> smoke
        random_state=42,
        fn_cost=10,
        fp_cost=1,
    )

    # Vérifier les artefacts
    assert (models_dir / "best_model.pkl").exists()
    assert (models_dir / "features.txt").exists()
    assert (models_dir / "threshold.json").exists()
    assert (models_dir / "params.json").exists()
    assert (models_dir / "metrics.json").exists()
