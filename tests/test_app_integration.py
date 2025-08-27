# tests/test_app_integration.py

from pathlib import Path
import sys
import numpy as np

# Ajouter le dossier parent au path pour l'import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importer la fonction de prédiction directement
from api.app_gradio import predict_credit_score

def test_predict_credit_score_valid_input():
    # Appel de la fonction avec des valeurs cohérentes
    result = predict_credit_score(
        EXT_SOURCE_2=0.5,
        EXT_SOURCE_3=0.4,
        EXT_SOURCE_1=0.6,
        NAME_EDUCATION_TYPE_Higher=True,
        NAME_EDUCATION_TYPE_Secondary=False,
        ATM_DRAWINGS=10,
        WORKING=True,
        GENDER_F=True,
        GENDER_M=False,
        DOC3=True,
        OVERDUE=500
    )

    # Vérification du type et contenu de la réponse
    assert isinstance(result, str), "La prédiction devrait retourner une chaîne de caractères"
    assert "Score :" in result, "La chaîne devrait contenir un score"
    assert "risque" in result.lower(), "La chaîne devrait indiquer le risque"



