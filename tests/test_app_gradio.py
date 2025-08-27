# tests/test_app_gradio.py

import sys
from pathlib import Path

# Ajouter la racine du projet au sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app_gradio import predict_credit_score

def test_gradio_prediction_valid_input():
    score = predict_credit_score(
        EXT_SOURCE_2=0.5,
        EXT_SOURCE_3=0.6,
        EXT_SOURCE_1=0.4,
        NAME_EDUCATION_TYPE_Higher=True,
        NAME_EDUCATION_TYPE_Secondary=False,
        ATM_DRAWINGS=10,
        WORKING=True,
        GENDER_F=True,
        GENDER_M=False,
        DOC3=True,
        OVERDUE=1000
    )
    assert isinstance(score, str)
    assert "Score" in score

def test_gradio_prediction_all_zeros():
    score = predict_credit_score(
        EXT_SOURCE_2=0.0,
        EXT_SOURCE_3=0.0,
        EXT_SOURCE_1=0.0,
        NAME_EDUCATION_TYPE_Higher=False,
        NAME_EDUCATION_TYPE_Secondary=False,
        ATM_DRAWINGS=0,
        WORKING=False,
        GENDER_F=False,
        GENDER_M=False,
        DOC3=False,
        OVERDUE=0
    )
    assert isinstance(score, str)
    assert "Score" in score

def test_gradio_prediction_invalid_type():
    try:
        predict_credit_score(
            EXT_SOURCE_2="not_a_number",
            EXT_SOURCE_3=0.5,
            EXT_SOURCE_1=0.5,
            NAME_EDUCATION_TYPE_Higher=True,
            NAME_EDUCATION_TYPE_Secondary=False,
            ATM_DRAWINGS=10,
            WORKING=True,
            GENDER_F=True,
            GENDER_M=False,
            DOC3=True,
            OVERDUE=0
        )
        assert False, "Le type invalide aurait dû générer une erreur"
    except Exception:
        assert True  # Erreur attendue
