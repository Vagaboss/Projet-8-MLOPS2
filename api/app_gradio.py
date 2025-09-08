# app_gradio.py

import gradio as gr
import pandas as pd
import numpy as np
import joblib, json
from pathlib import Path
from api.logger import log_prediction
from datetime import datetime, timezone
import time
import psutil

# === Constantes ===
MODELS = Path("models")
FEATURES_PATH = MODELS / "features.txt"
MODEL_PATH = MODELS / "best_model.pkl"
THRESHOLD_PATH = MODELS / "threshold.json"

# === Chargement des artefacts ===
model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as f:
    threshold = json.load(f)["threshold"]

with open(FEATURES_PATH, "r") as f:
    all_features = [line.strip() for line in f]

# === Liste des 10 features à exposer ===
selected_features = [
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "EXT_SOURCE_1",
    "NAME_EDUCATION_TYPE_Higher education",
    "NAME_EDUCATION_TYPE_Secondary / secondary special",
    "CRECARD_CNT_DRAWINGS_ATM_CURRENT_mean",
    "NAME_INCOME_TYPE_Working",
    "CODE_GENDER_F",
    "CODE_GENDER_M",
    "FLAG_DOCUMENT_3",
    "BUREAU_AMT_CREDIT_SUM_OVERDUE_max"
]

# === Fonction d'inférence ===

    # Dictionnaire des inputs
def predict_credit_score(
    EXT_SOURCE_2, EXT_SOURCE_3, EXT_SOURCE_1,
    NAME_EDUCATION_TYPE_Higher, NAME_EDUCATION_TYPE_Secondary,
    ATM_DRAWINGS, WORKING, GENDER_F, GENDER_M, DOC3, OVERDUE
):
    start_time = time.time()

    input_dict = {
        "EXT_SOURCE_2": EXT_SOURCE_2,
        "EXT_SOURCE_3": EXT_SOURCE_3,
        "EXT_SOURCE_1": EXT_SOURCE_1,
        "NAME_EDUCATION_TYPE_Higher education": int(NAME_EDUCATION_TYPE_Higher),
        "NAME_EDUCATION_TYPE_Secondary / secondary special": int(NAME_EDUCATION_TYPE_Secondary),
        "CRECARD_CNT_DRAWINGS_ATM_CURRENT_mean": ATM_DRAWINGS,
        "NAME_INCOME_TYPE_Working": int(WORKING),
        "CODE_GENDER_F": int(GENDER_F),
        "CODE_GENDER_M": int(GENDER_M),
        "FLAG_DOCUMENT_3": int(DOC3),
        "BUREAU_AMT_CREDIT_SUM_OVERDUE_max": OVERDUE
    }

    # === Optimisation NumPy ===
    X_np = np.zeros((1, len(all_features)), dtype=np.float32)
    for key, value in input_dict.items():
        if key in all_features:
            idx = all_features.index(key)
            X_np[0, idx] = value

    # === Prédiction
    score = model.predict_proba(X_np)[0, 1]
    prediction = int(score >= threshold)

    duration = round(time.time() - start_time, 4)
    cpu = psutil.cpu_percent(interval=None)
    memory = psutil.Process().memory_info().rss / (1024 * 1024)  # en Mo

    output_text = f"Score : {round(score, 4)} → " + ("❌ Risque élevé" if prediction == 1 else "✅ Faible risque")

    log_prediction(input_dict, output_text, duration, cpu=cpu, memory=memory)

    return output_text


   
# === Interface Gradio ===
demo = gr.Interface(
    fn=predict_credit_score,
    inputs=[
        gr.Slider(0, 1, label="EXT_SOURCE_2"),
        gr.Slider(0, 1, label="EXT_SOURCE_3"),
        gr.Slider(0, 1, label="EXT_SOURCE_1"),
        gr.Checkbox(label="Diplôme supérieur ?"),
        gr.Checkbox(label="Diplôme secondaire ?"),
        gr.Slider(0, 50, step=1, label="Nb retraits carte (ATM)"),
        gr.Checkbox(label="Actif professionnellement ?"),
        gr.Checkbox(label="Femme ?"),
        gr.Checkbox(label="Homme ?"),
        gr.Checkbox(label="Document 3 fourni ?"),
        gr.Slider(0, 100000, step=100, label="Montant max en retard (BUREAU)")
    ],
    outputs=gr.Text(label="Résultat de la prédiction"),
    title="🧮 Prédiction de risque client (Crédit Express)",
    description="Modèle de scoring simplifié à partir de 10 variables."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
