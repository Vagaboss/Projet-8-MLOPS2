import streamlit as st
import pandas as pd
import json
import numpy as np

# === Fichiers ===
LOG_PATH = "logs/predictions_log.jsonl"
DRIFT_REPORT = "logs/drift_report.html"

# === Titre ===
st.set_page_config(page_title="Monitoring API", layout="centered")
st.title("📊 Dashboard de Monitoring de l'API de Scoring Crédit")

# === Chargement des logs ===
records = []
with open(LOG_PATH, "r") as f:
    for line in f:
        log = json.loads(line)
        row = log.get("input", {})
        row["duration"] = log.get("duration", None)
        row["prediction"] = log.get("prediction", None)
        row["cpu_percent"] = log.get("cpu_percent", None)
        row["memory_usage_MB"] = log.get("memory_usage_MB", None)
        records.append(row)

df = pd.DataFrame(records)

# === Statistiques générales ===
st.header("📈 Statistiques Générales")
st.write(f"Nombre total de requêtes : `{len(df)}`")

if "duration" in df.columns:
    durations = df["duration"].astype(float)
    st.metric("🐢 Latence moyenne (s)", round(durations.mean(), 3))
    st.metric("⏱️ Latence max (s)", round(durations.max(), 3))
    st.metric("🐌 % requêtes lentes (>95e)", f"{round(100 * len(durations[durations > durations.quantile(0.95)]) / len(durations), 1)} %")
else:
    st.warning("⚠️ Colonne 'duration' absente des logs.")

if "prediction" in df.columns:
    erreurs = df["prediction"].astype(str).isin(["", "None", "nan"]).sum()
    st.metric("⚠️ Taux d’erreur", f"{round(100 * erreurs / len(df), 2)} %")
else:
    st.warning("⚠️ Colonne 'prediction' absente des logs.")

# === Statistiques CPU et RAM ===
st.header("🖥️ Utilisation CPU & RAM")

if "cpu_percent" in df.columns:
    cpu_vals = df["cpu_percent"].astype(float)
    st.metric("🧠 CPU moyen (%)", f"{cpu_vals.mean():.2f}")
    st.metric("📈 CPU max (%)", f"{cpu_vals.max():.2f}")
else:
    st.warning("⚠️ Colonne 'cpu_percent' absente des logs.")

if "memory_usage_MB" in df.columns:
    ram_vals = df["memory_usage_MB"].astype(float)
    st.metric("💾 RAM moyenne (MB)", f"{ram_vals.mean():.2f}")
    st.metric("📈 RAM max (MB)", f"{ram_vals.max():.2f}")
else:
    st.warning("⚠️ Colonne 'memory_usage_MB' absente des logs.")

# === Rapport de dérive (Data Drift) ===
st.header("🔍 Rapport de dérive (Data Drift)")

try:
    with open(DRIFT_REPORT, "r", encoding="utf-8") as f:
        html_content = f.read()
        # ✅ Permet le scroll horizontal et vertical
        st.components.v1.html(
            f"""
            <div style="overflow-x: auto; overflow-y: auto; width: 2000px; height: 1800px;">
                {html_content}
            </div>
            """,
            height=1000,
            scrolling=True
        )
except FileNotFoundError:
    st.error("Rapport de dérive non trouvé. Veuillez exécuter `analyse_logs.py` d'abord.")


