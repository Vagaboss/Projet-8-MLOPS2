import json
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# === Chemins ===
LOG_PATH = "logs/predictions_log.jsonl"
REFERENCE_PATH = "datasets/train_final.csv"
REPORT_PATH = "logs/drift_report.html"

# === Chargement des données ===

# Données de référence
df_ref = pd.read_csv(REFERENCE_PATH)

# === Partie 1 : Lecture complète
records = []
with open(LOG_PATH, "r") as f:
    for line in f:
        log = json.loads(line)
        row = log.get("input", {})
        row["duration"] = log.get("duration", None)
        row["prediction"] = log.get("prediction", None)
        records.append(row)

# DataFrame complet
df_full = pd.DataFrame(records)

# Sélectionner uniquement les colonnes communes pour la dérive
common_cols = [col for col in df_ref.columns if col in df_full.columns]
df_ref_filtered = df_ref[common_cols]
df_prod_filtered = df_full[common_cols]

# === Analyse du drift
report = Report([DataDriftPreset(method="psi")])
evaluation = report.run(reference_data=df_ref_filtered, current_data=df_prod_filtered)
evaluation.save_html(REPORT_PATH)
print(f"✅ Rapport de dérive généré : {REPORT_PATH}")

# === Analyse de la latence et des erreurs (avec toutes les colonnes)
durations = df_full["duration"].astype(float)

latence_moyenne = durations.mean()
latence_max = durations.max()
latence_seuil = durations.quantile(0.95)
requêtes_lentes = durations[durations > latence_seuil]
pourcentage_lentes = 100 * len(requêtes_lentes) / len(durations)

df_full["prediction"] = df_full["prediction"].astype(str)
taux_erreur = 100 * df_full["prediction"].isin(["", "None", "nan"]).sum() / len(df_full)

print("\n=== Analyse des performances opérationnelles ===")
print(f"🔁 Nombre total de requêtes : {len(df_full)}")
print(f"⚠️ Taux d’erreur : {taux_erreur:.2f} %")
print(f"🐢 Latence moyenne : {latence_moyenne:.3f} secondes")
print(f"⏱️ Latence maximale : {latence_max:.3f} secondes")
print(f"🐌 Requêtes lentes (> 95e percentile) : {pourcentage_lentes:.1f} % des requêtes")

# CPU
if "cpu_percent" in df_full.columns:
    df_full["cpu_percent"] = df_full["cpu_percent"].astype(float)
    cpu_moyenne = df_full["cpu_percent"].mean()
    cpu_max = df_full["cpu_percent"].max()
    print(f"🧠 CPU moyen : {cpu_moyenne:.2f} % | max : {cpu_max:.2f} %")

# RAM
if "memory_used_mb" in df_full.columns:
    df_full["memory_used_mb"] = df_full["memory_used_mb"].astype(float)
    ram_moyenne = df_full["memory_used_mb"].mean()
    ram_max = df_full["memory_used_mb"].max()
    print(f"💾 RAM moyenne : {ram_moyenne:.2f} MB | max : {ram_max:.2f} MB")








