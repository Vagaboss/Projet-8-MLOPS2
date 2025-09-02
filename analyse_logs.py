import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from evidently import Report
from evidently.metrics import DataDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset

# === Fichiers ===
LOG_PATH = "logs/predictions_log.jsonl"
REFERENCE_PATH = "datasets/train_final.csv"
REPORT_PATH = "logs/drift_report.html"
TESTSUITE_PATH = "logs/tests_report.html"

# === Chargement des données ===

# Charger les données de référence
df_ref = pd.read_csv(REFERENCE_PATH)

# Charger les données de production (logs)
records = []
with open(LOG_PATH, "r") as f:
    for line in f:
        log = json.loads(line)
        input_data = log["input"]
        records.append(input_data)

df_prod = pd.DataFrame(records)

# === Générer le rapport de dérive ===

report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset()
])
report.run(reference_data=df_ref, current_data=df_prod)
report.save_html(REPORT_PATH)

# === Générer la suite de tests Evidently ===

tests = TestSuite(tests=[
    DataStabilityTestPreset()
])
tests.run(reference_data=df_ref, current_data=df_prod)
tests.save_html(TESTSUITE_PATH)

print(f"✅ Analyse terminée. Rapport enregistré : {REPORT_PATH} & {TESTSUITE_PATH}")
