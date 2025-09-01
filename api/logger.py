# logger.py

import json
import os
from datetime import datetime, timezone

LOG_FILE = "logs/predictions_log.jsonl"

# S'assurer que le dossier existe
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log_prediction(input_data, prediction, duration):
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input_data,
        "prediction": prediction,
        "duration": duration  # en secondes
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

