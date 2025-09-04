# profiling_full_api.py

import cProfile
import pstats
import io
import requests

# Dummy input simulant un vrai appel
data = {
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.4,
    "has_secondary_education": True,
    "has_higher_education": True,
    "ATM_DRAWINGS_LAST_6_MONTHS": 12,
    "is_working": True,
    "is_male": True,
    "is_female": False,
    "has_document_3": True,
    "OVERDUE": 1000
}

pr = cProfile.Profile()
pr.enable()

response = requests.post("http://127.0.0.1:7860/predict", json=data)  # URL Gradio/API

pr.disable()

# Sauvegarde dans un fichier pour SnakeViz
pr.dump_stats("profiling_full_output.prof")
print("✅ Fichier 'profiling_full_output.prof' généré pour visualisation SnakeViz.")

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
ps.print_stats()

print(s.getvalue())
print("Réponse :", response.json())
