import cProfile
import pstats
import io
from api.app_gradio import predict_credit_score

# === Dummy inputs pour simuler une requête API ===
args = (
    0.5, 0.9, 0.4,  # EXT_SOURCEs
    True, True,     # Éducation
    12,             # ATM_DRAWINGS
    True,           # Working
    True, False,    # Genre
    True,           # Doc3
    10000            # OVERDUE
)

# === Profiling et sauvegarde du résultat pour SnakeViz ===
profiler = cProfile.Profile()
profiler.enable()

predict_credit_score(*args)

profiler.disable()

# Sauvegarde dans un fichier pour SnakeViz
profiler.dump_stats("profiling_output3.prof")
print("✅ Fichier 'profiling_output3.prof' généré pour visualisation SnakeViz.")

# Affichage textuel optionnel dans le terminal
s = io.StringIO()
stats = pstats.Stats(profiler, stream=s).sort_stats("cumtime")
stats.print_stats()
print(s.getvalue())
