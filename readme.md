#  Etape 1 

🗂️ Structure du projet

Projet 8 MLOPS2/
│
├── .git/                           # Dossier Git local
├── .github/                        # Configuration GitHub Actions (CI/CD)
│
│
├── api/                            # Code de l’API Gradio (logger, predict)
├── datasets/                       # Fichiers de données d'entraînement/test
├── htmlcov/                        # Rapport HTML de couverture de test (pytest-cov)
├── logs/                           # Fichier .jsonl des logs de prédictions
├── models/                         # Modèles sauvegardés (.pkl, .onnx, .json, etc.)
├── notebook/                       # Notebooks d'exploration et tests
├── tests/                          # Tests automatisés (Pytest)
│
├── .gitattributes                  # Fichier Git
├── .gitignore                      # Fichier Git d’exclusion
│
├── analyse_logs.py                 # Analyse des logs JSONL (durée, CPU, drift…)
├── dashboard.py                    # Dashboard (Streamlit ou Gradio)
├── Dockerfile                      # Fichier de conteneurisation Docker
├── inference.py                    # Inférence batch sur un dataset
├── profiling.py                    # Profiling de la fonction predict
├── profiling_output.prof           # Résultat 1 du profiling
├── profiling_output3.prof          # Résultat 2 du profiling
├── readme.md                       # Documentation principale du projet
├── requirements.txt                # Dépendances du projet
└── train.py                        # Script d'entraînement du modèle




#  Étape 2 – Déploiement du modèle via une API Gradio + Docker + CI/CD


🎯 Objectif

Cette étape vise à exposer le modèle de machine learning entraîné à l'étape 1 via une API accessible. 

⚙️ Fonctionnement général

1. Interface utilisateur (API Gradio)

Le fichier app_gradio.py contient la fonction principale predict_credit_score() qui :

Reçoit des entrées utilisateur (11 features sélectionnées)

Retourne le résultat dans une interface simple via Gradio

# lancement : python -m api.app_gradio.py

✅ L’interface Gradio s’ouvre dans un navigateur à l’adresse : http://localhost:7860

2. Conteneurisation avec Docker

Le Dockerfile permet de :

Créer une image contenant Python, les dépendances (requirements.txt) et les scripts


3. Automatisation via CI/CD

Le dossier .github/workflows/ contient le fichier YAML de configuration pour GitHub Actions. Ce pipeline CI/CD :

- Se déclenche automatiquement à chaque push sur la branche principale

- Installe les dépendances

- Exécute les tests 

- Build l’image Docker et déployer sur Hugging Face Spaces 


🧪 Logs & Monitoring

La fonction log_prediction() définit dans le fichier api/logger.py enregistre chaque appel à l’API dans le fichier predictions_log.jsonl avec :

- Entrées utilisateur

- Score et prédiction

- Temps d’exécution

- Utilisation CPU et RAM

Ces logs peuvent ensuite être analysés via le fichier analyse_logs.py ou visualisés dans un dashboard.py.

✅ Tests réalisés

Pour garantir le bon fonctionnement, la stabilité et les performances de l’API, plusieurs types de tests ont été mis en place tout au long de l’étape 2 

- => Les tests visibles dans le sous dossier tests

- => Rapport de couverture des tests dans le dossier /htmlcov


#  Étape 3 – Monitoring et détection d’anomalies
🎯 Objectif

Dans cette troisième étape, l'objectif est de surveiller automatiquement l'activité de l’API déployée 


🧠 Analyse des logs

Un script Python analyse_logs.py a été développé pour :

🔎 Lire et parser les logs :

Chargement des lignes JSON du fichier logs/predictions.log

Conversion en DataFrame pour une analyse plus simple

# lancement : python analyse_logs.py

📊 Calculer les statistiques clés :

Temps de réponse moyen / max / min

Distribution des scores de risque

Taux de "risques élevés"

Moyennes mobiles (latence, CPU…)

📉 Détection automatique :

Dérive des données : détection d’un changement significatif dans la distribution des inputs


📈 3. Visualisation 


Un dashboard.py avec Streamlit  :

- Afficher les  temps de réponse, CPU, mémoire

- Suivre en temps réel la distribution des scores

- Détecter visuellement les anomalies

# lancement : streamlit run dashboard.py


#  Étape 4 – Optimisation des performances du modèle en production
Objectif

🔍 Profiling de l’API

Un profiling de la fonction predict_credit_score() exposée dans app_gradio.py a été réalisé à l’aide du module cProfile. Le fichier profiling.py contient ce profilage.

Exemple de lancement depuis le terminal :

# python profiling.py

Les résultats sont sauvegardés au format .prof et visualisés via Snakeviz :

# snakeviz profiling_output3.prof



