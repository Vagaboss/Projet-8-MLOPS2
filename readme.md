# Etape 1 

🗂️ Structure du projet

Le projet est organisé de manière modulaire afin de séparer les différentes logiques : entraînement, inférence, API, monitoring et visualisation. Voici une description des principaux dossiers et fichiers :

📁 Dossiers

- api/ : Contient le code lié à l’API Gradio (par ex. logger.py, fonctions d'inférence, etc.).

- datasets/ : Données utilisées pour l’entraînement, la validation et les tests (train.csv, - test_final.csv, etc.).

- logs/ : Fichiers de logs générés automatiquement lors des prédictions (log_prediction()).

- models/ : Modèles sauvegardés (best_model.pkl, .onnx, quantifié, etc.), seuils et fichiers de features.

- notebook/ : Contient les notebooks d’exploration, de test ou de démonstration.

- tests/ : Scripts de test ou validation du modèle/API.

- htmlcov/ : Dossier généré automatiquement pour les rapports de couverture de test (pytest-cov).

- .github/ : Fichiers liés à la configuration GitHub Actions ou CI/CD.

📄 Fichiers racine

- train.py : Script principal pour l’entraînement du modèle.

- inference.py : Script pour exécuter l’inférence sur un dataset complet.

- app_gradio.py (non visible ici mais probablement dans api/) : Point d'entrée de l'interface utilisateur avec Gradio.

- profiling.py : Script de profiling de la fonction predict_credit_score pour détecter les goulots d’étranglement.

- profiling_output.prof, profiling_output3.prof : Résultats du profiling à visualiser avec SnakeViz.

- analyse_logs.py : Analyse et agrégation des logs (logs/predictions.log).

- dashboard.py : Script pour créer une visualisation ou un dashboard à partir des données de logs.

- requirements.txt : Liste des dépendances Python à installer.

- Dockerfile : Conteneurisation du projet pour un déploiement facile.

- README.md : Documentation principale du projet (actuellement vide).

- .gitignore, .gitattributes : Fichiers de configuration Git.

🔍 Fichiers de couverture & cache

- .coverage, .htmlcov/ : Générés par pytest pour suivre la couverture des tests.

__pycache__/ : Fichiers compilés automatiquement par Python (à ignorer via .gitignore).



#  Étape 2 – Déploiement du modèle via une API Gradio + Docker + CI/CD


🎯 Objectif

Cette étape vise à exposer le modèle de machine learning entraîné à l'étape 1 via une API accessible. L’objectif est triple :

Créer une interface utilisateur (UI) à l’aide de Gradio pour permettre à un utilisateur de faire des prédictions interactives.

Conteneuriser cette application avec Docker pour garantir un déploiement portable et reproductible.

Automatiser le déploiement via une pipeline CI/CD GitHub Actions qui build, teste et déploie automatiquement l’application dès qu’une modification est poussée.

⚙️ Fonctionnement général

1. Interface utilisateur (API Gradio)

Le fichier app_gradio.py contient la fonction principale predict_credit_score() qui :

Reçoit des entrées utilisateur (11 features sélectionnées)

Reconstruit un vecteur complet de features pour le modèle (full_input)

Applique model.predict_proba() pour générer un score

Traduit ce score en prédiction binaire selon un seuil

Retourne le résultat dans une interface simple via Gradio

✅ L’interface Gradio s’ouvre dans un navigateur à l’adresse : http://localhost:7860

2. Conteneurisation avec Docker

Le Dockerfile permet de :

Créer une image contenant Python, les dépendances (requirements.txt) et les scripts

Lancer automatiquement l’API Gradio au démarrage du conteneur

Exemple de build et lancement local :
# Build de l'image
docker build -t credit-risk-api .

# Lancement du conteneur
docker run -p 7860:7860 credit-risk-api

3. Automatisation via CI/CD

Le dossier .github/workflows/ contient le fichier YAML de configuration pour GitHub Actions. Ce pipeline CI/CD :

- Se déclenche automatiquement à chaque push ou pull request sur la branche principale

- Installe les dépendances

- Exécute les tests unitaires (s’il y en a dans le dossier tests/)

- Build l’image Docker et déployer sur Hugging Face Spaces 

# pip install -r requirements.txt

# python app_gradio.py

📁 Fichiers importants

Fichier / Dossier	Rôle
app_gradio.py	L’API Gradio exposant le modèle
models/best_model.pkl	Modèle XGBoost sauvegardé
models/features.txt	Liste ordonnée des colonnes attendues par le modèle
models/threshold.json	Seuil de classification binaire
Dockerfile	Instructions pour créer l’image Docker
.github/workflows/	Contient la configuration CI/CD
logs/	Stocke les prédictions faites via l’API

🧪 Logs & Monitoring

La fonction log_prediction() enregistre chaque appel à l’API dans le fichier predictions_log.jsonl avec :

- Entrées utilisateur

- Score et prédiction

- Temps d’exécution

- Utilisation CPU et RAM

Ces logs peuvent ensuite être analysés via analyse_logs.py ou visualisés dans un dashboard.py.

✅ Tests réalisés

Pour garantir le bon fonctionnement, la stabilité et les performances de l’API, plusieurs types de tests ont été mis en place tout au long de l’étape 2 :

🧪 1. Tests fonctionnels

Ces tests permettent de s'assurer que l'API Gradio retourne une prédiction correcte selon les entrées fournies par l'utilisateur :

✔️ Vérification que la fonction predict_credit_score() retourne une prédiction de score et le bon libellé associé (✅ Faible risque ou ❌ Risque élevé).

✔️ Tests avec des cas limites (par exemple : score proche du seuil, valeurs nulles ou extrêmes).

✔️ Contrôle que les cases à cocher (booleans) sont bien converties en 0 ou 1 pour correspondre aux features attendues par le modèle.

🧠 2. Tests techniques / unitaires

🔎 Test de bon chargement des artefacts :

features.txt contient toutes les colonnes nécessaires

Le model.pkl est bien chargé via joblib

Le threshold.json est correctement lu

🔎 Test de complétion automatique des features :
Vérifie que toutes les colonnes du modèle sont bien renseignées dans l’appel predict(), même si l'utilisateur ne fournit que les 10 exposées (les autres sont remplies par des 0).

🔎 Test de logging :

Chaque prédiction doit générer une ligne JSON valide dans le fichier logs/predictions.log.

⚙️ 3. Tests de performance

Pour optimiser l'inférence et détecter des goulots d’étranglement :

⏱️ Profiling de la fonction predict_credit_score() à l’aide de cProfile + snakeviz.

Permet d’identifier les fonctions les plus lentes (ex : to_numpy, isna)

✅ Suite à ces tests, la construction du dictionnaire d’entrée a été optimisée pour gagner plusieurs millisecondes par appel.

📉 Mesures de :

Temps d’exécution

% CPU utilisé

Mémoire RAM consommée
Ces données sont loggées automatiquement dans chaque appel.

🔁 4. Tests d'intégration (API + Docker)

🐳 Vérification que l’API Gradio fonctionne bien dans le conteneur Docker :

Le serveur démarre sans erreur (python app_gradio.py)

L'API retourne les mêmes résultats qu'en local

Les fichiers modèles et logs sont bien montés et accessibles

⚙️ Tests sur la CI/CD :

À chaque push vers repo distant github, la pipeline GitHub Actions installe les dépendances, lance des tests (ou checks de syntaxe / formatage) et peut effectuer un build Docker avant deploiement sur hugging face.

Cela permet de détecter rapidement les régressions ou erreurs d'importation.

- => Rapport de couverture des tests dans le dossier /htmlcov


🔍 Étape 3 – Monitoring et détection d’anomalies
🎯 Objectif

Dans cette troisième étape, l'objectif est de surveiller automatiquement l'activité de l’API déployée afin de :

Stocker et centraliser les données de prédiction (inputs, outputs, temps de réponse, ressources système, etc.).

Analyser ces données pour détecter des anomalies telles que :

Une dérive des données (data drift)

Une augmentation du temps de réponse (latence)

Une utilisation anormale du CPU ou de la RAM

Une hausse du taux d’erreur ou des scores inhabituels

📦 1. Collecte et stockage des logs
✅ Données collectées

À chaque appel de l’API, un log JSON est généré, contenant :

Clé	Description
timestamp	Date et heure UTC de la requête
input	Données saisies par l'utilisateur
prediction	Score + étiquette de risque
duration	Temps de réponse (en secondes)
cpu_percent	Utilisation CPU au moment de la prédiction
memory_usage_MB	Mémoire RAM utilisée (en mégaoctets)

Tous les logs sont stockés dans un fichier local : logs/predictions.log


🧼 2. Analyse des logs

Un script Python analyse_logs.py a été développé pour :

🔎 Lire et parser les logs :

Chargement des lignes JSON du fichier logs/predictions.log

Conversion en DataFrame pour une analyse plus simple

📊 Calculer les statistiques clés :

Temps de réponse moyen / max / min

Distribution des scores de risque

Taux de "risques élevés"

Moyennes mobiles (latence, CPU…)

📉 Détection automatique :

Dérive des données : détection d’un changement significatif dans la distribution des inputs

Exemple : chute ou pic soudain sur EXT_SOURCE_1 ou CRECARD_CNT_DRAWINGS_ATM_CURRENT_mean

Latence anormale : alerte si un appel dépasse un seuil critique (ex. : 0.5s)

Surconsommation CPU / RAM : si cpu_percent ou memory_usage_MB dépasse un seuil anormal

📈 3. Visualisation 


Un dashboard dashboard.py avec Streamlit  :

Afficher les  temps de réponse, CPU, mémoire

Suivre en temps réel la distribution des scores

Détecter visuellement les anomalies


Étape 4 – Optimisation des performances du modèle en production
Objectif

Maintenant que notre modèle est déployé et que nous collectons des données de monitoring, cette étape vise à analyser ses performances en conditions réelles, identifier les éventuels goulots d’étranglement, tester des stratégies d’optimisation, et intégrer les améliorations dans le pipeline CI/CD.

1. Analyse des performances en production

Les appels à l’API Gradio génèrent des logs automatiquement via la fonction log_prediction() définie dans le fichier api/logger.py. Chaque appel enregistre les éléments suivants : timestamp, input utilisateur, score, prédiction binaire, temps d’inférence, pourcentage d'utilisation CPU (cpu_percent), et consommation mémoire (memory_usage_MB).

Un exemple de log généré dans logs/predictions.log :

{"timestamp": "2025-09-08T08:50:30.359718+00:00", "input": {"EXT_SOURCE_2": 0.29, "EXT_SOURCE_3": 0.45, ...}, "prediction": "Score : 0.2159 → ✅ Faible risque", "duration": 0.0046, "cpu_percent": 0.0, "memory_usage_MB": 260.83}

2. Profiling de l’API

Un profiling de la fonction predict_credit_score() exposée dans app_gradio.py a été réalisé à l’aide du module cProfile. Le fichier profile_api.py contient ce profilage.

Exemple de lancement depuis le terminal :

python profile_api.py

Les résultats sont sauvegardés au format .prof et visualisés via Snakeviz :

snakeviz logs/profile_predict.prof

Le premier profilage a révélé que la ligne suivante était un goulot d’étranglement :

full_input = {col: 0 for col in all_features}

Cette ligne réinitialisait un dictionnaire de plus de 300 colonnes à chaque appel, ce qui créait une charge inutile.

3. Optimisations mises en œuvre
a) Optimisation de code dans Gradio

L’optimisation principale a consisté à créer une version préremplie du dictionnaire full_input une seule fois au chargement de l’API. Lors de chaque prédiction, on fait simplement une copie de ce dictionnaire :

input_template = {col: 0 for col in all_features}

Puis dans la fonction predict_credit_score :

full_input = input_template.copy()
full_input.update(input_dict)

Grâce à cette optimisation, le temps d’inférence est passé d’environ 0.045 secondes à 0.004 secondes.

b) Conversion ONNX

Nous avons converti le modèle XGBoost .pkl en ONNX avec la bibliothèque onnxmltools, dans le fichier convert_to_onnx.py. Le modèle est ensuite testé avec onnxruntime.InferenceSession pour l'inférence. Le code de chargement ressemble à :

session = onnxruntime.InferenceSession("models/best_model.onnx", providers=["CPUExecutionProvider"])

Puis pour l'inférence :

inputs = {session.get_inputs()[0].name: X.astype(np.float32).values}
probas = session.run(None, inputs)[0].ravel()

Cela est visible à la fin de mon notebook principal : dupli.ipynb dans le dossier notebook


4. Intégration dans le pipeline CI/CD

L’API optimisée (app_gradio.py) a été committée dans le dépôt GitHub. Le pipeline CI/CD défini dans .github/workflows/deploy.yml permet de :

Exécuter les tests,

Construire l’image Docker à jour,

Déployer automatiquement l’API.

Chaque mise à jour du code déclenche une reconstruction complète.

5. Résultats des optimisations

Avant optimisation, l’inférence sur un appel API prenait environ 0.045 secondes.

Après optimisation de code (copie du dictionnaire input_template), la durée est descendue à 0.004 secondes.

Le modèle ONNX quant à lui mettait 0.119 secondes pour inférer le même batch, confirmant que l’utilisation directe du modèle XGBoost .pkl est plus rapide dans notre cas.

Fichiers clés

app_gradio.py : expose l’API optimisée avec Gradio

api/logger.py : fonction log_prediction() qui trace les entrées, sorties et métriques

logs/predictions.log : fichier de logs des appels API

profile_api.py : script de profiling avec cProfile

convert_to_onnx.py : conversion du modèle vers ONNX

quantize_onnx.py : tentative de quantification

.github/workflows/deploy.yml : pipeline CI/CD complet


# Commandes terminales utiles

# 🧠 Credit Scoring API – Projet MLOps

Ce projet a pour objectif de développer une API de scoring crédit, basée sur un modèle XGBoost, avec les étapes clés suivantes : entraînement, déploiement, monitoring, optimisation, CI/CD.

---

## 📦 Commandes terminal utilisées dans le projet

### ⚙️ ENVIRONNEMENT & DÉPENDANCES

Installation des dépendances :
pip install -r requirements.txt


### 🧠 ENTRAÎNEMENT DU MODÈLE

Lancer le script d'entraînement :
python train.py

---

### 🔍 INFÉRENCE EN LOCAL

Lancer l’inférence sur un jeu de test :
python inference.py

---

### 🧪 TESTS UNITAIRES & COUVERTURE

Exécuter les tests avec coverage :
pytest --cov=api tests/

Générer un rapport HTML :
pytest --cov=api --cov-report=html


### 📈 PROFILING & GOULOTS D'ÉTRANGLEMENT

Lancer le profiling :
python profiling.py

Visualiser les résultats :
snakeviz profiling_output3.prof

---

### 🖼️ INTERFACE GRADIO

Lancer l’interface Gradio :
python app_gradio.py

Accessible sur : http://localhost:7860

---

### 🐳 DOCKER

Build de l’image Docker :
docker build -t credit-risk-api .

Exécuter l’image Docker :
docker run -p 7860:7860 credit-risk-api

---

### 🤖 CI/CD – GITHUB ACTIONS

Déclenché automatiquement à chaque push.
Fichier : .github/workflows/deploy.yml

---

### 📊 MONITORING & ANALYSE DES LOGS

Analyser les logs :
python analyse_logs.py

Lancer le dashboard (optionnel) :
streamlit run dashboard.py

---



## ✅ Fichiers liés

- train.py : Entraînement du modèle
- inference.py : Inférence sur données
- app_gradio.py : API Gradio
- profiling.py : Profiling de l’inférence
- analyse_logs.py : Analyse automatique des logs
- convert_to_onnx.py : Conversion du modèle
- dashboard.py : Dashboard de monitoring
- Dockerfile : Conteneurisation
- requirements.txt : Dépendances
- .github/workflows/ : Pipeline CI/CD
