# Dockerfile (placé à la racine du projet)

# 1. Image de base officielle
FROM python:3.10-slim

# 2. Définir le dossier de travail
WORKDIR /app

# 3. Copier les fichiers nécessaires
COPY api/app_gradio.py ./api/app_gradio.py
COPY models/ ./models/
COPY requirements.txt .

# 4. Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5. Exposer le port utilisé par Gradio
EXPOSE 7860

# 6. Commande pour démarrer Gradio
CMD ["python", "api/app_gradio.py"]
