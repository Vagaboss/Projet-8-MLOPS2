# Dockerfile (placé à la racine du projet)

# 1. Image de base officielle
FROM python:3.10-slim

# 2. Définir le dossier de travail
WORKDIR /app

# Créer le dossier logs avec les bons droits
RUN mkdir -p /app/logs && chmod -R 777 /app/logs

# 3. Copier les fichiers nécessaires
COPY api/ ./api/
COPY models/ ./models/
COPY requirements.txt .

# 4. Installer les dépendances
RUN python -m pip install --no-cache-dir -r requirements.txt

# 5. Exposer le port utilisé par Gradio
EXPOSE 7860


ENV PYTHONPATH="${PYTHONPATH}:/app"

# 6. Commande pour démarrer Gradio
CMD ["python", "api/app_gradio.py"]
