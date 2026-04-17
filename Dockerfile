# Utiliser une image Python légère
FROM python:3.11-slim

# Empêcher Python de générer des fichiers .pyc et activer l'affichage immédiat des logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système légères
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier requirements.txt (celui que j'ai allégé sans docling)
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --progress-bar on -r requirements.txt

# Copier le reste de l'application
COPY . .

# Créer les dossiers nécessaires
RUN adduser --disabled-password --gecos "" appuser \
    && mkdir -p logs output temp_api temp_unpack \
    && chown -R appuser:appuser /app

USER appuser

# Healthcheck
ENV HEALTHCHECK_URL=http://localhost:8000/health
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request,sys; disabled=os.getenv('DISABLE_HEALTHCHECK','false').lower()=='true'; url=os.getenv('HEALTHCHECK_URL',''); sys.exit(0) if (disabled or not url) else None; urllib.request.urlopen(url, timeout=3).read()"

# Exposer le port de l'API
EXPOSE 8000

# Commande pour démarrer l'API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
