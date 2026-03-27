# Utiliser une image Python légère
FROM python:3.11-slim

# Empêcher Python de générer des fichiers .pyc et activer l'affichage immédiat des logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système (Tesseract et Docling libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier requirements.txt
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --progress-bar on -r requirements.txt

# --- PRÉ-TÉLÉCHARGEMENT DES MODÈLES DOCLING ---
RUN python3 -c "from docling.document_converter import DocumentConverter; \
    from docling.datamodel.pipeline_options import PdfPipelineOptions; \
    from docling.datamodel.base_models import InputFormat; \
    from docling.document_converter import PdfFormatOption; \
    pipeline_options = PdfPipelineOptions(); \
    pipeline_options.do_ocr = True; \
    pipeline_options.do_table_structure = True; \
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}); \
    print('Docling models pre-downloaded successfully!')"

# Copier le reste de l'application
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p logs output temp_api temp_unpack

# Exposer le port de l'API
EXPOSE 8000

# Commande pour démarrer l'API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
