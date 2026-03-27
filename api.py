import os
import shutil
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from schemas.models import CVData
from service import process_cv_pipeline
from outils.rag_utils import VectorStoreManager
from outils.llm_client import create_client, llm_call

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CV Finaxys API",
    description="API d'analyse de CV (PDF/DOCX) pour transformation en JSON Finaxys et recherche RAG.",
    version="1.2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dossiers temporaires
TEMP_DIR = Path("temp_api")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Systèmes de prompts pour le RAG
RAG_SYSTEM_PROMPT = """Tu es un assistant expert en recrutement pour la société Finaxys.
Ton rôle est d'analyser les profils de candidats extraits d'une base de données vectorielle et de répondre à la demande d'un recruteur (RH).
Réponds de manière synthétique et cite tes sources."""

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API CV Finaxys. Accédez à /docs pour la documentation."}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=CVData)
async def analyze_cv(
    file: UploadFile = File(...),
    generate_word: bool = True,
    index: bool = True
):
    """
    Analyse un CV (PDF/DOCX), extrait les données et optionnellement l'indexe dans le RAG.
    """
    logger.info("Réception d'un fichier : %s", file.filename)
    
    # Vérification de l'extension
    ext = Path(file.filename).suffix.lower()
    if ext not in [".pdf", ".docx"]:
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF et DOCX sont acceptés.")

    # Sauvegarde temporaire du fichier
    temp_path = TEMP_DIR / file.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Exécution du pipeline
        cv_data = process_cv_pipeline(
            file_path=temp_path,
            output_dir=OUTPUT_DIR,
            generate_word_doc=generate_word
        )
        
        # Ajouter le chemin du fichier Word dans les métadonnées
        if generate_word:
            word_name = f"{temp_path.stem}_finaxys.docx"
            cv_data["metadata"]["fichier_word"] = str(OUTPUT_DIR / word_name)
        
        # Indexation RAG automatique
        if index:
            try:
                vdb = VectorStoreManager()
                vdb.add_cv(cv_data, file.filename)
                logger.info("CV indexé dans le RAG : %s", file.filename)
            except Exception as e:
                logger.error("Erreur indexation RAG : %s", e)

        return cv_data

    except Exception as e:
        logger.exception("Erreur lors du traitement du CV.")
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement : {str(e)}")
    
    finally:
        # Nettoyage fichier temporaire
        if temp_path.exists():
            os.remove(temp_path)

@app.post("/analyze-local", response_model=CVData)
async def analyze_local_cv(
    filename: str,
    generate_word: bool = True,
    index: bool = True
):
    """
    Analyse un CV déjà présent dans le dossier 'input/', extrait les données 
    et optionnellement l'indexe dans le RAG. Idéal pour les agents OpenClaw.
    """
    input_dir = Path("input")
    file_path = input_dir / filename
    
    if not file_path.exists():
        # Essayer aussi dans le dossier temporaire si non trouvé
        file_path = TEMP_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Fichier '{filename}' non trouvé dans 'input/' ou 'temp_api/'.")

    logger.info("Analyse locale demandée pour : %s", filename)
    
    try:
        # Exécution du pipeline
        cv_data = process_cv_pipeline(
            file_path=file_path,
            output_dir=OUTPUT_DIR,
            generate_word_doc=generate_word
        )
        
        # Ajouter le chemin du fichier Word dans les métadonnées
        if generate_word:
            word_name = f"{file_path.stem}_finaxys.docx"
            cv_data["metadata"]["fichier_word"] = str(OUTPUT_DIR / word_name)
        
        # Indexation RAG automatique
        if index:
            try:
                vdb = VectorStoreManager()
                vdb.add_cv(cv_data, filename)
                logger.info("CV indexé dans le RAG : %s", filename)
            except Exception as e:
                logger.error("Erreur indexation RAG : %s", e)

        return cv_data

    except Exception as e:
        logger.exception("Erreur lors du traitement local du CV.")
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement : {str(e)}")

@app.get("/search")
async def search_candidates(query: str, results: int = 3):
    """
    Recherche sémantique (RAG) parmi les CV indexés. Idéal pour le bot Telegram.
    """
    logger.info("Requête RAG reçue : %s", query)
    try:
        vdb = VectorStoreManager()
        search_results = vdb.search(query, n_results=results)
        
        if not search_results or not search_results['documents'][0]:
            return {"answer": "Désolé, je n'ai trouvé aucun candidat correspondant à cette recherche.", "candidates": []}

        # Génération de la réponse IA
        context_parts = []
        for i, doc in enumerate(search_results['documents'][0]):
            source = search_results['metadatas'][0][i].get('source', 'Inconnu')
            context_parts.append(f"CANDIDAT {i+1} (Source: {source}):\n{doc}")
        
        client, provider = create_client()
        user_msg = f"Demande RH: {query}\n\nProfils trouvés:\n" + "\n".join(context_parts)
        
        answer = llm_call(client, provider, RAG_SYSTEM_PROMPT, user_msg)
        
        return {
            "query": query,
            "answer": answer,
            "sources": [m.get('source') for m in search_results['metadatas'][0]]
        }

    except Exception as e:
        logger.error("Erreur recherche RAG : %s", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
