import logging
import os
from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions
from schemas.models import CVData

logger = logging.getLogger("rag_utils")

def cv_to_searchable_text(cv: CVData) -> str:
    """
    Transforme un objet CVData en une description textuelle riche pour le RAG.
    Extraction déterministe sans LLM.
    """
    sections = []
    
    # Identité & Titre
    identite = f"Candidat : {cv.identite.prenom} {cv.identite.nom}"
    if cv.titre_professionnel:
        identite += f" ({cv.titre_professionnel})"
    sections.append(identite)
    
    # Profil
    if cv.profil:
        sections.append(f"Profil résumé : {cv.profil}")
    
    # Compétences
    if cv.competences:
        skills = []
        for s in cv.competences:
            if isinstance(s, dict):
                skills.append(f"{s.get('nom')} ({s.get('niveau', 'Intermédiaire')})")
            else:
                skills.append(s.nom if hasattr(s, 'nom') else str(s))
        sections.append(f"Compétences techniques : {', '.join(skills)}.")

    # Expériences (Dernières expériences les plus importantes)
    if cv.experiences:
        exp_list = []
        for exp in cv.experiences[:5]: # On prend les 5 dernières
            exp_text = f"{exp.titre} chez {exp.entreprise} ({exp.date_debut} à {exp.date_fin or 'Présent'})"
            if exp.missions:
                exp_text += f". Missions : {' '.join(exp.missions[:3])}"
            exp_list.append(exp_text)
        sections.append(f"Parcours professionnel : {' | '.join(exp_list)}")

    # Certifications
    if cv.certifications:
        certs = [f"{c.nom} ({c.organisme or 'N/A'})" for c in cv.certifications]
        sections.append(f"Certifications : {', '.join(certs)}.")

    # Langues
    if cv.langues:
        langs = [f"{l.langue} ({l.niveau})" for l in cv.langues]
        sections.append(f"Langues : {', '.join(langs)}.")

    return "\n".join(sections)

class VectorStoreManager:
    def __init__(self, db_path: str = "./vector_db"):
        db_path = os.getenv("CHROMA_DB_PATH", db_path)
        self.client = chromadb.PersistentClient(path=db_path)
        # On utilise un modèle d'embedding local gratuit (all-MiniLM-L6-v2)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="cv_collection",
            embedding_function=self.embedding_fn
        )

    def add_cv(self, cv: CVData, doc_id: str):
        """Ajoute ou met à jour un CV dans la base vectorielle."""
        text = cv_to_searchable_text(cv)
        
        # Métadonnées pour filtrage (optionnel mais puissant)
        metadata = {
            "nom": cv.identite.nom,
            "prenom": cv.identite.prenom,
            "titre": cv.titre_professionnel or "",
            "source": doc_id
        }
        
        self.collection.upsert(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        logger.info(f"CV de {cv.identite.nom} indexé avec succès.")

    def search(self, query: str, n_results: int = 3):
        """Recherche les CV les plus pertinents pour une requête donnée."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
