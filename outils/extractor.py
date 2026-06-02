"""
Extraction structurée : texte brut → JSON Finaxys via LLM.

Envoie le texte parsé au LLM avec un system prompt spécialisé
et récupère un JSON structuré conforme au schéma Finaxys.
"""

import json
import logging

from outils.prompt_loader import load_instruction_prompt_by_name

logger = logging.getLogger(__name__)

_FALLBACK_PROMPT = """Tu es un expert en extraction de données de CVs. Ton but est de transformer un texte de CV en un JSON STRRICTEMENT conforme au schéma.

IMPORTANT : Tu dois utiliser EXACTEMENT les noms de clés suivants. Ne les traduis pas, ne les change pas.

STRUCTURE JSON REQUISE :
{
  "identite": { 
    "nom": "string", "prenom": "string", "email": "string|null", 
    "localisation": "string|null", "telephone": "string|null", "linkedin": "string|null" 
  },
  "titre_professionnel": "string|null (max 6 mots)",
  "type_poste": "string|null",
  "profil": "string|null (rédigé à la 3e personne, sans 'je', 'j ai', 'mon')",
  "competences": [
    {
      "nom": "string (nom normalisé : JS→JavaScript, K8s→Kubernetes, Spring boot→Spring Boot)",
      "categorie": "string (parmi : Langages de programmation | Frameworks & Librairies | Bases de données | Cloud & DevOps | Outils & Méthodologies | Compétences fonctionnelles | Soft skills)",
      "niveau": "string (Débutant | Intermédiaire | Confirmé | Expert)",
      "annees_experience": null,
      "aliases_experiences": ["string (toutes les variantes de ce nom dans les technologies des expériences)"]
    }
  ],
  "experiences": [
    {
      "titre": "string", "entreprise": "string", "date_debut": "YYYY-MM", "date_fin": "YYYY-MM|null",
      "en_cours": bool, "projet": "string|null", "equipe": "string|null", "methodologie": "string|null",
      "missions": ["string (commence par un verbe d action au passé composé : A conçu..., A développé..., A mis en place...)"],
      "technologies": ["string"], "resultats": ["string (quantifié si possible : Réduction de 30%...)"]
    }
  ],
  "projets_academiques": [
    {
      "nom": "string", "etablissement": "string|null", "equipe": "string|int|null",
      "duree": "string|null", "missions": ["string"], "technologies": ["string"]
    }
  ],
  "formations": [
    { "diplome": "string", "etablissement": "string", "annee": int|null }
  ],
  "certifications": [
    { "nom": "string", "organisme": "string|null", "annee": "string|int|null", "score": "string|null" }
  ],
  "langues": [
    { "langue": "string", "niveau": "string", "certification": "string|null" }
  ],
  "centres_interet": "string|null",
  "metadata": {
    "score_completude": float,
    "champs_incertains": ["string"]
  }
}

RÈGLES D'OR :
- Pour "competences" : inclus AUSSI les technologies des expériences absentes de la liste principale. Déduplique.
- Pour "competences.aliases_experiences" : liste toutes les variantes du nom dans les technologies brutes (ex: ["K8s", "Kubernetes"] pour Kubernetes).
- Pour "competences" : ne renvoie JAMAIS une liste de strings. Renvoie toujours une liste d'OBJETS.
- Pour "profil" : rédige à la 3e personne, sans "je", "j'ai", "mon", "ma".
- Pour "missions" : commence par un verbe d'action au passé composé (A conçu, A développé, A piloté...).
- Si le texte est en Markdown, utilise les titres (#) pour repérer les sections.
- Réponds UNIQUEMENT avec le JSON."""

SYSTEM_PROMPT = load_instruction_prompt_by_name(
  "cv-extractor",
    _FALLBACK_PROMPT,
) + "\n\nIMPORTANT: Réponds UNIQUEMENT avec le JSON valide."


def _clean_json_response(raw: str) -> str:
    """Supprime les balises markdown ```json ... ``` si présentes."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return cleaned

from datetime import datetime, timezone
from pydantic import ValidationError

from schemas.models import CVData


def extract_cv_to_json(
    text: str,
    source_filename: str,
    client,
    provider: str,
) -> dict:
    """Envoie le texte du CV au LLM et retourne un dictionnaire validé par Pydantic."""
    from outils.llm_client import llm_call

    logger.info("Extraction LLM (%s) pour : %s", provider, source_filename)

    user_message = (
        f"Fichier source : {source_filename}\n\n"
        f"--- TEXTE DU CV ---\n{text}\n--- FIN DU TEXTE ---"
    )

    raw_response = llm_call(
        client,
        provider,
        SYSTEM_PROMPT,
        user_message,
        max_tokens=4096,
        operation="cv_extract",
    )
    cleaned = _clean_json_response(raw_response)

    try:
        data = json.loads(cleaned)
        
        # Normalisations légères avant validation
        centres_interet = data.get("centres_interet")
        if isinstance(centres_interet, list):
            data["centres_interet"] = ", ".join(str(item) for item in centres_interet if str(item).strip()) or None

        # Injecter les métadonnées avant validation
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["source_fichier"] = source_filename
        data["metadata"]["date_extraction"] = datetime.now(timezone.utc).isoformat()
        data["metadata"]["version_pipeline"] = "1.1.0"

        # VALIDATION PYDANTIC
        cv_obj = CVData(**data)
        
        # On retourne un dict pour la compatibilité avec le reste du pipeline, 
        # mais on est SÛR que les données sont valides.
        validated_data = cv_obj.model_dump()
        
    except json.JSONDecodeError as e:
        logger.error("JSON invalide retourné par le LLM : %s", e)
        raise ValueError(f"Le LLM a retourné un JSON invalide : {e}")
    except ValidationError as e:
        logger.error("Erreur de validation Pydantic : %s", e)
        # Optionnel : on pourrait tenter une réparation ici, mais pour l'instant on raise
        raise ValueError(f"Données extraites non conformes au schéma : {e}")

    logger.info("Extraction réussie et validée : %d compétences, %d expériences",
                len(validated_data.get("competences", [])), 
                len(validated_data.get("experiences", [])))
    
    return validated_data
