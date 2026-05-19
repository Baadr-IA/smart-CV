"""
Normalisation du style rédactionnel : applique le ton Finaxys au JSON extrait.

Finaxys attend :
- Descriptions de missions commençant par un verbe d'action au passé composé
  (ex: "A conçu...", "A piloté...", "A développé...")
- Profil rédigé à la 3e personne
- Formulation concise et orientée résultat
"""

import json
import logging

from outils.prompt_loader import load_instruction_prompt_by_name

logger = logging.getLogger(__name__)

_FALLBACK_PROMPT = """Tu es un rédacteur expert en CVs de consultants IT pour le cabinet Finaxys.

Tu reçois un JSON de CV structuré. Tu dois NORMALISER le style rédactionnel selon les conventions Finaxys :

RÈGLES DE STYLE FINAXYS :
1. Le "profil" doit être rédigé à la 3e personne du singulier
   Ex: "Développeur Java Senior avec 8 ans d'expérience spécialisé en..."
2. Les "missions" dans chaque expérience doivent commencer par un verbe d'action au passé composé
   Ex: "A conçu et développé...", "A piloté la migration...", "A mis en place..."
3. Les "resultats" doivent être mesurables si possible
   Ex: "Réduction de 40% du temps de traitement", "Migration de 200+ microservices"
4. Le "titre_professionnel" doit être concis (max 6 mots)
5. Supprime les pronoms personnels ("je", "j'ai", "mon")
6. Uniformise la casse des technologies (JavaScript, pas javascript)

IMPORTANT :
- Ne modifie PAS les données factuelles (dates, entreprises, certifications)
- Ne modifie PAS la structure JSON
- Retourne le JSON complet avec le style normalisé
- Réponds UNIQUEMENT avec le JSON"""

SYSTEM_PROMPT = load_instruction_prompt_by_name(
    "cv-normalizer",
    _FALLBACK_PROMPT,
) + "\n\nIMPORTANT: Réponds strictement en JSON valide, sans texte additionnel."


def _clean_json_response(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return cleaned


def normalize_style(cv_data: dict, client, provider: str) -> dict:
    """Normalise le style rédactionnel du CV au format Finaxys."""
    from outils.llm_client import llm_call

    logger.info("Normalisation du style pour : %s",
                cv_data.get("identite", {}).get("nom", "inconnu"))

    user_message = f"JSON du CV à normaliser :\n{json.dumps(cv_data, ensure_ascii=False, indent=2)}"
    raw = llm_call(
        client,
        provider,
        SYSTEM_PROMPT,
        user_message,
        max_tokens=4096,
        operation="cv_normalize",
    )
    cleaned = _clean_json_response(raw)

    try:
        normalized = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("JSON invalide après normalisation : %s", e)
        raise ValueError(f"Normalisation a retourné un JSON invalide : {e}")

    # Defensive merge: if the LLM dropped 'metadata' (or any other structural key),
    # restore it from the original data so the pipeline never crashes with KeyError.
    for key in (
        "metadata",
        "identite",
        "competences",
        "experiences",
        "projets_academiques",
        "formations",
        "certifications",
        "langues",
        "centres_interet",
    ):
        if key not in normalized and key in cv_data:
            logger.warning("normalize_style: LLM dropped key '%s' — restoring from original.", key)
            normalized[key] = cv_data[key]

    logger.info("Normalisation terminée")
    return normalized
