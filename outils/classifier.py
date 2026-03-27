"""
Classification des compétences : normalise et catégorise les compétences extraites via LLM.

Catégories Finaxys :
- Langages de programmation
- Frameworks & Librairies
- Bases de données
- Cloud & DevOps
- Outils & Méthodologies
- Compétences fonctionnelles
- Soft skills
"""

import json
import logging

from outils.prompt_loader import load_instruction_prompt_by_name

logger = logging.getLogger(__name__)

CATEGORIES = [
    "Langages de programmation",
    "Frameworks & Librairies",
    "Bases de données",
    "Cloud & DevOps",
    "Outils & Méthodologies",
    "Compétences fonctionnelles",
    "Soft skills",
]

_FALLBACK_PROMPT = f"""Tu es un expert en classification de compétences techniques et fonctionnelles.

Tu reçois une liste de compétences extraites d'un CV.
Pour chaque compétence, tu dois :
1. Normaliser le nom (ex: "JS" → "JavaScript", "K8s" → "Kubernetes")
2. Assigner une catégorie parmi : {json.dumps(CATEGORIES, ensure_ascii=False)}
3. Estimer un niveau parmi : "Débutant", "Intermédiaire", "Confirmé", "Expert"
4. Conserver les années d'expérience si disponibles (sinon null)

RÈGLES :
- Déduplique les compétences (même techno mentionnée différemment)
- Extrais aussi les technologies mentionnées dans les expériences mais absentes de la liste
- Réponds UNIQUEMENT avec un tableau JSON de compétences

FORMAT :
[
  {{ "nom": "Python", "categorie": "Langages de programmation", "niveau": "Expert", "annees_experience": 5 }},
  ...
]"""

SYSTEM_PROMPT = load_instruction_prompt_by_name(
    "skills-classifier",
        _FALLBACK_PROMPT,
) + "\n\nIMPORTANT: Réponds strictement avec un tableau JSON valide, sans texte additionnel."


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


def classify_skills(cv_data: dict, client, provider: str) -> list[dict]:
    """Classifie et normalise les compétences du CV."""
    from outils.llm_client import llm_call

    competences_brutes = cv_data.get("competences", [])
    techs_from_exp = []
    for exp in cv_data.get("experiences", []):
        techs_from_exp.extend(exp.get("technologies", []))

    user_content = (
        f"Compétences extraites : {json.dumps(competences_brutes, ensure_ascii=False)}\n\n"
        f"Technologies mentionnées dans les expériences : {json.dumps(list(set(techs_from_exp)), ensure_ascii=False)}"
    )

    logger.info("Classification de %d compétences + %d technologies",
                len(competences_brutes), len(set(techs_from_exp)))

    raw = llm_call(client, provider, SYSTEM_PROMPT, user_content, max_tokens=2048)
    cleaned = _clean_json_response(raw)

    try:
        classified = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("JSON invalide pour classification : %s", e)
        raise ValueError(f"Classification a retourné un JSON invalide : {e}")

    logger.info("Classification terminée : %d compétences normalisées", len(classified))
    return classified
