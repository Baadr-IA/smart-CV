"""
Classification des compétences : normalise et catégorise les compétences extraites via LLM.
Calcule les années d'expérience par compétence à partir des dates réelles des expériences (T1).

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
from datetime import datetime

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

Tu reçois :
1. Une liste de compétences extraites d'un CV
2. Une liste de technologies brutes mentionnées dans les expériences professionnelles

Pour chaque compétence, tu dois :
1. Normaliser le nom (ex: "JS" → "JavaScript", "K8s" → "Kubernetes", "Spring boot" → "Spring Boot")
2. Assigner une catégorie parmi : {json.dumps(CATEGORIES, ensure_ascii=False)}
3. Estimer un niveau parmi : "Débutant", "Intermédiaire", "Confirmé", "Expert"
4. Lister dans "aliases_experiences" toutes les variantes de cette compétence présentes
   dans la liste des technologies brutes des expériences (abréviations, casse différente,
   noms alternatifs). Laisser [] si aucune variante trouvée.
5. Mettre annees_experience à null (les années sont calculées séparément depuis les dates)

RÈGLES :
- Déduplique les compétences (même techno mentionnée différemment)
- Extrais aussi les technologies des expériences absentes de la liste principale
- Réponds UNIQUEMENT avec un tableau JSON valide, sans texte additionnel

FORMAT :
[
  {{
    "nom": "Kubernetes",
    "categorie": "Cloud & DevOps",
    "niveau": "Expert",
    "annees_experience": null,
    "aliases_experiences": ["K8s", "Kubernetes"]
  }},
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
    """Classifie et normalise les compétences du CV, puis calcule les années depuis les dates."""
    from outils.llm_client import llm_call

    competences_brutes = cv_data.get("competences", [])
    experiences = cv_data.get("experiences", [])

    techs_from_exp = []
    for exp in experiences:
        techs_from_exp.extend(exp.get("technologies", []))
    techs_uniques = list(set(techs_from_exp))

    user_content = (
        f"Compétences extraites : {json.dumps(competences_brutes, ensure_ascii=False)}\n\n"
        f"Technologies brutes dans les expériences : {json.dumps(techs_uniques, ensure_ascii=False)}"
    )

    logger.info("Classification de %d compétences + %d technologies",
                len(competences_brutes), len(techs_uniques))

    raw = llm_call(client, provider, SYSTEM_PROMPT, user_content, max_tokens=2048)
    cleaned = _clean_json_response(raw)

    try:
        classified = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("JSON invalide pour classification : %s", e)
        raise ValueError(f"Classification a retourné un JSON invalide : {e}")

    _inject_calculated_years(classified, experiences)

    logger.info("Classification terminée : %d compétences normalisées", len(classified))
    return classified


def _niveau_from_years(years: float) -> str:
    """Détermine le niveau d'une compétence à partir des années d'expérience réelles."""
    if years < 1:
        return "Débutant"
    elif years < 2:
        return "Intermédiaire"
    elif years < 4:
        return "Confirmé"
    else:
        return "Expert"


def _inject_calculated_years(skills: list[dict], experiences: list[dict]) -> None:
    """Calcule annees_experience pour chaque skill depuis les dates réelles des expériences."""
    try:
        from dateutil.parser import parse as parse_date
    except ImportError:
        logger.warning("python-dateutil absent — annees_experience restera null")
        return

    for skill in skills:
        aliases = [a.lower().strip() for a in skill.get("aliases_experiences", []) if a]
        if not aliases:
            continue

        periods = []
        for exp in experiences:
            exp_techs = [t.lower().strip() for t in exp.get("technologies", []) if t]
            if not any(alias in exp_techs for alias in aliases):
                continue
            try:
                start = parse_date(exp["date_debut"], default=datetime(2000, 1, 1)).date()
                if exp.get("en_cours") or not exp.get("date_fin"):
                    end = datetime.now().date()
                else:
                    end = parse_date(exp["date_fin"], default=datetime(2000, 12, 31)).date()
                if end > start:
                    periods.append((start, end))
            except Exception as e:
                logger.debug("Dates non parsables pour expérience '%s' : %s", exp.get("titre", "?"), e)
                continue

        if periods:
            merged = _merge_overlapping(sorted(periods))
            total_days = sum((e - s).days for s, e in merged)
            skill["annees_experience"] = round(total_days / 365.25, 1)
            skill["niveau"] = _niveau_from_years(skill["annees_experience"])
            logger.debug("'%s' → %.1f ans → niveau=%s (depuis %d expérience(s))",
                         skill.get("nom"), skill["annees_experience"],
                         skill["niveau"], len(merged))


def _merge_overlapping(periods: list[tuple]) -> list[tuple]:
    """Fusionne les périodes qui se chevauchent pour éviter le double comptage."""
    if not periods:
        return []
    merged = [periods[0]]
    for start, end in periods[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged
