"""
Classification des compétences : calcule les années d'expérience depuis les dates réelles.

Le LLM n'est plus nécessaire ici — l'extracteur produit directement les compétences
avec aliases_experiences. Cette étape est désormais entièrement en Python.
"""

import logging
from datetime import datetime

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


def classify_skills(cv_data: dict, client, provider: str) -> list[dict]:
    """Injecte les années d'expérience calculées depuis les dates réelles des expériences.
    
    Le LLM n'est plus appelé ici — la classification et les aliases sont produits
    directement par l'extracteur (fusion Option A).
    """
    competences = list(cv_data.get("competences", []))
    experiences = cv_data.get("experiences", [])

    _inject_calculated_years(competences, experiences)

    logger.info("Calcul des années terminé : %d compétences normalisées", len(competences))
    return competences


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
