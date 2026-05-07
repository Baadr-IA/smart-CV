"""
Matching intelligent CV ↔ référentiel de compétences (T2).

Les noms de compétences du CV sont déjà normalisés par le LLM (T1).
Ce module fait uniquement de la comparaison textuelle :

1. EXACT    : libellés identiques après normalisation ASCII
2. FUZZY    : similarité textuelle (rapidfuzz) > seuil
3. NO_MATCH : aucune correspondance suffisante
"""

import logging
import unicodedata
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

FUZZY_THRESHOLD = 0.85  # score rapidfuzz normalisé entre 0 et 1


def _normalize(name: str) -> str:
    """Normalise un nom de compétence : lowercase, sans accents, tirets → espaces."""
    if not name:
        return ""
    nfd = unicodedata.normalize("NFD", name)
    sans_accents = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return sans_accents.lower().strip().replace("-", " ").replace("_", " ")


@dataclass
class SkillMatchResult:
    skill_cv: str
    skill_ref: Optional[str]
    score: float
    match_type: str  # EXACT | FUZZY | NO_MATCH


def match_skills(skills_cv: list[str], skills_ref: list[str]) -> list[SkillMatchResult]:
    """
    Pour chaque compétence du CV, trouve la meilleure correspondance dans le référentiel.
    Les noms sont supposés déjà normalisés par le LLM (ex: Kubernetes, Spring Boot).
    Retourne une liste ordonnée dans le même ordre que skills_cv.
    """
    if not skills_cv:
        return []
    if not skills_ref:
        return [SkillMatchResult(s, None, 0.0, "NO_MATCH") for s in skills_cv]

    normalized_ref = [(s, _normalize(s)) for s in skills_ref]
    results: list[SkillMatchResult] = []

    for skill_cv in skills_cv:
        norm_cv = _normalize(skill_cv)
        best: Optional[SkillMatchResult] = None

        for skill_ref, norm_ref in normalized_ref:

            # Niveau 1 — exact (après normalisation ASCII)
            if norm_cv == norm_ref:
                best = SkillMatchResult(skill_cv, skill_ref, 1.0, "EXACT")
                break

            # Niveau 2 — fuzzy (gère les fautes de frappe et variantes mineures)
            fuzzy_score = fuzz.token_sort_ratio(norm_cv, norm_ref) / 100.0
            if fuzzy_score >= FUZZY_THRESHOLD:
                candidate = SkillMatchResult(skill_cv, skill_ref, round(fuzzy_score, 3), "FUZZY")
                if best is None or candidate.score > best.score:
                    best = candidate

        results.append(best or SkillMatchResult(skill_cv, None, 0.0, "NO_MATCH"))

    logger.info(
        "Matching terminé : %d/%d compétences matchées",
        sum(1 for r in results if r.match_type != "NO_MATCH"),
        len(results),
    )
    return results

