"""
Chargement et résolution du référentiel de postes (referentiel_postes.yaml).
Utilisé par l'endpoint /search/job pour enrichir les requêtes RAG.
"""
from __future__ import annotations

import logging
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("job_referentiel")

_YAML_PATH = Path(__file__).parent.parent / "referentiel_postes.yaml"


def _normalize(text: str) -> str:
    """Minuscule + suppression des accents pour comparaison tolérante."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c)).strip()


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _normalize(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(value.strip())
    return result


@lru_cache(maxsize=1)
def _load_referentiel() -> dict:
    try:
        with open(_YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        logger.info("Référentiel postes chargé : %d postes", len(data.get("postes", {})))
        return data.get("postes", {})
    except FileNotFoundError:
        logger.warning("referentiel_postes.yaml introuvable — fallback libre activé")
        return {}


def resolve_poste(job_title: str) -> Optional[dict]:
    """
    Résout un intitulé de poste vers une entrée du référentiel.
    Cherche d'abord par clé exacte, puis par alias normalisé.
    Retourne None si le poste n'est pas dans le référentiel.
    """
    referentiel = _load_referentiel()
    normalized_input = _normalize(job_title)

    # 1. Match exact sur la clé
    if job_title in referentiel:
        return referentiel[job_title]

    # 2. Match normalisé sur les clés
    for key, poste in referentiel.items():
        if _normalize(key) == normalized_input:
            return poste

    # 3. Match normalisé sur le label
    for poste in referentiel.values():
        if _normalize(poste.get("label", "")) == normalized_input:
            return poste

    # 4. Match normalisé sur les alias
    for poste in referentiel.values():
        for alias in poste.get("aliases", []):
            if _normalize(alias) == normalized_input:
                return poste

    return None


def build_enriched_query(job_title: str, poste: Optional[dict]) -> str:
    """
    Construit une requête enrichie pour le moteur vectoriel.
    Si le poste est dans le référentiel, on injecte les skills requis.
    Sinon, on utilise le titre brut.
    """
    if poste is None:
        return job_title
    skills = poste.get("required_skills", [])
    return f"{job_title} {' '.join(skills)}"


def compute_skill_gap(
    required_skills: list[str],
    candidate_text: str,
) -> tuple[list[str], list[str]]:
    """
    Calcule les compétences trouvées / manquantes dans le texte du candidat.
    Comparaison insensible à la casse et aux accents.
    """
    normalized_text = _normalize(candidate_text)
    matched: list[str] = []
    missing: list[str] = []
    for skill in required_skills:
        if _normalize(skill) in normalized_text:
            matched.append(skill)
        else:
            missing.append(skill)
    return matched, missing


def build_prefilter_terms(
    job_title: str,
    poste: Optional[dict],
    *,
    max_aliases: int = 4,
    max_skills: int = 8,
) -> list[str]:
    """
    Construit une liste courte de termes métier pour le préfiltrage SQL et
    la recherche lexicale hybride.
    """
    terms = [job_title]
    if poste is None:
        return _dedupe_keep_order(terms)

    label = poste.get("label")
    if label:
        terms.append(label)

    terms.extend(poste.get("aliases", [])[:max_aliases])
    terms.extend(poste.get("required_skills", [])[:max_skills])
    return _dedupe_keep_order([term for term in terms if term])
