"""
Normalisation légère en Python pur — le style Finaxys est produit directement
par l'extracteur (fusion Option B). Cette étape assure uniquement le nettoyage
mécanique : casse des technologies, suppression des pronoms résiduels, troncature du titre.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Mapping casse standard des technos courantes
_TECH_CASING: dict[str, str] = {
    "javascript": "JavaScript", "typescript": "TypeScript",
    "python": "Python", "java": "Java",
    "kubernetes": "Kubernetes", "k8s": "Kubernetes",
    "docker": "Docker", "react": "React",
    "angular": "Angular", "vue": "Vue.js",
    "nodejs": "Node.js", "node.js": "Node.js",
    "postgresql": "PostgreSQL", "mysql": "MySQL",
    "mongodb": "MongoDB", "aws": "AWS",
    "azure": "Azure", "gcp": "GCP",
    "git": "Git", "linux": "Linux",
    "fastapi": "FastAPI", "django": "Django",
    "flask": "Flask", "spring boot": "Spring Boot",
    "springboot": "Spring Boot",
}

_PRONOUN_RE = re.compile(
    r"\b(je|j'ai|j'|mon|ma|mes|nous avons|nous)\b", re.IGNORECASE
)


def _clean_pronouns(text: str) -> str:
    return re.sub(r" {2,}", " ", _PRONOUN_RE.sub("", text)).strip()


def normalize_style(cv_data: dict, client, provider: str) -> dict:
    """Normalisation Python : casse des techs, suppression de pronoms résiduels, troncature du titre."""
    logger.info("Normalisation Python pour : %s",
                cv_data.get("identite", {}).get("nom", "inconnu"))

    # Normaliser la casse des compétences
    for comp in cv_data.get("competences", []):
        nom = comp.get("nom", "")
        comp["nom"] = _TECH_CASING.get(nom.lower(), nom)

    # Troncature du titre à 6 mots
    title = cv_data.get("titre_professionnel")
    if title:
        words = title.split()
        if len(words) > 6:
            cv_data["titre_professionnel"] = " ".join(words[:6])

    # Nettoyage des pronoms résiduels dans le profil
    profil = cv_data.get("profil")
    if profil:
        cv_data["profil"] = _clean_pronouns(profil)

    logger.info("Normalisation Python terminée")
    return cv_data
