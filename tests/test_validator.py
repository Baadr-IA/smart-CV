"""Tests de validation JSON."""

import pytest
from skills.validator import validate_structure


VALID_CV = {
    "identite": {"nom": "Dupont", "prenom": "Jean", "email": None, "localisation": "Paris"},
    "titre_professionnel": "Développeur Java Senior",
    "profil": "Expert Java avec 10 ans d'expérience",
    "competences": [
        {"nom": "Java", "categorie": "Langages de programmation", "niveau": "Expert", "annees_experience": 10}
    ],
    "experiences": [
        {
            "titre": "Développeur Senior",
            "entreprise": "Finaxys",
            "client": None,
            "date_debut": "2020-01",
            "date_fin": None,
            "en_cours": True,
            "missions": ["A développé des microservices"],
            "technologies": ["Java", "Spring"],
            "resultats": [],
        }
    ],
    "formations": [
        {"diplome": "Master Informatique", "etablissement": "Université Paris-Saclay", "annee": 2014}
    ],
    "certifications": ["AWS Solutions Architect"],
    "langues": [{"langue": "Français", "niveau": "Natif"}],
    "metadata": {
        "date_extraction": "2026-03-12T10:00:00Z",
        "source_fichier": "dupont_jean.pdf",
        "score_completude": 0.85,
        "champs_incertains": [],
        "version_pipeline": "1.0.0",
    },
}


class TestStructuralValidation:
    def test_valid_cv(self):
        errors = validate_structure(VALID_CV)
        assert errors == []

    def test_missing_required_field(self):
        cv = {**VALID_CV}
        del cv["identite"]
        errors = validate_structure(cv)
        assert len(errors) > 0
        assert any("identite" in e for e in errors)

    def test_invalid_date_format(self):
        cv = {**VALID_CV, "experiences": [{
            "titre": "Dev",
            "entreprise": "Corp",
            "date_debut": "mars 2020",  # Format invalide
            "missions": [],
            "technologies": [],
            "resultats": [],
        }]}
        errors = validate_structure(cv)
        assert len(errors) > 0
