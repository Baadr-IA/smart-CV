import pytest
import json
from pathlib import Path
from schemas.models import CVData

def test_full_pipeline_mock(tmp_path):
    """
    Test d'intégration simulant une sortie LLM pour vérifier
    la validité du modèle Pydantic et de la génération.
    """
    mock_json = {
        "identite": {
            "nom": "DOE",
            "prenom": "John",
            "email": "john.doe@email.com"
        },
        "titre_professionnel": "Ingénieur IA",
        "competences": [
            {"nom": "Python", "categorie": "Langages", "niveau": "Expert"}
        ],
        "experiences": [
            {
                "titre": "Data Scientist",
                "entreprise": "Finaxys",
                "date_debut": "2022-01",
                "en_cours": True,
                "missions": ["Développement de pipelines RAG"]
            }
        ],
        "formations": [],
        "langues": [],
        "metadata": {
            "date_extraction": "2026-03-16T12:00:00Z",
            "source_fichier": "test.pdf"
        }
    }

    # 1. Validation Pydantic (si ça échoue, le test s'arrête)
    cv_data = CVData(**mock_json)
    assert cv_data.identite.nom == "DOE"
    assert len(cv_data.experiences) == 1

    # 2. Test de sauvegarde/rechargement
    output_file = tmp_path / "test_output.json"
    output_file.write_text(cv_data.model_dump_json(indent=2), encoding="utf-8")
    
    reloaded_data = CVData.model_validate_json(output_file.read_text(encoding="utf-8"))
    assert reloaded_data.identite.prenom == "John"

def test_pydantic_validation_error():
    """Vérifie que Pydantic détecte bien les données invalides."""
    bad_data = {
        "identite": {"nom": "Doe"}, # Manque le prénom obligatoire
        "metadata": {"date_extraction": "now", "source_fichier": "test.pdf"}
    }
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        CVData(**bad_data)
