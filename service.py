import logging
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from outils.llm_client import create_client
from outils.parser import parse_file
from outils.extractor import extract_cv_to_json
from outils.classifier import classify_skills
from outils.normalizer import normalize_style
from outils.validator import validate
from outils.generator import generate_word

logger = logging.getLogger(__name__)

def process_cv_pipeline(
    file_path: Path, 
    output_dir: Optional[Path] = None, 
    generate_word_doc: bool = True
) -> Dict[str, Any]:
    """
    Exécute le pipeline complet d'analyse de CV.
    Retourne le dictionnaire de données validé.
    """
    logger.info("Démarrage du pipeline pour %s", file_path.name)
    
    client, provider = create_client()
    
    # Étape 1 : Parsing
    parsed = parse_file(file_path)
    if parsed["char_count"] < 50:
        raise ValueError("Texte trop court ou fichier illisible.")

    # Étape 2 : Extraction JSON
    cv_data = extract_cv_to_json(parsed["text"], file_path.name, client, provider)
    
    # Réinjection des métadonnées de parsing
    cv_data.setdefault("metadata", {})
    cv_data["metadata"]["char_count"] = parsed["char_count"]
    cv_data["metadata"]["parsing_method"] = parsed["method"]

    # Étape 3 : Classification compétences
    classified = classify_skills(cv_data, client, provider)
    cv_data["competences"] = classified

    # Étape 4 : Normalisation style
    cv_data = normalize_style(cv_data, client, provider)

    # Étape 5 : Validation
    report = validate(cv_data, client, provider)
    # On ajoute le rapport de validation aux métadonnées ou à la réponse
    cv_data["metadata"]["validation_report"] = report

    # Sauvegarde optionnelle du document Word seulement si demandé
    if output_dir and generate_word_doc:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = file_path.stem
        word_path = output_dir / f"{stem}_finaxys.docx"
        generate_word(cv_data, word_path)
        cv_data["metadata"]["word_path"] = word_path.name
        cv_data["metadata"]["word_filename"] = word_path.name

    return cv_data
