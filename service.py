import logging
import time
from pathlib import Path
from typing import Dict, Any, Callable, Optional, TypeVar

from schemas.models import CVData
from outils.llm_client import create_client
from outils.parser import parse_file
from outils.extractor import extract_cv_to_json
from outils.classifier import classify_skills
from outils.normalizer import normalize_style
from outils.validator import validate
from outils.generator import generate_word
from outils.metrics import observe_cv_input, observe_cv_pages, observe_cv_stage, observe_validation_report

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _guess_mime_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if suffix == ".doc":
        return "application/msword"
    return "application/octet-stream"


def _run_stage(stage: str, operation: Callable[[], T]) -> T:
    start = time.perf_counter()
    try:
        result = operation()
    except Exception as exc:
        observe_cv_stage(
            stage=stage,
            status="error",
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            error_type=exc.__class__.__name__,
        )
        raise
    observe_cv_stage(
        stage=stage,
        status="success",
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
    )
    return result

def process_cv_pipeline(
    file_path: Path,
    output_dir: Optional[Path] = None,
    generate_word_doc: bool = True,
    source: str = "unknown",
) -> Dict[str, Any]:
    """
    Exécute le pipeline complet d'analyse de CV.
    Retourne le dictionnaire de données validé.
    """
    logger.info("Démarrage du pipeline pour %s", file_path.name)

    try:
        observe_cv_input(
            source=source,
            mime_type=_guess_mime_type(file_path),
            size_bytes=file_path.stat().st_size,
        )
    except OSError:
        logger.debug("Impossible de lire la taille du fichier pour les métriques: %s", file_path)

    client, provider = create_client()

    def _parse_stage() -> Dict[str, Any]:
        parsed = parse_file(file_path)
        if parsed["char_count"] < 50:
            raise ValueError("Texte trop court ou fichier illisible.")
        return parsed

    parsed = _run_stage("parse", _parse_stage)
    observe_cv_pages(
        source=source,
        page_count=parsed.get("quality", {}).get("page_count"),
    )

    # Étape 2 : Extraction JSON
    cv_data = _run_stage(
        "extract",
        lambda: extract_cv_to_json(parsed["text"], file_path.name, client, provider),
    )

    # Réinjection des métadonnées de parsing
    cv_data.setdefault("metadata", {})
    cv_data["metadata"]["char_count"] = parsed["char_count"]
    cv_data["metadata"]["parsing_method"] = parsed["method"]

    # Étape 3 : Classification compétences
    classified = _run_stage(
        "classify",
        lambda: classify_skills(cv_data, client, provider),
    )
    cv_data["competences"] = classified

    # Étape 4 : Normalisation style
    cv_data = _run_stage(
        "normalize",
        lambda: normalize_style(cv_data, client, provider),
    )

    # Étape 5 : Validation
    def _validate_stage() -> Dict[str, Any]:
        report = validate(cv_data, client, provider)
        cv_data["metadata"]["validation_report"] = report
        observe_validation_report(report)
        return CVData(**cv_data).model_dump()

    cv_data = _run_stage("validate", _validate_stage)

    # Sauvegarde optionnelle du document Word seulement si demandé
    if output_dir and generate_word_doc:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = file_path.stem
        word_path = output_dir / f"{stem}_finaxys.docx"
        _run_stage("generate_word", lambda: generate_word(cv_data, word_path))
        cv_data["metadata"]["word_path"] = word_path.name
        cv_data["metadata"]["word_filename"] = word_path.name

    return cv_data
