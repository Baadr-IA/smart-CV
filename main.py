"""
Pipeline CV → JSON → CV Finaxys
Point d'entrée CLI.

Usage :
  python main.py parse <fichier>             # Parser seul (dry-run, pas d'API)
  python main.py process <fichier>           # Pipeline complet
  python main.py process-dir <dossier>       # Traiter tous les CVs d'un dossier
  python main.py test-api                    # Tester la connexion LLM
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

# Configuration logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def _setup_logging(verbose: bool = False):
    """Configure le logging console + fichier."""
    level = logging.DEBUG if verbose else logging.INFO

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)

    # Fichier
    log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)

    return log_file


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Mode debug")
@click.pass_context
def cli(ctx, verbose):
    """Pipeline CV → JSON → CV Finaxys"""
    ctx.ensure_object(dict)
    log_file = _setup_logging(verbose)
    ctx.obj["verbose"] = verbose
    ctx.obj["log_file"] = log_file


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
def parse(file_path):
    """Parse un CV (PDF/DOCX) et affiche le texte extrait. """
    from outils.parser import parse_file

    result = parse_file(file_path)

    meta = {k: v for k, v in result.items() if k != "text"}
    click.echo(json.dumps(meta, ensure_ascii=False, indent=2))
    click.echo(f"\n--- Texte extrait ({result['char_count']} caractères) ---")
    click.echo(result["text"][:2000])
    if result["char_count"] > 2000:
        click.echo(f"\n... ({result['char_count'] - 2000} caractères supplémentaires)")


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default="output", help="Dossier de sortie")
@click.option("--skip-word", is_flag=True, help="Ne pas générer le Word")
@click.option("--index", is_flag=True, help="Indexer le CV dans la base vectorielle (RAG)")
def process(file_path, output_dir, skip_word, index):
    """Pipeline complet : CV brut → JSON → Word Finaxys."""
    logger = logging.getLogger("pipeline")
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from outils.llm_client import create_client
    client, provider = create_client()
    stem = file_path.stem

    # Étape 1 : Parsing
    # ... (inchangé)
    logger.info("=" * 60)
    logger.info("ÉTAPE 1/6 : PARSING — %s", file_path.name)
    logger.info("=" * 60)
    from outils.parser import parse_file
    parsed = parse_file(file_path)
    click.echo(f"[OK] Parsing : {parsed['char_count']} caractères ({parsed['method']})")

    if parsed["char_count"] < 50:
        click.echo("[ERR] Texte trop court. Le fichier est peut-être corrompu ou vide.")
        sys.exit(1)

    # Étape 2 : Extraction JSON
    logger.info("=" * 60)
    logger.info("ÉTAPE 2/6 : EXTRACTION JSON")
    logger.info("=" * 60)
    from outils.extractor import extract_cv_to_json
    cv_data = extract_cv_to_json(parsed["text"], file_path.name, client, provider)
    click.echo(f"[OK] Extraction : {len(cv_data.get('competences', []))} compétences, "
               f"{len(cv_data.get('experiences', []))} expériences")

    # Étape 3 : Classification compétences
    logger.info("=" * 60)
    logger.info("ÉTAPE 3/6 : CLASSIFICATION COMPÉTENCES")
    logger.info("=" * 60)
    from outils.classifier import classify_skills
    classified = classify_skills(cv_data, client, provider)
    cv_data["competences"] = classified
    click.echo(f"[OK] Classification : {len(classified)} compétences normalisées")

    # Étape 4 : Normalisation style
    logger.info("=" * 60)
    logger.info("ÉTAPE 4/6 : NORMALISATION STYLE FINAXYS")
    logger.info("=" * 60)
    from outils.normalizer import normalize_style
    cv_data = normalize_style(cv_data, client, provider)
    click.echo("[OK] Normalisation du style terminée")

    # Étape 5 : Validation
    logger.info("=" * 60)
    logger.info("ÉTAPE 5/6 : DOUBLE VALIDATION")
    logger.info("=" * 60)
    from outils.validator import validate
    report = validate(cv_data, client, provider)

    if report["structural_errors"]:
        click.echo("[!]  Erreurs structurelles :")
        for err in report["structural_errors"]:
            click.echo(f"   • {err}")

    if report["semantic_warnings"]:
        click.echo("[!]  Warnings sémantiques :")
        for w in report["semantic_warnings"]:
            click.echo(f"   • {w}")

    click.echo(f"[OK] Validation : structurelle={'OK' if report['structural_valid'] else 'KO'} | "
               f"sémantique={'OK' if report['semantic_valid'] else 'KO'} "
               f"(score={report['semantic_score']:.2f})")

    # Étape EXTRA : Indexation RAG
    if index:
        logger.info("=" * 60)
        logger.info("ÉTAPE EXTRA : INDEXATION VECTORIELLE (RAG)")
        logger.info("=" * 60)
        from outils.rag_utils import VectorStoreManager
        from schemas.models import CVData
        try:
            vdb = VectorStoreManager()
            # On reconvertit en objet Pydantic pour la sécurité
            cv_obj = CVData(**cv_data)
            vdb.add_cv(cv_obj, file_path.name)
            click.echo(f"[RAG] CV indexé dans ChromaDB (ID: {file_path.name})")
        except Exception as e:
            logger.error("Erreur lors de l'indexation RAG : %s", e)
            click.echo(f"[ERR] Échec de l'indexation : {e}")

    # Sauvegarder le JSON
    json_path = output_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cv_data, f, ensure_ascii=False, indent=2)
    click.echo(f"[>>] JSON sauvegardé : {json_path}")

    # Étape 6 : Génération Word
    if not skip_word:
        logger.info("=" * 60)
        logger.info("ÉTAPE 6/6 : GÉNÉRATION WORD")
        logger.info("=" * 60)
        from outils.generator import generate_word
        word_path = output_dir / f"{stem}_finaxys.docx"
        generate_word(cv_data, word_path)
        click.echo(f"[DOC] CV Word généré : {word_path}")

    click.echo("\n[***] Pipeline terminé avec succès !")


@cli.command("process-dir")
@click.argument("dir_path", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default="output", help="Dossier de sortie")
def process_dir(dir_path, output_dir):
    """Traite tous les CVs d'un dossier."""
    dir_path = Path(dir_path)
    files = list(dir_path.glob("*.pdf")) + list(dir_path.glob("*.docx"))

    if not files:
        click.echo(f"Aucun fichier PDF/DOCX trouvé dans {dir_path}")
        return

    click.echo(f"[DIR] {len(files)} fichier(s) trouvé(s) dans {dir_path}")
    for i, f in enumerate(files, 1):
        click.echo(f"\n{'='*60}")
        click.echo(f"[{i}/{len(files)}] {f.name}")
        click.echo(f"{'='*60}")
        try:
            ctx = click.get_current_context()
            ctx.invoke(process, file_path=str(f), output_dir=output_dir, skip_word=False)
        except Exception as e:
            click.echo(f"[ERR] Erreur sur {f.name} : {e}")
            logging.getLogger("pipeline").exception("Erreur sur %s", f.name)


@cli.command("test-api")
def test_api():
    """Teste la connexion au LLM configuré."""
    from outils.llm_client import create_client, get_model, llm_call

    click.echo("[>] Test de connexion LLM...")
    try:
        client, provider = create_client()
        model = get_model(provider)
        click.echo(f"   Provider : {provider} | Modèle : {model}")

        response = llm_call(
            client, provider,
            system_prompt="Tu es un assistant.",
            user_message="Réponds uniquement 'OK' si tu reçois ce message.",
            max_tokens=20,
        )
        click.echo(f"[OK] Connexion réussie ! Réponse : {response.strip()}")
    except Exception as e:
        click.echo(f"[ERR] Erreur de connexion : {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
