"""Charge les prompts LLM depuis le dossier instructions/ à la racine du projet."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTRUCTIONS_DIR = PROJECT_ROOT / "instructions"


def _strip_yaml_frontmatter(text: str) -> str:
    """Supprime le frontmatter YAML s'il est présent en tête du markdown."""
    content = text.lstrip()
    if not content.startswith("---\n"):
        return text.strip()

    parts = content.split("\n---\n", 1)
    if len(parts) != 2:
        return text.strip()
    return parts[1].strip()


def load_instruction_prompt(filename: str, fallback: str) -> str:
    """
    Charge un prompt markdown depuis le dossier instructions/.

    Si le fichier n'existe pas ou est vide, retourne le fallback fourni.
    """
    path = INSTRUCTIONS_DIR / filename
    try:
        content = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return fallback
    except OSError:
        return fallback

    return content or fallback


def load_instruction_prompt_by_name(name: str, fallback: str) -> str:
    """
    Charge instructions/<name>/SKILL.md et retourne son corps sans frontmatter.
    
    Note : Le fichier s'appelle toujours SKILL.md par commodité de formatage markdown,
    mais il est traité ici comme une instruction pour un script automatique.

    Retourne fallback si le fichier est absent, vide ou illisible.
    """
    path = INSTRUCTIONS_DIR / name / "SKILL.md"
    try:
        content = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return fallback
    except OSError:
        return fallback

    if not content:
        return fallback

    stripped = _strip_yaml_frontmatter(content)
    return stripped or fallback
