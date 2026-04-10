from __future__ import annotations

from pathlib import Path
from typing import Iterable


_ILLEGAL_CHARS = {"\x00", "\n", "\r"}


def sanitize_client_filename(filename: str, *, max_length: int = 255) -> str:
    """
    Clean a user-provided filename so it cannot escape directories or inject control characters.
    """
    if not filename or not isinstance(filename, str):
        raise ValueError("Le nom de fichier est vide.")

    if len(filename) > max_length:
        raise ValueError("Nom de fichier trop long.")

    name = Path(filename).name.strip()
    if not name or name in {".", ".."}:
        raise ValueError("Nom de fichier invalide.")

    if any(ch in name for ch in _ILLEGAL_CHARS):
        raise ValueError("Nom de fichier contenant des caractères non autorisés.")

    return name


def ensure_path_within_directory(
    base_dir: Path,
    candidate: Path,
    *,
    strict: bool = False
) -> Path:
    """
    Resolve a candidate path and ensure it stays inside base_dir.
    """
    base_resolved = base_dir.resolve()
    resolved = candidate.resolve(strict=strict)
    if not resolved.is_relative_to(base_resolved):
        raise ValueError(f"{candidate} sort du dossier autorisé {base_dir}.")
    return resolved


def resolve_path_in_directory(
    base_dir: Path,
    relative_path: str,
    *,
    strict: bool = False
) -> Path:
    """
    Build a path from base_dir + relative_path and ensure it stays inside base_dir.
    """
    if not relative_path or not relative_path.strip():
        raise ValueError("Le chemin est vide.")

    candidate = base_dir / relative_path
    return ensure_path_within_directory(base_dir, candidate, strict=strict)


def ensure_extension(filename: str, allowed: Iterable[str]) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed:
        raise ValueError("Extension non autorisée.")
    return suffix
