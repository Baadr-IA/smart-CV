"""Root conftest.py — adds project root to sys.path for all test modules."""
import sys
from pathlib import Path

# Ensure project root is on the path so `outils`, `schemas`, `service`, `api` are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
