"""Root conftest.py — adds project root to sys.path and stable test env defaults."""
import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on the path so `outils`, `schemas`, `service`, `api` are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("API_KEY_DISABLED", "true")
os.environ.setdefault("DISABLE_DOCLING", "1")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")


@pytest.fixture(autouse=True)
def _stabilize_api_test_flags():
    api_module = sys.modules.get("api")
    if api_module is not None:
        api_module.API_KEY_DISABLED = True
        api_module.RATE_LIMIT_ENABLED = False
        api_module.MAX_CONCURRENT_JOBS = 0
    yield
