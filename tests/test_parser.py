"""Tests du parser CV."""

import pytest
from pathlib import Path
from outils.parser import _clean_text, _compute_text_quality, _is_quality_sufficient, parse_file


class TestCleanText:
    """Tests de la fonction de nettoyage de texte."""

    def test_empty(self):
        assert _clean_text("") == ""

    def test_multiple_spaces(self):
        assert _clean_text("hello   world") == "hello world"

    def test_multiple_newlines(self):
        result = _clean_text("a\n\n\n\n\nb")
        assert result == "a\n\nb"

    def test_trailing_spaces(self):
        result = _clean_text("hello   \nworld  ")
        assert result == "hello\nworld"

    def test_windows_line_endings(self):
        result = _clean_text("hello\r\nworld")
        assert result == "hello\nworld"

    def test_control_characters(self):
        result = _clean_text("hello\x00\x01world")
        assert result == "helloworld"

    def test_hyphenated_words(self):
        result = _clean_text("déve-\nloppement")
        assert result == "développement"


class TestTextQuality:
    """Tests des métriques de qualité de texte."""

    def test_compute_text_quality_empty(self):
        metrics = _compute_text_quality("")
        assert metrics["char_count"] == 0
        assert metrics["alpha_ratio"] == 0.0

    def test_compute_text_quality_normal_text(self):
        metrics = _compute_text_quality("Bonjour le monde 2026")
        assert metrics["char_count"] > 0
        assert metrics["word_count"] == 4
        assert 0.0 <= metrics["alpha_ratio"] <= 1.0

    def test_is_quality_sufficient_true(self):
        assert _is_quality_sufficient({"char_count": 300, "alpha_ratio": 0.9})

    def test_is_quality_sufficient_false(self):
        assert not _is_quality_sufficient({"char_count": 50, "alpha_ratio": 0.4})


class TestParseFile:
    """Tests du parsing de fichiers."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_file("inexistant.pdf")

    def test_unsupported_format(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="Format non supporté"):
            parse_file(f)


class TestAPIConnection:
    """Test de connexion API (lancé manuellement, skip en CI)."""

    @pytest.mark.skipif(
        not Path(".env").exists(),
        reason=".env absent",
    )
    def test_anthropic_import(self):
        """Vérifie que le SDK Anthropic est installé."""
        import anthropic
        assert hasattr(anthropic, "Anthropic")
