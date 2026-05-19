from pathlib import Path
from unittest.mock import call, patch

import pytest

from service import process_cv_pipeline


def _mock_cv_data() -> dict:
    return {
        "identite": {"nom": "Doe", "prenom": "Jane"},
        "titre_professionnel": "Consultante Data",
        "competences": [],
        "experiences": [],
        "formations": [],
        "langues": [],
        "metadata": {
            "source_fichier": "cv.pdf",
            "date_extraction": "2026-05-18T12:00:00Z",
        },
    }


def test_process_cv_pipeline_observes_stage_metrics(tmp_path):
    fake = tmp_path / "cv.pdf"
    fake.write_bytes(b"%PDF-1.4\n" + b"a" * 4096)
    parse_result = {
        "char_count": 300,
        "text": "cv text " * 40,
        "method": "pypdf",
        "quality": {"page_count": 2},
    }
    mock_cv = _mock_cv_data()
    validation_report = {
        "structural_valid": True,
        "structural_errors": [],
        "semantic_valid": True,
        "semantic_score": 0.95,
        "semantic_warnings": [],
        "semantic_errors": [],
        "overall_valid": True,
    }

    with patch("service.observe_cv_input") as mock_input, patch("service.observe_cv_pages") as mock_pages, patch(
        "service.observe_cv_stage"
    ) as mock_stage, patch("service.observe_validation_report") as mock_validation_metrics, patch(
        "service.parse_file", return_value=parse_result
    ), patch("service.create_client", return_value=(object(), "openai")), patch(
        "service.extract_cv_to_json", return_value=mock_cv
    ), patch("service.classify_skills", return_value=[]), patch(
        "service.normalize_style", return_value=mock_cv
    ), patch("service.validate", return_value=validation_report), patch(
        "service.generate_word"
    ) as mock_generate_word:
        result = process_cv_pipeline(
            fake,
            output_dir=tmp_path,
            generate_word_doc=True,
            source="local",
        )

    assert result["metadata"]["parsing_method"] == "pypdf"
    mock_input.assert_called_once_with(
        source="local",
        mime_type="application/pdf",
        size_bytes=fake.stat().st_size,
    )
    mock_pages.assert_called_once_with(source="local", page_count=2)
    mock_validation_metrics.assert_called_once_with(validation_report)
    mock_generate_word.assert_called_once()

    expected_stages = ["parse", "extract", "classify", "normalize", "validate", "generate_word"]
    observed_stages = [call_args.kwargs["stage"] for call_args in mock_stage.call_args_list]
    assert observed_stages == expected_stages
    assert all(call_args.kwargs["status"] == "success" for call_args in mock_stage.call_args_list)


def test_process_cv_pipeline_records_stage_error(tmp_path):
    fake = tmp_path / "cv.pdf"
    fake.write_bytes(b"%PDF-1.4\n" + b"a" * 2048)
    parse_result = {
        "char_count": 300,
        "text": "cv text " * 40,
        "method": "pypdf",
        "quality": {"page_count": 1},
    }

    with patch("service.observe_cv_stage") as mock_stage, patch(
        "service.parse_file", return_value=parse_result
    ), patch("service.create_client", return_value=(object(), "openai")), patch(
        "service.extract_cv_to_json", side_effect=ValueError("bad llm json")
    ):
        with pytest.raises(ValueError, match="bad llm json"):
            process_cv_pipeline(fake)

    assert mock_stage.call_args_list[0].kwargs["stage"] == "parse"
    assert mock_stage.call_args_list[0].kwargs["status"] == "success"
    assert mock_stage.call_args_list[1].kwargs == {
        "stage": "extract",
        "status": "error",
        "latency_ms": mock_stage.call_args_list[1].kwargs["latency_ms"],
        "error_type": "ValueError",
    }
    assert mock_stage.call_args_list[1].kwargs["latency_ms"] >= 0
