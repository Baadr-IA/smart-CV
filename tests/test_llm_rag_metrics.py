from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from outils import llm_client
from schemas.models import CVData


def test_llm_call_observes_usage_metrics():
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=30, total_tokens=150)
    response = MagicMock()
    response.choices = [SimpleNamespace(message=SimpleNamespace(content="OK"))]
    response.usage = usage
    client = MagicMock()
    client.chat.completions.create.return_value = response

    with patch("outils.llm_client.observe_llm_call") as mock_observe:
        result = llm_client.llm_call(
            client,
            "openai",
            "sys",
            "user",
            max_tokens=10,
            operation="cv_extract",
        )

    assert result == "OK"
    mock_observe.assert_called_once()
    kwargs = mock_observe.call_args.kwargs
    assert kwargs["provider"] == "openai"
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["operation"] == "cv_extract"
    assert kwargs["status"] == "success"
    assert kwargs["prompt_tokens"] == 120
    assert kwargs["completion_tokens"] == 30
    assert kwargs["total_tokens"] == 150
    assert kwargs["cost_usd"] == pytest.approx(0.00105)


def test_llm_call_observes_error_metrics():
    client = MagicMock()
    client.chat.completions.create.side_effect = ValueError("boom")

    with patch("outils.llm_client.observe_llm_call") as mock_observe:
        with pytest.raises(ValueError, match="boom"):
            llm_client.llm_call(client, "openai", "sys", "user", operation="cv_extract")

    mock_observe.assert_called_once()
    kwargs = mock_observe.call_args.kwargs
    assert kwargs["status"] == "error"
    assert kwargs["error_type"] == "ValueError"


def test_add_cv_observes_rag_index_metrics():
    from outils.rag_utils import VectorStoreManager

    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager.collection_name = "cv_collection"
    manager.embedding_model_name = "BAAI/bge-m3"
    manager.document_table = "rag_vector_documents"
    manager._encode_documents = MagicMock(return_value=[[0.1, 0.2]])

    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    conn.__enter__.return_value = conn
    manager._connect = MagicMock(return_value=conn)

    cv = CVData(
        identite={"nom": "Doe", "prenom": "Jane"},
        metadata={"source_fichier": "cv.pdf"},
        competences=[{"nom": "Python"}],
    )

    with patch("outils.rag_utils.observe_rag_index") as mock_index:
        manager.add_cv(cv, "cv.pdf", index_source="local")

    mock_index.assert_called_once()
    kwargs = mock_index.call_args.kwargs
    assert kwargs["collection"] == "cv_collection"
    assert kwargs["source"] == "local"
    assert kwargs["status"] == "success"
    assert kwargs["chunks_indexed"] == 1


def test_add_cv_observes_rag_index_errors():
    from outils.rag_utils import VectorStoreManager

    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager.collection_name = "cv_collection"
    manager.embedding_model_name = "BAAI/bge-m3"
    manager.document_table = "rag_vector_documents"
    manager._encode_documents = MagicMock(side_effect=RuntimeError("embed failed"))

    cv = CVData(
        identite={"nom": "Doe", "prenom": "Jane"},
        metadata={"source_fichier": "cv.pdf"},
    )

    with patch("outils.rag_utils.observe_rag_index") as mock_index:
        with pytest.raises(RuntimeError, match="embed failed"):
            manager.add_cv(cv, "cv.pdf", index_source="local")

    mock_index.assert_called_once()
    kwargs = mock_index.call_args.kwargs
    assert kwargs["status"] == "error"
    assert kwargs["source"] == "local"
