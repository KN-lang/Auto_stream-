"""tests/test_rag_service.py — unit tests for RAGService."""

import pytest
from unittest.mock import MagicMock, patch


def _make_svc_with_vectorstore(mock_docs):
    """
    Build a RAGService instance by bypassing __init__ entirely and
    injecting a mock vectorstore directly onto the instance.
    This avoids patching os.path/listdir which conflicts with Python
    import machinery under pytest.
    """
    from services.rag_service import RAGService

    svc = object.__new__(RAGService)
    svc._settings = MagicMock(retrieval_k=3, chroma_persist_dir="/fake/dir")
    svc._embeddings = MagicMock()

    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = mock_docs
    svc._vectorstore = mock_vs

    return svc


class TestRAGService:

    def test_retrieve_returns_string(self):
        """retrieve() must return a non-empty string when documents are found."""
        doc = MagicMock()
        doc.page_content = "Basic Plan costs $29/month."
        svc = _make_svc_with_vectorstore([doc])

        result = svc.retrieve("How much is the Basic plan?")
        assert isinstance(result, str)
        assert "29" in result

    def test_retrieve_empty_returns_fallback(self):
        """retrieve() must return a graceful fallback when no docs match."""
        svc = _make_svc_with_vectorstore([])

        result = svc.retrieve("xyzzy not in kb")
        assert "No relevant" in result

    def test_retrieve_joins_chunks_with_separator(self):
        """Multiple chunks must be joined by the markdown separator."""
        doc1 = MagicMock(); doc1.page_content = "Chunk A"
        doc2 = MagicMock(); doc2.page_content = "Chunk B"
        svc = _make_svc_with_vectorstore([doc1, doc2])

        result = svc.retrieve("pricing")
        assert "---" in result
        assert "Chunk A" in result
        assert "Chunk B" in result

    def test_retrieve_calls_vectorstore_with_correct_k(self):
        """retrieve() must pass retrieval_k to similarity_search."""
        doc = MagicMock(); doc.page_content = "something"
        svc = _make_svc_with_vectorstore([doc])
        svc._settings.retrieval_k = 5

        svc.retrieve("query")
        svc._vectorstore.similarity_search.assert_called_once_with("query", k=5)
