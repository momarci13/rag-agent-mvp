"""Tests for scholar integration: arXiv search and dynamic KB augmentation.

These tests mock the arXiv API to avoid network dependencies.
"""
from __future__ import annotations

import tempfile
import shutil
from unittest.mock import patch, MagicMock

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")
pytest.importorskip("rank_bm25")

from rag.hybrid import LiteHybridRAG
from rag.embeddings import DeterministicHashEmbeddings
from tools.scholar import scholar_augment_task, ArxivPaper


@pytest.fixture
def tmp_rag():
    """Temporary RAG instance for testing."""
    d = tempfile.mkdtemp(prefix="scholar_test_")
    rag = LiteHybridRAG(
        db_path=d,
        collection="scholar_test",
        alpha_dense=0.6,
        embedding_client=DeterministicHashEmbeddings(),
    )
    rag.reset()
    yield rag
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def mock_arxiv_response():
    """Mock arXiv API response XML."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2401.12345</id>
    <title>Momentum Strategies in Quantitative Finance</title>
    <author><name>Smith, John</name></author>
    <author><name>Doe, Jane</name></author>
    <published>2024-01-15T10:00:00Z</published>
    <summary>This paper examines momentum strategies and their effectiveness in modern markets.</summary>
  </entry>
  <entry>
    <id>https://arxiv.org/abs/2402.67890</id>
    <title>Risk Management in Algorithmic Trading</title>
    <author><name>Johnson, Bob</name></author>
    <published>2024-02-20T12:00:00Z</published>
    <summary>We propose new methods for risk management in high-frequency trading systems.</summary>
  </entry>
</feed>"""


def test_scholar_augment_task_with_mock(tmp_rag, mock_arxiv_response):
    """Test full scholar augmentation pipeline with mocked API."""
    task_description = "Implement a momentum trading strategy with risk management"

    # Mock the requests.get call
    with patch('tools.scholar.requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.content = mock_arxiv_response.encode('utf-8')
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Run scholar augmentation
        papers, context = scholar_augment_task(
            task_description=task_description,
            n_papers=2,
            category="q-fin"
        )

        # Verify papers were found
        assert len(papers) == 2
        assert isinstance(papers[0], ArxivPaper)
        assert "momentum" in papers[0].title.lower()
        assert papers[0].arxiv_id == "2401.12345"
        assert papers[0].authors == ["Smith, John", "Doe, Jane"]

        # Verify context string
        assert "Recently Retrieved Academic Papers" in context
        assert "momentum" in context.lower()

        # Test ingestion
        chunks_added = tmp_rag.ingest_papers(papers)
        assert chunks_added > 0  # Should add multiple chunks

        # Test retrieval includes scholar content
        results = tmp_rag.retrieve("momentum strategies", k=5)
        scholar_results = [r for r in results if r['meta'].get('source', '').startswith('arxiv:')]
        assert len(scholar_results) > 0

        # Verify metadata
        for result in scholar_results:
            assert result['meta']['kind'] == 'doc'
            assert 'arxiv:' in result['meta']['source']
            assert result['meta']['year'] is not None


def test_scholar_augment_task_no_results(tmp_rag):
    """Test scholar augmentation when no papers are found."""
    task_description = "invalid query that should return no results"

    with patch('tools.scholar.requests.get') as mock_get:
        mock_response = MagicMock()
        # Empty feed
        mock_response.content = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        papers, context = scholar_augment_task(task_description, n_papers=5)

        assert papers == []
        assert context == ""


def test_scholar_augment_task_api_error(tmp_rag):
    """Test scholar augmentation handles API errors gracefully."""
    task_description = "momentum strategy"

    with patch('tools.scholar.requests.get') as mock_get:
        mock_get.side_effect = Exception("Network error")

        papers, context = scholar_augment_task(task_description, n_papers=5)

        assert papers == []
        assert context == ""


def test_arxiv_paper_to_markdown():
    """Test ArxivPaper markdown formatting."""
    paper = ArxivPaper(
        arxiv_id="2401.12345",
        title="Test Paper Title",
        authors=["Author One", "Author Two", "Author Three", "Author Four"],
        published="2024-01-15",
        summary="This is a test abstract.",
        url="https://arxiv.org/abs/2401.12345"
    )

    markdown = paper.to_markdown()

    assert "# Test Paper Title" in markdown
    assert "**Authors:** Author One, Author Two, et al." in markdown
    assert "**Published:** 2024-01-15" in markdown
    assert "This is a test abstract." in markdown
    assert "*Source: arXiv. Retrieved dynamically*" in markdown


def test_search_arxiv_accepts_multiple_categories(monkeypatch, mock_arxiv_response):
    """Test search_arxiv with multiple math/stat categories."""
    called = {}

    def fake_retry(url, params, max_retries=3, backoff=1.0):
        called['params'] = params
        mock_response = MagicMock()
        mock_response.content = mock_arxiv_response.encode('utf-8')
        return mock_response

    monkeypatch.setattr('tools.scholar._retry_request', fake_retry)

    from tools.scholar import search_arxiv
    papers = search_arxiv(
        query="bayesian inference",
        n=1,
        category=["math.PR", "stat.AP"],
    )

    assert len(papers) == 1
    assert "bayesian" in called['params']['search_query']
    assert "cat:math.PR" in called['params']['search_query']
    assert "cat:stat.AP" in called['params']['search_query']


def test_arxiv_paper_to_bibtex():
    """Test ArxivPaper BibTeX formatting."""
    paper = ArxivPaper(
        arxiv_id="2401.12345",
        title="Test Paper Title",
        authors=["Author One", "Author Two"],
        published="2024-01-15",
        summary="This is a test abstract.",
        url="https://arxiv.org/abs/2401.12345"
    )

    bibtex = paper.to_bibtex()

    assert "@article{" in bibtex
    assert "author = {Author One and Author Two}" in bibtex
    assert "title = {Test Paper Title}" in bibtex
    assert "year = {2024}" in bibtex
    assert "eprint = {2401.12345}" in bibtex
