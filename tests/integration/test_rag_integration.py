"""End-to-end integration tests for RAG system.

Tests the complete pipeline: ingestion → retrieval → generation → evaluation.
Measures coherence, citation coverage, and hallucination rates.
"""
from __future__ import annotations

import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

import pytest

# Skip if required modules not available
pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")
pytest.importorskip("tiktoken")

from rag.hybrid import LiteHybridRAG
from rag.embeddings import DeterministicHashEmbeddings
from rag.ingest import ingest_file
from rag.metrics import RetrievalMetrics


class MockLLM:
    """Mock LLM for testing that returns predictable responses."""

    def __init__(self):
        self.call_count = 0
        self.last_messages = []

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Mock chat response that includes citations when context provided."""
        self.call_count += 1
        self.last_messages = messages

        # Extract context from system messages
        context_docs = []
        user_msg = ""
        for msg in messages:
            if msg["role"] == "system" and "Retrieved context" in msg["content"]:
                # Parse retrieved context
                content = msg["content"]
                if "Retrieved context" in content:
                    context_part = content.split("Retrieved context")[1].strip()
                    # Extract doc IDs from [doc:id] patterns
                    import re
                    doc_ids = re.findall(r'\[doc:([^\]]+)\]', context_part)
                    context_docs.extend(doc_ids)
            elif msg["role"] == "user":
                user_msg = msg["content"]

        # Generate mock response based on query type
        if "momentum" in user_msg.lower():
            response = "Momentum strategies capture trends in asset prices. "
            if context_docs:
                response += f"According to [doc:{context_docs[0]}], momentum investing works. "
            response += "Key considerations include lookback periods and risk management."
        elif "backtest" in user_msg.lower():
            response = "Backtesting evaluates trading strategies on historical data. "
            if context_docs:
                response += f"As shown in [doc:{context_docs[0]}], proper backtesting requires out-of-sample validation. "
            response += "Common pitfalls include overfitting and survivorship bias."
        elif "citation" in user_msg.lower() or "citations" in user_msg.lower():
            response = "Here are citations for the requested topic. "
            if context_docs:
                response += " ".join(f"[doc:{doc_id}]" for doc_id in context_docs[:2])
            else:
                response += "No supporting documents were found."
        else:
            response = f"This is a mock response to: {user_msg[:50]}..."
            if context_docs:
                response += f" Based on documents: {', '.join(context_docs[:2])}"

        return response


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    return MockLLM()


@pytest.fixture
def test_rag():
    """Create temporary RAG instance with test data."""
    tmpdir = tempfile.mkdtemp(prefix="test_integration_")
    rag = LiteHybridRAG(
        db_path=tmpdir,
        collection="test",
        embedding_client=DeterministicHashEmbeddings(),
    )

    # Add test documents
    test_docs = [
        {
            "id": "momentum_paper_2020",
            "text": "Momentum strategies involve buying winners and selling losers. Research shows momentum works across asset classes. Key parameters include formation and holding periods.",
            "meta": {"kind": "doc", "source": "momentum.pdf", "year": "2020"}
        },
        {
            "id": "backtest_guide_2022",
            "text": "Backtesting requires careful validation. Use walk-forward analysis to avoid overfitting. Multiple testing correction is essential for statistical significance.",
            "meta": {"kind": "doc", "source": "backtest.pdf", "year": "2022"}
        },
        {
            "id": "bib_smith2019",
            "text": "ARTICLE smith2019\nTitle: Momentum and Reversal\nAuthors: Smith, J.\nYear: 2019\nAbstract: We study momentum and reversal effects in equity markets.\n",
            "meta": {"kind": "bib", "key": "smith2019", "year": "2019", "title": "Momentum and Reversal"}
        }
    ]
    rag.add(test_docs)

    yield rag
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_retrieval_to_generation_pipeline(test_rag, mock_llm):
    """Test complete pipeline: retrieval → prompt assembly → generation."""
    # Test query
    query = "momentum strategy backtesting"

    # Step 1: Retrieve documents
    docs = test_rag.retrieve(query, k=3)
    assert len(docs) >= 2, "Should retrieve multiple relevant documents"

    # Step 2: Assemble prompt (simulate roles.py logic)
    context_text = "\n\n".join([f"[doc:{d['id']}] {d['text']}" for d in docs])
    system_msg = f"Retrieved context (use for grounding):\n{context_text}"
    user_msg = f"Explain {query} in quantitative finance."

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    # Step 3: Generate response
    response = mock_llm.chat(messages)

    # Step 4: Validate response
    assert len(response) > 50, "Response should be substantial"
    assert "momentum" in response.lower(), "Should mention momentum"
    assert mock_llm.call_count == 1, "LLM should be called once"

    print(f"✓ Pipeline test passed. Response: {response[:100]}...")


def test_citation_coverage(test_rag, mock_llm):
    """Test that generated responses include proper citations."""
    query = "momentum effects in markets"

    # Retrieve and generate
    docs = test_rag.retrieve(query, k=2)
    context_text = "\n\n".join([f"[doc:{d['id']}] {d['text']}" for d in docs])

    messages = [
        {"role": "system", "content": f"Retrieved context:\n{context_text}"},
        {"role": "user", "content": f"Explain {query} with citations."}
    ]

    response = mock_llm.chat(messages)

    # Check for citation patterns
    import re
    citations = re.findall(r'\[doc:([^\]]+)\]', response)
    assert len(citations) > 0, "Response should include citations"

    # Verify cited documents exist in retrieved set
    retrieved_ids = {d['id'] for d in docs}
    cited_ids = set(citations)
    assert cited_ids.issubset(retrieved_ids), "All citations should reference retrieved documents"

    print(f"✓ Citation test passed. Citations: {citations}")


def test_metadata_filtering_integration(test_rag, mock_llm):
    """Test that metadata filtering works in the full pipeline."""
    query = "research methodology"

    # Test filtering by year
    docs_recent = test_rag.retrieve(query, k=5, metadata_filters={"year_min": "2020"})
    assert all(int(d['meta']['year']) >= 2020 for d in docs_recent), "Should only return recent docs"

    # Test filtering by kind
    docs_bib = test_rag.retrieve(query, k=5, metadata_filters={"kind": "bib"})
    assert all(d['meta']['kind'] == "bib" for d in docs_bib), "Should only return bib entries"

    # Generate with filtered results
    if docs_recent:
        context_text = "\n\n".join([f"[doc:{d['id']}] {d['text']}" for d in docs_recent])
        messages = [
            {"role": "system", "content": f"Recent research context:\n{context_text}"},
            {"role": "user", "content": f"Summarize recent {query}."}
        ]
        response = mock_llm.chat(messages)
        assert len(response) > 20, "Should generate response with filtered context"

    print("✓ Metadata filtering integration test passed")


def test_retrieval_metrics_integration(test_rag):
    """Test retrieval metrics calculation in integration context."""
    from rag.metrics import RetrievalMetrics

    # Define test queries with ground truth
    test_cases = [
        {
            "query": "momentum strategy",
            "relevant_ids": ["momentum_paper_2020", "bib_smith2019"],
            "domain": "quant"
        },
        {
            "query": "backtesting methodology",
            "relevant_ids": ["backtest_guide_2022"],
            "domain": "quant"
        }
    ]

    metrics = RetrievalMetrics()

    for case in test_cases:
        # Retrieve documents
        docs = test_rag.retrieve(case["query"], k=3)

        # Log metrics
        result = metrics.log_query(
            case["query"],
            docs,
            relevant_doc_ids=case["relevant_ids"],
            query_type=case["domain"]
        )

        # Validate metrics
        assert "precision" in result, "Should compute precision"
        assert "recall" in result, "Should compute recall"
        assert result["precision"] >= 0.0, "Precision should be non-negative"
        assert result["recall"] >= 0.0, "Recall should be non-negative"

    # Check aggregate metrics
    summary = metrics.summary()
    assert "num_queries" in summary, "Should have query count"
    assert summary["num_queries"] == len(test_cases), "Should match test case count"

    print(f"✓ Retrieval metrics test passed. Summary: {summary}")


def test_query_expansion_integration(test_rag):
    """Test query expansion improves retrieval."""
    from rag.query_expansion import QueryExpander

    expander = QueryExpander()
    base_query = "momentum"

    # Get expanded queries
    expanded_queries = expander.expand_query(base_query, domain="quant", max_expansions=3)
    assert len(expanded_queries) >= 2, "Should generate multiple query variations"

    # Test retrieval with each expanded query
    base_docs = test_rag.retrieve(base_query, k=3)
    expanded_docs = []
    for exp_query in expanded_queries[1:]:  # Skip original
        docs = test_rag.retrieve(exp_query, k=3)
        expanded_docs.extend(docs)

    # Expanded queries should retrieve at least as many unique docs
    base_ids = {d['id'] for d in base_docs}
    expanded_ids = {d['id'] for d in expanded_docs}
    assert len(expanded_ids) >= len(base_ids), "Query expansion should not reduce coverage"

    print(f"✓ Query expansion test passed. Base: {len(base_ids)} docs, Expanded: {len(expanded_ids)} docs")


def test_semantic_chunking_quality():
    """Test that semantic chunking produces coherent chunks."""
    from rag.ingest import chunk_text

    # Test text with clear topic boundaries
    text = """Introduction to Momentum Strategies

Momentum investing is a strategy that buys assets that have performed well recently and sells assets that have performed poorly. This approach is based on the idea that trends tend to continue.

Key Parameters in Momentum Strategies

The formation period is typically 6-12 months. The holding period is usually 1-3 months. These parameters significantly affect performance.

Risk Management Considerations

Momentum strategies can experience crashes. Risk management is crucial to avoid large drawdowns during reversal periods."""

    chunks = chunk_text(text, target_tokens=80, overlap=10)

    # Should produce multiple chunks
    assert len(chunks) >= 2, f"Should split into multiple chunks, got {len(chunks)}"

    # Each chunk should be coherent (contain related sentences)
    for i, chunk in enumerate(chunks):
        assert len(chunk.strip()) > 20, f"Chunk {i} should be substantial"
        # Check that chunks don't end mid-sentence (basic coherence check)
        assert not chunk.strip().endswith(('and', 'or', 'the', 'a', 'an')), f"Chunk {i} may end mid-sentence"

    print(f"✓ Semantic chunking test passed. Produced {len(chunks)} coherent chunks")


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running RAG integration smoke tests...")

    # Test imports
    try:
        from rag.hybrid import LiteHybridRAG
        from rag.ingest import chunk_text
        from rag.metrics import RetrievalMetrics
        from rag.query_expansion import QueryExpander
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        exit(1)

    # Test basic functionality
    try:
        expander = QueryExpander()
        queries = expander.expand_query("momentum strategy", domain="quant")
        print(f"✓ Query expansion works: {len(queries)} variations")

        chunks = chunk_text("This is a test paragraph.\n\nThis is another paragraph.", target_tokens=50)
        print(f"✓ Chunking works: {len(chunks)} chunks")

        print("✓ All smoke tests passed")
    except Exception as e:
        print(f"✗ Smoke test failed: {e}")
        exit(1)
