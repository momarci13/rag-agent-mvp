"""Tests for the RAG hybrid retriever.

These tests use deterministic synthetic vectors and never call a model.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import pytest

pytest.importorskip("chromadb")
from rag.embeddings import DeterministicHashEmbeddings
from rag.hybrid import LiteHybridRAG
from rag.ingest import chunk_text, ingest_path, parse_bibtex


@pytest.fixture
def tmp_rag():
    d = tempfile.mkdtemp(prefix="rag_test_")
    rag = LiteHybridRAG(
        db_path=d,
        collection="tmp",
        alpha_dense=0.6,
        embedding_client=DeterministicHashEmbeddings(),
    )
    rag.reset()
    yield rag
    shutil.rmtree(d, ignore_errors=True)


def test_chunk_text_respects_size():
    text = "Para one.\n\n" + ("Para two has many sentences. " * 50) + "\n\nPara three."
    chunks = chunk_text(text, target_tokens=64, overlap=8)
    assert len(chunks) >= 2
    # Rough check: no chunk grossly oversize
    for c in chunks:
        assert len(c) <= 64 * 4 + 100


def test_parse_bibtex_basic(tmp_path):
    bib = """
@article{smith2020,
  title = {Momentum and Value Together},
  author = {Smith, John and Doe, Jane},
  year = {2020},
  journal = {Journal of Finance},
  abstract = {We show that combining factors improves Sharpe.}
}
@book{lopez2018,
  title = {Advances in Financial ML},
  author = {Lopez de Prado, Marcos},
  year = {2018}
}
"""
    p = tmp_path / "refs.bib"
    p.write_text(bib)
    entries = parse_bibtex(p)
    assert len(entries) == 2
    keys = {e["meta"]["key"] for e in entries}
    assert keys == {"smith2020", "lopez2018"}


@pytest.mark.slow
def test_retrieval_finds_relevant_doc(tmp_rag):
    """Smoke test: clearly distinct docs, retrieval should rank correctly."""
    docs = [
        {"id": "fin1", "text": "The Sharpe ratio measures risk-adjusted returns of a portfolio.",
         "meta": {"kind": "doc"}},
        {"id": "fin2", "text": "Mean-variance optimization balances expected return against variance.",
         "meta": {"kind": "doc"}},
        {"id": "cook1", "text": "To make sourdough bread, mix flour, water, salt and starter.",
         "meta": {"kind": "doc"}},
        {"id": "cook2", "text": "Pasta carbonara uses eggs, pecorino, guanciale and black pepper.",
         "meta": {"kind": "doc"}},
    ]
    tmp_rag.add(docs)
    assert len(tmp_rag) == 4

    res = tmp_rag.retrieve("Sharpe portfolio return variance", k=2)
    ids = [r["id"] for r in res]
    # At least one finance doc should appear in top 2
    assert any(i.startswith("fin") for i in ids)


def test_metadata_iter(tmp_rag):
    tmp_rag.add([
        {"id": "a", "text": "alpha", "meta": {"kind": "bib", "key": "alpha2020"}},
        {"id": "b", "text": "beta",  "meta": {"kind": "doc"}},
    ])
    metas = list(tmp_rag.iter_metadata())
    assert any(m.get("kind") == "bib" for m in metas)


def test_ingest_path_skips_duplicates(tmp_rag, tmp_path):
    file_path = tmp_path / "example.md"
    file_path.write_text("First paragraph.\n\nSecond paragraph about alpha and beta.\n")

    first_pass = ingest_path(
        file_path,
        tmp_rag,
        chunk_tokens=40,
        overlap=8,
        skip_existing=True,
    )
    assert first_pass["added"] > 0
    assert first_pass["skipped"] == 0

    second_pass = ingest_path(
        file_path,
        tmp_rag,
        chunk_tokens=40,
        overlap=8,
        skip_existing=True,
    )
    assert second_pass["added"] == 0
    assert second_pass["skipped"] == second_pass["total"]
    assert len(tmp_rag) == first_pass["added"]
