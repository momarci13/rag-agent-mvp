"""Quick tests for Phase 1 implementations."""
import sys
import tempfile
import shutil
from pathlib import Path

# Test 1: BM25 preprocessing with stemming and stopwords
print("\n" + "="*60)
print("TEST 1: BM25 Preprocessing (Stemming + Stopwords)")
print("="*60)
try:
    from rag.hybrid import _tokenize_stem
    
    # Test stemming
    result = _tokenize_stem("running runs ran runner")
    print(f"✓ Stemming works: 'running runs ran runner' → {result}")
    assert "run" in result, "Stemming should reduce to 'run'"
    
    # Test stopword removal
    result = _tokenize_stem("the quick brown fox and the lazy dog")
    print(f"✓ Stopwords removed: 'the quick brown fox and the lazy dog' → {result}")
    assert "the" not in result, "Stopwords should be removed"
    assert "brown" in result, "Content words should remain"
    
    print("✓ TEST 1 PASSED")
except Exception as e:
    print(f"✗ TEST 1 FAILED: {e}")
    sys.exit(1)

# Test 2: Accurate token counting with tiktoken
print("\n" + "="*60)
print("TEST 2: Accurate Token Counting (tiktoken)")
print("="*60)
try:
    from rag.ingest import _count_tokens
    
    text1 = "Hello world"
    tokens1 = _count_tokens(text1)
    print(f"✓ Token count for '{text1}': {tokens1} tokens")
    assert tokens1 > 0, "Token count should be positive"
    
    text2 = "Hello world. This is a longer sentence with more words."
    tokens2 = _count_tokens(text2)
    print(f"✓ Token count for longer text: {tokens2} tokens")
    assert tokens2 > tokens1, "Longer text should have more tokens"
    
    print("✓ TEST 2 PASSED")
except Exception as e:
    print(f"✗ TEST 2 FAILED: {e}")
    sys.exit(1)

# Test 3: Metadata filtering in retrieval
print("\n" + "="*60)
print("TEST 3: Metadata Filtering")
print("="*60)
try:
    from rag.hybrid import LiteHybridRAG
    from rag.embeddings import DeterministicHashEmbeddings
    
    # Create temporary RAG instance
    tmpdir = tempfile.mkdtemp(prefix="test_phase1_")
    try:
        rag = LiteHybridRAG(
            db_path=tmpdir,
            collection="test",
            embedding_client=DeterministicHashEmbeddings(),
        )
        
        # Add test documents with metadata
        docs = [
            {
                "id": "doc1",
                "text": "Momentum strategy from 2020",
                "meta": {"kind": "doc", "source": "paper1.pdf", "year": "2020"}
            },
            {
                "id": "doc2",
                "text": "Value strategy from 2022",
                "meta": {"kind": "doc", "source": "paper2.pdf", "year": "2022"}
            },
            {
                "id": "bib1",
                "text": "Smith 2019 momentum paper",
                "meta": {"kind": "bib", "key": "smith2019", "year": "2019"}
            },
        ]
        rag.add(docs)
        
        # Test basic retrieval
        results = rag.retrieve("momentum strategy", k=3)
        print(f"✓ Basic retrieval: found {len(results)} documents")
        
        # Test metadata filtering by kind
        results_doc = rag.retrieve(
            "strategy",
            k=3,
            metadata_filters={"kind": "doc"}
        )
        print(f"✓ Filter by kind='doc': found {len(results_doc)} documents")
        assert all(d["meta"]["kind"] == "doc" for d in results_doc), "Should only return docs"
        
        # Test metadata filtering by year
        results_recent = rag.retrieve(
            "strategy",
            k=3,
            metadata_filters={"year_min": "2021"}
        )
        print(f"✓ Filter by year >= 2021: found {len(results_recent)} documents")
        for d in results_recent:
            year = d["meta"].get("year")
            if year:
                assert int(year) >= 2021, f"Year {year} should be >= 2021"
        
        # Test metadata filtering by source
        results_source = rag.retrieve(
            "strategy",
            k=3,
            metadata_filters={"source": "paper2.pdf"}
        )
        print(f"✓ Filter by source='paper2.pdf': found {len(results_source)} documents")
        assert all(d["meta"].get("source") == "paper2.pdf" for d in results_source), "Should only return from paper2.pdf"
        
        print("✓ TEST 3 PASSED")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        
except Exception as e:
    print(f"✗ TEST 3 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Chunk text with accurate token counting
print("\n" + "="*60)
print("TEST 4: Semantic Chunking with Token Awareness")
print("="*60)
try:
    from rag.ingest import chunk_text
    
    text = "First paragraph about momentum strategy.\n\n" + \
           "Second paragraph about value investing. " * 20 + \
           "\n\nThird paragraph about risk management."
    
    chunks = chunk_text(text, target_tokens=64, overlap=8)
    print(f"✓ Chunked text into {len(chunks)} chunks (target: 64 tokens, overlap: 8)")
    
    for i, chunk in enumerate(chunks):
        from rag.ingest import _count_tokens
        tokens = _count_tokens(chunk)
        print(f"  Chunk {i+1}: {tokens} tokens, {len(chunk)} chars")
        assert tokens <= 64 * 1.2, f"Chunk {i} exceeds soft limit"
    
    print("✓ TEST 4 PASSED")
except Exception as e:
    print(f"✗ TEST 4 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Retrieval metrics logging
print("\n" + "="*60)
print("TEST 5: Retrieval Metrics Logging")
print("="*60)
try:
    from rag.metrics import RetrievalMetrics
    import json
    
    tmpdir = tempfile.mkdtemp(prefix="test_metrics_")
    try:
        metrics = RetrievalMetrics(output_dir=tmpdir)
        
        # Log a query
        retrieved = [
            {"id": "doc1", "score": 0.95},
            {"id": "doc2", "score": 0.87},
        ]
        relevant = {"doc1", "doc2", "doc3"}
        
        result = metrics.log_query(
            "test query",
            retrieved,
            relevant,
            query_type="test"
        )
        print(f"✓ Logged query with metrics")
        print(f"  Precision: {result.get('precision', 0):.2f}")
        print(f"  Recall: {result.get('recall', 0):.2f}")
        
        # Save metrics
        path = metrics.save("test_metrics.json")
        print(f"✓ Saved metrics to {path}")
        
        # Verify file exists and is valid JSON
        with open(path) as f:
            data = json.load(f)
        assert "queries" in data, "Saved JSON should have 'queries' key"
        print(f"✓ Metrics file contains {len(data['queries'])} queries")
        
        print("✓ TEST 5 PASSED")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        
except Exception as e:
    print(f"✗ TEST 5 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("PHASE 1 QUICK TESTS: ALL PASSED ✓")
print("="*60)
print("\nImplemented features:")
print("  ✓ BM25 preprocessing (stemming + stopwords)")
print("  ✓ Accurate token counting (tiktoken)")
print("  ✓ Metadata filtering (kind, year, source)")
print("  ✓ Token-aware semantic chunking")
print("  ✓ Retrieval metrics logging")
