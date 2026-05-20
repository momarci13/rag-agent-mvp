"""Lite hybrid RAG: dense (bge-small) + BM25, weighted fusion.

Deliberately no cross-encoder reranker — it costs another 300 MB VRAM
we don't have. For <100k chunks, fusion alone is ~95% as good.

Math:
  s_dense(q,d) = cos(phi(q), phi(d))
  s_bm25(q,d)  = sum_t IDF(t) * tf(t,d)(k1+1) / (tf(t,d) + k1(1-b+b|d|/avgL))
  s(q,d)       = alpha * s_dense + (1-alpha) * s_bm25_norm
"""
from __future__ import annotations

import logging
import os
import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from agents.llm import OllamaLLM

logger = logging.getLogger(__name__)

import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
except Exception as e:  # pragma: no cover
    raise ImportError(
        "chromadb is required. Install with: pip install chromadb"
    ) from e

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    raise ImportError(
        "sentence-transformers is required. Install: pip install sentence-transformers"
    ) from e

try:
    from rank_bm25 import BM25Okapi
except Exception as e:  # pragma: no cover
    raise ImportError("rank-bm25 is required. Install: pip install rank-bm25") from e

try:
    from functools import lru_cache
except ImportError:
    # Python < 3.2
    def lru_cache(maxsize=128):
        def decorator(func):
            return func
        return decorator

try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
except Exception as e:  # pragma: no cover
    raise ImportError("nltk is required. Install: pip install nltk") from e


# -------------- tokenization & stemming ----------------

try:
    import tiktoken
    _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _tiktoken_enc = None

_stemmer = PorterStemmer()
_stopwords = set(stopwords.words('english'))


def _tokenize_stem(text: str) -> list[str]:
    """Tokenize, stem, and remove stopwords."""
    tokens = text.lower().split()
    stemmed = [_stemmer.stem(t) for t in tokens]
    return [t for t in stemmed if t not in _stopwords and len(t) > 1]


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken (more accurate than ÷4 heuristic)."""
    if _tiktoken_enc is not None:
        try:
            return len(_tiktoken_enc.encode(text))
        except Exception:
            pass
    # Fallback to rough estimate if tiktoken is unavailable or encoding fails
    return max(1, len(text) // 4)


class LiteHybridRAG:
    def __init__(
        self,
        db_path: str = "./kb/chroma",
        collection: str = "main",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        alpha_dense: float = 0.6,
        query_expansion_enabled: bool = False,
        query_expansion_method: str = "local_model",
        max_expansions: int = 3,
        reranking_enabled: bool = False,
        reranking_model: str = "local_cross_encoder",
        top_k_before_rerank: int = 50,
        top_k_after_rerank: int = 6,
    ):
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.collection_name = collection
        self.alpha = alpha_dense

        # New RAG enhancement options
        self.query_expansion_enabled = query_expansion_enabled
        self.query_expansion_method = query_expansion_method
        self.max_expansions = max_expansions
        self.reranking_enabled = reranking_enabled
        self.reranking_model = reranking_model
        self.top_k_before_rerank = top_k_before_rerank
        self.top_k_after_rerank = top_k_after_rerank

        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.col = self.client.get_or_create_collection(collection)

        self.emb = SentenceTransformer(embedding_model)

        # Load reranking model if enabled
        self.reranker = None
        if self.reranking_enabled:
            self._load_reranker()

        # BM25 is rebuilt from chroma on load, cached to disk.
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._bm25_path = Path(db_path) / f"{collection}_bm25.pkl"
        
        # Query result cache for performance
        self._query_cache: dict[str, list[dict]] = {}
        self._cache_max_size = 100
        
        self._load_bm25_cache()

    def _load_reranker(self):
        """Load the reranking model."""
        try:
            if self.reranking_model == "local_cross_encoder":
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            # TODO: Add Cohere API support
            elif self.reranking_model.startswith("cohere"):
                # Placeholder for Cohere integration
                self.reranker = None
            else:
                print(f"Unknown reranking model: {self.reranking_model}")
                self.reranker = None
        except Exception as e:
            logger.warning("Failed to load reranker: %s", e)
            self.reranker = None

    def _expand_query(self, query: str, llm: "OllamaLLM | None" = None) -> list[str]:
        """Expand the query with related terms, optionally via LLM."""
        if not self.query_expansion_enabled:
            return [query]

        expansions = [query]
        method = self.query_expansion_method

        try:
            # LLM-based expansion (preferred when available)
            if llm is not None and "llm" in method:
                from rag.query_expansion import expand_query_with_llm
                variants = expand_query_with_llm(query, llm, n_variants=self.max_expansions)
                expansions.extend(variants)

            # Rule-based expansion as supplement / fallback
            if len(expansions) < self.max_expansions + 1:
                expansions.extend(self._rule_based_expansion(query))
        except Exception as exc:
            logger.warning("Query expansion failed: %s", exc)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for e in expansions:
            if e not in seen:
                seen.add(e)
                unique.append(e)

        return unique[:self.max_expansions + 1]

    def _rule_based_expansion(self, query: str) -> list[str]:
        """Simple rule-based query expansion."""
        # Basic synonym expansion
        synonyms = {
            "stock": ["equity", "share", "security"],
            "return": ["performance", "yield", "profit"],
            "volatility": ["risk", "variance", "fluctuation"],
            "strategy": ["approach", "method", "tactic"],
            "market": ["exchange", "trading", "financial"],
        }

        expanded = []
        words = query.lower().split()
        for word in words:
            if word in synonyms:
                for syn in synonyms[word][:2]:  # Limit to 2 synonyms per word
                    new_query = query.lower().replace(word, syn)
                    expanded.append(new_query)

        return expanded

    def _rerank_results(self, query: str, candidates: list[dict]) -> list[dict]:
        """Re-rank candidates using cross-encoder."""
        if not self.reranking_enabled or self.reranker is None or len(candidates) <= self.top_k_after_rerank:
            return candidates

        try:
            # Prepare query-doc pairs
            query_doc_pairs = [(query, doc["text"]) for doc in candidates[:self.top_k_before_rerank]]

            # Get reranker scores
            scores = self.reranker.predict(query_doc_pairs)

            # Add scores to candidates
            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = float(score)

            # Sort by rerank score
            candidates.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)

            return candidates[:self.top_k_after_rerank]

        except Exception as e:
            logger.warning("Reranking failed: %s", e)
            return candidates

    # ---------------- ingestion ----------------

    def add(self, docs: list[dict]) -> None:
        """docs: [{"id": str, "text": str, "meta": dict}]"""
        if not docs:
            return
        texts = [d["text"] for d in docs]
        vecs = self.emb.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
        ids = [d["id"] for d in docs]
        metas = [d.get("meta", {}) or {} for d in docs]

        self.col.add(ids=ids, embeddings=vecs, documents=texts, metadatas=metas)

        self._ids.extend(ids)
        self._texts.extend(texts)
        self._metas.extend(metas)
        self._rebuild_bm25()
        
        # Clear query cache since KB changed
        self._query_cache.clear()

    def ingest_papers(self, papers: list) -> int:
        """Ingest arXiv papers into the KB.
        
        Args:
            papers: List of ArxivPaper objects
            
        Returns:
            Number of chunks added
        """
        if not papers:
            return 0
            
        total_chunks = 0
        for paper in papers:
            try:
                content = paper.to_markdown()
                chunks = self._chunk_text(content, target_tokens=256, overlap=32)
                docs = []
                for i, chunk in enumerate(chunks):
                    chunk_id = f"arxiv::{paper.arxiv_id}::{i}"
                    docs.append({
                        "id": chunk_id,
                        "text": chunk,
                        "meta": {
                            "kind": "doc",
                            "source": f"arxiv:{paper.arxiv_id}",
                            "chunk": i,
                            "title": paper.title,
                            "authors": ", ".join(paper.authors),
                            "year": paper.published.split("-")[0] if paper.published != "unknown" else None,
                            "url": paper.url,
                        },
                    })
                self.add(docs)
                total_chunks += len(docs)
            except Exception as e:
                logger.warning("[RAG] Skipping paper %s: %s", getattr(paper, 'arxiv_id', '?'), e)
                continue

        return total_chunks

    def _chunk_text(self, text: str, target_tokens: int = 256, overlap: int = 32) -> list[str]:
        """Token-aware chunker using tiktoken for accurate token counting."""
        paragraphs = re.split(r"\n\s*\n", text.strip())
        chunks: list[str] = []
        buf = ""
        overlap_budget = overlap  # tokens to preserve for overlap

        for p in paragraphs:
            if not p.strip():
                continue

            # Check if adding this paragraph to buffer would exceed target
            if buf:
                candidate = buf + "\n\n" + p
            else:
                candidate = p
            
            tokens_candidate = _count_tokens(candidate)

            if tokens_candidate <= target_tokens:
                # Fits within budget, add to buffer
                buf = candidate
            else:
                # Would exceed budget
                if buf:
                    chunks.append(buf)
                
                # Check if paragraph alone fits
                tokens_p = _count_tokens(p)
                if tokens_p > target_tokens:
                    # Paragraph too long; split on sentences
                    sentences = re.split(r'(?<=[.!?])\s+', p)
                    sub_buf = ""
                    for sent in sentences:
                        cand = (sub_buf + " " + sent).strip() if sub_buf else sent
                        if _count_tokens(cand) <= target_tokens:
                            sub_buf = cand
                        else:
                            if sub_buf:
                                chunks.append(sub_buf)
                            sub_buf = sent
                    if sub_buf:
                        chunks.append(sub_buf)
                    buf = ""
                else:
                    # Paragraph fits alone; prepare overlap
                    # Take tail of previous buffer for overlap context
                    words = buf.split()
                    overlap_words = []
                    overlap_tokens_current = 0
                    for word in reversed(words):
                        test_seq = " ".join([word] + overlap_words)
                        if _count_tokens(test_seq) <= overlap_budget:
                            overlap_words.insert(0, word)
                            overlap_tokens_current = _count_tokens(test_seq)
                        else:
                            break
                    
                    if overlap_words:
                        buf = " ".join(overlap_words) + "\n\n" + p
                    else:
                        buf = p

        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]

    def _rebuild_bm25(self) -> None:
        tokens = [_tokenize_stem(t) for t in self._texts]
        self._bm25 = BM25Okapi(tokens) if tokens else None
        self._save_bm25_cache()

    def _save_bm25_cache(self) -> None:
        with open(self._bm25_path, "wb") as f:
            pickle.dump(
                {"ids": self._ids, "texts": self._texts, "metas": self._metas}, f,
            )

    def _load_bm25_cache(self) -> None:
        if self._bm25_path.exists():
            with open(self._bm25_path, "rb") as f:
                cache = pickle.load(f)
            self._ids = cache.get("ids", [])
            self._texts = cache.get("texts", [])
            self._metas = cache.get("metas", [])
            if self._texts:
                self._bm25 = BM25Okapi([_tokenize_stem(t) for t in self._texts])

    # ---------------- retrieval ----------------

    def retrieve(
        self,
        query: str,
        k: int = 6,
        m: int = 30,
        token_budget: int = 3500,
        metadata_filters: dict | None = None,
        llm: "OllamaLLM | None" = None,
    ) -> list[dict]:
        """Retrieve top-k documents using hybrid dense+BM25 retrieval with optional query expansion and reranking.
        
        Args:
            query: search query
            k: max documents to return
            m: candidates before fusion
            token_budget: max tokens for context packing
            metadata_filters: dict with optional keys:
                - kind: "doc" or "bib"
                - year_min, year_max: year range (as strings)
                - source: filename filter
        
        Returns:
            list of {"id", "text", "meta", "score"} dicts
        """
        if not self._ids:
            return []

        # Query expansion (LLM-assisted when llm is provided)
        expanded_queries = self._expand_query(query, llm=llm)
        all_candidates = []

        for exp_query in expanded_queries:
            # Create cache key
            cache_key = f"{exp_query}|{k}|{m}|{token_budget}|{str(metadata_filters)}"
            if cache_key in self._query_cache:
                candidates = self._query_cache[cache_key]
            else:
                candidates = self._retrieve_single_query(exp_query, k, m, token_budget, metadata_filters)
                # Cache result
                if len(self._query_cache) >= self._cache_max_size:
                    oldest_key = next(iter(self._query_cache))
                    del self._query_cache[oldest_key]
                self._query_cache[cache_key] = candidates

            all_candidates.extend(candidates)

        # Remove duplicates and sort by score
        seen_ids = set()
        unique_candidates = []
        for cand in all_candidates:
            if cand["id"] not in seen_ids:
                unique_candidates.append(cand)
                seen_ids.add(cand["id"])

        unique_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Apply reranking
        final_candidates = self._rerank_results(query, unique_candidates)

        return final_candidates[:k]

    def _retrieve_single_query(
        self,
        query: str,
        k: int,
        m: int,
        token_budget: int,
        metadata_filters: dict | None = None,
    ) -> list[dict]:
        """Core retrieval logic for a single query."""
        m = min(m, len(self._ids))

        # Apply metadata filtering
        valid_indices = set(range(len(self._ids)))
        if metadata_filters:
            if metadata_filters.get("kind"):
                kind = metadata_filters["kind"]
                valid_indices &= {
                    i for i in valid_indices
                    if i < len(self._metas) and self._metas[i].get("kind") == kind
                }
            if "year_min" in metadata_filters or "year_max" in metadata_filters:
                year_min = metadata_filters.get("year_min")
                year_max = metadata_filters.get("year_max")
                def in_year_range(y):
                    if not y:
                        return False
                    try:
                        y_int = int(y)
                        if year_min and y_int < int(year_min):
                            return False
                        if year_max and y_int > int(year_max):
                            return False
                        return True
                    except (ValueError, TypeError):
                        return False
                valid_indices &= {
                    i for i in valid_indices
                    if i < len(self._metas) and in_year_range(self._metas[i].get("year"))
                }
            if metadata_filters.get("source"):
                source = metadata_filters["source"]
                valid_indices &= {
                    i for i in valid_indices
                    if i < len(self._metas) and self._metas[i].get("source") == source
                }

        if not valid_indices:
            return []

        # Dense retrieval: get top-m candidates
        qv = self.emb.encode([query], normalize_embeddings=True).tolist()
        dres = self.col.query(query_embeddings=qv, n_results=m)
        # chroma returns distances in cosine distance space (1 - cos)
        d_ids = dres["ids"][0]
        d_dist = dres["distances"][0]
        
        # Filter by metadata and build dense scores
        dense_scores = {}
        for i, dist in zip(d_ids, d_dist):
            idx = self._ids.index(i)
            if idx in valid_indices:
                dense_scores[i] = max(0.0, 1.0 - float(dist))

        # BM25
        bm25_scores: dict[str, float] = {}
        if self._bm25 is not None:
            query_tokens = _tokenize_stem(query)
            if query_tokens:
                raw = np.asarray(self._bm25.get_scores(query_tokens), dtype=float)
                if raw.size:
                    mn, mx = float(raw.min()), float(raw.max())
                    norm = (raw - mn) / (mx - mn + 1e-9)
                    for idx in valid_indices:
                        if idx < len(self._ids):
                            bm25_scores[self._ids[idx]] = float(norm[idx])

        # Fuse
        keys = set(dense_scores) | set(bm25_scores)
        fused = {
            i: self.alpha * dense_scores.get(i, 0.0)
            + (1 - self.alpha) * bm25_scores.get(i, 0.0)
            for i in keys
        }
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        # Greedy knapsack pack under token budget
        id_to_idx = {i: j for j, i in enumerate(self._ids)}
        selected: list[dict] = []
        used = 0
        for doc_id, score in ordered:
            if len(selected) >= k:
                break
            idx = id_to_idx.get(doc_id)
            if idx is None:
                continue
            text = self._texts[idx]
            approx_tokens = _count_tokens(text)  # Use tiktoken instead of ÷4
            if used + approx_tokens > token_budget and selected:
                continue
            selected.append({
                "id": doc_id,
                "text": text,
                "meta": self._metas[idx] if idx < len(self._metas) else {},
                "score": score,
            })
            used += approx_tokens
        
        return selected

    # ---------------- introspection ----------------

    def iter_metadata(self) -> Iterator[dict]:
        yield from self._metas

    def __len__(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        """Nuke the collection and cache. Useful for tests."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.col = self.client.get_or_create_collection(self.collection_name)
        self._ids.clear()
        self._texts.clear()
        self._metas.clear()
        self._bm25 = None
        if self._bm25_path.exists():
            os.remove(self._bm25_path)
        
        # Clear query cache
        self._query_cache.clear()
