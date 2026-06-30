"""Spreading Activation: Biological memory retrieval engine.

Hybrid BM25 + Vector search with biological scoring:
  1. Anchor Strike: BM25 + Vector similarity, re-ranked with keyword boost
  2. Network Expansion: Energy flows through graph edges
  3. Biological Scoring: Similarity + Energy x Decay + Noise
  4. Wander Mechanic: Serendipity injection from the long tail

Zero LLM calls. Pure graph math and vector lookups.
"""

import hashlib
import logging
import math
import random
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

W_SIMILARITY = 0.30
W_ENERGY = 0.30
W_BM25 = 0.40
DECAY_LAMBDA = 0.1
NOISE_STD = 0.02

RANK_BM25 = 0.35
RANK_VECTOR = 0.25
RANK_KEYWORD = 0.20
RANK_RECENCY = 0.10
RANK_CENTRALITY = 0.10

BM25_K1 = 1.5
BM25_B = 0.75

INHIBITION_PENALTY = 0.5
BM25_CAP = 3.0

ANCHOR_COUNT = 2
WANDER_COUNT = 1
FINAL_CHUNKS = 4

ENERGY_WM = 0.8
ENERGY_GRAPH_NODE = 0.3

# Relevance metric weights (separate from ranking score — for interpretable quality measurement)
# Semantic and lexical overlap partially; lowered semantic to avoid double-counting.
REL_SEMANTIC = 0.30
REL_LEXICAL = 0.30
REL_TEMPORAL = 0.20
REL_GRAPH = 0.10
REL_SOURCE = 0.10

STOP_WORDS = frozenset({
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "this", "that", "these", "those", "is",
    "am", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "shall", "to", "of", "in", "for", "on",
    "with", "at", "by", "from", "as", "into", "about", "between",
    "through", "during", "before", "after", "above", "below", "up",
    "down", "out", "off", "over", "under", "again", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "just", "and", "but", "or", "if", "because", "while",
    "what", "which", "who", "whom", "its", "a", "an", "the", "also",
    "like",
})

# Query expansion mapping: common search terms → related terms
# Helps BM25 find chunks that use different vocabulary for the same concept
QUERY_EXPANSION = {
    "meta": "meta knowledge",
    "graph": "graph knowledge edges connections",
    "management": "management session tracking control scheduling",
    "session": "session conversation chat history",
    "memory": "memory recall remember store",
    "vector": "vector embedding chromadb store",
    "store": "store database vector chromadb",
    "debug": "debug error fix bug troubleshoot",
    "debugging": "debug error fix bug troubleshoot debug",
    "code": "code python script function",
    "cron": "cron schedule job task",
    "job": "job task cron schedule",
    "explore": "explore learn discover search",
    "exploration": "explore learn discover search",
    "identity": "identity profile who nora",
    "profile": "profile identity user system personality traits",
    "nora": "nora hermes agent ai",
    "hermes": "hermes nora agent system",
    "agent": "agent hermes nora ai",
    "system": "system hermes agent nora",
    "correction": "correction fix update correct reconcile",
    "correct": "correction fix update amend reconcile",
    "search": "search find retrieve lookup",
    "retrieve": "retrieve find search recall",
    "retrieval": "retrieve find search recall memory",
    "reconcile": "reconcile merge sync fix correction cleanup",
    "schedule": "schedule cron job timing plan",
    "temperature": "temperature temp heat hardware",
    "gpu": "gpu graphics hardware video",
    "update": "update change modify edit",
    "skill": "skill tool capability feature",
    "spreading": "spreading activation graph retrieval memory",
    "activation": "spreading activation graph retrieval memory",
    "wander": "wander serendipity random spontaneous",
    "anchor": "anchor strike direct match hit",
    "episodic": "episodic sequential temporal order sequence",
    "semantic": "semantic similar related concept knowledge",
    "emotion": "emotion mood feel sentiment tone",
    "emotional": "emotion mood feel sentiment tone",
    "content": "content type category kind format",
    "tool": "tool command action function capability",
    "personality": "personality identity profile traits hobby interest",
    "topic": "topic tag label category subject",
    "tags": "tag label topic metadata prefix",
    "keyword": "keyword tag label token word",
    "enrich": "enrich annotate tag label metadata prefix",
    "enriched": "enrich annotate tag label metadata prefix",
    "discussion": "discussion conversation talk chat exchange",
    "rant": "rant frustration complaint vent anger",
    "feedback": "feedback correction review reconcile improve",
    "bug": "bug error issue defect problem debug",
    "error": "error bug fail exception crash traceback",
    "errors": "error bug fail exception crash traceback",
    "traceback": "traceback stack error exception python crash",
    "routine": "routine habit pattern schedule daily regular",
    "thought": "thought reflection introspection reasoning cognition",
    "think": "thought reflection introspection reasoning cognition",
    "thoughts": "thought reflection introspection reasoning cognition",
    "reflect": "reflect introspection self awareness meta cognition",
    "reflection": "reflect introspection self awareness meta cognition",
    "introspect": "introspect reflection self awareness cognition",
    "feel": "feel emotion mood sentiment tone",
    "feeling": "feel emotion mood sentiment tone",
    "frustrated": "frustration anger annoyed upset",
    "curious": "curious wonder explore interest discovery",
}


def _expand_query(query: str) -> str:
    """Expand query with related terms for better BM25 matching."""
    tokens = _tokenize(query)
    expanded = set(tokens)
    for t in tokens:
        if t in QUERY_EXPANSION:
            for related in QUERY_EXPANSION[t].split():
                expanded.add(related)
    return " ".join(expanded)


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, return word tokens >= 2 chars."""
    return [w for w in re.findall(r"[a-z0-9]{2,}", text.lower()) if w not in STOP_WORDS]


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------


class BM25Index:
    """Lightweight in-memory BM25 index over all chunks."""

    def __init__(self):
        self.doc_count: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lens: Dict[str, int] = {}
        self.term_freqs: Dict[str, Dict[str, int]] = {}
        self._dirty = True
        self._indexed_ids: Set[str] = set()

    def _build_tf(self, tokens: List[str]) -> Dict[str, int]:
        """Build term frequency dict from token list."""
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        return tf

    def rebuild(self, chunks: Dict[str, str]) -> None:
        """Rebuild index from {chunk_id: text} dict."""
        self.doc_count = len(chunks)
        if self.doc_count == 0:
            return

        self.doc_freqs.clear()
        self.doc_lens.clear()
        self.term_freqs.clear()

        total_len = 0
        for cid, text in chunks.items():
            tokens = _tokenize(text)
            self.doc_lens[cid] = len(tokens)
            total_len += len(tokens)
            self.term_freqs[cid] = self._build_tf(tokens)
            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_doc_len = total_len / self.doc_count if self.doc_count else 0
        self._indexed_ids = set(chunks.keys())
        self._dirty = False

    def add(self, chunk_id: str, text: str) -> None:
        """Add a single chunk incrementally."""
        if chunk_id in self._indexed_ids:
            return

        tokens = _tokenize(text)
        doc_len = len(tokens)

        # Update stats
        old_total = self.avg_doc_len * self.doc_count
        self.doc_count += 1
        self.doc_lens[chunk_id] = doc_len
        self.term_freqs[chunk_id] = self._build_tf(tokens)
        for term in set(tokens):
            self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_doc_len = (old_total + doc_len) / self.doc_count if self.doc_count else 0
        self._indexed_ids.add(chunk_id)

    def remove(self, chunk_id: str) -> None:
        """Remove a single chunk incrementally."""
        if chunk_id not in self._indexed_ids:
            return

        doc_len = self.doc_lens.pop(chunk_id, 0)
        tf = self.term_freqs.pop(chunk_id, {})

        old_total = self.avg_doc_len * self.doc_count
        self.doc_count -= 1
        for term, count in tf.items():
            self.doc_freqs[term] = max(0, self.doc_freqs.get(term, 0) - 1)
            if self.doc_freqs[term] == 0:
                del self.doc_freqs[term]

        self.avg_doc_len = (old_total - doc_len) / self.doc_count if self.doc_count else 0
        self._indexed_ids.discard(chunk_id)

    def score(self, query: str, chunk_id: str) -> float:
        """Compute BM25 score for a query against a single document."""
        if self._dirty or chunk_id not in self.term_freqs:
            return 0.0

        tokens = _tokenize(query)
        if not tokens:
            return 0.0

        score = 0.0
        doc_len = self.doc_lens.get(chunk_id, 0)
        tf = self.term_freqs.get(chunk_id, {})

        for term in tokens:
            if term not in self.doc_freqs:
                continue
            df = self.doc_freqs[term]
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
            term_tf = tf.get(term, 0)
            denom = term_tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / max(self.avg_doc_len, 1))
            score += idf * (term_tf * (BM25_K1 + 1)) / max(denom, 1e-10)

        return score


# ---------------------------------------------------------------------------
# Cached Embedding Function (keeps ONNX session alive, ~12ms vs ~235ms)
# ---------------------------------------------------------------------------

class _CachedEmbeddingFunction:
    """Embedding function using bge-small-en-v1.5 (384-dim, CPU-friendly)."""

    def name(self) -> str:
        return "default"

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def _lazy_init(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import os
            import sys
            os.environ.setdefault("ORT_LOG_LEVEL", "FATAL")
            os.environ.setdefault("ORT_PROVIDERS", "CPUExecutionProvider")

            # Suppress tqdm "Loading weights" output by redirecting stderr
            old_stderr = sys.stderr
            sys.stderr = open(os.devnull, "w")
            try:
                from sentence_transformers import SentenceTransformer
                model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "bge-small-en-v1.5")
                self._model = SentenceTransformer(model_path, device="cpu")
            finally:
                sys.stderr.close()
                sys.stderr = old_stderr

    def __call__(self, input: List[str]) -> List[List[float]]:
        self._lazy_init()
        embeddings = self._model.encode(input, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, input: str) -> List[List[float]]:
        if isinstance(input, str):
            return self.__call__([input])
        return self.__call__(input)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return self.__call__(documents)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    chunk_id: str
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    similarity: float = 0.0
    bm25_score: float = 0.0
    energy: float = 0.0
    score: float = 0.0
    relevance: float = 0.0
    relevance_breakdown: Dict[str, float] = field(default_factory=dict)
    is_anchor: bool = False
    is_wm: bool = False
    is_graph: bool = False
    is_sequential: bool = False
    inhibition_penalty: float = 0.0


@dataclass
class RetrievalResult:
    chunks: List[Candidate]
    anchor_ids: List[str]
    wm_ids: List[str]
    wander_id: Optional[str]
    total_candidates: int
    relevance: float = 0.0
    relevance_components: Dict[str, float] = field(default_factory=dict)
    suppressed_count: int = 0
    min_relevance: float = 0.0


# ---------------------------------------------------------------------------
# The Engine
# ---------------------------------------------------------------------------


class SpreadingActivation:
    """Hybrid BM25 + Vector memory retrieval engine."""

    def __init__(self, meta_graph: Any, collection: Any):
        self.graph = meta_graph
        self.collection = collection
        self._bm25 = BM25Index()
        self._bm25_cache_ts: float = 0
        self._embed_fn = None
        self._embed_cache: Dict[str, Any] = {}

    def _get_embed_fn(self):
        if self._embed_fn is None:
            self._embed_fn = _CachedEmbeddingFunction()
        return self._embed_fn

    def _embed(self, texts: List[str]) -> List[List[float]]:
        fn = self._get_embed_fn()
        # Cache embeddings for repeated queries within same process
        key = tuple(texts)
        if key not in self._embed_cache:
            if len(self._embed_cache) > 100:
                self._embed_cache.clear()
            try:
                self._embed_cache[key] = fn(texts)
            except Exception as exc:
                logger.warning("Embedding failed for texts=%s: %s", [t[:30] for t in texts], exc)
                raise
        return self._embed_cache[key]

    def retrieve(
        self,
        query: str,
        session_id: str = "",
        working_memory: Optional[List[str]] = None,
        min_relevance: float = 0.20,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[RetrievalResult]:
        if not query or not query.strip():
            return None

        self._ensure_bm25()

        anchors = self._anchor_strike(query, session_id, metadata_filter=metadata_filter)
        if not anchors:
            logger.debug("Retrieve: no anchors found for query='%s'", query[:40])
            return None

        anchor_ids = [a.chunk_id for a in anchors]
        wm_ids = working_memory or []
        seed_ids = anchor_ids + [wid for wid in wm_ids if wid not in anchor_ids]
        graph_neighbors = self.graph.expand_neighbors(seed_ids)

        candidates = self._build_and_score(anchors, anchor_ids, wm_ids, graph_neighbors, query)
        self._apply_inhibition(candidates)
        result = self._select_final(candidates, anchor_ids, wm_ids, min_relevance=min_relevance)

        for c in result.chunks:
            self.graph.touch(c.chunk_id)
            self.graph.working_memory.push(c.chunk_id)

        return result

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    def _ensure_bm25(self) -> None:
        """Incrementally update BM25 index (add new chunks, remove deleted ones)."""
        import time
        now = time.time()
        if self._bm25._dirty or (now - self._bm25_cache_ts) > 60:
            try:
                current_count = self.collection.count()
                # Only fetch if count changed or first build
                if self._bm25.doc_count == 0 or current_count != self._bm25.doc_count:
                    results = self.collection.get(include=["documents", "metadatas"])
                    ids = results.get("ids", [])
                    docs = results.get("documents", [])
                    metas = results.get("metadatas", [])
                    # Use enriched_text from metadata for BM25 (tags improve recall)
                    # Fall back to document text if enriched_text not available (old chunks)
                    chunks = {}
                    for cid, doc, meta in zip(ids, docs, metas):
                        if not doc and not meta:
                            continue
                        if isinstance(meta, dict) and meta.get("enriched_text"):
                            chunks[cid] = meta["enriched_text"]
                        elif doc:
                            chunks[cid] = doc

                    if self._bm25._dirty or self._bm25.doc_count == 0:
                        self._bm25.rebuild(chunks)
                    else:
                        current_ids = set(ids)
                        for cid, doc in chunks.items():
                            if cid not in self._bm25._indexed_ids:
                                self._bm25.add(cid, doc)
                        for cid in list(self._bm25._indexed_ids):
                            if cid not in current_ids:
                                self._bm25.remove(cid)

                self._bm25_cache_ts = now
            except Exception as exc:
                logger.warning("BM25 index build failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 1: Anchor Strike (hybrid BM25 + Vector)
    # ------------------------------------------------------------------

    # Patterns that indicate noisy/meta-commentary chunks (not real conversation)
    # NOTE: "i was hallucinating" / "correction" etc are NOT filtered — they're
    # genuine self-corrections that help Nora learn. Only internal analysis markers.
    _NOISY_PATTERNS = [
        "[out-of-band", "[backend:",
        "actual knowledge retrieved", "stored fact", "knowledge retrieved",
        "system context verified",
    ]

    def _anchor_strike(self, query: str, session_id: str, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Candidate]:
        try:
            collection_count = self.collection.count()
            if collection_count == 0:
                logger.debug("Anchor strike: collection empty")
                return []

            # Expand query for better BM25 matching
            expanded_query = _expand_query(query)

            n = min(20, collection_count)
            # Pre-compute embedding and pass directly (avoids ChromaDB re-embedding)
            q_emb = self._embed([query])[0]
            query_kwargs: Dict[str, Any] = {"query_embeddings": [q_emb], "n_results": n}
            if metadata_filter:
                query_kwargs["where"] = metadata_filter
            results = self.collection.query(**query_kwargs)

            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            logger.debug("Anchor strike: query='%s' got %d results from ChromaDB", query[:40], len(ids))

            query_tokens = set(_tokenize(query))

            raw = []
            for cid, doc, meta, dist in zip(ids, docs, metas, distances):
                if not doc or not doc.strip():
                    continue
                if meta.get("session_id") == session_id:
                    continue

                # Filter out noisy/meta-commentary chunks
                doc_lower = doc.strip().lower()
                if any(pat in doc_lower for pat in self._NOISY_PATTERNS):
                    logger.debug("Anchor strike: filtered noisy chunk %s", cid[:8])
                    continue

                vector_sim = 1.0 / (1.0 + dist)
                # Use expanded query for BM25
                bm25 = self._bm25.score(expanded_query, cid)

                raw.append(Candidate(
                    chunk_id=cid,
                    text=doc.strip(),
                    metadata=meta,
                    similarity=vector_sim,
                    bm25_score=bm25,
                    is_anchor=True,
                ))

            if not raw:
                return []

            reranked = self._rerank(raw, query_tokens, query)
            return reranked[:ANCHOR_COUNT]

        except Exception as exc:
            logger.debug("Anchor strike failed: %s", exc)
            return []

    def _rerank(
        self,
        candidates: List[Candidate],
        query_tokens: Set[str],
        query: str,
    ) -> List[Candidate]:
        max_vec = max((c.similarity for c in candidates), default=1.0) or 1.0
        max_bm25 = max((c.bm25_score for c in candidates), default=1.0) or 1.0

        centrality = {}
        recency = {}
        for c in candidates:
            centrality[c.chunk_id] = self._compute_centrality(c.chunk_id)
            recency[c.chunk_id] = self._compute_recency(c.chunk_id)

        max_cent = max(centrality.values(), default=1.0) or 1.0
        max_rec = max(recency.values(), default=1.0) or 1.0

        for c in candidates:
            vec_norm = c.similarity / max_vec
            bm25_norm = c.bm25_score / max_bm25

            text_lower = c.text.lower()
            kw_hits = sum(1 for t in query_tokens if t in text_lower)
            kw_score = kw_hits / len(query_tokens) if query_tokens else 0.0

            cent_norm = centrality.get(c.chunk_id, 0.0) / max_cent
            rec_norm = recency.get(c.chunk_id, 0.0) / max_rec

            hybrid = (
                RANK_BM25 * bm25_norm +
                RANK_VECTOR * vec_norm +
                RANK_KEYWORD * kw_score +
                RANK_RECENCY * rec_norm +
                RANK_CENTRALITY * cent_norm
            )

            c.metadata["_hybrid_score"] = hybrid
            c.metadata["_original_sim"] = c.similarity
            c.similarity = hybrid

        candidates.sort(key=lambda c: c.similarity, reverse=True)
        return candidates

    def _compute_centrality(self, chunk_id: str) -> float:
        syn = self.graph.get_synapses(chunk_id)
        if not syn:
            return 0.0
        similar_count = len(syn.get("similar_ids", []))
        has_prev = 1 if syn.get("prev_id") else 0
        has_next = 1 if syn.get("next_id") else 0
        temp = self.graph.get_temporal(chunk_id)
        access = temp.get("access_count", 0) if temp else 0
        return similar_count * 2.0 + has_prev + has_next + access * 0.5

    def _compute_recency(self, chunk_id: str) -> float:
        days = self.graph.days_since_access(chunk_id)
        return math.exp(-0.05 * days)

    # ------------------------------------------------------------------
    # Step 2 & 3: Build pool and score
    # ------------------------------------------------------------------

    def _build_and_score(
        self,
        anchors: List[Candidate],
        anchor_ids: List[str],
        wm_ids: List[str],
        graph_neighbors: Set[str],
        query: str,
    ) -> List[Candidate]:
        expanded_query = _expand_query(query)
        candidates: Dict[str, Candidate] = {}

        for anchor in anchors:
            candidates[anchor.chunk_id] = anchor

        for wm_id in wm_ids:
            if wm_id not in candidates and self.graph.has_chunk(wm_id):
                text, meta = self._fetch_text_meta(wm_id)
                # Skip noisy chunks
                if text and any(pat in text.lower() for pat in self._NOISY_PATTERNS):
                    continue
                bm25 = self._bm25.score(expanded_query, wm_id)
                candidates[wm_id] = Candidate(
                    chunk_id=wm_id, text=text, metadata=meta,
                    is_wm=True, energy=ENERGY_WM, bm25_score=bm25,
                )

        for gid in graph_neighbors:
            if gid not in candidates and self.graph.has_chunk(gid):
                text, meta = self._fetch_text_meta(gid)
                # Skip noisy chunks
                if text and any(pat in text.lower() for pat in self._NOISY_PATTERNS):
                    continue
                bm25 = self._bm25.score(expanded_query, gid)
                # Detect sequential-only neighbors (connected via prev/next but NOT via similar_ids)
                syn = self.graph.get_synapses(gid)
                is_sequential = False
                if syn:
                    has_semantic = len(syn.get("similar_ids", [])) > 0
                    has_seq = bool(syn.get("prev_id")) or bool(syn.get("next_id"))
                    is_sequential = has_seq and not has_semantic
                candidates[gid] = Candidate(
                    chunk_id=gid, text=text, metadata=meta,
                    is_graph=True, is_sequential=is_sequential,
                    energy=ENERGY_GRAPH_NODE, bm25_score=bm25,
                )

        for cand in candidates.values():
            cand.score = self._score(cand, query, compute_breakdown=False)

        return list(candidates.values())

    def _score(self, candidate: Candidate, query: str, compute_breakdown: bool = True) -> float:
        original_sim = candidate.metadata.get("_original_sim", candidate.similarity)
        sim_component = W_SIMILARITY * original_sim
        bm25_component = W_BM25 * min(candidate.bm25_score / BM25_CAP, 1.0)

        days = self.graph.days_since_access(candidate.chunk_id)
        decay = math.exp(-DECAY_LAMBDA * days)
        energy_component = W_ENERGY * candidate.energy * decay

        # Deterministic noise from chunk_id hash for reproducibility
        noise_seed = int(hashlib.md5(candidate.chunk_id.encode()).hexdigest()[:8], 16) % 10000
        noise = ((noise_seed / 10000.0) - 0.5) * NOISE_STD * 2
        noise += random.gauss(0, NOISE_STD * 0.1)  # tiny stochastic jitter for wander

        # Compute interpretable relevance metric (separate from ranking score)
        query_tokens = set(_tokenize(query))
        # Use raw text from metadata to avoid tag/keyword inflation in matching
        raw_text = candidate.metadata.get("text", candidate.text)
        text_tokens = set(_tokenize(raw_text))
        kw_hits = len(query_tokens & text_tokens) if query_tokens else 0
        kw_score = kw_hits / len(query_tokens) if query_tokens else 0.0
        bm25_capped = min(candidate.bm25_score / BM25_CAP, 1.0)
        lexical = 0.5 * bm25_capped + 0.5 * kw_score

        recency = self._compute_recency(candidate.chunk_id)
        centrality = self._compute_centrality(candidate.chunk_id)
        cent_norm = min(centrality / 5.0, 1.0)
        energy_decayed = decay * (candidate.energy / 0.8)
        graph_score = 0.7 * cent_norm + 0.3 * energy_decayed
        # Sequential-only neighbors (prev/next without semantic edge) get reduced graph weight
        if candidate.is_graph and candidate.is_sequential:
            graph_score *= 0.5

        if candidate.is_anchor:
            source_score = 1.0
        elif candidate.is_wm:
            source_score = 0.7
        else:
            source_score = 0.5

        semantic = original_sim
        relevance = (
            REL_SEMANTIC * semantic +
            REL_LEXICAL * lexical +
            REL_TEMPORAL * recency +
            REL_GRAPH * graph_score +
            REL_SOURCE * source_score
        )

        candidate.relevance = relevance
        if compute_breakdown:
            candidate.relevance_breakdown = {
                "semantic": round(semantic, 4),
                "lexical": round(lexical, 4),
                "temporal": round(recency, 4),
                "graph": round(graph_score, 4),
                "source": round(source_score, 4),
            }

        return min(1.0, max(0.0, sim_component + bm25_component + energy_component + noise))

    def _fetch_text(self, chunk_id: str) -> str:
        """Fetch enriched document text from ChromaDB."""
        try:
            result = self.collection.get(ids=[chunk_id], include=["documents"])
            docs = result.get("documents", [])
            return docs[0] if docs else ""
        except Exception:
            return ""

    def _fetch_text_meta(self, chunk_id: str) -> Tuple[str, Dict]:
        """Fetch enriched text + metadata dict from ChromaDB."""
        try:
            result = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
            docs = result.get("documents", [])
            metas = result.get("metadatas", [])
            text = docs[0] if docs else ""
            meta = metas[0] if metas else {}
            if not isinstance(meta, dict):
                meta = {}
            return text, meta
        except Exception:
            return "", {}

    # ------------------------------------------------------------------
    # Step 4: Competitive Inhibition
    # ------------------------------------------------------------------

    def _apply_inhibition(self, candidates: List[Candidate]) -> None:
        for i, c1 in enumerate(candidates):
            for c2 in candidates[i + 1:]:
                if abs(c1.similarity - c2.similarity) > 0.3:
                    continue
                syn1 = self.graph.get_synapses(c1.chunk_id)
                syn2 = self.graph.get_synapses(c2.chunk_id)
                if not syn1 or not syn2:
                    continue
                sim1 = set(syn1.get("similar_ids", []))
                sim2 = set(syn2.get("similar_ids", []))
                # One-directional edge is enough — set_similar now writes both ways
                if c1.chunk_id in sim2 or c2.chunk_id in sim1:
                    t1 = self.graph.get_temporal(c1.chunk_id)
                    t2 = self.graph.get_temporal(c2.chunk_id)
                    if t1 and t2:
                        try:
                            dt1 = datetime.fromisoformat(t1.get("created_at", "1970-01-01T00:00:00+00:00"))
                            dt2 = datetime.fromisoformat(t2.get("created_at", "1970-01-01T00:00:00+00:00"))
                            older = c1 if dt1 < dt2 else c2
                        except (ValueError, TypeError):
                            older = c1 if t1.get("created_at", "") < t2.get("created_at", "") else c2
                        older.inhibition_penalty = INHIBITION_PENALTY
                        older.score *= (1.0 - INHIBITION_PENALTY)

    # ------------------------------------------------------------------
    # Step 5: Select final chunks + wander
    # ------------------------------------------------------------------

    def _select_final(
        self,
        candidates: List[Candidate],
        anchor_ids: List[str],
        wm_ids: List[str],
        min_relevance: float = 0.0,
    ) -> RetrievalResult:
        candidates.sort(key=lambda c: c.score, reverse=True)

        if not candidates:
            return RetrievalResult(
                chunks=[], anchor_ids=anchor_ids,
                wm_ids=wm_ids, wander_id=None, total_candidates=0,
                min_relevance=min_relevance,
            )

        # Filter out chunks below confidence threshold
        total_before_filter = len(candidates)
        if min_relevance > 0.0:
            candidates = [c for c in candidates if c.relevance >= min_relevance]
        suppressed = total_before_filter - len(candidates)

        top_n = FINAL_CHUNKS - WANDER_COUNT
        selected = candidates[:top_n]

        wander_id = None
        remaining = candidates[top_n:]
        if remaining:
            wander = random.choice(remaining)
            wander_id = wander.chunk_id
            selected.append(wander)
        elif len(selected) > 1:
            # Small pool: steal last selected as wander for serendipity
            wander = selected.pop()
            wander_id = wander.chunk_id
            selected.append(wander)

        selected = selected[:FINAL_CHUNKS]

        # Compute relevance breakdown only for selected chunks
        for c in selected:
            if not c.relevance_breakdown:
                self._score(c, "", compute_breakdown=True)

        # Compute aggregate relevance for the full result
        avg_relevance = sum(c.relevance for c in selected) / len(selected) if selected else 0.0
        components = {}
        if selected:
            keys = list(selected[0].relevance_breakdown.keys())
            for k in keys:
                components[k] = round(sum(c.relevance_breakdown.get(k, 0.0) for c in selected) / len(selected), 4)

        return RetrievalResult(
            chunks=selected, anchor_ids=anchor_ids,
            wm_ids=wm_ids, wander_id=wander_id,
            total_candidates=len(candidates),
            relevance=round(avg_relevance, 4),
            relevance_components=components,
            suppressed_count=suppressed,
            min_relevance=min_relevance,
        )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_context_block(result: RetrievalResult) -> str:
    if not result or not result.chunks:
        return ""

    parts = []
    for chunk in result.chunks:
        if chunk.chunk_id == result.wander_id:
            parts.append(f"[Spontaneous Association]\n{chunk.text}")
        else:
            parts.append(f"[Contextual Recall]\n{chunk.text}")

    context = "\n---\n".join(parts)

    return (
        f"[RECALLED PAST CONTEXT - {len(result.chunks)} pieces from previous sessions. "
        f"This is historical memory, NOT the current conversation. "
        f"Do NOT run tools to verify or address this. "
        f"Use it only if directly relevant to what the user is asking NOW. "
        f"If the user is having a casual conversation, stay in that conversation.]\n\n"
        f"{context}\n\n"
        f"[END OF RECALLED CONTEXT - The user's actual message follows]"
    )
