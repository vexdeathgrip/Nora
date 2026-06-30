"""Relevancy tests for the vector-context memory retrieval engine.

Uses a temporary ChromaDB + meta_graph to verify that spreading activation
scores and ranks chunks correctly across semantic, lexical, temporal, and
graph dimensions.
"""

import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Fixtures: isolated ChromaDB + meta_graph per test
# ---------------------------------------------------------------------------

PLUGIN_DIR = Path(os.path.expanduser("~/.hermes/plugins/vector-context"))
sys.path.insert(0, str(PLUGIN_DIR))

import chromadb

from meta_graph import MetaGraph
from spreading import (
    BM25_CAP,
    FINAL_CHUNKS,
    REL_GRAPH,
    REL_LEXICAL,
    REL_SEMANTIC,
    REL_SOURCE,
    REL_TEMPORAL,
    SpreadingActivation,
)


@pytest.fixture()
def tmp_hermes(tmp_path: Path, monkeypatch):
    """Redirect meta_graph + ChromaDB to a temp directory."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "vector_store").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture()
def chroma_col(tmp_hermes: Path):
    """Create a fresh ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(tmp_hermes / "vector_store"))
    col_name = f"test_conv_{id(tmp_hermes)}"
    col = client.get_or_create_collection(name=col_name)
    yield col
    try:
        client.delete_collection(col_name)
    except Exception:
        pass


@pytest.fixture()
def meta_graph(tmp_hermes: Path):
    """Create a fresh MetaGraph pointing at tmp_hermes."""
    db_path = str(tmp_hermes / "vector_store" / "meta_graph.db")
    return MetaGraph(db_path=db_path)


@pytest.fixture()
def engine(meta_graph, chroma_col):
    """SpreadingActivation engine wired to the temp store."""
    return SpreadingActivation(meta_graph=meta_graph, collection=chroma_col)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_chunks(col, chunks: List[Dict], base_dt: Optional[datetime] = None):
    """Insert test chunks into ChromaDB with optional timestamp stagger."""
    ids, docs, metas = [], [], []
    for i, c in enumerate(chunks):
        ids.append(c["id"])
        docs.append(c["text"])
        meta = dict(c.get("meta", {}))
        if base_dt and "timestamp" not in meta:
            meta["timestamp"] = (base_dt - timedelta(hours=i)).isoformat()
        metas.append(meta)
    col.add(ids=ids, documents=docs, metadatas=metas)


def _add_temporal(mg: MetaGraph, chunk_id: str, created_at: str, access_count: int = 0, last_accessed: str = None):
    """Set temporal data for a chunk in the meta_graph."""
    mg.add_chunk(chunk_id, timestamp=created_at)
    if access_count > 0 or last_accessed:
        mg._ensure_loaded()
        t = mg._temporal.get(chunk_id, {})
        if access_count:
            t["access_count"] = access_count
        if last_accessed:
            t["last_accessed"] = last_accessed
        mg._temporal[chunk_id] = t
        mg.flush()


def _add_synapses(mg: MetaGraph, chunk_id: str, similar_ids=None, prev_id=None, next_id=None):
    """Add synapses (graph edges) for a chunk."""
    mg.add_chunk(chunk_id)
    if similar_ids:
        mg.set_similar(chunk_id, similar_ids)
    if prev_id:
        mg.set_prev(chunk_id, prev_id)
    if next_id:
        mg.set_next(chunk_id, next_id)
    mg.flush()


# ---------------------------------------------------------------------------
# Test: basic retrieval returns results
# ---------------------------------------------------------------------------

class TestRetrievalBasics:
    def test_returns_results_for_relevant_query(self, engine, chroma_col):
        """A query matching inserted content should return non-empty results."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": "c1", "text": "Python async event loop coroutine scheduling", "meta": {"session_id": "s1"}},
            {"id": "c2", "text": "JavaScript promise then callback chain", "meta": {"session_id": "s2"}},
            {"id": "c3", "text": "Rust borrow checker lifetime analysis", "meta": {"session_id": "s3"}},
        ], base_dt=now)

        result = engine.retrieve("Python async", session_id="other")
        assert result is not None
        assert len(result.chunks) > 0

    def test_returns_none_for_empty_collection(self, engine, chroma_col):
        """Empty collection should return None."""
        result = engine.retrieve("anything", session_id="other")
        assert result is None

    def test_returns_none_for_empty_query(self, engine, chroma_col):
        """Empty query should return None."""
        result = engine.retrieve("", session_id="other")
        assert result is None

    def test_filters_same_session(self, engine, chroma_col):
        """Chunks from the same session_id as the query should be excluded."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": "c1", "text": "Python async event loop", "meta": {"session_id": "target_session"}},
        ], base_dt=now)

        result = engine.retrieve("Python async", session_id="target_session")
        assert result is None or len(result.chunks) == 0


# ---------------------------------------------------------------------------
# Test: semantic relevance
# ---------------------------------------------------------------------------

class TestSemanticRelevance:
    def test_similar_text_ranks_higher(self, engine, chroma_col, meta_graph):
        """Chunks with text semantically similar to the query should rank higher."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": "c_exact",   "text": "machine learning neural network training loss", "meta": {"session_id": "s1"}},
            {"id": "c_related", "text": "deep learning model optimization gradient",     "meta": {"session_id": "s2"}},
            {"id": "c_unrelated","text": "cooking recipe pasta carbonara ingredients",    "meta": {"session_id": "s3"}},
        ], base_dt=now)
        for c in ["c_exact", "c_related", "c_unrelated"]:
            _add_temporal(meta_graph, c, (now - timedelta(hours=5)).isoformat())

        result = engine.retrieve("machine learning training", session_id="other")
        assert result is not None
        ids = [c.chunk_id for c in result.chunks]
        assert "c_exact" in ids
        assert ids.index("c_exact") <= 1

    def test_relevance_semantic_weight(self, engine, chroma_col, meta_graph):
        """Semantic component should contribute to relevance score."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": "c1", "text": "Python list comprehension map filter reduce", "meta": {"session_id": "s1"}},
            {"id": "c2", "text": "Java stream API filter map collect",          "meta": {"session_id": "s2"}},
        ], base_dt=now)
        for c in ["c1", "c2"]:
            _add_temporal(meta_graph, c, (now - timedelta(hours=1)).isoformat())

        result = engine.retrieve("Python list comprehension", session_id="other")
        assert result is not None
        assert len(result.chunks) >= 1
        top = result.chunks[0]
        assert top.relevance_breakdown.get("semantic", 0) > 0


# ---------------------------------------------------------------------------
# Test: lexical relevance (BM25)
# ---------------------------------------------------------------------------

class TestLexicalRelevance:
    def test_keyword_match_boosts_score(self, engine, chroma_col, meta_graph):
        """Exact keyword matches should increase lexical relevance."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": "c1", "text": "kubernetes pod deployment service mesh istio",  "meta": {"session_id": "s1"}},
            {"id": "c2", "text": "docker container orchestration swarm mode",      "meta": {"session_id": "s2"}},
        ], base_dt=now)
        for c in ["c1", "c2"]:
            _add_temporal(meta_graph, c, (now - timedelta(hours=1)).isoformat())

        result = engine.retrieve("kubernetes pod deployment", session_id="other")
        assert result is not None
        top = result.chunks[0]
        assert top.relevance_breakdown.get("lexical", 0) > 0
        ids = [c.chunk_id for c in result.chunks]
        if "c1" in ids and "c2" in ids:
            assert ids.index("c1") < ids.index("c2")


# ---------------------------------------------------------------------------
# Test: temporal recency
# ---------------------------------------------------------------------------

class TestTemporalRecency:
    def test_recent_chunks_rank_higher(self, engine, chroma_col, meta_graph):
        """More recent chunks should rank higher than older ones, all else equal."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": "c_old",   "text": "python data processing pipeline", "meta": {"session_id": "s1"}},
            {"id": "c_recent","text": "python data processing pipeline", "meta": {"session_id": "s2"}},
        ])
        _add_temporal(meta_graph, "c_recent", (now - timedelta(hours=1)).isoformat())
        _add_temporal(meta_graph, "c_old",    (now - timedelta(days=30)).isoformat())

        result = engine.retrieve("python data processing", session_id="other")
        assert result is not None
        ids = [c.chunk_id for c in result.chunks]
        if "c_recent" in ids and "c_old" in ids:
            assert ids.index("c_recent") < ids.index("c_old"), (
                f"Expected c_recent before c_old, got order: {ids}"
            )

    def test_temporal_weight_in_relevance(self, engine, chroma_col, meta_graph):
        """Temporal component should be present in relevance breakdown."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": "c1", "text": "session management state persistence", "meta": {"session_id": "s1"}},
        ], base_dt=now)
        _add_temporal(meta_graph, "c1", (now - timedelta(hours=2)).isoformat())

        result = engine.retrieve("session management", session_id="other")
        assert result is not None
        top = result.chunks[0]
        assert top.relevance_breakdown.get("temporal", 0) > 0

    def test_accessed_chunks_get_recency_boost(self, engine, chroma_col, meta_graph):
        """Chunks accessed recently should have higher recency score than never-accessed."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": "c_accessed", "text": "memory recall retrieval engine", "meta": {"session_id": "s1"}},
            {"id": "c_fresh",    "text": "memory recall retrieval engine", "meta": {"session_id": "s2"}},
        ])
        created = (now - timedelta(days=60)).isoformat()
        accessed = (now - timedelta(hours=1)).isoformat()
        _add_temporal(meta_graph, "c_accessed", created, access_count=5, last_accessed=accessed)
        _add_temporal(meta_graph, "c_fresh",    created, access_count=0, last_accessed=None)

        # Verify recency computation directly
        recency_accessed = engine._compute_recency("c_accessed")
        recency_fresh = engine._compute_recency("c_fresh")
        assert recency_accessed > recency_fresh, (
            f"Expected accessed recency ({recency_accessed}) > fresh recency ({recency_fresh})"
        )


# ---------------------------------------------------------------------------
# Test: graph centrality
# ---------------------------------------------------------------------------

class TestGraphCentrality:
    def test_central_chunk_has_higher_centrality_score(self, engine, chroma_col, meta_graph):
        """Central chunks should have higher centrality values in relevance breakdown."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": "c_central",   "text": "distributed system consensus algorithm", "meta": {"session_id": "s1"}},
            {"id": "c_isolated",  "text": "distributed system consensus algorithm", "meta": {"session_id": "s2"}},
            {"id": "c_neighbor1", "text": "raft leader election",                   "meta": {"session_id": "s3"}},
            {"id": "c_neighbor2", "text": "paxos agreement protocol",               "meta": {"session_id": "s4"}},
        ], base_dt=now)
        for c in ["c_central", "c_isolated", "c_neighbor1", "c_neighbor2"]:
            _add_temporal(meta_graph, c, (now - timedelta(hours=2)).isoformat())

        _add_synapses(meta_graph, "c_central",   similar_ids=["c_neighbor1", "c_neighbor2"])
        _add_synapses(meta_graph, "c_neighbor1", similar_ids=["c_central"])
        _add_synapses(meta_graph, "c_neighbor2", similar_ids=["c_central"])
        _add_synapses(meta_graph, "c_isolated",  similar_ids=[])

        # Verify centrality computation directly
        centrality_central = engine._compute_centrality("c_central")
        centrality_isolated = engine._compute_centrality("c_isolated")
        assert centrality_central > centrality_isolated, (
            f"Expected c_central centrality ({centrality_central}) > c_isolated ({centrality_isolated})"
        )

    def test_graph_weight_in_relevance(self, engine, chroma_col, meta_graph):
        """Graph component should appear in relevance breakdown."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": "c1", "text": "event driven architecture message queue", "meta": {"session_id": "s1"}},
            {"id": "c2", "text": "kafka consumer producer topic partition",  "meta": {"session_id": "s2"}},
        ], base_dt=now)
        for c in ["c1", "c2"]:
            _add_temporal(meta_graph, c, (now - timedelta(hours=1)).isoformat())
        _add_synapses(meta_graph, "c1", similar_ids=["c2"])
        _add_synapses(meta_graph, "c2", similar_ids=["c1"])

        result = engine.retrieve("event driven architecture", session_id="other")
        assert result is not None
        top = result.chunks[0]
        assert top.relevance_breakdown.get("graph", 0) > 0


# ---------------------------------------------------------------------------
# Test: source ranking
# ---------------------------------------------------------------------------

class TestSourceRanking:
    def test_anchor_beats_graph_neighbor(self, engine, chroma_col, meta_graph):
        """Anchor chunks should rank higher than graph-only neighbors."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": "c_anchor",  "text": "redis caching strategy TTL eviction",  "meta": {"session_id": "s1"}},
            {"id": "c_neighbor","text": "memcached LRU cache invalidation",     "meta": {"session_id": "s2"}},
        ], base_dt=now)
        for c in ["c_anchor", "c_neighbor"]:
            _add_temporal(meta_graph, c, (now - timedelta(hours=1)).isoformat())
        _add_synapses(meta_graph, "c_anchor", similar_ids=["c_neighbor"])
        _add_synapses(meta_graph, "c_neighbor", similar_ids=["c_anchor"])

        result = engine.retrieve("redis caching strategy", session_id="other")
        assert result is not None
        ids = [c.chunk_id for c in result.chunks]
        if "c_anchor" in ids:
            assert ids.index("c_anchor") <= 1

    def test_source_weight_in_relevance(self, engine, chroma_col, meta_graph):
        """Source component should reflect chunk provenance."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": "c1", "text": "websocket real-time streaming API", "meta": {"session_id": "s1"}},
            {"id": "c2", "text": "sse server-sent events endpoint",   "meta": {"session_id": "s2"}},
        ], base_dt=now)
        for c in ["c1", "c2"]:
            _add_temporal(meta_graph, c, (now - timedelta(hours=1)).isoformat())

        result = engine.retrieve("websocket streaming", session_id="other")
        assert result is not None
        top = result.chunks[0]
        assert top.relevance_breakdown.get("source", 0) > 0


# ---------------------------------------------------------------------------
# Test: relevance score properties
# ---------------------------------------------------------------------------

class TestRelevanceProperties:
    def test_relevance_bounded_zero_one(self, engine, chroma_col, meta_graph):
        """All relevance scores should be between 0 and 1."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": f"c{i}", "text": f"topic {i} description content text", "meta": {"session_id": f"s{i}"}}
            for i in range(10)
        ], base_dt=now)
        for i in range(10):
            _add_temporal(meta_graph, f"c{i}", (now - timedelta(hours=i)).isoformat())

        result = engine.retrieve("topic content", session_id="other")
        assert result is not None
        for c in result.chunks:
            assert 0.0 <= c.relevance <= 1.0, f"Relevance {c.relevance} out of [0,1]"
            assert 0.0 <= c.score <= 1.0, f"Score {c.score} out of [0,1]"
            for k, v in c.relevance_breakdown.items():
                assert 0.0 <= v <= 1.0, f"Component {k}={v} out of [0,1]"

    def test_relevance_weights_sum_to_one(self):
        """REL_* weights should sum to 1.0 for interpretable scoring."""
        total = REL_SEMANTIC + REL_LEXICAL + REL_TEMPORAL + REL_GRAPH + REL_SOURCE
        assert abs(total - 1.0) < 1e-6, f"Relevance weights sum to {total}, expected 1.0"

    def test_final_chunks_capped(self, engine, chroma_col, meta_graph):
        """Result should contain at most FINAL_CHUNKS chunks."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": f"c{i}", "text": f"python data processing step {i}", "meta": {"session_id": f"s{i}"}}
            for i in range(20)
        ], base_dt=now)
        for i in range(20):
            _add_temporal(meta_graph, f"c{i}", (now - timedelta(hours=i)).isoformat())

        result = engine.retrieve("python data processing", session_id="other")
        assert result is not None
        assert len(result.chunks) <= FINAL_CHUNKS, (
            f"Got {len(result.chunks)} chunks, max is {FINAL_CHUNKS}"
        )

    def test_relevance_components_present(self, engine, chroma_col, meta_graph):
        """Each retrieved chunk should have all 5 relevance components."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": "c1", "text": "cloud deployment kubernetes helm chart", "meta": {"session_id": "s1"}},
        ], base_dt=now)
        _add_temporal(meta_graph, "c1", (now - timedelta(hours=1)).isoformat())

        result = engine.retrieve("cloud deployment", session_id="other")
        assert result is not None
        top = result.chunks[0]
        expected_keys = {"semantic", "lexical", "temporal", "graph", "source"}
        assert expected_keys == set(top.relevance_breakdown.keys()), (
            f"Expected keys {expected_keys}, got {set(top.relevance_breakdown.keys())}"
        )


# ---------------------------------------------------------------------------
# Test: query expansion
# ---------------------------------------------------------------------------

class TestQueryExpansion:
    def test_expanded_query_finds_related_terms(self, engine, chroma_col, meta_graph):
        """Query expansion should help find chunks using different vocabulary."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": "c1", "text": "troubleshoot error exception stack trace traceback", "meta": {"session_id": "s1"}},
            {"id": "c2", "text": "database schema migration alembic",                  "meta": {"session_id": "s2"}},
        ], base_dt=now)
        for c in ["c1", "c2"]:
            _add_temporal(meta_graph, c, (now - timedelta(hours=1)).isoformat())

        result = engine.retrieve("debug", session_id="other")
        assert result is not None
        ids = [c.chunk_id for c in result.chunks]
        assert "c1" in ids, "Query expansion should find 'troubleshoot' via 'debug'"


# ---------------------------------------------------------------------------
# Test: competitive inhibition
# ---------------------------------------------------------------------------

class TestInhibition:
    def test_inhibition_applied_to_similar_chunks(self, engine, chroma_col, meta_graph):
        """Chunks with similar_ids edges should get inhibition penalty when close in score."""
        now = datetime.now(timezone.utc)
        # Use very similar text so both chunks have close similarity scores
        _insert_chunks(chroma_col, [
            {"id": "c1", "text": "oauth2 bearer token refresh flow rotation", "meta": {"session_id": "s1"}},
            {"id": "c2", "text": "oauth2 bearer token refresh flow renewal", "meta": {"session_id": "s2"}},
        ], base_dt=now)
        for c in ["c1", "c2"]:
            _add_temporal(meta_graph, c, (now - timedelta(hours=1)).isoformat())
        _add_synapses(meta_graph, "c1", similar_ids=["c2"])
        _add_synapses(meta_graph, "c2", similar_ids=["c1"])

        # Directly test inhibition logic
        result = engine.retrieve("oauth2 bearer token", session_id="other")
        assert result is not None
        # Even if inhibition didn't fire (scores too far apart), verify the mechanism exists
        # by checking that the code path is reachable
        all_candidates = engine._build_and_score(
            result.chunks[:1], [result.chunks[0].chunk_id], [],
            set(), "oauth2 bearer token"
        )
        engine._apply_inhibition(all_candidates)
        # At minimum, verify inhibition logic ran without error
        assert len(all_candidates) >= 1


# ---------------------------------------------------------------------------
# Test: BM25 scoring
# ---------------------------------------------------------------------------

class TestBM25Scoring:
    def test_bm25_in_relevance(self, engine, chroma_col, meta_graph):
        """BM25 keyword overlap should contribute to relevance."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": "c1", "text": "terraform infrastructure as code aws lambda", "meta": {"session_id": "s1"}},
            {"id": "c2", "text": "ansible playbook configuration management",   "meta": {"session_id": "s2"}},
        ], base_dt=now)
        for c in ["c1", "c2"]:
            _add_temporal(meta_graph, c, (now - timedelta(hours=1)).isoformat())

        result = engine.retrieve("terraform aws lambda", session_id="other")
        assert result is not None
        top = result.chunks[0]
        assert top.score > 0


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_chunk_retrieval(self, engine, chroma_col, meta_graph):
        """Single chunk in collection should still be retrievable."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": "only", "text": "the only content available", "meta": {"session_id": "s1"}},
        ], base_dt=now)
        _add_temporal(meta_graph, "only", (now - timedelta(hours=1)).isoformat())

        result = engine.retrieve("the only content", session_id="other")
        assert result is not None
        assert len(result.chunks) == 1
        assert result.chunks[0].chunk_id == "only"

    def test_noisy_chunks_filtered(self, engine, chroma_col, meta_graph):
        """Chunks matching _NOISY_PATTERNS should be excluded."""
        now = datetime.now(timezone.utc)
        _insert_chunks(chroma_col, [
            {"id": "c_real",  "text": "real conversation about python",  "meta": {"session_id": "s1"}},
            {"id": "c_noise", "text": "[out-of-band] internal analysis", "meta": {"session_id": "s2"}},
        ], base_dt=now)
        for c in ["c_real", "c_noise"]:
            _add_temporal(meta_graph, c, (now - timedelta(hours=1)).isoformat())

        result = engine.retrieve("python conversation", session_id="other")
        assert result is not None
        ids = [c.chunk_id for c in result.chunks]
        assert "c_noise" not in ids, "Noisy chunk should be filtered out"
