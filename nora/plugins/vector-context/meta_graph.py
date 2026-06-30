"""Meta-Graph: SQLite-backed structural memory with LRU cache.

Dual-system coexistence with ChromaDB:
  System A: ChromaDB (vector index, semantic similarity)
  System B: Meta-Graph (synapses, temporal metadata, working memory)

Uses LRU cache for 1M+ scale. Hot data stays in memory, cold data
loads on demand from SQLite.
"""

import json
import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WMQ_MAX_SIZE = 5
FLUSH_INTERVAL = 100
DEDUP_THRESHOLD = 0.96
SIMILAR_EDGES = 3
EVICT_UTILITY_MIN = 0.01
EVICT_AGE_MIN_DAYS = 30
LRU_MAX_SIZE = 200000  # 200K entries ~100MB RAM, good for 1M+ scale

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS synapses (
    chunk_id TEXT PRIMARY KEY,
    next_id TEXT,
    prev_id TEXT,
    similar_ids TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS temporal (
    chunk_id TEXT PRIMARY KEY,
    last_accessed TEXT,
    access_count INTEGER DEFAULT 0,
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_synapses_next ON synapses(next_id);
CREATE INDEX IF NOT EXISTS idx_synapses_prev ON synapses(prev_id);
CREATE INDEX IF NOT EXISTS idx_temporal_access ON temporal(access_count);
CREATE INDEX IF NOT EXISTS idx_temporal_created ON temporal(created_at);
"""


# ---------------------------------------------------------------------------
# Working Memory Queue
# ---------------------------------------------------------------------------


class WorkingMemoryQueue:
    """Rolling queue of recently retrieved chunk IDs."""

    def __init__(self, max_size: int = WMQ_MAX_SIZE):
        self._queue: OrderedDict[str, None] = OrderedDict()
        self._max_size = max_size
        self._dirty = False
        self._lock = threading.Lock()

    def push(self, chunk_id: str) -> None:
        with self._lock:
            if chunk_id in self._queue:
                self._queue.move_to_end(chunk_id)
            self._queue[chunk_id] = None
            while len(self._queue) > self._max_size:
                self._queue.popitem(last=False)
            self._dirty = True

    def push_many(self, chunk_ids: List[str]) -> None:
        with self._lock:
            for cid in chunk_ids:
                if cid in self._queue:
                    self._queue.move_to_end(cid)
                self._queue[cid] = None
                while len(self._queue) > self._max_size:
                    self._queue.popitem(last=False)
            self._dirty = True

    def get_ids(self) -> List[str]:
        with self._lock:
            return list(self._queue.keys())

    def contains(self, chunk_id: str) -> bool:
        with self._lock:
            return chunk_id in self._queue

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
            self._dirty = True

    def is_dirty(self) -> bool:
        with self._lock:
            return self._dirty

    def mark_clean(self) -> None:
        with self._lock:
            self._dirty = False

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)

    def __contains__(self, chunk_id: str) -> bool:
        with self._lock:
            return chunk_id in self._queue


# ---------------------------------------------------------------------------
# Meta-Graph
# ---------------------------------------------------------------------------


class MetaGraph:
    """Persistent graph with LRU cache for 1M+ scale."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            try:
                from hermes_constants import get_hermes_home
                db_path = str(get_hermes_home() / "vector_store" / "meta_graph.db")
            except Exception:
                db_path = str(Path.home() / ".hermes" / "vector_store" / "meta_graph.db")

        self._db_path = db_path
        self._wm_path = db_path + ".working_memory.json"  # persisted working memory
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._dirty_count = 0
        self._total_writes = 0

        # LRU cache for synapses and temporal data
        self._synapses: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._temporal: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._loaded = False
        self._synapses_in_db: Set[str] = set()  # track what's persisted
        self._temporal_in_db: Set[str] = set()

        # Working memory queue
        self.working_memory = WorkingMemoryQueue()

        self._init_db()
        self._restore_working_memory()

    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self._db_path, timeout=10, check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-64000")  # 64MB page cache
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load_all()

    def _load_all(self) -> None:
        """Load all data into memory. For 1M+ this uses LRU eviction."""
        try:
            rows = self._conn.execute(
                "SELECT chunk_id, next_id, prev_id, similar_ids, created_at FROM synapses"
            ).fetchall()
            for row in rows:
                self._synapses[row[0]] = {
                    "next_id": row[1], "prev_id": row[2],
                    "similar_ids": row[3].split(",") if row[3] else [],
                    "created_at": row[4],
                }
                self._synapses_in_db.add(row[0])

            rows = self._conn.execute(
                "SELECT chunk_id, last_accessed, access_count, created_at FROM temporal"
            ).fetchall()
            for row in rows:
                self._temporal[row[0]] = {
                    "last_accessed": row[1], "access_count": row[2],
                    "created_at": row[3],
                }
                self._temporal_in_db.add(row[0])

            self._loaded = True
            logger.info(
                "Meta-graph loaded: %d synapses, %d temporal",
                len(self._synapses), len(self._temporal),
            )
        except Exception as exc:
            logger.error("Failed to load meta-graph: %s", exc)

    def _evict_lru(self) -> None:
        """Evict least recently used entries when cache exceeds limit."""
        while len(self._synapses) > LRU_MAX_SIZE:
            _key, _ = self._synapses.popitem(last=False)
        while len(self._temporal) > LRU_MAX_SIZE:
            _key, _ = self._temporal.popitem(last=False)

    # ------------------------------------------------------------------
    # Synapse operations
    # ------------------------------------------------------------------

    def add_chunk(
        self, chunk_id: str, prev_id: Optional[str] = None,
        next_id: Optional[str] = None, similar_ids: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        self._ensure_loaded()
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        self._synapses[chunk_id] = {
            "next_id": next_id, "prev_id": prev_id,
            "similar_ids": similar_ids or [], "created_at": timestamp,
        }
        self._temporal[chunk_id] = {
            "last_accessed": None, "access_count": 0, "created_at": timestamp,
        }
        self._dirty_count += 1
        if self._dirty_count >= FLUSH_INTERVAL:
            self.flush()

    def get_synapses(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Return synapse dict for chunk, loading from SQLite on cache miss."""
        self._ensure_loaded()
        syn = self._synapses.get(chunk_id)
        if syn is not None:
            return syn
        # Cache miss — try loading from SQLite
        try:
            row = self._conn.execute(
                "SELECT next_id, prev_id, similar_ids, created_at FROM synapses WHERE chunk_id=?",
                (chunk_id,),
            ).fetchone()
            if row:
                syn = {
                    "next_id": row[0], "prev_id": row[1],
                    "similar_ids": row[2].split(",") if row[2] else [],
                    "created_at": row[3],
                }
                # Re-insert into LRU cache (evicts oldest if at capacity)
                self._synapses[chunk_id] = syn
                self._synapses_in_db.add(chunk_id)
                if len(self._synapses) > LRU_MAX_SIZE:
                    self._synapses.popitem(last=False)
                return syn
        except Exception:
            pass
        return None

    def set_next(self, chunk_id: str, next_id: str) -> None:
        self._ensure_loaded()
        if chunk_id in self._synapses:
            self._synapses[chunk_id]["next_id"] = next_id
            self._dirty_count += 1

    def set_prev(self, chunk_id: str, prev_id: str) -> None:
        self._ensure_loaded()
        if chunk_id in self._synapses:
            self._synapses[chunk_id]["prev_id"] = prev_id
            self._dirty_count += 1

    def set_similar(self, chunk_id: str, similar_ids: List[str]) -> None:
        self._ensure_loaded()
        if chunk_id in self._synapses:
            self._synapses[chunk_id]["similar_ids"] = similar_ids[:SIMILAR_EDGES]
            self._dirty_count += 1
        # Bidirectional edges: each similar chunk also lists this chunk
        for sid in similar_ids[:SIMILAR_EDGES]:
            if sid != chunk_id and sid in self._synapses:
                existing = self._synapses[sid].get("similar_ids", [])
                if chunk_id not in existing:
                    if len(existing) >= SIMILAR_EDGES:
                        existing.pop(0)  # evict oldest to make room
                    existing.append(chunk_id)
                    self._synapses[sid]["similar_ids"] = existing
                    self._dirty_count += 1

    def get_similar(self, chunk_id: str) -> List[str]:
        self._ensure_loaded()
        syn = self._synapses.get(chunk_id)
        return syn["similar_ids"] if syn else []

    def get_next(self, chunk_id: str) -> Optional[str]:
        self._ensure_loaded()
        syn = self._synapses.get(chunk_id)
        return syn["next_id"] if syn else None

    def get_prev(self, chunk_id: str) -> Optional[str]:
        self._ensure_loaded()
        syn = self._synapses.get(chunk_id)
        return syn["prev_id"] if syn else None

    # ------------------------------------------------------------------
    # Temporal operations
    # ------------------------------------------------------------------

    def touch(self, chunk_id: str) -> None:
        self._ensure_loaded()
        now = datetime.now(timezone.utc).isoformat()
        if chunk_id in self._temporal:
            self._temporal[chunk_id]["last_accessed"] = now
            self._temporal[chunk_id]["access_count"] += 1
            self._dirty_count += 1

    def get_temporal(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_loaded()
        return self._temporal.get(chunk_id)

    def get_access_count(self, chunk_id: str) -> int:
        self._ensure_loaded()
        t = self._temporal.get(chunk_id)
        return t["access_count"] if t else 0

    def days_since_access(self, chunk_id: str) -> float:
        self._ensure_loaded()
        t = self._temporal.get(chunk_id)
        if not t:
            return 999.0
        if not t["last_accessed"]:
            # Never accessed — use creation age instead of treating as ancient
            return self.chunk_age_days(chunk_id)
        try:
            last = datetime.fromisoformat(t["last_accessed"])
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return max(0.0, (now - last).total_seconds() / 86400)
        except Exception:
            return self.chunk_age_days(chunk_id)

    def chunk_age_days(self, chunk_id: str) -> float:
        self._ensure_loaded()
        t = self._temporal.get(chunk_id)
        if not t or not t["created_at"]:
            return 999.0
        try:
            created = datetime.fromisoformat(t["created_at"])
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return max(0.0, (now - created).total_seconds() / 86400)
        except Exception:
            return 999.0

    def utility(self, chunk_id: str) -> float:
        self._ensure_loaded()
        t = self._temporal.get(chunk_id)
        if not t:
            return 0.0
        import math
        access = t.get("access_count", 0)
        age = self.days_since_access(chunk_id)
        # High-access chunks decay slower: every access halves effective decay rate
        access_boost = math.log(1 + access) if access > 0 else 0
        effective_age = age / (1 + access_boost)
        return (1 + access) * math.exp(-0.1 * effective_age)

    # ------------------------------------------------------------------
    # Neighbor expansion
    # ------------------------------------------------------------------

    def expand_neighbors(self, chunk_ids: List[str]) -> Set[str]:
        self._ensure_loaded()
        candidates: Set[str] = set()
        for cid in chunk_ids:
            syn = self._synapses.get(cid)
            if not syn:
                continue
            for sim_id in syn.get("similar_ids", []):
                if sim_id and sim_id != cid:
                    candidates.add(sim_id)
            nxt = syn.get("next_id")
            if nxt and nxt != cid:
                candidates.add(nxt)
            prv = syn.get("prev_id")
            if prv and prv != cid:
                candidates.add(prv)
        candidates -= set(chunk_ids)
        return candidates

    # ------------------------------------------------------------------
    # Competitive inhibition
    # ------------------------------------------------------------------

    def find_inhibited_pairs(
        self, chunk_ids: List[str], similarity_threshold: float = 0.85
    ) -> List[Tuple[str, str]]:
        self._ensure_loaded()
        pairs = []
        for i, cid_a in enumerate(chunk_ids):
            for cid_b in chunk_ids[i + 1:]:
                syn_a = self._synapses.get(cid_a, {})
                syn_b = self._synapses.get(cid_b, {})
                sim_a = set(syn_a.get("similar_ids", []))
                sim_b = set(syn_b.get("similar_ids", []))
                if cid_a in sim_b or cid_b in sim_a:
                    t_a = self._temporal.get(cid_a, {})
                    t_b = self._temporal.get(cid_b, {})
                    if t_a.get("created_at", "") < t_b.get("created_at", ""):
                        pairs.append((cid_a, cid_b))
                    else:
                        pairs.append((cid_b, cid_a))
        return pairs

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def compute_eviction_candidates(self) -> List[str]:
        self._ensure_loaded()
        candidates = []
        for cid in self._temporal:
            age = self.chunk_age_days(cid)
            if age < EVICT_AGE_MIN_DAYS:
                continue
            util = self.utility(cid)
            if util < EVICT_UTILITY_MIN:
                candidates.append((cid, util))
        candidates.sort(key=lambda x: x[1])
        return [cid for cid, _ in candidates]

    def evict(self, chunk_ids: List[str]) -> int:
        self._ensure_loaded()
        evicted: Set[str] = set()
        for cid in chunk_ids:
            if cid in self._synapses:
                syn = self._synapses[cid]
                nxt = syn.get("next_id")
                prv = syn.get("prev_id")
                if nxt and nxt in self._synapses:
                    self._synapses[nxt]["prev_id"] = prv
                if prv and prv in self._synapses:
                    self._synapses[prv]["next_id"] = nxt
                for other_id, other_syn in self._synapses.items():
                    if cid in other_syn.get("similar_ids", []):
                        other_syn["similar_ids"] = [
                            s for s in other_syn["similar_ids"] if s != cid
                        ]
                del self._synapses[cid]
                self._synapses_in_db.discard(cid)
                evicted.add(cid)
            if cid in self._temporal:
                del self._temporal[cid]
                self._temporal_in_db.discard(cid)
                evicted.add(cid)
        self._dirty_count += 1
        if evicted:
            self.flush()
        return len(evicted)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def find_duplicates(
        self, similarity_map: Dict[str, List[Tuple[str, float]]]
    ) -> List[Tuple[str, str, float]]:
        self._ensure_loaded()
        duplicates = []
        seen = set()
        for cid, neighbors in similarity_map.items():
            for other_id, score in neighbors:
                if score < DEDUP_THRESHOLD:
                    continue
                pair_key = tuple(sorted([cid, other_id]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                t_cid = self._temporal.get(cid, {})
                t_other = self._temporal.get(other_id, {})
                if t_cid.get("created_at", "") >= t_other.get("created_at", ""):
                    duplicates.append((cid, other_id, score))
                else:
                    duplicates.append((other_id, cid, score))
        return duplicates

    def merge_duplicates(self, keep_id: str, discard_id: str) -> None:
        self._ensure_loaded()
        syn_keep = self._synapses.get(keep_id, {})
        syn_discard = self._synapses.get(discard_id, {})
        temp_keep = self._temporal.get(keep_id, {})
        temp_discard = self._temporal.get(discard_id, {})

        if syn_keep and syn_discard:
            existing = set(syn_keep.get("similar_ids", []))
            for sim_id in syn_discard.get("similar_ids", []):
                if sim_id != keep_id and sim_id not in existing:
                    syn_keep["similar_ids"].append(sim_id)
                    existing.add(sim_id)
            syn_keep["similar_ids"] = list(dict.fromkeys(syn_keep["similar_ids"]))[:SIMILAR_EDGES]

        if temp_keep and temp_discard:
            temp_keep["access_count"] += temp_discard["access_count"]

        discard_next = syn_discard.get("next_id") if syn_discard else None
        discard_prev = syn_discard.get("prev_id") if syn_discard else None
        if discard_next and discard_next in self._synapses:
            self._synapses[discard_next]["prev_id"] = keep_id
        if discard_prev and discard_prev in self._synapses:
            self._synapses[discard_prev]["next_id"] = keep_id

        for other_id, other_syn in self._synapses.items():
            if other_id == keep_id:
                continue
            sims = other_syn.get("similar_ids", [])
            if discard_id in sims:
                other_syn["similar_ids"] = [
                    keep_id if s == discard_id else s for s in sims
                ]

        self.evict([discard_id])

    def _persist_working_memory(self) -> None:
        """Write working memory to a sidecar JSON file."""
        try:
            ids = self.working_memory.get_ids()
            Path(self._wm_path).write_text(json.dumps(ids))
        except Exception as exc:
            logger.debug("Failed to persist working memory: %s", exc)

    def _restore_working_memory(self) -> None:
        """Restore working memory from the sidecar JSON file."""
        try:
            p = Path(self._wm_path)
            if p.exists():
                ids = json.loads(p.read_text())
                if isinstance(ids, list):
                    for cid in ids:
                        self.working_memory.push(cid)
                    self.working_memory.mark_clean()
                    logger.debug("Restored %d working memory entries", len(ids))
        except Exception as exc:
            logger.debug("Failed to restore working memory: %s", exc)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self) -> None:
        with self._lock:
            try:
                # Persist working memory to a separate JSON file (avoids WAL visibility issues)
                if self.working_memory.is_dirty():
                    self._persist_working_memory()
                    self.working_memory.mark_clean()

                if not self._loaded:
                    return

                for cid, syn in self._synapses.items():
                    self._conn.execute(
                        "INSERT OR REPLACE INTO synapses (chunk_id, next_id, prev_id, similar_ids, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (cid, syn.get("next_id"), syn.get("prev_id"),
                         ",".join(syn.get("similar_ids", [])), syn.get("created_at")),
                    )

                for cid, tmp in self._temporal.items():
                    self._conn.execute(
                        "INSERT OR REPLACE INTO temporal (chunk_id, last_accessed, access_count, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (cid, tmp.get("last_accessed"), tmp.get("access_count", 0),
                         tmp.get("created_at")),
                    )

                self._conn.commit()
                self._dirty_count = 0
                self._total_writes += 1
                self._evict_lru()
                logger.debug(
                    "Meta-graph flushed: %d syn, %d temp, %d wm (writes=%d)",
                    len(self._synapses), len(self._temporal),
                    len(self.working_memory), self._total_writes,
                )
            except Exception as exc:
                logger.error("Failed to flush meta-graph: %s", exc)

    def size(self) -> int:
        self._ensure_loaded()
        return len(self._synapses)

    def has_chunk(self, chunk_id: str) -> bool:
        self._ensure_loaded()
        if chunk_id in self._synapses:
            return True
        # Check SQLite on cache miss
        try:
            row = self._conn.execute(
                "SELECT 1 FROM synapses WHERE chunk_id=?", (chunk_id,)
            ).fetchone()
            return row is not None
        except Exception:
            return False

    def get_all_ids(self) -> List[str]:
        self._ensure_loaded()
        return list(self._synapses.keys())

    def sync_with_collection(self, collection) -> int:
        """Remove graph entries for chunks that no longer exist in ChromaDB.
        Returns number of evicted chunks."""
        self._ensure_loaded()
        try:
            db_ids = set(self._synapses.keys())
            if not db_ids:
                return 0
            # Sample: ChromaDB returns all IDs — use get() with limit chunks
            results = collection.get(limit=len(db_ids))
            chroma_ids = set(results.get("ids", []))
            orphaned = list(db_ids - chroma_ids)
            if orphaned:
                logger.info("Graph sync: evicting %d orphans from meta-graph", len(orphaned))
                return self.evict(orphaned)
            return 0
        except Exception as exc:
            logger.debug("Graph sync failed: %s", exc)
            return 0

    def garbage_collect(self, max_evict: int = 200) -> int:
        """Evict old/low-utility chunks. Call during nightly maintenance.
        Returns number evicted."""
        candidates = self.compute_eviction_candidates()
        if not candidates:
            return 0
        to_evict = candidates[:max_evict]
        return self.evict(to_evict)

    def close(self) -> None:
        self.flush()
        if self._conn:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_graph: Optional[MetaGraph] = None
_graph_lock = threading.Lock()


def get_graph(db_path: Optional[str] = None) -> MetaGraph:
    global _graph
    if _graph is not None:
        return _graph
    with _graph_lock:
        if _graph is not None:
            return _graph
        _graph = MetaGraph(db_path)
    return _graph
