"""Skill Vector Store: Embed and search skills by content similarity.

Uses the same bge-small-en-v1.5 embedding model as memory but stores
in a separate ChromaDB collection (hermes_skills). Skills are chunked
by ## section headers for granular retrieval.

Zero LLM calls. Pure vector search.
"""

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "hermes_skills"
SKILL_CACHE_FILE = ".recent.json"
MAX_MRU_SIZE = 10

# ---------------------------------------------------------------------------
# Confidence threshold
# ---------------------------------------------------------------------------

# Very low threshold — having some skills is better than having none.
# Only filters out complete garbage (score < 0.2).
MIN_CONFIDENCE_THRESHOLD = 0.2

# ---------------------------------------------------------------------------
# Emotional query detection — route to companion
# ---------------------------------------------------------------------------

_EMOTIONAL_PATTERNS = [
    r"\b(sad|depressed|anxious|stressed|angry|upset|lonely|tired|exhausted|overwhelmed|frustrated|worried|scared|hurt)\b",
    r"\b(happy|excited|grateful|proud|relieved|content|joyful|glad|cheerful)\b",
    r"\bi (feel|am feeling|'m feeling)\b",
    r"\bfeeling\s+(a bit|a little|very|really|so|kind of)\b",
    r"\bneed (someone to talk|support|to vent|a friend|comfort)\b",
]


def _is_emotional_query(query: str) -> bool:
    """Check if query is about emotional support."""
    q = query.lower()
    return any(re.search(p, q) for p in _EMOTIONAL_PATTERNS)

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_client = None
_collection = None
_embed_fn = None
_lock = threading.Lock()


def _get_hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home()
    except Exception:
        return Path.home() / ".hermes"


def _get_embed_fn():
    """Return the shared embedding function (reuses vector-context's)."""
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn
    with _lock:
        if _embed_fn is not None:
            return _embed_fn
        # Import works both as package member and standalone module
        try:
            from .spreading import _CachedEmbeddingFunction
        except ImportError:
            from spreading import _CachedEmbeddingFunction
        _embed_fn = _CachedEmbeddingFunction()
    return _embed_fn


def _get_collection():
    """Return the skills ChromaDB collection (lazy init)."""
    global _client, _collection
    if _collection is not None:
        return _collection
    # Get embed fn first (has its own lock)
    embed_fn = _get_embed_fn()
    with _lock:
        if _collection is not None:
            return _collection
        import chromadb
        data_path = str(_get_hermes_home() / "vector_store")
        _client = chromadb.PersistentClient(path=data_path)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )
        logger.info("Skill vector store ready (%d docs)", _collection.count())
    return _collection


# ---------------------------------------------------------------------------
# Skill file discovery
# ---------------------------------------------------------------------------


def _find_all_skill_files() -> List[Tuple[str, str, str]]:
    """Find all SKILL.md files across active, archive, and hub dirs.

    Returns list of (skill_name, file_path, source) tuples.
    Deduplicates by skill name: active > archive > hub.
    """
    hermes_home = _get_hermes_home()
    skills_dir = hermes_home / "skills"
    seen_names = set()
    results = []

    # Active skills (highest priority)
    if skills_dir.exists():
        for skill_dir in sorted(skills_dir.iterdir()):
            if skill_dir.is_dir() and not skill_dir.name.startswith("."):
                skill_md = skill_dir / "SKILL.md"
                if skill_md.exists():
                    results.append((skill_dir.name, str(skill_md), "active"))
                    seen_names.add(skill_dir.name)

    # Archive skills (may be nested)
    archive_dir = skills_dir / ".archive"
    if archive_dir.exists():
        for item in archive_dir.rglob("SKILL.md"):
            skill_name = item.parent.name
            if skill_name not in seen_names:
                results.append((skill_name, str(item), "archive"))
                seen_names.add(skill_name)

    # Hub skills (downloaded from marketplace)
    hub_dir = skills_dir / ".hub"
    if hub_dir.exists():
        for item in hub_dir.rglob("SKILL.md"):
            skill_name = item.parent.name
            if skill_name not in seen_names:
                results.append((skill_name, str(item), "hub"))
                seen_names.add(skill_name)

    return results


# ---------------------------------------------------------------------------
# Skill chunking by ## sections
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from skill content."""
    return _FRONTMATTER_RE.sub("", content, count=1)


def _chunk_skill(file_path: str, skill_name: str, source: str) -> List[Dict[str, Any]]:
    """Chunk a SKILL.md file by ## section headers.

    Returns list of {id, text, metadata} dicts.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (OSError, IOError) as e:
        logger.warning("Failed to read skill '%s': %s", skill_name, e)
        return []

    content = _strip_frontmatter(content).strip()
    if not content:
        return []

    chunks = []
    # Split by ## headers
    sections = re.split(r"(?=^## )", content, flags=re.MULTILINE)

    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        # Extract section title
        title_match = re.match(r"^## (.+)$", section, re.MULTILINE)
        section_title = title_match.group(1).strip() if title_match else f"section_{i}"

        # Sanitize section title for use as ID
        safe_title = re.sub(r"[^a-zA-Z0-9_-]", "_", section_title.lower())
        # Include source and sequence to avoid ID collisions
        safe_source = re.sub(r"[^a-zA-Z0-9]", "_", source)[:8]
        chunk_id = f"{skill_name}:{safe_source}:{safe_title}:{i}"

        metadata = {
            "type": "skill",
            "name": skill_name,
            "section": section_title,
            "source": source,
            "file_path": file_path,
        }

        chunks.append({
            "id": chunk_id,
            "text": section,
            "metadata": metadata,
        })

    # If no ## sections found, treat whole file as one chunk
    if not chunks:
        safe_source = re.sub(r"[^a-zA-Z0-9]", "_", source)[:8]
        chunks.append({
            "id": f"{skill_name}:{safe_source}:full:0",
            "text": content,
            "metadata": {
                "type": "skill",
                "name": skill_name,
                "section": "full",
                "source": source,
                "file_path": file_path,
            },
        })

    return chunks


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def index_all_skills() -> int:
    """Index all SKILL.md files into the vector store.

    Returns count of chunks indexed.
    """
    collection = _get_collection()
    skill_files = _find_all_skill_files()

    all_chunks = []
    for skill_name, file_path, source in skill_files:
        chunks = _chunk_skill(file_path, skill_name, source)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.info("No skill files found to index")
        return 0

    # Batch upsert
    ids = [c["id"] for c in all_chunks]
    documents = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    batch_size = 4500
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_docs = documents[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
        )

    logger.info("Indexed %d skill chunks from %d files", len(all_chunks), len(skill_files))
    return len(all_chunks)


def index_skill(skill_name: str, file_path: str, source: str = "active") -> int:
    """Index a single skill into the vector store.

    Returns count of chunks indexed. Returns 0 if file doesn't exist.
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning("Skill file not found: %s", file_path)
        return 0

    collection = _get_collection()

    # Remove old chunks for this skill first (handles content changes)
    try:
        old = collection.get(where={"name": skill_name})
        if old and old["ids"]:
            collection.delete(ids=old["ids"])
    except Exception:
        pass

    chunks = _chunk_skill(file_path, skill_name, source)
    if not chunks:
        return 0

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Indexed skill '%s' (%d chunks)", skill_name, len(chunks))
    return len(chunks)


def remove_skill(skill_name: str) -> int:
    """Remove all chunks for a skill from the vector store.

    Also removes from MRU cache.
    Returns count of chunks removed.
    """
    collection = _get_collection()

    try:
        results = collection.get(where={"name": skill_name})
    except Exception:
        return 0

    if results and results["ids"]:
        collection.delete(ids=results["ids"])
        # Clean MRU cache
        recent = load_recent()
        recent = [s for s in recent if s != skill_name]
        save_recent(recent)
        logger.info("Removed skill '%s' (%d chunks)", skill_name, len(results["ids"]))
        return len(results["ids"])
    return 0


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_skills(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search skills by content similarity.

    Returns list of {name, section, score, source} dicts.
    Score is 0-1 where higher is more similar.
    """
    if not query or not query.strip():
        return []

    collection = _get_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(max_results * 2, 20),
        include=["metadatas", "distances"],
    )

    if not results or not results["metadatas"]:
        return []

    # Deduplicate by skill name, keep highest scoring section
    seen = {}
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        name = meta["name"]
        # ChromaDB default is L2 distance. Convert to similarity score.
        # For L2: score = 1 / (1 + dist) — ranges from 0 (far) to 1 (identical)
        score = 1.0 / (1.0 + dist)

        if name not in seen or score > seen[name]["score"]:
            seen[name] = {
                "name": name,
                "section": meta.get("section", "full"),
                "score": round(score, 4),
                "source": meta.get("source", "unknown"),
            }

    ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:max_results]


def get_top_skills(query: str, count: int = 3) -> List[str]:
    """Get top N skill names by similarity to query.

    Returns list of skill name strings.
    """
    results = search_skills(query, max_results=count)
    return [r["name"] for r in results]


def get_contextual_skills(query: str, count: int = 3) -> List[str]:
    """Get skills with emotional routing and confidence filtering.

    - Emotional queries: always route to companion (useful for users).
    - Normal queries: inject if score >= 0.2 (very low threshold).
    """
    if not query or not query.strip():
        return []

    # Emotional queries always get companion — it's useful
    if _is_emotional_query(query):
        return ["companion"]

    results = search_skills(query, max_results=count)

    if not results:
        return []

    # Filter by confidence threshold (very low — any match is better than none)
    confident = [r for r in results if r["score"] >= MIN_CONFIDENCE_THRESHOLD]

    if not confident:
        return []

    return [r["name"] for r in confident]


# ---------------------------------------------------------------------------
# MRU Cache (recently used skills)
# ---------------------------------------------------------------------------


def _get_recent_file() -> Path:
    return _get_hermes_home() / SKILL_CACHE_FILE


def load_recent() -> List[str]:
    """Load MRU skill list from disk."""
    recent_file = _get_recent_file()
    if not recent_file.exists():
        return []
    try:
        data = json.loads(recent_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [s for s in data if isinstance(s, str)][:MAX_MRU_SIZE]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def save_recent(skills: List[str]):
    """Save MRU skill list to disk (thread-safe)."""
    recent_file = _get_recent_file()
    trimmed = [s for s in skills if isinstance(s, str)][:MAX_MRU_SIZE]
    recent_file.write_text(json.dumps(trimmed), encoding="utf-8")


def add_to_recent(skill_name: str):
    """Add a skill to the MRU cache (most recent first, thread-safe)."""
    with _lock:
        recent = load_recent()
        recent = [s for s in recent if s != skill_name]
        recent.insert(0, skill_name)
        save_recent(recent)


def get_recent_skills() -> List[str]:
    """Get MRU skill list."""
    return load_recent()


# ---------------------------------------------------------------------------
# Skill path resolution
# ---------------------------------------------------------------------------


def resolve_skill_path(skill_name: str) -> Optional[str]:
    """Resolve skill path across active, archive, and hub.

    Returns file path or None if not found.
    Priority: active > archive (flat) > archive (nested) > hub
    """
    hermes_home = _get_hermes_home()
    skills_dir = hermes_home / "skills"

    # 1. Active directory
    active_path = skills_dir / skill_name / "SKILL.md"
    if active_path.exists():
        return str(active_path)

    # 2. Archive (flat)
    archive_path = skills_dir / ".archive" / skill_name / "SKILL.md"
    if archive_path.exists():
        return str(archive_path)

    # 3. Archive (nested - only match exact directory name)
    archive_dir = skills_dir / ".archive"
    if archive_dir.exists():
        for category_dir in archive_dir.iterdir():
            if category_dir.is_dir():
                nested_path = category_dir / skill_name / "SKILL.md"
                if nested_path.exists():
                    return str(nested_path)

    # 4. Hub
    hub_dir = skills_dir / ".hub"
    if hub_dir.exists():
        for item in hub_dir.iterdir():
            if item.is_dir():
                hub_path = item / "SKILL.md"
                if hub_path.exists() and item.name == skill_name:
                    return str(hub_path)

    return None


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def init_skill_store(embed_fn=None):
    """Initialize the skill vector store on startup.
    
    Args:
        embed_fn: Optional shared embedding function to avoid loading model twice.
    """
    global _embed_fn
    if embed_fn is not None:
        _embed_fn = embed_fn
    
    collection = _get_collection()
    
    # Only index if collection is empty (first run)
    if collection.count() == 0:
        try:
            count = index_all_skills()
            logger.info("Skill vector store initialized with %d chunks", count)
        except Exception as e:
            logger.error("Failed to initialize skill vector store: %s", e)
    else:
        logger.info("Skill vector store ready (%d docs)", collection.count())
