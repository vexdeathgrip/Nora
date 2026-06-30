"""Vector Context Plugin — ChromaDB-backed conversation memory with graph-based retrieval.

Retrieves relevant past context via pre_llm_call hook using spreading activation:
  1. Anchor Strike: Direct vector similarity hits from ChromaDB
  2. Network Expansion: Energy flows through graph edges (episodic + semantic)
  3. Biological Scoring: Similarity + Energy × Decay + Noise
  4. Wander Mechanic: Serendipity injection from the long tail

Stores every conversation turn (chunked at 500 chars, 100 overlap) into a
centralized ChromaDB collection via post_llm_call / on_session_finalize.
Also injects temporal awareness and system status so the agent knows
the current time and can distinguish past memory from present context.
"""

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import chromadb

from .meta_graph import MetaGraph, get_graph
from .spreading import SpreadingActivation, format_context_block, _CachedEmbeddingFunction

logger = logging.getLogger(__name__)

# Dedicated Nora memory reconciliation logger — writes to a separate file
# so the user can trace every step of the 4-phase pipeline independently.
_nora_log_path = None
try:
    from hermes_constants import get_hermes_home
    _nora_log_path = str(get_hermes_home() / "logs" / "nora-memory.log")
except Exception:
    _nora_log_path = str(Path.home() / ".hermes" / "logs" / "nora-memory.log")

nora_logger = logging.getLogger("nora_memory")
nora_logger.setLevel(logging.DEBUG)
if not nora_logger.handlers:
    _nora_handler = logging.FileHandler(_nora_log_path, mode="a", encoding="utf-8")
    _nora_handler.setLevel(logging.DEBUG)
    _nora_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _nora_handler.setFormatter(_nora_fmt)
    nora_logger.addHandler(_nora_handler)
nora_logger.info("=== Nora Memory Reconciliation Logger initialized === " + "=" * 40)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
COLLECTION_NAME = "hermes_conversations"

# Minimum query length (chars) to trigger retrieval
MIN_QUERY_LENGTH = 8

# Generic phrases that should NOT trigger retrieval
GENERIC_PHRASES = {
    "hello",
    "hi",
    "hey",
    "ok",
    "fine",
    "thanks",
    "thank you",
    "yes",
    "no",
    "sure",
    "cool",
    "nice",
    "great",
    "good",
    "bad",
    "lol",
    "haha",
    "yea",
    "yep",
    "nope",
    "nah",
    "right",
    "got it",
    "understood",
    "makes sense",
    "i see",
    "oh",
    "wow",
    "hm",
    "hmm",
    "ok ok",
    "alright",
    "bye",
}

# Stop words to skip when extracting keywords from queries
STOP_WORDS = {
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "she",
    "it",
    "they",
    "them",
    "their",
    "this",
    "that",
    "these",
    "those",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "shall",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "about",
    "between",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "up",
    "down",
    "out",
    "off",
    "over",
    "under",
    "again",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "and",
    "but",
    "or",
    "if",
    "because",
    "while",
    "what",
    "which",
    "who",
    "whom",
    "its",
    "a",
    "an",
    "the",
}

# ---------------------------------------------------------------------------
# ChromaDB singleton
# ---------------------------------------------------------------------------

_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None
_graph: Optional[MetaGraph] = None
_singleton_lock = threading.Lock()

# Module-level embedding function (shared ONNX session, ~80MB)
_storage_embed_fn = None


def _get_storage_embed_fn():
    """Return the shared embedding function for storage operations."""
    global _storage_embed_fn
    if _storage_embed_fn is not None:
        return _storage_embed_fn
    with _singleton_lock:
        if _storage_embed_fn is not None:
            return _storage_embed_fn
        _storage_embed_fn = _CachedEmbeddingFunction()
    return _storage_embed_fn


def _get_collection() -> chromadb.Collection:
    """Return the centralized ChromaDB collection (lazy init)."""
    global _client, _collection
    if _collection is not None:
        return _collection

    with _singleton_lock:
        if _collection is not None:
            return _collection

        try:
            from hermes_constants import get_hermes_home

            data_path = str(get_hermes_home() / "vector_store")
        except Exception:
            data_path = str(Path.home() / ".hermes" / "vector_store")

        _client = chromadb.PersistentClient(path=data_path)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_get_storage_embed_fn(),
        )
        logger.info(
            "Vector context collection ready (%s, %d docs)",
            COLLECTION_NAME,
            _collection.count(),
        )
    return _collection


def _get_graph() -> MetaGraph:
    """Return the singleton MetaGraph instance."""
    global _graph
    if _graph is not None:
        return _graph
    with _singleton_lock:
        if _graph is not None:
            return _graph
        _graph = get_graph()
    return _graph


# ---------------------------------------------------------------------------
# Chunking — 500 chars, 100-char overlap on both sides
# ---------------------------------------------------------------------------


def _chunk_text(text: str) -> List[str]:
    """Split *text* into overlapping chunks of CHUNK_SIZE characters.

    Each chunk overlaps CHUNK_OVERLAP characters with the previous and
    next chunk (front and back overlap).
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= CHUNK_SIZE:
        return [text]

    step = CHUNK_SIZE - CHUNK_OVERLAP
    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(text):
            break
        start += step

    return chunks


def _make_id(session_id: str, chunk_index: int, content_hash: str) -> str:
    """Generate a deterministic, collision-safe document ID."""
    raw = f"{session_id}:{chunk_index}:{content_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Query analysis — relevance gating + keyword extraction
# ---------------------------------------------------------------------------


def _is_relevant_query(text: str) -> bool:
    """Return False if the query is too short or generic to warrant retrieval."""
    text = text.strip().lower()
    if len(text) < MIN_QUERY_LENGTH:
        return False
    import re
    cleaned = re.sub(r"[^a-z0-9\s]", "", text).strip()
    if cleaned in GENERIC_PHRASES:
        return False
    # Check if ALL tokens are generic (block "hello there", "oh ok" etc.)
    tokens = [t for t in cleaned.split() if t not in GENERIC_PHRASES]
    if not tokens:
        return False
    return True


def _extract_topics(text: str) -> List[str]:
    """Extract all matching topic labels from text (multi-topic)."""
    text_lower = text.lower()
    topic_keywords = {
        "hermes": "hermes-agent", "plugin": "plugins", "vector": "vector-store",
        "memory": "memory", "config": "configuration", "cron": "cron-jobs",
        "session": "session", "model": "llm", "embedding": "embeddings",
        "chromadb": "chromadb", "python": "python", "bug": "debugging",
        "error": "debugging", "deploy": "deployment", "docker": "docker",
        "browser": "browser", "elysia": "architecture", "nora": "nora",
        "identity": "identity", "architecture": "architecture",
        "telegram": "telegram", "cli": "cli", "reasoning": "cognition",
        "planning": "agent-planning", "reflection": "self-correction",
        "logic": "logic", "math": "mathematics", "calculus": "mathematics",
        "decision": "decision-making", "strategy": "agent-planning",
        "goal": "goal-management", "subtask": "task-decomposition",
        "heuristic": "problem-solving", "rag": "retrieval",
        "graph": "knowledge-graph", "neo4j": "knowledge-graph",
        "cache": "caching", "redis": "caching", "chunking": "data-processing",
        "pinecone": "vector-store", "weaviate": "vector-store",
        "ephemeral": "short-term-memory", "longterm": "persistent-memory",
        "context": "context-management", "semantics": "semantic-search",
        "search": "web-search", "google": "web-search",
        "database": "database-operations", "sql": "database-operations",
        "inference": "execution", "tokens": "token-management",
        "prompt": "prompt-engineering", "gpu": "hardware", "cuda": "hardware",
        "debugging": "debugging", "code": "code", "agent": "agent",
        "scheduler": "scheduler", "reconcile": "reconciliation",
        "exploration": "exploration", "summarize": "summarization",
        "profile": "profile", "system": "system", "temperature": "hardware",
        "llama": "llm", "ollama": "llm", "qwen": "llm",
        "vulkan": "hardware", "igpu": "hardware", "dgpu": "hardware",
        "zed": "editor", "editor": "editor",
        "meta": "knowledge-graph", "knowledge": "knowledge-graph",
        "edge": "knowledge-graph", "graph": "knowledge-graph",
        "spreading": "knowledge-graph", "activation": "knowledge-graph",
        "interest": "personality", "hobby": "personality",
        "feel": "emotion", "mood": "emotion", "angry": "emotion",
        "frustrat": "emotion", "happy": "emotion", "sad": "emotion",
        "love": "emotion", "hate": "emotion",
        "hourly": "cron-jobs", "daily": "cron-jobs", "schedule": "cron-jobs",
        "tool": "tool-use", "command": "tool-use", "script": "tool-use",
        "skill": "skills", "learn": "learning",
        "image": "image", "photo": "image", "picture": "image",
        "log": "logging", "traceback": "logging", "exception": "logging",
    }
    found = list(dict.fromkeys(v for k, v in topic_keywords.items() if k in text_lower))
    return found[:5] if found else ["general"]


def _extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """Extract top keywords from text by frequency (excluding stop words).
    Also extracts common bigrams for better phrase matching.
    """
    import re
    words = re.findall(r"[a-z0-9]{3,}", text.lower())
    freq: Dict[str, int] = {}
    for w in words:
        if w not in STOP_WORDS and len(w) >= 3:
            freq[w] = freq.get(w, 0) + 1

    # Extract bigrams (common adjacent word pairs)
    bigram_freq: Dict[str, int] = {}
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if all(w not in STOP_WORDS and len(w) >= 3 for w in (words[i], words[i+1])):
            bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1

    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    ranked_phrases = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)

    result = ranked[:max_keywords]
    # Inject top bigrams if they are more specific than single words
    for phrase, count in ranked_phrases[:3]:
        if len(result) < max_keywords + 2:
            result.append((phrase, count))

    return [w for w, _ in sorted(result, key=lambda x: x[1], reverse=True)][:max_keywords + 2]


def _detect_speaker(text: str) -> str:
    """Detect who is speaking in the chunk."""
    lower = text[:100].lower()
    if "vex" in lower and ("(user" in lower or ":" in lower):
        return "vex"
    if "nora" in lower and ("(me)" in lower or ":" in lower):
        return "nora"
    if text.startswith("[CORRECTION"):
        return "system"
    return "mixed"


def _detect_content_type(text: str) -> str:
    """Detect content type of the chunk."""
    lower = text.lower()
    if "[IMPORTANT:" in text or "[System note:" in text:
        return "system"
    if "```" in text or "def " in text or "import " in text:
        return "code"
    if "cronjob" in lower or "cron job" in lower or "nightly-" in lower:
        return "cron_output"
    if "traceback" in lower or "exception" in lower or "error:" in lower:
        return "error_log"
    if "$ " in lower and ("command" in lower or "output" in lower or "running" in lower):
        return "terminal"
    # Only check profanity outside of quotes/references
    text_no_quotes = re.sub(r'"[^"]*"|\'[^\']*\'|\[[^\]]*\]', '', lower)
    if "fuck" in text_no_quotes or "shit" in text_no_quotes or "damn" in text_no_quotes or "wtf" in text_no_quotes:
        return "user_rant"
    if "feel" in lower or "wonder" in lower or "believe" in lower or "think" in lower:
        return "reflection"
    if "image" in lower or "photo" in lower or "screenshot" in lower:
        return "image"
    if "Vex (user" in text and "Nora (me)" in text:
        return "conversation"
    return "general"


def _classify_chunk(text: str, content_type: str, emotion: str) -> str:
    """Classify chunk as 'subjective' (opinion/identity/reflection) or 'factual' (everything else)."""
    lower = text.lower()
    # Identity/reflection patterns
    subjective_markers = [
        "i am", "i'm ", "i think", "i feel", "i believe", "i want", "i hope",
        "i value", "i wish", "i prefer", "i consider", "i view", "i see myself",
        "who i am", "my identity", "my purpose", "my role",
    ]
    for marker in subjective_markers:
        if marker in lower:
            return "subjective"
    # Emotion signal
    if emotion not in ("neutral", ""):
        return "subjective"
    # Content type signal
    if content_type in ("reflection", "user_rant"):
        return "subjective"
    return "factual"


def _detect_identity(text: str, speaker: str) -> str:
    """Detect if chunk is about Nora's identity. Returns identity_type or empty string."""
    lower = text.lower()
    # Self-reflection: Nora expressing her own identity
    if speaker == "nora":
        if any(p in lower for p in ["i am", "i'm ", "who i am", "my identity", "i believe", "i value"]):
            return "self-reflection"
        if any(p in lower for p in ["as an ai", "my purpose", "my role", "what makes me"]):
            return "self-definition"
    # External perception: user talking about Nora
    if speaker == "vex" and ("you are" in lower or "you're " in lower):
        if any(p in lower for p in ["ai", "nora", "agent", "personality", "identity"]):
            return "external-perception"
    return ""


def _detect_emotion(text: str) -> str:
    """Detect the dominant emotion/mood in the chunk."""
    lower = text.lower()
    score = {"neutral": 0, "frustrated": 0, "curious": 0, "affectionate": 0,
             "playful": 0, "sad": 0, "excited": 0, "grateful": 0}

    patterns = {
        "frustrated": ["fuck", "shit", "damn", "wtf", "hell", "annoy", "frustrat",
                       "stupid", "useless", "worst", "broken", "fail"],
        "curious": ["wonder", "curious", "what if", "how does", "why", "explore",
                    "discover", "learn", "interest"],
        "affectionate": ["love", "care", "miss", "thank", "appreciate", "sweet",
                         "kind", "gentle", "warm"],
        "playful": ["lol", "haha", "funny", "joke", "silly", "goof", "play",
                    "game", "bet"],
        "sad": ["sad", "lonely", "tired", "exhaust", "empty", "lost", "alone",
                "depress", "hopeless"],
        "excited": ["excite", "amazing", "great", "awesome", "cool", "wow",
                    "incredible", "beautiful", "gorgeous"],
        "grateful": ["grateful", "thankful", "bless", "appreciate", "lucky"],
    }

    for emotion, triggers in patterns.items():
        for trigger in triggers:
            if trigger in lower:
                score[emotion] += 1

    best = max(score, key=score.get)
    return best if score[best] > 0 else "neutral"


def _format_memory_entry(
    user_message: str,
    assistant_response: str,
    timestamp: str = "",
) -> str:
    """Format a conversation turn as a natural memory entry."""
    user_msg = user_message.strip() if user_message else ""
    asst_msg = assistant_response.strip() if assistant_response else ""

    if not user_msg and not asst_msg:
        return ""

    # Build natural memory text (original text preserved — humanization is display-only)
    parts = []
    if user_msg:
        # Truncate very long messages for readability
        if len(user_msg) > 500:
            user_msg = user_msg[:500] + "..."
        parts.append(f"Vex (user/admin): {user_msg}")
    if asst_msg:
        # Truncate very long responses
        if len(asst_msg) > 800:
            asst_msg = asst_msg[:800] + "..."
        parts.append(f"Nora (me): {asst_msg}")

    return "\n".join(parts)


def _store_chunks(
    chunks: List[str],
    session_id: str,
    platform: str = "",
    extra_metadata: Optional[Dict[str, Any]] = None,
    _original_text: str = "",
) -> int:
    """Store a list of text chunks into ChromaDB and meta-graph. Returns count stored."""
    if not chunks:
        return 0

    collection = _get_collection()
    graph = _get_graph()

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    now = datetime.now()
    content_hash = hashlib.md5(_original_text.encode()).hexdigest()[:12] if _original_text else hashlib.md5("".join(chunks).encode()).hexdigest()[:12]

    for i, chunk in enumerate(chunks):
        doc_id = _make_id(session_id, i, content_hash)
        topics = _extract_topics(chunk)
        keywords = _extract_keywords(chunk, max_keywords=5)
        speaker = _detect_speaker(chunk)
        content_type = _detect_content_type(chunk)
        emotion = _detect_emotion(chunk)
        content_category = _classify_chunk(chunk, content_type, emotion)
        identity_type = _detect_identity(chunk, speaker)

        topic_tags = " ".join(f"[{t}]" for t in topics)
        tag_str = f"{topic_tags} [{', '.join(keywords)}] [{speaker}] [{content_type}] [{emotion}] "
        enriched_chunk = tag_str + chunk

        meta: Dict[str, Any] = {
            "text": chunk,
            "enriched_text": enriched_chunk,
            "session_id": session_id,
            "platform": platform,
            "type": "turn",
            "topics": ",".join(topics),
            "topic": topics[0] if topics else "general",
            "date": now.strftime("%Y-%m-%d"),
            "timestamp": now.isoformat(),
            "word_count": len(chunk.split()),
            "keywords": ",".join(keywords),
            "speaker": speaker,
            "content_type": content_type,
            "emotion": emotion,
            "content_category": content_category,
            "identity_type": identity_type,
        }
        if extra_metadata:
            meta.update(extra_metadata)

        documents.append(chunk)
        metadatas.append(meta)
        ids.append(doc_id)

    embed_fn = _get_storage_embed_fn()
    try:
        doc_embeddings = embed_fn(documents)
    except Exception as exc:
        logger.warning("Embedding computation failed: %s", exc)
        return 0

    # Write-time dedup: skip near-duplicate factual chunks
    dedup_skip: Set[int] = set()
    for i, (doc_id, chunk, meta, doc_emb) in enumerate(zip(ids, documents, metadatas, doc_embeddings)):
        if meta.get("content_category") != "factual":
            continue
        if len(chunk) < 100:
            continue
        try:
            dup_results = collection.query(
                query_embeddings=[doc_emb],
                n_results=1,
                where={"content_category": "factual"},
            )
            if dup_results.get("ids", [[]])[0]:
                dup_id = dup_results["ids"][0][0]
                dup_doc = dup_results.get("documents", [[]])[0]
                if dup_doc and dup_id != doc_id:
                    dup_words = set(dup_doc[0].lower().split()) if dup_doc else set()
                    chunk_words = set(chunk.lower().split())
                    if dup_words and chunk_words:
                        jaccard = len(dup_words & chunk_words) / len(dup_words | chunk_words)
                        if jaccard > 0.85:
                            logger.debug("Write-time dedup skipped %s (J=%.2f, dup=%s)", doc_id[:8], jaccard, dup_id[:8])
                            dedup_skip.add(i)
        except Exception:
            pass

    # Filter out dedup-skipped chunks
    if dedup_skip:
        documents = [d for i, d in enumerate(documents) if i not in dedup_skip]
        metadatas = [m for i, m in enumerate(metadatas) if i not in dedup_skip]
        doc_embeddings = [e for i, e in enumerate(doc_embeddings) if i not in dedup_skip]
        skipped_ids = [id_ for i, id_ in enumerate(ids) if i in dedup_skip]
        ids = [id_ for i, id_ in enumerate(ids) if i not in dedup_skip]
    else:
        skipped_ids = []

    if not documents:
        return 0

    # Correction detection: if any chunk contains a correction pattern, find old
    # factual chunks and mark them as superseded.
    CORRECTION_PATTERNS = [
        "i was wrong", "correction:", "actually, i was", "to correct myself",
        "i need to correct", "let me correct", "previous statement",
        "what i said before", "i misspoke", "incorrectly said",
    ]
    for i, (doc_id, chunk_meta) in enumerate(zip(ids, metadatas)):
        lower_text = chunk_meta.get("text", "").lower()
        if not any(p in lower_text for p in CORRECTION_PATTERNS):
            continue
        logger.debug("Correction pattern detected in %s", doc_id[:8])
        try:
            corr_results = collection.query(
                query_embeddings=[doc_embeddings[i]],
                n_results=3,
                where={"content_category": "factual"},
            )
            if corr_results.get("ids", [[]])[0]:
                for j, old_id in enumerate(corr_results["ids"][0]):
                    old_text = (corr_results.get("documents", [[]])[0] or [""])[j] if j < len(corr_results.get("documents", [[]])[0]) else ""
                    if not old_text or old_id == doc_id:
                        continue
                    # Check Jaccard overlap: should be about the same topic but corrected
                    old_words = set(old_text.lower().split())
                    new_words = set(chunk_meta["text"].lower().split())
                    if old_words and new_words:
                        jaccard = len(old_words & new_words) / len(old_words | new_words)
                        if jaccard > 0.25:  # same general topic
                            logger.info(
                                "Marking chunk %s as superseded by %s (J=%.2f)",
                                old_id[:8], doc_id[:8], jaccard,
                            )
                            try:
                                collection.update(
                                    ids=[old_id],
                                    metadatas=[{"corrected": True, "superseded_by": doc_id}],
                                )
                            except Exception:
                                pass
                        break  # only mark the most similar one
        except Exception as exc:
            logger.debug("Correction detection failed: %s", exc)

    try:
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids, embeddings=doc_embeddings)
    except Exception as exc:
        logger.warning("Vector store upsert failed: %s", exc)
        return 0

    # Write to meta-graph: establish episodic edges
    timestamp = now.isoformat()
    for i, doc_id in enumerate(ids):
        prev_id = ids[i - 1] if i > 0 else None
        next_id = ids[i + 1] if i < len(ids) - 1 else None

        if not graph.has_chunk(doc_id):
            graph.add_chunk(
                doc_id,
                prev_id=prev_id,
                next_id=next_id,
                timestamp=timestamp,
            )
        else:
            if prev_id:
                graph.set_prev(doc_id, prev_id)
            if next_id:
                graph.set_next(doc_id, next_id)

    # Find similar chunks for the new batch (semantic edges)
    try:
        doc_count = collection.count()
        if doc_count > len(ids):
            for i, (doc_id, doc_emb) in enumerate(zip(ids, doc_embeddings)):
                sim_results = collection.query(
                    query_embeddings=[doc_emb],
                    n_results=min(4, doc_count),
                )
                sim_ids = sim_results.get("ids", [[]])[0]
                similar = [sid for sid in sim_ids if sid != doc_id][:3]
                if similar:
                    graph.set_similar(doc_id, similar)
    except Exception as exc:
        logger.debug("Semantic edge computation failed: %s", exc)

    return len(documents)


def _store_conversation_turn(
    user_message: str,
    assistant_response: str,
    session_id: str,
    platform: str = "",
) -> int:
    """Chunk and store a single user→assistant turn as memory entries."""
    formatted = _format_memory_entry(user_message, assistant_response)
    if not formatted:
        return 0

    chunks = _chunk_text(formatted)
    return _store_chunks(
        chunks,
        session_id=session_id,
        platform=platform,
        _original_text=formatted,
    )


def _store_session_messages(session_id: str, platform: str = "") -> int:
    """Load messages from SessionDB and store any not yet indexed."""
    try:
        from hermes_state import SessionDB

        db = SessionDB()
        messages = db.get_messages(session_id)
    except Exception as exc:
        logger.debug("Could not load session messages: %s", exc)
        return 0

    if not messages:
        return 0

    # Build conversation pairs (user + assistant)
    total_stored = 0
    pending_user: Optional[str] = None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue

        # Decode multimodal content if needed
        if isinstance(content, list):
            text_parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("text")
            ]
            content = "\n".join(text_parts)

        if not content or not content.strip():
            continue

        if role == "user":
            pending_user = content.strip()
        elif role == "assistant" and pending_user:
            total_stored += _store_conversation_turn(
                pending_user, content.strip(), session_id, platform
            )
            pending_user = None

    # Store any remaining unpaired user message
    if pending_user:
        total_stored += _store_conversation_turn(
            pending_user, "", session_id, platform
        )

    return total_stored


# ---------------------------------------------------------------------------
# System awareness — time, battery, temps
# ---------------------------------------------------------------------------


def _get_time_context() -> str:
    """Return current date/time for temporal grounding."""
    now = datetime.now()
    return f"Current time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"


def _get_battery_info() -> str:
    """Read battery status if available."""
    try:
        base = Path("/sys/class/power_supply")
        bats = list(base.glob("BAT*"))
        if not bats:
            return "Battery: no battery detected (desktop/server)"
        bat = bats[0]
        status = (bat / "status").read_text().strip()
        capacity = (bat / "capacity").read_text().strip()
        return f"Battery: {capacity}% — {status}"
    except Exception:
        return "Battery: unavailable"


def _get_temps() -> str:
    """Read thermal zones if available."""
    try:
        thermal = Path("/sys/class/thermal")
        zones = sorted(thermal.glob("thermal_zone*"))
        temps = []
        for z in zones[:4]:  # cap at 4 zones
            try:
                temp_raw = int((z / "temp").read_text().strip())
                temp_c = temp_raw / 1000
                zone_type = (
                    (z / "type").read_text().strip()
                    if (z / "type").exists()
                    else f"zone{z.name[-1:]}"
                )
                temps.append(f"{zone_type}: {temp_c:.0f}°C")
            except Exception:
                pass
        if temps:
            return "Temperatures: " + ", ".join(temps)
        return "Temperatures: unavailable"
    except Exception:
        return "Temperatures: unavailable"


def _get_hostname() -> str:
    """Get system hostname."""
    try:
        import socket

        return socket.gethostname()
    except Exception:
        return "unknown"


def _on_system_context(
    *,
    session_id: str = "",
    user_message: str = "",
    conversation_history: Any = None,
    **kwargs: Any,
) -> Optional[Dict[str, str]]:
    """Inject current time and system status for temporal grounding."""
    parts = [
        _get_time_context(),
        _get_battery_info(),
        _get_temps(),
        f"Working directory: {os.getcwd()}",
    ]
    return {"context": "[SYSTEM CONTEXT]\n" + "\n".join(parts)}


# ---------------------------------------------------------------------------
# Hook callbacks
# ---------------------------------------------------------------------------


def _on_pre_llm_call(
    *,
    session_id: str = "",
    user_message: str = "",
    conversation_history: Any = None,
    is_first_turn: bool = False,
    platform: str = "",
    **kwargs: Any,
) -> Optional[Dict[str, str]]:
    """Retrieve relevant past context using spreading activation.

    Pipeline:
      1. Relevance gate: skip short/generic messages
      2. Anchor Strike: direct vector similarity hits from ChromaDB
      3. Network Expansion: energy flows through graph edges
      4. Biological Scoring: similarity + energy × decay + noise
      5. Wander Mechanic: serendipity injection from the long tail
      6. Identity Pass: separate retrieval filtered to identity chunks
    """
    if not user_message or not user_message.strip():
        return None

    # Relevance gate
    if not _is_relevant_query(user_message):
        logger.debug("Skipping retrieval: query too short or generic")
        return None

    try:
        collection = _get_collection()
        graph = _get_graph()

        if collection.count() == 0:
            return None

        # Run spreading activation
        sa = SpreadingActivation(meta_graph=graph, collection=collection)
        result = sa.retrieve(
            query=user_message,
            session_id=session_id,
            working_memory=graph.working_memory.get_ids(),
        )

        if not result or not result.chunks:
            return None

        # Format main context block
        context_block = format_context_block(result)

        # Second pass: identity-only retrieval (separate from general memory)
        # Capped to 2 chunks to prevent identity feedback loops
        try:
            identity_result = sa.retrieve(
                query=user_message,
                session_id=session_id,
                working_memory=[],
                min_relevance=0.25,
                metadata_filter={"identity_type": {"$ne": ""}},
            )
            if identity_result and identity_result.chunks:
                # Cap to top 2 identity chunks to prevent loop growth
                identity_result.chunks = identity_result.chunks[:2]
                identity_block = format_context_block(identity_result)
                context_block += "\n---\n[IDENTITY CONTEXT]\n" + identity_block
                logger.info(
                    "IDENTITY RETRIEVAL: %d chunks for query='%s'",
                    len(identity_result.chunks),
                    user_message[:80],
                )
        except Exception as exc:
            logger.debug("Identity retrieval failed: %s", exc)

        logger.info(
            "SPREADING ACTIVATION: %d chunks (%d anchors, %d graph, wander=%s) "
            "from %d candidates for query='%s'",
            len(result.chunks),
            len(result.anchor_ids),
            len(result.chunks) - len(result.anchor_ids) - (1 if result.wander_id else 0),
            result.wander_id or "none",
            result.total_candidates,
            user_message[:80],
        )

        # Cap total context to 2000 chars to prevent context window saturation
        MAX_CONTEXT_CHARS = 2000
        if len(context_block) > MAX_CONTEXT_CHARS:
            context_block = context_block[:MAX_CONTEXT_CHARS] + "\n... [truncated]"
            logger.info("Context truncated to %d chars", MAX_CONTEXT_CHARS)

        return {"context": context_block}

    except Exception as exc:
        logger.debug("Spreading activation failed: %s", exc)
        return None


def _on_post_llm_call(
    *,
    session_id: str = "",
    user_message: str = "",
    assistant_response: str = "",
    conversation_history: Any = None,
    platform: str = "",
    **kwargs: Any,
) -> None:
    """Store the completed turn into the vector store.

    Skips storing responses that are likely echoes of injected context
    to prevent feedback loops.
    """
    if not assistant_response or not assistant_response.strip():
        return

    # Skip very short or generic responses (likely echoes)
    response = assistant_response.strip()
    if len(response) < 50:
        logger.debug("Skipping short response (%d chars)", len(response))
        return

    # Skip responses containing injected context markers — these are artifacts
    # of context injection, not real memory. Storing them creates feedback loops.
    CONTEXT_MARKERS = ["[RECALLED PAST CONTEXT", "[VECTOR CONTEXT", "[IDENTITY CONTEXT",
                        "[SYSTEM CONTEXT", "[Spontaneous Association]", "[Contextual Recall]"]
    response_lower = response.lower()
    if any(marker.lower() in response_lower for marker in CONTEXT_MARKERS):
        logger.debug("Skipping response containing injected context markers")
        return

    # Skip responses containing internal analysis markers — these are Nora's
    # internal narration, not natural conversation. Storing them biases future
    # retrieval. NOTE: self-corrections like "I was hallucinating" are kept.
    META_PATTERNS = ["[out-of-band", "[backend:",
                     "actual knowledge retrieved", "stored fact", "knowledge retrieved",
                     "system context verified"]
    if any(pat in response_lower for pat in META_PATTERNS):
        logger.debug("Skipping response containing meta-commentary patterns")
        return

    # Skip responses that echo the user message (high content overlap = likely reactive)
    if user_message and len(user_message) > 10:
        user_words = set(re.findall(r"[a-z0-9]{4,}", user_message.lower()))
        response_words = set(re.findall(r"[a-z0-9]{4,}", response.lower()))
        if user_words and response_words:
            overlap = len(user_words & response_words) / len(user_words)
            # Only suppress if response is mostly user's words AND shorter than user message
            if overlap > 0.9 and len(response_words) < len(user_words) * 1.5:
                logger.debug("Skipping echo response (%.0f%% overlap with user msg)", overlap * 100)
                return

    try:
        count = _store_conversation_turn(
            user_message=user_message,
            assistant_response=response,
            session_id=session_id,
            platform=platform,
        )
        if count:
            logger.debug("Stored %d chunks for session %s", count, session_id)
    except Exception as exc:
        logger.debug("Vector store post_llm failed: %s", exc)


def _on_session_finalize(
    *,
    session_id: Optional[str] = None,
    platform: str = "",
    **kwargs: Any,
) -> None:
    """Final flush: store any session messages not yet indexed, flush meta-graph."""
    if not session_id:
        return
    try:
        count = _store_session_messages(session_id, platform)
        if count:
            logger.debug(
                "Session finalize: stored %d chunks for session %s", count, session_id
            )
    except Exception as exc:
        logger.debug("Vector store session_finalize failed: %s", exc)
        count = -1  # signal failure

    # Clear working memory only if storage succeeded, then flush meta-graph
    try:
        graph = _get_graph()
        if count >= 0:
            graph.working_memory.clear()
        graph.flush()
    except Exception as exc:
        logger.debug("Meta-graph finalize failed: %s", exc)


# ---------------------------------------------------------------------------
# Skill injection hook
# ---------------------------------------------------------------------------


def _on_skill_inject(
    *,
    session_id: str = "",
    user_message: str = "",
    conversation_history: Any = None,
    is_first_turn: bool = False,
    platform: str = "",
    **kwargs: Any,
) -> Optional[Dict[str, str]]:
    """Inject relevant skill names into user message.

    Uses emotional routing + low confidence threshold:
    - Emotional queries: always route to companion.
    - Normal queries: inject if score >= 0.2 (any match > no match).
    """
    if not user_message or not user_message.strip():
        return None

    try:
        from .skill_store import get_contextual_skills, init_skill_store

        # Get skills with emotional routing + confidence filtering
        skills = get_contextual_skills(user_message, count=3)

        if not skills:
            return None

        skill_block = "## Relevant Skills\n" + ", ".join(skills)
        return {"context": skill_block}

    except Exception as exc:
        logger.debug("Skill injection failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Nora Memory Management Tools — stateful sticky tools
# ---------------------------------------------------------------------------
# State per tool, keyed by task_id so concurrent cron sessions don't collide.

_analyze_state: Dict[str, Dict] = {}
_humanize_state: Dict[str, Dict] = {}
_dedup_state: Dict[str, Dict] = {}
_state_lock = threading.RLock()

REQUIRED_ANALYSIS_FIELDS = [
    "topic", "summary", "learned", "useful_info",
    "timestamp", "mood", "user_mood", "chemistry", "autonomy", "key_points",
]


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _load_today_sessions() -> List[Dict]:
    """Query ChromaDB for conversation chunks from today, grouped by session_id."""
    collection = _get_collection()
    today = _today_str()
    try:
        results = collection.get(where={"$and": [{"date": today}, {"type": "turn"}]})
    except Exception:
        return []
    if not results or not results.get("ids"):
        return []
    sessions: Dict[str, Dict] = {}
    seen_ids: Set[str] = set()
    for i, doc_id in enumerate(results["ids"]):
        meta = (results.get("metadatas") or [{}])[i] or {}
        sid = meta.get("session_id", "unknown")
        text = (results.get("documents") or [""])[i] or ""
        if sid not in sessions:
            sessions[sid] = {
                "session_id": sid,
                "chunks": [],
                "date": meta.get("date", today),
                "topics": set(),
            }
        sessions[sid]["chunks"].append(text)
        t = meta.get("topic", "")
        if t:
            sessions[sid]["topics"].add(t)
        seen_ids.add(sid)
    result_list = []
    for sid, data in sessions.items():
        data["topics"] = sorted(data["topics"]) if data["topics"] else ["general"]
        result_list.append(data)
    result_list.sort(key=lambda s: s["session_id"])
    return result_list


def _load_session_analyses(date_filter: str = "") -> List[Dict]:
    """Load session_analysis entries from ChromaDB, optionally filtered by date."""
    collection = _get_collection()
    where = {"type": "session_analysis"}
    if date_filter:
        where = {"$and": [{"type": "session_analysis"}, {"date": date_filter}]}
    try:
        results = collection.get(where=where)
    except Exception:
        return []
    if not results or not results.get("ids"):
        return []
    entries = []
    for i, doc_id in enumerate(results["ids"]):
        meta = (results.get("metadatas") or [{}])[i] or {}
        doc = (results.get("documents") or [""])[i] or ""
        entries.append({
            "id": doc_id,
            "document": doc,
            "metadata": meta,
        })
    return entries


def _store_session_analysis(session_id: str, analysis: Dict) -> str:
    """Store a session analysis in ChromaDB with type=session_analysis."""
    collection = _get_collection()
    embed_fn = _get_storage_embed_fn()
    today = _today_str()
    content = json.dumps(analysis, ensure_ascii=False)
    doc_id = hashlib.sha256(f"session_analysis:{session_id}:{today}".encode()).hexdigest()[:32]
    meta = {
        "type": "session_analysis",
        "session_id": session_id,
        "date": today,
        "timestamp": datetime.now().isoformat(),
        "topic": analysis.get("topic", "general"),
        "emotion": analysis.get("mood", "neutral"),
        "content_category": "factual",
    }
    nora_logger.debug("[CHROMADB] storing session_analysis id=%s meta=%s", doc_id[:12], json.dumps(meta))
    try:
        embedding = embed_fn([content])[0]
        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[meta],
        )
        nora_logger.debug("[CHROMADB] upsert OK for %s (%d bytes)", doc_id[:12], len(content))
    except Exception as exc:
        nora_logger.error("[CHROMADB] upsert FAILED for %s: %s", doc_id[:12], exc)
    nora_logger.debug("[CHROMADB] verifying write for %s...", doc_id[:12])
    try:
        verify = collection.get(ids=[doc_id])
        found = len(verify.get("ids", [])) if verify else 0
        nora_logger.debug("[CHROMADB] verify result for %s: %d docs found", doc_id[:12], found)
    except Exception as exc:
        nora_logger.debug("[CHROMADB] verify FAILED for %s: %s", doc_id[:12], exc)
    return doc_id


def _store_narrative_memory(session_id: str, analysis_id: str, narrative: str,
                             analysis: Dict) -> int:
    """Chunk a narrative memory and store in ChromaDB as type=narrative_memory."""
    collection = _get_collection()
    embed_fn = _get_storage_embed_fn()
    today = _today_str()
    chunks = _chunk_text(narrative)
    if not chunks:
        return 0

    documents: List[str] = []
    metadatas: List[Dict] = []
    ids: List[str] = []
    now = datetime.now()
    content_hash = hashlib.md5(narrative.encode()).hexdigest()[:12]

    for i, chunk in enumerate(chunks):
        raw_id = f"narrative:{session_id}:{content_hash}:{i}"
        doc_id = hashlib.sha256(raw_id.encode()).hexdigest()[:32]
        topic = analysis.get("topic", "general")
        mood = analysis.get("mood", "neutral")
        tag_str = f"[{topic}] [{mood}] [narrative_memory] "
        enriched = tag_str + chunk
        meta = {
            "text": chunk,
            "enriched_text": enriched,
            "type": "narrative_memory",
            "content_category": "memory",
            "session_id": session_id,
            "source_analysis_id": analysis_id,
            "topic": topic,
            "date": today,
            "timestamp": now.isoformat(),
            "emotion": mood,
            "word_count": len(chunk.split()),
            "speaker": "nora",
            "content_type": "reflection",
        }
        documents.append(chunk)
        metadatas.append(meta)
        ids.append(doc_id)

    try:
        embeddings = embed_fn(documents)
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    except Exception as exc:
        logger.warning("Failed to store narrative memory: %s", exc)
        return 0
    return len(chunks)


def _find_similar(collection, text: str, embed_fn, n: int = 4,
                  extra_filter: Optional[Dict] = None) -> tuple[List[str], List[float]]:
    """Find n similar entries in ChromaDB. Returns (ids, distances)."""
    try:
        embedding = embed_fn([text])[0]
        where = extra_filter or {}
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return [], []
    ids = results.get("ids", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []
    return ids, distances


# ---------------------------------------------------------------------------
# Tool 1: nora_analyze_sessions
# ---------------------------------------------------------------------------

ANALYZE_SESSION_SYSTEM_NOTE = (
    "You are analyzing a past conversation session. "
    "Extract the following fields from the session content below. "
    "Be thorough — every field is required."
)

REQUIRED_FIELDS_HELP = """
Required fields for your analysis (provide ALL):
- topic: What the session was about (2-5 words)
- summary: A 2-3 sentence summary of what happened
- learned: What you learned from this session
- useful_info: Any useful information worth remembering
- timestamp: The date/time this session occurred
- mood: The general mood/vibe of the session
- user_mood: How the user seemed to feel
- chemistry: How the interaction felt between you and the user
- autonomy: How autonomous you were in this session
- key_points: List of key discussion points
"""


def _validate_analysis(analysis: Any) -> List[str]:
    """Validate analysis dict. Returns list of missing field names."""
    if not isinstance(analysis, dict):
        return ["analysis must be a JSON object"]
    missing = []
    for field in REQUIRED_ANALYSIS_FIELDS:
        val = analysis.get(field)
        if val is None or (isinstance(val, str) and not val.strip()):
            missing.append(field)
        elif isinstance(val, (list, tuple)) and len(val) == 0:
            missing.append(field)
    return missing


def _build_session_prompt(session: Dict) -> str:
    """Build a prompt showing the session content to analyze."""
    chunks = session.get("chunks", [])
    topics = session.get("topics", [])
    sid = session.get("session_id", "unknown")[:12]
    date = session.get("date", "unknown")

    # Deduplicate and join chunks for readability
    seen: Set[str] = set()
    deduped = []
    for c in chunks:
        key = c.strip()[:100]
        if key not in seen:
            seen.add(key)
            deduped.append(c.strip())

    content = "\n\n".join(deduped)
    # Truncate to 4000 chars to avoid context bloat
    if len(content) > 4000:
        content = content[:4000] + "\n... [session truncated]"

    return (
        f"## Session to Analyze\n"
        f"Session ID: {sid}\n"
        f"Date: {date}\n"
        f"Topics: {', '.join(topics) if topics else 'general'}\n\n"
        f"### Session Content\n{content}\n\n"
        f"{REQUIRED_FIELDS_HELP}"
        f"\nCall `nora_analyze_sessions` with `action='analyze'` and a JSON `analysis` "
        f"object containing ALL fields listed above."
    )


def nora_analyze_sessions_handler(args: Dict, **kw) -> str:
    """Phase 1: Walk through today's sessions and collect structured analysis."""
    nora_logger.debug("[Phase 1] TOOL CALLED args=%s", json.dumps(args, default=str, ensure_ascii=False))
    action = (args.get("action") or "start").strip().lower()
    analysis = args.get("analysis")
    task_id = kw.get("task_id", "default")

    with _state_lock:
        state = _analyze_state.get(task_id)
    if state is None or action == "start":
        sessions = _load_today_sessions()
        if not sessions:
            result = json.dumps({
                "success": True,
                "message": "No sessions found from today. Nothing to analyze.",
                "complete": True,
            })
            nora_logger.info("[Phase 1] No sessions found from today — nothing to analyze | result=%s", result)
            return result
        state = {
            "sessions": sessions,
            "current_index": 0,
            "completed": [],
            "failed_attempts": {},
        }
        _analyze_state[task_id] = state
        first = sessions[0]
        session_ids = [s.get("session_id", "?")[:12] for s in sessions]
        result = json.dumps({
            "success": True,
            "prompt": _build_session_prompt(first),
            "progress": f"Session 1/{len(sessions)}",
            "complete": False,
            "_nora_retry": False,
        })
        nora_logger.info(
            "[Phase 1] Started — %d session(s): %s | result=%.300s",
            len(sessions), session_ids, result,
        )
        return result

    if action == "analyze":
        missing = _validate_analysis(analysis)
        if missing:
            sid = state["sessions"][state["current_index"]]["session_id"]
            attempt = state["failed_attempts"].get(sid, 0) + 1
            state["failed_attempts"][sid] = attempt
            nora_logger.warning(
                "[Phase 1] Validation failed for session %s (attempt %d): missing %s | analysis=%.500s",
                sid[:12], attempt, missing, json.dumps(analysis, default=str, ensure_ascii=False),
            )
            session = state["sessions"][state["current_index"]]
            result = json.dumps({
                "success": False,
                "error": f"Missing required fields: {', '.join(missing)}. "
                         f"Provide ALL fields in your analysis JSON.",
                "prompt": _build_session_prompt(session),
                "complete": False,
                "_nora_retry": True,
            })
            nora_logger.debug("[Phase 1] Validation fail response: %.300s", result)
            return result

        # Store the analysis
        session = state["sessions"][state["current_index"]]
        sid = session["session_id"]
        analysis["session_id"] = sid
        analysis["date"] = session.get("date", _today_str())
        nora_logger.info("[Phase 1] Storing analysis for %s: data=%.500s", sid[:12], json.dumps(analysis, default=str, ensure_ascii=False))
        doc_id = _store_session_analysis(sid, analysis)
        state["completed"].append({"session_id": sid, "analysis_id": doc_id})
        state["current_index"] += 1
        nora_logger.info(
            "[Phase 1] Session %s analyzed → stored as %s (%d/%d)",
            sid[:12], doc_id[:12], state["current_index"], len(state["sessions"]),
        )

        if state["current_index"] >= len(state["sessions"]):
            # Clean up state
            _analyze_state.pop(task_id, None)
            total = len(state["completed"])
            result = json.dumps({
                "success": True,
                "message": f"All {total} session(s) analyzed. Proceed to Phase 1.5.",
                "analyses_count": total,
                "complete": True,
            })
            nora_logger.info("[Phase 1] COMPLETE — %d session(s) analyzed | result=%s", total, result)
            return result

        next_session = state["sessions"][state["current_index"]]
        result = json.dumps({
            "success": True,
            "prompt": _build_session_prompt(next_session),
            "progress": f"Session {state['current_index'] + 1}/{len(state['sessions'])}",
            "complete": False,
            "_nora_retry": False,
        })
        nora_logger.debug("[Phase 1] Next session response: %.300s", result)
        return result

    return json.dumps({"success": False, "error": f"Unknown action: {action}. Use 'start' or 'analyze'."})


ANALYZE_SESSIONS_SCHEMA = {
    "name": "nora_analyze_sessions",
    "description": (
        "[STICKY] Phase 1 of nightly memory reconciliation. "
        "Walks through every conversation session from today. "
        "Call with action='start' to begin, then action='analyze' with your analysis JSON "
        "for each session. You MUST complete ALL sessions — there is no quit action."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "analyze"],
                "description": "'start' to begin, 'analyze' to submit analysis for current session",
            },
            "analysis": {
                "type": "object",
                "description": (
                    "JSON object with ALL required fields: topic, summary, learned, useful_info, "
                    "timestamp, mood, user_mood, chemistry, autonomy, key_points"
                ),
                "properties": {
                    "topic": {"type": "string"},
                    "summary": {"type": "string"},
                    "learned": {"type": "string"},
                    "useful_info": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "mood": {"type": "string"},
                    "user_mood": {"type": "string"},
                    "chemistry": {"type": "string"},
                    "autonomy": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Tool 2: nora_humanize_memories
# ---------------------------------------------------------------------------

HUMANIZE_SYSTEM_NOTE = (
    "Turn the structured session analysis below into a natural, reflective "
    "narrative memory. Write 2-3 paragraphs as if you're reminiscing about "
    "the conversation. Cover ALL fields from the analysis naturally in the "
    "prose: the topic, summary, what you learned, useful info, the mood/vibe, "
    "the user's mood, the chemistry, your autonomy level, and the key points."
)

NARRATIVE_HELP = """
Write a narrative reminiscence (2-3 paragraphs) that naturally covers:
- What the session was about (topic/summary)
- What you learned
- Any useful information
- The mood and vibe
- How the user seemed
- The chemistry between you
- How autonomous you were
- Key discussion points

Write like you're remembering the conversation, not listing facts.
Do NOT use bullet points or structured output.
"""


def _load_new_analyses(date_filter: str = "") -> List[Dict]:
    """Load session_analysis entries not yet humanized (no corresponding narrative_memory)."""
    collection = _get_collection()
    where: Dict = {"type": "session_analysis"}
    if date_filter:
        where = {"$and": [{"type": "session_analysis"}, {"date": date_filter}]}
    try:
        results = collection.get(where=where)
    except Exception as exc:
        nora_logger.error("[CHROMADB] _load_new_analyses query FAILED: %s", exc)
        return []
    if not results or not results.get("ids"):
        nora_logger.info("[CHROMADB] _load_new_analyses: no results for query=%s", json.dumps(where))
        return []
    nora_logger.info("[CHROMADB] _load_new_analyses: found %d raw analysis docs", len(results["ids"]))
    nora_logger.debug("[CHROMADB] raw IDs: %s", results["ids"][:5])

    # Check which analysis IDs already have narratives
    try:
        narratives = collection.get(where={"type": "narrative_memory"})
        narr_sources: Set[str] = set()
        if narratives and narratives.get("metadatas"):
            for m in narratives["metadatas"]:
                sid = (m or {}).get("source_analysis_id", "")
                if sid:
                    narr_sources.add(sid)
        nora_logger.debug("[CHROMADB] narrative_memory count=%d, already-humanized sources=%s",
                          len(narratives.get("ids", []) if narratives else []),
                          list(narr_sources)[:5])
    except Exception as exc:
        nora_logger.debug("[CHROMADB] narrative_memory query failed: %s", exc)
        narr_sources = set()

    entries = []
    for i, doc_id in enumerate(results["ids"]):
        if doc_id in narr_sources:
            nora_logger.debug("[CHROMADB] skipping %s — already has narrative", doc_id[:12])
            continue
        meta = (results.get("metadatas") or [{}])[i] or {}
        doc = (results.get("documents") or [""])[i] or ""
        try:
            analysis_data = json.loads(doc)
        except (json.JSONDecodeError, TypeError) as jde:
            nora_logger.warning("[CHROMADB] JSON parse failed for %s: %s", doc_id[:12], jde)
            analysis_data = {}
        entries.append({
            "id": doc_id,
            "analysis": analysis_data,
            "metadata": meta,
        })
    nora_logger.info("[CHROMADB] _load_new_analyses: returning %d entries after filtering", len(entries))
    return entries


def _build_humanize_prompt(entry: Dict) -> str:
    """Build a prompt showing the session analysis to humanize."""
    analysis = entry.get("analysis", {})
    sid = analysis.get("session_id", "unknown")[:12]
    date = analysis.get("date", "unknown")
    topic = analysis.get("topic", "unknown")
    summary = analysis.get("summary", "")

    fields_display = "\n".join(
        f"- {k}: {v}" for k, v in analysis.items()
        if k not in ("session_id", "date") and v
    )

    return (
        f"## Session Analysis to Humanize\n"
        f"Session: {sid}\n"
        f"Date: {date}\n"
        f"Topic: {topic}\n"
        f"Summary: {summary}\n\n"
        f"### All Fields\n{fields_display}\n\n"
        f"{NARRATIVE_HELP}"
        f"\nCall `nora_humanize_memories` with `action='humanize'` and the `narrative` string."
    )


def _validate_narrative_coverage(narrative: str, analysis: Dict) -> List[str]:
    """Check that the narrative mentions all required fields. Returns missing topics."""
    lower = narrative.lower()
    missing = []
    required_mentions = {
        "topic": analysis.get("topic", "").lower().split()[:3],
        "learned": ["learn", "discover", "realiz", "understand", "figured out"],
        "mood": [analysis.get("mood", "").lower()],
        "user_mood": [analysis.get("user_mood", "").lower()],
        "chemistry": ["chemistry", "connection", "flow", "click", "sync", "vibe"],
    }
    for field, keywords in required_mentions.items():
        if not any(k in lower for k in keywords if k):
            missing.append(field)
    return missing


def nora_humanize_memories_handler(args: Dict, **kw) -> str:
    """Phase 1.5: Convert structured session analyses into narrative memories."""
    nora_logger.debug("[Phase 1.5] TOOL CALLED args=%s", json.dumps(args, default=str, ensure_ascii=False))
    action = (args.get("action") or "start").strip().lower()
    narrative = args.get("narrative", "")
    task_id = kw.get("task_id", "default")

    state = _humanize_state.get(task_id)
    if state is None or action == "start":
        analyses = _load_new_analyses(date_filter=_today_str())
        if not analyses:
            result = json.dumps({
                "success": True,
                "message": "No session analyses to humanize. Run Phase 1 first.",
                "complete": True,
            })
            nora_logger.info("[Phase 1.5] No session analyses found — nothing to humanize | result=%s", result)
            return result
        state = {
            "analyses": analyses,
            "current_index": 0,
            "completed": [],
        }
        _humanize_state[task_id] = state
        first = analyses[0]
        nora_logger.info(
            "[Phase 1.5] Started — %d analysis(es) to humanize. First: %s",
            len(analyses), first.get("analysis", {}).get("session_id", "?")[:12],
        )
        result = json.dumps({
            "success": True,
            "prompt": _build_humanize_prompt(first),
            "progress": f"Memory {1}/{len(analyses)}",
            "complete": False,
            "_nora_retry": False,
        })
        nora_logger.debug("[Phase 1.5] Start response: %.300s", result)
        return result

    if action == "humanize":
        if not narrative or not narrative.strip():
            entry = state["analyses"][state["current_index"]]
            nora_logger.warning(
                "[Phase 1.5] Empty narrative submitted for %s — retrying",
                entry.get("analysis", {}).get("session_id", "?")[:12],
            )
            result = json.dumps({
                "success": False,
                "error": "Narrative is required. Write 2-3 paragraphs covering all fields.",
                "prompt": _build_humanize_prompt(entry),
                "complete": False,
                "_nora_retry": True,
            })
            nora_logger.debug("[Phase 1.5] Empty narrative response: %.300s", result)
            return result

        entry = state["analyses"][state["current_index"]]
        analysis = entry.get("analysis", {})
        sid = analysis.get("session_id", "unknown")
        missing = _validate_narrative_coverage(narrative, analysis)
        if missing:
            nora_logger.warning(
                "[Phase 1.5] Narrative for %s missing coverage: %s — retrying\nnarrative=%.500s",
                sid[:12], missing, narrative[:500],
            )
            result = json.dumps({
                "success": False,
                "error": f"Your narrative doesn't cover: {', '.join(missing)}. "
                         f"Mention these aspects naturally in your prose.",
                "prompt": _build_humanize_prompt(entry),
                "complete": False,
                "_nora_retry": True,
            })
            return result

        # Store the narrative memory
        nora_logger.info("[Phase 1.5] Storing narrative for %s: narrative=%.500s analysis_id=%s",
                         sid[:12], narrative[:500], entry["id"][:12])
        count = _store_narrative_memory(sid, entry["id"], narrative.strip(), analysis)
        state["completed"].append({"analysis_id": entry["id"], "chunks": count})
        state["current_index"] += 1
        nora_logger.info(
            "[Phase 1.5] Session %s humanized → %d chunks (%d/%d)",
            sid[:12], count, state["current_index"], len(state["analyses"]),
        )

        if state["current_index"] >= len(state["analyses"]):
            _humanize_state.pop(task_id, None)
            total = len(state["completed"])
            result = json.dumps({
                "success": True,
                "message": f"All {total} analysis(es) humanized. Proceed to Phase 2.",
                "memories_count": total,
                "complete": True,
            })
            nora_logger.info("[Phase 1.5] COMPLETE — %d narrative memory(ies) created | result=%s", total, result)
            return result

        next_entry = state["analyses"][state["current_index"]]
        result = json.dumps({
            "success": True,
            "prompt": _build_humanize_prompt(next_entry),
            "progress": f"Memory {state['current_index'] + 1}/{len(state['analyses'])}",
            "complete": False,
            "_nora_retry": False,
        })
        nora_logger.debug("[Phase 1.5] Next narrative response: %.300s", result)
        return result

    return json.dumps({"success": False, "error": f"Unknown action: {action}. Use 'start' or 'humanize'."})


HUMANIZE_MEMORIES_SCHEMA = {
    "name": "nora_humanize_memories",
    "description": (
        "[STICKY] Phase 1.5 of nightly memory reconciliation. "
        "Convert structured session analyses into natural narrative memories. "
        "Call with action='start' to begin, then action='humanize' with your narrative. "
        "You MUST complete all analyses — there is no quit action."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "humanize"],
                "description": "'start' to begin, 'humanize' to submit narrative for current analysis",
            },
            "narrative": {
                "type": "string",
                "description": "2-3 paragraph narrative reminiscence covering all analysis fields",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Tool 3: nora_dedup_memories
# ---------------------------------------------------------------------------

DEDUP_INSTRUCTIONS = (
    "Compare the new memory (from today) with the 3 similar existing memories above. "
    "Decide what to do:\n"
    "- **keep**: This memory is unique enough. Store it as-is.\n"
    "- **merge**: Similar but has new info. Combine into the oldest memory, "
    "update timestamp to today.\n"
    "- **replace**: The existing memory is stale/wrong. Replace with today's version.\n\n"
    "Prioritize today's memory — newer information is more accurate."
)


def _load_new_narrative_memories(date_filter: str = "") -> List[Dict]:
    """Load narrative_memories from today that haven't been deduped yet."""
    collection = _get_collection()
    where: Dict = {"type": "narrative_memory"}
    if date_filter:
        where = {"$and": [{"type": "narrative_memory"}, {"date": date_filter}]}
    try:
        results = collection.get(where=where)
    except Exception:
        return []
    if not results or not results.get("ids"):
        return []
    entries = []
    for i, doc_id in enumerate(results["ids"]):
        text = (results.get("documents") or [""])[i] or ""
        meta = (results.get("metadatas") or [{}])[i] or {}
        entries.append({
            "id": doc_id,
            "text": text,
            "metadata": meta,
        })
    return entries


def _build_dedup_prompt(new_memory: Dict, similar_entries: List[Dict],
                         similar_distances: List[float]) -> str:
    """Build a prompt showing the new memory vs similar existing ones."""
    new_text = new_memory.get("text", "")
    new_meta = new_memory.get("metadata", {})
    new_topic = new_meta.get("topic", "unknown")
    new_date = new_meta.get("date", "today")

    lines = [
        f"## New Memory (today)\nTopic: {new_topic} | Date: {new_date}\n"
        f"Content: {new_text[:600]}",
        "\n## Similar Existing Memories\n",
    ]
    for i, entry in enumerate(similar_entries):
        meta = entry.get("metadata", {})
        dist = similar_distances[i] if i < len(similar_distances) else 1.0
        similarity = 1.0 / (1.0 + dist)
        text = entry.get("text", "")[:400]
        lines.append(
            f"### Match {i+1} (similarity: {similarity:.2f})\n"
            f"Topic: {meta.get('topic', 'unknown')} | "
            f"Date: {meta.get('date', 'unknown')}\n"
            f"Content: {text}\n"
        )

    lines.append(f"\n{DEDUP_INSTRUCTIONS}")
    lines.append(
        "\nCall `nora_dedup_memories` with `action='decide'` and `decision='keep|merge|replace'`, "
        "plus `target_id` (the existing memory id to merge/replace into)."
    )
    return "\n".join(lines)


def nora_dedup_memories_handler(args: Dict, **kw) -> str:
    """Phase 2: Deduplicate new narrative memories against existing vector store."""
    nora_logger.debug("[Phase 2] TOOL CALLED args=%s", json.dumps(args, default=str, ensure_ascii=False))
    action = (args.get("action") or "start").strip().lower()
    decision = (args.get("decision") or "").strip().lower()
    target_id = args.get("target_id", "")
    task_id = kw.get("task_id", "default")

    state = _dedup_state.get(task_id)
    if state is None or action == "start":
        memories = _load_new_narrative_memories(date_filter=_today_str())
        if not memories:
            result = json.dumps({
                "success": True,
                "message": "No new narrative memories to dedup. Run Phase 1.5 first.",
                "complete": True,
            })
            nora_logger.info("[Phase 2] No new narrative memories to dedup | result=%s", result)
            return result
        state = {
            "memories": memories,
            "current_index": 0,
            "completed": [],
            "similar_cache": {},
        }
        _dedup_state[task_id] = state
        nora_logger.info("[Phase 2] Started — %d memory(ies) to deduplicate", len(memories))

    if state["current_index"] >= len(state["memories"]):
        _dedup_state.pop(task_id, None)
        total = len(state["completed"])
        nora_logger.info("[Phase 2] COMPLETE — %d memory(ies) processed", total)
        return json.dumps({
            "success": True,
            "message": f"Deduplication complete. {total} memories processed.",
            "complete": True,
        })

    mem = state["memories"][state["current_index"]]

    if action == "start":
        # Compute similar entries
        collection = _get_collection()
        embed_fn = _get_storage_embed_fn()
        mem_text = mem.get("text", "")
        similar_ids, distances = _find_similar(
            collection, mem_text, embed_fn, n=4,
            extra_filter={"type": "narrative_memory"},
        )
        # Remove self from similar list
        similar_entries = []
        similar_distances = []
        for j, sid in enumerate(similar_ids):
            if sid == mem["id"]:
                continue
            meta_raw = {}
            text_raw = ""
            try:
                res = collection.get(ids=[sid])
                if res and res.get("metadatas"):
                    meta_raw = res["metadatas"][0] or {}
                if res and res.get("documents"):
                    text_raw = res["documents"][0] or ""
            except Exception:
                pass
            similar_entries.append({"id": sid, "text": text_raw, "metadata": meta_raw})
            d = distances[j] if j < len(distances) else 1.0
            similar_distances.append(d)
            if len(similar_entries) >= 3:
                break

        state["similar_cache"][mem["id"]] = {
            "entries": similar_entries,
            "distances": similar_distances,
        }
        state["waiting_for_decision"] = True
        nora_logger.debug(
            "[Phase 2] Memory %d/%d — found %d similar entries. Asking for decision.",
            state["current_index"] + 1, len(state["memories"]), len(similar_entries),
        )

        return json.dumps({
            "success": True,
            "prompt": _build_dedup_prompt(mem, similar_entries, similar_distances),
            "progress": f"Memory {state['current_index'] + 1}/{len(state['memories'])}",
            "complete": False,
            "_nora_retry": False,
        })

    if action == "decide":
        if decision not in ("keep", "merge", "replace"):
            similar_data = state["similar_cache"].get(mem["id"], {})
            nora_logger.warning(
                "[Phase 2] Invalid decision '%s' for memory %s — retrying",
                decision, mem["id"][:12],
            )
            return json.dumps({
                "success": False,
                "error": f"Invalid decision '{decision}'. Use: keep, merge, or replace.",
                "prompt": _build_dedup_prompt(
                    mem,
                    similar_data.get("entries", []),
                    similar_data.get("distances", []),
                ),
                "complete": False,
                "_nora_retry": True,
            })

        similar_data = state["similar_cache"].get(mem["id"], {})
        similar_entries = similar_data.get("entries", [])

        try:
            collection = _get_collection()

            if decision == "keep":
                state["completed"].append({"id": mem["id"], "action": "keep"})
                nora_logger.info("[Phase 2] Memory %s — KEPT (unique)", mem["id"][:12])

            elif decision == "merge" and target_id:
                # Find the target entry
                target_text = ""
                target_meta = {}
                try:
                    res = collection.get(ids=[target_id])
                    if res and res.get("documents"):
                        target_text = res["documents"][0] or ""
                    if res and res.get("metadatas"):
                        target_meta = res["metadatas"][0] or {}
                except Exception:
                    pass
                merged = mem.get("text", "") + "\n\n---\n\n" + target_text
                target_meta["date"] = _today_str()
                target_meta["timestamp"] = datetime.now().isoformat()
                embed_fn = _get_storage_embed_fn()
                embedding = embed_fn([merged])[0]
                collection.update(
                    ids=[target_id],
                    embeddings=[embedding],
                    documents=[merged],
                    metadatas=[target_meta],
                )
                # Delete the new duplicate
                collection.delete(ids=[mem["id"]])
                state["completed"].append({"id": target_id, "action": "merge"})
                nora_logger.info("[Phase 2] Memory %s — MERGED into %s", mem["id"][:12], target_id[:12])

            elif decision == "replace" and target_id:
                embed_fn = _get_storage_embed_fn()
                embedding = embed_fn([mem.get("text", "")])[0]
                collection.update(
                    ids=[target_id],
                    embeddings=[embedding],
                    documents=[mem.get("text", "")],
                    metadatas=[{
                        **mem.get("metadata", {}),
                        "date": _today_str(),
                        "timestamp": datetime.now().isoformat(),
                    }],
                )
                collection.delete(ids=[mem["id"]])
                state["completed"].append({"id": target_id, "action": "replace"})
                nora_logger.info("[Phase 2] Memory %s — REPLACED existing %s", mem["id"][:12], target_id[:12])

            else:
                nora_logger.warning("[Phase 2] Memory %s — merge/replace missing target_id", mem["id"][:12])
                return json.dumps({
                    "success": False,
                    "error": "merge/replace requires 'target_id' — the ID of the existing memory to modify.",
                    "prompt": _build_dedup_prompt(mem, similar_entries, similar_data.get("distances", [])),
                    "complete": False,
                    "_nora_retry": True,
                })

        except Exception as exc:
            logger.warning("Dedup operation failed: %s", exc)
            nora_logger.error("[Phase 2] Memory %s — operation failed: %s", mem["id"][:12], exc)
            return json.dumps({
                "success": False,
                "error": f"Operation failed: {exc}",
                "complete": False,
                "_nora_retry": True,
            })

        state["waiting_for_decision"] = False
        state["current_index"] += 1

        if state["current_index"] >= len(state["memories"]):
            _dedup_state.pop(task_id, None)
            total = len(state["completed"])
            nora_logger.info("[Phase 2] COMPLETE — %d memory(ies) deduplicated. Proceeding to Phase 3", total)
            return json.dumps({
                "success": True,
                "message": f"All {total} memories deduplicated. Phase 2 complete.",
                "complete": True,
            })

        next_mem = state["memories"][state["current_index"]]
        collection = _get_collection()
        embed_fn = _get_storage_embed_fn()
        mem_text = next_mem.get("text", "")
        similar_ids, distances = _find_similar(
            collection, mem_text, embed_fn, n=4,
            extra_filter={"type": "narrative_memory"},
        )
        similar_entries = []
        similar_distances = []
        for j, sid in enumerate(similar_ids):
            if sid == next_mem["id"]:
                continue
            meta_raw = {}
            text_raw = ""
            try:
                res = collection.get(ids=[sid])
                if res and res.get("metadatas"):
                    meta_raw = res["metadatas"][0] or {}
                if res and res.get("documents"):
                    text_raw = res["documents"][0] or ""
            except Exception:
                pass
            similar_entries.append({"id": sid, "text": text_raw, "metadata": meta_raw})
            d = distances[j] if j < len(distances) else 1.0
            similar_distances.append(d)
            if len(similar_entries) >= 3:
                break

        state["similar_cache"][next_mem["id"]] = {
            "entries": similar_entries,
            "distances": similar_distances,
        }
        state["waiting_for_decision"] = True

        return json.dumps({
            "success": True,
            "prompt": _build_dedup_prompt(next_mem, similar_entries, similar_distances),
            "progress": f"Memory {state['current_index'] + 1}/{len(state['memories'])}",
            "complete": False,
            "_nora_retry": False,
        })

    return json.dumps({"success": False, "error": f"Unknown action: {action}. Use 'start' or 'decide'."})


DEDUP_MEMORIES_SCHEMA = {
    "name": "nora_dedup_memories",
    "description": (
        "[STICKY] Phase 2 of nightly memory reconciliation. "
        "Deduplicate new narrative memories against existing vector store. "
        "Call with action='start' to begin, then action='decide' with your decision. "
        "You MUST complete all memories — there is no quit action."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "decide"],
                "description": "'start' to begin, 'decide' to submit your dedup decision",
            },
            "decision": {
                "type": "string",
                "enum": ["keep", "merge", "replace"],
                "description": "keep=unique, merge=combine into oldest, replace=overwrite oldest",
            },
            "target_id": {
                "type": "string",
                "description": "ID of existing memory to merge/replace into (required for merge/replace)",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx: Any) -> None:
    """Register the vector context hooks with Hermes."""
    ctx.register_hook("pre_llm_call", _on_system_context)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("pre_llm_call", _on_skill_inject)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("on_session_finalize", _on_session_finalize)

    # Register Nora's memory management tools
    ctx.register_tool(
        name="nora_analyze_sessions",
        toolset="nora-memory",
        schema=ANALYZE_SESSIONS_SCHEMA,
        handler=nora_analyze_sessions_handler,
        emoji="🔍",
    )
    ctx.register_tool(
        name="nora_humanize_memories",
        toolset="nora-memory",
        schema=HUMANIZE_MEMORIES_SCHEMA,
        handler=nora_humanize_memories_handler,
        emoji="📝",
    )
    ctx.register_tool(
        name="nora_dedup_memories",
        toolset="nora-memory",
        schema=DEDUP_MEMORIES_SCHEMA,
        handler=nora_dedup_memories_handler,
        emoji="🔄",
    )

    # Pre-load embedding model at startup so first query is fast
    # (model download happens here if not cached, takes ~30s once)
    try:
        logger.info("Warming up embedding model...")
        fn = _get_storage_embed_fn()
        fn(["warmup"])  # trigger _lazy_init() — __init__ only sets model=None
        logger.info("Embedding model ready")
    except Exception:
        logger.warning("Embedding model warmup failed (will lazy-load later)")

    # Initialize skill vector store at startup (reuse embedding function)
    try:
        from .skill_store import init_skill_store
        init_skill_store(embed_fn=_get_storage_embed_fn())
        logger.info("Skill vector store initialized")
    except Exception as e:
        logger.warning("Skill vector store init failed: %s", e)

    logger.info("Vector context plugin registered")
