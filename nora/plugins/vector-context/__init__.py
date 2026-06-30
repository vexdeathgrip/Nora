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
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx: Any) -> None:
    """Register the vector context hooks with Hermes."""
    ctx.register_hook("pre_llm_call", _on_system_context)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("pre_llm_call", _on_skill_inject)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("on_session_finalize", _on_session_finalize)

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
