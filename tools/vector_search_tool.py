"""vector_search — search long-term memory via spreading activation.

Wraps the vector-context plugin's ChromaDB + meta-graph retrieval engine
so the agent can explicitly query its recalled past context.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def vector_search(
    query: str,
    limit: int = 4,
    min_relevance: float = 0.20,
    include_context: bool = True,
) -> str:
    """Search long-term memory using spreading activation retrieval.

    Returns the top matching memory chunks with relevance scores.
    """
    if not query or not query.strip():
        return json.dumps({"success": False, "error": "Query is required."})

    try:
        import chromadb
    except ImportError:
        return json.dumps({"success": False, "error": "ChromaDB not installed."})

    try:
        from hermes_constants import get_hermes_home
        data_path = str(get_hermes_home() / "vector_store")
    except Exception:
        from pathlib import Path
        data_path = str(Path.home() / ".hermes" / "vector_store")

    try:
        import sys as _sys
        from pathlib import Path as _P
        _plugin_dir = str(_P.home() / ".hermes" / "plugins" / "vector-context")
        if _plugin_dir not in _sys.path:
            _sys.path.insert(0, _plugin_dir)
        from spreading import SpreadingActivation, _CachedEmbeddingFunction
        from meta_graph import get_graph
    except ImportError:
        return json.dumps({"success": False, "error": "Vector context plugin not available."})

    try:
        client = chromadb.PersistentClient(path=data_path)
        collection = client.get_or_create_collection(
            name="hermes_conversations",
            embedding_function=_CachedEmbeddingFunction(),
        )

        if collection.count() == 0:
            return json.dumps({
                "success": True,
                "query": query,
                "results": [],
                "message": "No memories stored yet.",
            })

        graph = get_graph()
        sa = SpreadingActivation(meta_graph=graph, collection=collection)

        result = sa.retrieve(
            query=query,
            min_relevance=min_relevance,
        )

        if not result or not result.chunks:
            return json.dumps({
                "success": True,
                "query": query,
                "results": [],
                "message": "No relevant memories found.",
            })

        chunks_out = []
        for chunk in result.chunks[:limit]:
            entry = {
                "text": chunk.text[:500],
                "relevance": round(chunk.relevance, 3),
                "chunk_id": chunk.chunk_id,
            }
            if chunk.relevance_breakdown:
                entry["breakdown"] = {
                    k: round(v, 3) for k, v in chunk.relevance_breakdown.items()
                }
            meta = chunk.metadata or {}
            if meta.get("session_id"):
                entry["session_id"] = meta["session_id"]
            if meta.get("timestamp"):
                entry["timestamp"] = meta["timestamp"]
            if meta.get("source"):
                entry["source"] = meta["source"]
            chunks_out.append(entry)

        response: Dict[str, Any] = {
            "success": True,
            "query": query,
            "results": chunks_out,
            "total_candidates": result.total_candidates,
            "relevance": round(result.relevance, 3),
        }

        if include_context:
            from spreading import format_context_block
            response["context_block"] = format_context_block(result)

        return json.dumps(response, ensure_ascii=False)

    except Exception as exc:
        logger.warning("vector_search failed: %s", exc, exc_info=True)
        return json.dumps({"success": False, "error": str(exc)})


# --- Schema ---

VECTOR_SEARCH_SCHEMA = {
    "name": "vector_search",
    "description": (
        "Search long-term memory using spreading activation (hybrid BM25 + vector + graph). "
        "Returns the most relevant recalled memories from past sessions. "
        "Use this when you need to find something from previous conversations — "
        "facts the user told you, things you learned, or context from older sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in memory.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 4).",
            },
            "min_relevance": {
                "type": "number",
                "description": "Minimum relevance score 0-1 (default 0.20). Lower = more results.",
            },
        },
        "required": ["query"],
    },
}


# --- Registry ---

from tools.registry import registry

registry.register(
    name="vector_search",
    toolset="nora-minimal",
    schema=VECTOR_SEARCH_SCHEMA,
    handler=lambda args, **kw: vector_search(
        query=args.get("query") or "",
        limit=args.get("limit", 4),
        min_relevance=args.get("min_relevance", 0.20),
    ),
    emoji="🧠",
)
