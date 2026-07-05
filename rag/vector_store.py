from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings

from rag.embedding import embed_text, embed_texts, embedding_info

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("KNOWFLOW_DATA_DIR", str(BASE_DIR / "data"))).expanduser()
CHROMA_DIR = DATA_DIR / "chroma"
BASE_COLLECTION_NAME = "knowflow_chunks"


def collection_name() -> str:
    info = embedding_info()
    provider_slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(info["provider"])).strip("_").lower()
    return f"{BASE_COLLECTION_NAME}_{provider_slug}_{info['dim']}"


def get_collection():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=collection_name(),
        metadata={"hnsw:space": "cosine"},
    )


def chunk_document(title: str, category: str, text: str) -> str:
    return f"{title}\n分类：{category}\n{text}"


def upsert_chunks(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        return

    collection = get_collection()
    ids = [f"{item['note_id']}:{item['chunk_index']}" for item in chunks]
    documents = [
        chunk_document(
            str(item.get("title") or ""),
            str(item.get("category") or ""),
            str(item.get("text") or ""),
        )
        for item in chunks
    ]
    metadatas = [
        {
            "note_id": int(item["note_id"]),
            "chunk_index": int(item["chunk_index"]),
            "title": str(item.get("title") or ""),
            "category": str(item.get("category") or "未分类"),
            "type": str(item.get("type") or ""),
            "section": str(item.get("section") or ""),
            "char_start": int(item.get("char_start") or 0),
            "char_end": int(item.get("char_end") or 0),
            "text": str(item.get("text") or ""),
        }
        for item in chunks
    ]
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embed_texts(documents),
        metadatas=metadatas,
    )


def query_chunks(query: str, *, category: Optional[str] = None, limit: int = 5) -> list[dict[str, Any]]:
    collection = get_collection()
    where = {"category": category} if category else None
    result = collection.query(
        query_embeddings=[embed_text(query)],
        n_results=limit,
        where=where,
        include=["metadatas", "distances"],
    )

    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    matches: list[dict[str, Any]] = []
    for metadata, distance in zip(metadatas, distances):
        item = dict(metadata)
        item["score"] = 1 - float(distance)
        item["vector_distance"] = float(distance)
        item["text"] = str(item.get("text") or "")
        item["title"] = str(item.get("title") or "")
        item["category"] = str(item.get("category") or "未分类")
        item["note_id"] = int(item.get("note_id") or 0)
        item["chunk_index"] = int(item.get("chunk_index") or 0)
        item["char_start"] = int(item.get("char_start") or 0)
        item["char_end"] = int(item.get("char_end") or 0)
        matches.append(item)
    return matches


def reset_collection() -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(collection_name())
    except Exception:
        pass
    client.get_or_create_collection(
        name=collection_name(),
        metadata={"hnsw:space": "cosine"},
    )
