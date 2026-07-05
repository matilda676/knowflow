from __future__ import annotations

import hashlib
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Protocol


DEFAULT_MODEL = "shibing624/text2vec-base-chinese"
HASH_EMBED_DIM = 384
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("KNOWFLOW_DATA_DIR", str(BASE_DIR / "data"))).expanduser()
MODEL_CACHE_DIR = DATA_DIR / "model_cache"


class Embedder(Protocol):
    name: str
    dim: int

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        ...

    def embed_text(self, text: str) -> list[float]:
        ...


class HashEmbedder:
    name = "local-hash-embedding"
    dim = HASH_EMBED_DIM

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature in _features(text):
            digest = hashlib.md5(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))

        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(_resolve_model_path(model_name))
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [embedding.astype(float).tolist() for embedding in embeddings]


def _features(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", lowered)
    chars = [char for char in lowered if "\u4e00" <= char <= "\u9fff"]
    bigrams = [lowered[i:i + 2] for i in range(max(0, len(lowered) - 1))]

    synonym_map = {
        "必点": ["推荐", "招牌", "值得", "点什么", "吃什么"],
        "推荐": ["必点", "值得", "招牌"],
        "人均": ["价格", "预算", "多少钱", "花费"],
        "避雷": ["不推荐", "踩雷", "别点"],
        "日料": ["日本料理", "拉面", "寿司"],
    }
    synonyms: list[str] = []
    for key, values in synonym_map.items():
        if key in lowered:
            synonyms.extend(values)

    return words + chars + bigrams + synonyms


def _resolve_model_path(model_name: str) -> str:
    cache_root = MODEL_CACHE_DIR / "hub" / f"models--{model_name.replace('/', '--')}"
    ref_path = cache_root / "refs" / "main"
    if ref_path.exists():
        snapshot_id = ref_path.read_text(encoding="utf-8").strip()
        snapshot_path = cache_root / "snapshots" / snapshot_id
        if (snapshot_path / "config.json").exists():
            return str(snapshot_path)
    return model_name


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    provider = os.getenv("KNOWFLOW_EMBEDDING_PROVIDER", "auto").lower()
    model_name = os.getenv("KNOWFLOW_EMBEDDING_MODEL", DEFAULT_MODEL)

    if provider in {"auto", "sentence-transformers", "sentence_transformers"}:
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception:
            if provider != "auto":
                raise

    return HashEmbedder()


def embedding_info() -> dict[str, object]:
    embedder = get_embedder()
    return {
        "provider": embedder.name,
        "dim": embedder.dim,
        "model_ready": embedder.name != HashEmbedder.name,
    }


def embed_text(text: str) -> list[float]:
    return get_embedder().embed_text(text)


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    return get_embedder().embed_texts(texts)
