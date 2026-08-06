"""Embedding-based Retriever — Semantic retrieval using fastembed (ONNX runtime).

Replaces rule-based keyword matching with embedding cosine similarity.
Solves the core limitation of ACSRetriever: inability to match semantically
equivalent but lexically different text (e.g., "开门盘点" vs "公开评估").

Uses fastembed (ONNX runtime) instead of sentence-transformers (PyTorch),
making it ~10x lighter to install and compatible with Intel Macs where
PyTorch >= 2.4 wheels are unavailable.

Model download tips for users in China:
  - fastembed downloads from HuggingFace by default
  - Set environment variable: export HF_ENDPOINT=https://hf-mirror.com
"""

from __future__ import annotations
import hashlib
import re
from typing import Optional

from .base import BaseRetriever
from .types import ContextBlock, TaskProfile, InfoType
from .retriever import INFO_TYPE_PATTERNS


class EmbeddingRetriever(BaseRetriever):
    """Semantic retrieval using embedding models (fastembed/ONNX).

    Uses fastembed to encode messages and query into dense vectors,
    then selects top-k by cosine similarity. Embeddings are cached in-memory
    for reuse across queries within the same session.

    Args:
        model_name: Model identifier. Default: BAAI/bge-small-zh-v1.5
            (~95MB, supports Chinese + English, CPU-friendly, ONNX format)
        top_k: Maximum blocks to retrieve. Default: 20
        recency_weight: Weight for recency bonus (0-1). Default: 0.15
            When two blocks have similar semantic relevance, the more recent
            one gets a slight boost. Set to 0 for pure semantic ranking.

    Usage:
        # In ShenlunForge
        forge = ShenlunForge(model="deepseek-chat", api_key="...", retriever_type="embedding")

        # Standalone
        retriever = EmbeddingRetriever(model_name="BAAI/bge-small-zh-v1.5")
        blocks = await retriever.retrieve(messages, task_profile)

        # Lightweight mode (no LLM at all)
        forge = ShenlunForge(retriever_type="embedding")
        result = await forge.compress(messages, query, lightweight=True)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        top_k: int = 20,
        recency_weight: float = 0.15,
        **kwargs,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.recency_weight = recency_weight
        self._model = None
        self._cache: dict[str, object] = {}

    def _get_model(self):
        """Lazy load the embedding model on first use."""
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError:
                raise ImportError(
                    "fastembed is required for EmbeddingRetriever.\n"
                    "Install with: pip install fastembed"
                )
            print(
                f"[ShenlunForge] Loading embedding model: {self.model_name}"
                " (first run downloads ~95MB ONNX model)..."
            )
            self._model = TextEmbedding(model_name=self.model_name)
            print(f"[ShenlunForge] Model loaded.")
        return self._model

    @staticmethod
    def _content_hash(text: str) -> str:
        """Generate MD5 hash for caching."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _normalize(self, vec):
        """L2 normalize a vector."""
        import numpy as np
        vec = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _embed_query(self, text: str):
        """Embed a single query string."""
        model = self._get_model()
        embedding = list(model.embed([text]))[0]
        return self._normalize(embedding)

    def _embed_blocks(self, texts: list[str]):
        """Embed multiple block texts with in-memory caching.

        Already-seen texts are served from cache; only uncached texts
        are passed to the model. This makes repeated queries on the same
        conversation nearly free after the first run.
        """
        import numpy as np

        results = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            h = self._content_hash(text)
            if h in self._cache:
                results[i] = self._cache[h]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            model = self._get_model()
            embeddings = list(model.embed(uncached_texts))
            for idx, text, emb in zip(uncached_indices, uncached_texts, embeddings):
                normalized = self._normalize(emb)
                h = self._content_hash(text)
                self._cache[h] = normalized
                results[idx] = normalized

        return np.array(results, dtype=np.float32)

    async def retrieve(
        self,
        messages: list[ContextBlock],
        task_profile: TaskProfile,
    ) -> list[ContextBlock]:
        """Retrieve relevant blocks using embedding cosine similarity.

        Score = cosine_similarity * (1 - recency_weight) + recency * recency_weight

        Embeddings are L2-normalized, so dot product equals cosine similarity.
        """
        if not messages:
            return []

        import numpy as np

        # Build query text from current_query + key_elements
        query_text = task_profile.current_query
        if task_profile.key_elements:
            query_text += " " + " ".join(task_profile.key_elements)

        # Embed query and all blocks
        query_vec = self._embed_query(query_text)
        block_vecs = self._embed_blocks([m.content for m in messages])

        # Cosine similarity (normalized vectors -> dot product)
        similarities = np.dot(block_vecs, query_vec)

        # Score with recency bonus
        n = len(messages)
        scored: list[tuple[ContextBlock, float]] = []
        for i, block in enumerate(messages):
            sim_score = float(similarities[i])
            recency = (i + 1) / n
            final_score = sim_score * (1 - self.recency_weight) + recency * self.recency_weight
            block.relevance_score = final_score
            block.info_type = self._classify_info_type(block.content)
            scored.append((block, final_score))

        # Sort by score (descending), select top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        k = min(self.top_k, len(scored))
        selected = [b for b, _ in scored[:k]]

        # Re-sort by original position to maintain conversation flow
        selected.sort(key=lambda x: x.msg_index)

        return selected

    @staticmethod
    def _classify_info_type(content: str) -> InfoType:
        """Classify info type using the same rule-based patterns as ACSRetriever.

        This is a lightweight heuristic -- in full pipeline mode, the Analyzer
        step provides more accurate classification via LLM.
        """
        content_lower = content.lower()
        scores: dict[InfoType, int] = {}

        for info_type, patterns in INFO_TYPE_PATTERNS.items():
            match_count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content_lower)
                match_count += len(matches)
            scores[info_type] = match_count

        if not scores or max(scores.values()) == 0:
            return InfoType.OTHER

        return max(scores, key=scores.get)
