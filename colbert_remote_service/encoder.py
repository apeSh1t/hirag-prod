"""负责文本编码的后端实现。"""

from __future__ import annotations

import hashlib
import logging
from typing import Iterable, List, Optional

import numpy as np

from .settings import ServerSettings

try:  # pragma: no cover - 可选依赖
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - 未安装时忽略
    SentenceTransformer = None  # type: ignore[misc]

try:  # pragma: no cover - 可选依赖
    from colbert.inference import ColBERTInference  # type: ignore
except Exception:  # pragma: no cover - 路径在不同版本中可能变化
    try:  # pragma: no cover
        from colbert.inference.inference import (  # type: ignore
            Inference as ColBERTInference,
        )
    except Exception:  # pragma: no cover
        ColBERTInference = None  # type: ignore

logger = logging.getLogger(__name__)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """对向量进行 L2 归一化，避免零向量导致的 NaN。"""

    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class TextEncoder:
    """封装不同向量化后端的统一接口。"""

    def __init__(self, settings: ServerSettings):
        self._settings = settings
        self._backend = settings.backend
        self._model = None
        self._dimension: Optional[int] = None

        if self._backend == "sentence-transformer":
            if SentenceTransformer is None:
                raise RuntimeError(
                    "未安装 sentence-transformers，无法使用该后端，请改用 hashing 或安装依赖"
                )
            logger.info("加载 SentenceTransformer 模型 %s", settings.model_name)
            self._model = SentenceTransformer(settings.model_name, device=settings.device)
            try:
                self._dimension = int(self._model.get_sentence_embedding_dimension())
            except Exception:  # pragma: no cover - 兼容旧版本
                sample = self._model.encode(["hello"], convert_to_numpy=True)
                self._dimension = int(sample.shape[1])
        elif self._backend == "colbert":
            if ColBERTInference is None:
                logger.warning(
                    "ColBERT 推理依赖未安装，自动回退到 hashing 后端。"
                )
                self._backend = "hashing"
            else:  # pragma: no cover - 依赖较重，在 CI 中通常不会执行
                logger.info("加载 ColBERT checkpoint %s", settings.model_name)
                try:
                    self._model = ColBERTInference(
                        checkpoint=settings.model_name,
                        device=settings.device,
                    )
                except Exception as exc:  # pragma: no cover
                    logger.warning("初始化 ColBERTInference 失败，回退到 hashing: %s", exc)
                    self._backend = "hashing"
                    self._model = None
                else:
                    self._dimension = getattr(self._model, "dim", None)
        else:
            logger.info(
                "使用 hashing 编码后端，维度=%s", settings.fallback_dimension
            )

        if not self._dimension:
            self._dimension = int(settings.fallback_dimension)

    @property
    def dimension(self) -> int:
        return int(self._dimension or self._settings.fallback_dimension)

    def encode_passages(
        self, texts: Iterable[str], dimension: Optional[int] = None
    ) -> np.ndarray:
        """对段落进行编码。"""

        data = list(texts)
        if not data:
            dim = dimension or self.dimension
            return np.zeros((0, dim), dtype=np.float32)

        if self._backend == "sentence-transformer" and self._model is not None:
            emb = self._model.encode(  # type: ignore[no-any-return]
                data,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            return _normalize(np.asarray(emb, dtype=np.float32))

        if self._backend == "colbert" and self._model is not None:
            encode_fn = None
            for attr in ("encode_passages", "encode_documents", "encode_texts"):
                encode_fn = getattr(self._model, attr, None)
                if encode_fn:
                    break
            if encode_fn is None:
                raise RuntimeError("ColBERTInference 实例未暴露 encode_* 接口")
            emb = encode_fn(data)
            return _normalize(np.asarray(emb, dtype=np.float32))

        # 默认 hashing 编码
        dim = int(dimension or self.dimension)
        return self._hash_encode(data, dim)

    def encode_queries(
        self, texts: Iterable[str], dimension: Optional[int] = None
    ) -> np.ndarray:
        """对查询文本编码。"""

        data = list(texts)
        if not data:
            dim = dimension or self.dimension
            return np.zeros((0, dim), dtype=np.float32)

        if self._backend == "sentence-transformer" and self._model is not None:
            emb = self._model.encode(  # type: ignore[no-any-return]
                data,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            return _normalize(np.asarray(emb, dtype=np.float32))

        if self._backend == "colbert" and self._model is not None:
            encode_fn = None
            for attr in ("encode_queries", "encode_questions", "encode"):
                encode_fn = getattr(self._model, attr, None)
                if encode_fn:
                    break
            if encode_fn is None:
                raise RuntimeError("ColBERTInference 实例未暴露 encode_* 接口")
            emb = encode_fn(data)
            return _normalize(np.asarray(emb, dtype=np.float32))

        dim = int(dimension or self.dimension)
        return self._hash_encode(data, dim)

    @staticmethod
    def _hash_encode(texts: List[str], dimension: int) -> np.ndarray:
        """使用可重现的哈希技巧获得稀疏向量，便于本地演示。"""

        vectors = np.zeros((len(texts), dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            if not text:
                continue
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], byteorder="little") % dimension
                vectors[row, idx] += 1.0
        return _normalize(vectors)
