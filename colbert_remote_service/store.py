"""管理索引、元数据与图谱的内存实现。"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import numpy as np

from .encoder import TextEncoder
from .settings import ServerSettings

logger = logging.getLogger(__name__)


def _coalesce(data: Dict[str, Any], *keys: str, default: Optional[Any] = None) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default


def _normalize_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _matches_filters(payload: Dict[str, Any], where: Dict[str, Any]) -> bool:
    if not where:
        return True
    for key, expected in where.items():
        value = payload.get(key)
        if isinstance(expected, dict):
            if "$in" in expected:
                candidates = set(expected.get("$in", []) or [])
                if value not in candidates:
                    return False
            elif "$eq" in expected:
                if value != expected.get("$eq"):
                    return False
            else:
                if value != expected:
                    return False
        elif isinstance(expected, (list, tuple, set)):
            if value not in expected:
                return False
        else:
            if value != expected:
                return False
    return True


@dataclass
class StoredRecord:
    document_key: str
    workspace_id: str
    knowledge_base_id: str
    table_name: str
    metadata: Dict[str, Any]
    vector: Optional[np.ndarray]
    text: Optional[str]
    updated_at: datetime

    def to_dict(
        self,
        columns: Optional[Iterable[str]] = None,
        *,
        include_vector: bool = False,
        distance: Optional[float] = None,
    ) -> Dict[str, Any]:
        base = dict(self.metadata)
        base.setdefault("documentKey", self.document_key)
        base.setdefault("workspaceId", self.workspace_id)
        base.setdefault("knowledgeBaseId", self.knowledge_base_id)
        if self.text is not None:
            base.setdefault("text", self.text)
        base["updatedAt"] = self.updated_at.isoformat()

        if columns is not None:
            wanted = set(columns)
            payload = {k: base.get(k) for k in wanted if k in base}
            if "documentKey" in wanted:
                payload.setdefault("documentKey", self.document_key)
            if "text" in wanted and self.text is not None:
                payload.setdefault("text", self.text)
        else:
            payload = base

        if include_vector or (columns and "vector" in columns):
            payload["vector"] = (
                self.vector.tolist() if self.vector is not None else None
            )

        if distance is not None:
            payload["distance"] = float(distance)

        return payload


class MemoryIndex:
    """存储单个 (workspace, knowledgeBase, table) 的记录。"""

    def __init__(self, table_name: str, dimension: Optional[int] = None):
        self.table_name = table_name
        self.dimension = dimension
        self.records: Dict[str, StoredRecord] = {}

    def upsert(self, record: StoredRecord) -> StoredRecord:
        self.dimension = record.vector.shape[0] if record.vector is not None else self.dimension
        self.records[record.document_key] = record
        return record

    def select(self, keys: Iterable[str]) -> List[StoredRecord]:
        return [self.records[k] for k in keys if k in self.records]

    def values(self) -> Iterable[StoredRecord]:
        return self.records.values()

    def remove_matching(self, where: Dict[str, Any]) -> int:
        removed = 0
        for key, rec in list(self.records.items()):
            if _matches_filters(rec.metadata, where):
                self.records.pop(key, None)
                removed += 1
        return removed

    def existing_keys(self, uri: Optional[str] = None) -> List[str]:
        if uri is None:
            return list(self.records)
        return [
            key
            for key, rec in self.records.items()
            if rec.metadata.get("uri") == uri
        ]

    def is_empty(self) -> bool:
        return not self.records


@dataclass
class GraphRelation:
    source: str
    target: str
    metadata: Dict[str, Any]


class GraphData:
    """维护特定 workspace/kb 下的关系图。"""

    def __init__(self) -> None:
        self.edges: Dict[Tuple[str, str], GraphRelation] = {}
        self.adjacency: Dict[str, set[str]] = {}
        self.node_chunks: Dict[str, set[str]] = {}
        self.node_metadata: Dict[str, Dict[str, Any]] = {}

    def upsert(self, relation: GraphRelation) -> None:
        key = (relation.source, relation.target)
        self.edges[key] = relation
        self.adjacency.setdefault(relation.source, set()).add(relation.target)
        self.node_chunks.setdefault(relation.source, set())
        self.node_chunks.setdefault(relation.target, set())

        chunk_id = _coalesce(relation.metadata, "chunk_id", "chunkId")
        if chunk_id:
            self.node_chunks.setdefault(relation.source, set()).add(str(chunk_id))
            self.node_chunks.setdefault(relation.target, set()).add(str(chunk_id))

        for node, meta_key in (
            (relation.source, "source"),
            (relation.target, "target"),
        ):
            name_hint = relation.metadata.get(meta_key)
            if name_hint:
                node_meta = self.node_metadata.setdefault(node, {})
                node_meta.setdefault("entityName", name_hint)
            document_id = _coalesce(
                relation.metadata, "document_id", "documentId"
            )
            if document_id:
                node_meta = self.node_metadata.setdefault(node, {})
                node_meta["documentId"] = document_id

    def remove_matching(self, where: Dict[str, Any]) -> int:
        removed = 0
        for (src, tgt), rel in list(self.edges.items()):
            payload = {**rel.metadata, "source": src, "target": tgt}
            if _matches_filters(payload, where):
                self._remove_edge(src, tgt)
                removed += 1
        if removed:
            self._rebuild_chunks()
        return removed

    def _remove_edge(self, source: str, target: str) -> None:
        self.edges.pop((source, target), None)
        if source in self.adjacency:
            self.adjacency[source].discard(target)
            if not self.adjacency[source]:
                self.adjacency.pop(source)
        # target 无需在 adjacency 中删除，因为字典以 source 为键

    def _rebuild_chunks(self) -> None:
        self.node_chunks = {}
        for (src, tgt), rel in self.edges.items():
            chunk_id = _coalesce(rel.metadata, "chunk_id", "chunkId")
            if chunk_id:
                self.node_chunks.setdefault(src, set()).add(str(chunk_id))
                self.node_chunks.setdefault(tgt, set()).add(str(chunk_id))

    def nodes(self) -> List[str]:
        nodes = set(self.node_chunks.keys()) | set(self.adjacency.keys())
        for _, tgt in self.edges:
            nodes.add(tgt)
        return list(nodes)

    def outgoing(self, node_id: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for tgt in sorted(self.adjacency.get(node_id, [])):
            payload = self.edges.get((node_id, tgt))
            if not payload:
                continue
            data = {**payload.metadata}
            data.setdefault("source", node_id)
            data.setdefault("target", tgt)
            out.append(data)
        return out

    def incoming(self, node_id: str) -> List[Dict[str, Any]]:
        incoming: List[Dict[str, Any]] = []
        for (src, tgt), payload in self.edges.items():
            if tgt != node_id:
                continue
            data = {**payload.metadata}
            data.setdefault("source", src)
            data.setdefault("target", tgt)
            incoming.append(data)
        return incoming

    def chunk_ids(self, node_id: str) -> List[str]:
        return sorted(self.node_chunks.get(node_id, set()))

    def metadata_for(self, node_id: str) -> Dict[str, Any]:
        return dict(self.node_metadata.get(node_id, {}))

    def is_empty(self) -> bool:
        return not self.edges


def _compute_pagerank(
    adjacency: Dict[str, set[str]],
    reset_weights: Dict[str, float],
    alpha: float,
    chunk_prefix: str = "chunk-",
) -> Dict[str, float]:
    nodes = set(adjacency.keys())
    for targets in adjacency.values():
        nodes.update(targets)
    nodes.update(reset_weights.keys())
    if not nodes:
        return {}

    ordered = sorted(nodes)
    index = {node: i for i, node in enumerate(ordered)}
    size = len(ordered)
    matrix = np.zeros((size, size), dtype=np.float64)

    for src, targets in adjacency.items():
        src_idx = index[src]
        if not targets:
            matrix[:, src_idx] = 1.0 / size
            continue
        weight = 1.0 / len(targets)
        for tgt in targets:
            matrix[index[tgt], src_idx] = weight

    personalization = np.ones(size, dtype=np.float64) / size
    total = sum(reset_weights.values())
    if total > 0:
        vec = np.zeros(size, dtype=np.float64)
        for node, weight in reset_weights.items():
            if node not in index:
                continue
            try:
                val = float(weight)
            except Exception:
                continue
            if np.isfinite(val) and val > 0:
                vec[index[node]] = val
        if vec.sum() > 0:
            personalization = vec / vec.sum()

    ranks = np.ones(size, dtype=np.float64) / size
    for _ in range(100):
        new_rank = alpha * matrix @ ranks + (1 - alpha) * personalization
        if np.linalg.norm(new_rank - ranks, ord=1) < 1e-6:
            ranks = new_rank
            break
        ranks = new_rank

    return {
        node: float(ranks[idx])
        for node, idx in index.items()
        if str(node).startswith(chunk_prefix)
    }


class GraphStore:
    def __init__(self) -> None:
        self._graphs: Dict[Tuple[str, str], GraphData] = {}

    def _key(self, workspace_id: str, knowledge_base_id: str) -> Tuple[str, str]:
        return (workspace_id, knowledge_base_id)

    def upsert(self, relations: List[Dict[str, Any]], mode: str = "append") -> Dict[str, int]:
        if not relations:
            return {"edges": 0, "nodes": 0}

        inserted_edges = 0
        nodes_seen: set[str] = set()
        for rel in relations:
            props = rel.get("properties") or {}
            workspace_id = str(
                _coalesce(props, "workspace_id", "workspaceId", default="")
            )
            knowledge_base_id = str(
                _coalesce(props, "knowledge_base_id", "knowledgeBaseId", default="")
            )
            if not workspace_id or not knowledge_base_id:
                logger.warning(
                    "忽略缺少 workspace/knowledgeBase 的图关系: %s", rel
                )
                continue
            key = self._key(workspace_id, knowledge_base_id)
            graph = self._graphs.setdefault(key, GraphData())
            relation = GraphRelation(
                source=str(rel.get("source")),
                target=str(rel.get("target")),
                metadata=dict(props),
            )
            if mode == "overwrite":
                graph.remove_matching(
                    {
                        "source": relation.source,
                        "target": relation.target,
                        "documentId": _coalesce(props, "document_id", "documentId"),
                    }
                )
            graph.upsert(relation)
            inserted_edges += 1
            nodes_seen.update([relation.source, relation.target])

        return {"edges": inserted_edges, "nodes": len(nodes_seen)}

    def clean(self, where: Dict[str, Any]) -> int:
        workspace_id = str(where.get("workspaceId", ""))
        knowledge_base_id = str(where.get("knowledgeBaseId", ""))
        key = self._key(workspace_id, knowledge_base_id)
        graph = self._graphs.get(key)
        if not graph:
            return 0
        removed = graph.remove_matching(where)
        if graph.is_empty():
            self._graphs.pop(key, None)
        return removed

    def pagerank(
        self,
        workspace_id: str,
        knowledge_base_id: str,
        reset_weights: Dict[str, float],
        topk: int,
        alpha: float,
    ) -> List[Tuple[str, float]]:
        graph = self._graphs.get(self._key(workspace_id, knowledge_base_id))
        if not graph:
            return []
        scores = _compute_pagerank(graph.adjacency, reset_weights, alpha)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if topk > 0:
            ordered = ordered[:topk]
        return ordered

    def query_node(
        self, workspace_id: str, knowledge_base_id: str, node_id: str
    ) -> Dict[str, Any]:
        graph = self._graphs.get(self._key(workspace_id, knowledge_base_id))
        if not graph:
            return {}
        meta = graph.metadata_for(node_id)
        payload = {
            "id": node_id,
            "workspaceId": workspace_id,
            "knowledgeBaseId": knowledge_base_id,
            "entityName": meta.get("entityName"),
            "entityType": meta.get("entityType", "entity"),
            "chunkIds": graph.chunk_ids(node_id),
            "documentId": meta.get("documentId"),
            "outgoing": graph.outgoing(node_id),
            "incoming": graph.incoming(node_id),
        }
        return payload


class ColbertIndexStore:
    """在内存中模拟远端 ColBERT VDB 的最小实现。"""

    def __init__(self, settings: ServerSettings):
        self._settings = settings
        self._encoder = TextEncoder(settings)
        self._default_dimension = self._encoder.dimension
        self._indices: Dict[Tuple[str, str, str], MemoryIndex] = {}
        self._graph = GraphStore()
        self._lock = asyncio.Lock()

    def _index_key(self, workspace_id: str, knowledge_base_id: str, table: str) -> Tuple[str, str, str]:
        return (workspace_id, knowledge_base_id, table)

    async def configure(self, embedding_dimension: Optional[int]) -> None:
        if embedding_dimension and embedding_dimension > 0:
            self._default_dimension = int(embedding_dimension)
            logger.info("设置默认向量维度为 %s", self._default_dimension)

    async def upsert_records(
        self,
        table_name: str,
        records: List[Dict[str, Any]],
        texts: Optional[List[str]] = None,
        *,
        with_translation: bool = False,
        mode: str = "append",
    ) -> List[Dict[str, Any]]:
        if len(records) > self._settings.max_batch_size:
            raise ValueError(
                f"批量大小 {len(records)} 超过限制 {self._settings.max_batch_size}"
            )
        texts = texts or [None] * len(records)
        if len(texts) < len(records):
            texts = texts + [None] * (len(records) - len(texts))

        async with self._lock:
            results: List[Dict[str, Any]] = []
            for idx, payload in enumerate(records):
                record = dict(payload or {})
                workspace_id = str(
                    _coalesce(record, "workspaceId", "workspace_id", default="")
                )
                knowledge_base_id = str(
                    _coalesce(
                        record, "knowledgeBaseId", "knowledge_base_id", default=""
                    )
                )
                if not workspace_id or not knowledge_base_id:
                    logger.warning(
                        "忽略缺少 workspace/knowledgeBase 的记录: %s", record
                    )
                    continue

                document_key = str(
                    _coalesce(
                        record,
                        "documentKey",
                        "document_key",
                        default=f"{table_name}-{uuid4().hex}",
                    )
                )
                record["documentKey"] = document_key
                record["workspaceId"] = workspace_id
                record["knowledgeBaseId"] = knowledge_base_id
                record["updatedAt"] = datetime.utcnow().isoformat()
                if with_translation:
                    record.setdefault("withTranslation", True)

                text = texts[idx] if idx < len(texts) else None
                if not text:
                    text = record.get("text")
                if text is not None:
                    record["text"] = text

                record.pop("vector", None)
                vector: Optional[np.ndarray] = None
                if text:
                    key = self._index_key(workspace_id, knowledge_base_id, table_name)
                    index = self._indices.get(key)
                    dimension = index.dimension if index else self._default_dimension
                    vector = self._encoder.encode_passages([text], dimension=dimension)[0]
                else:
                    key = self._index_key(workspace_id, knowledge_base_id, table_name)
                    index = self._indices.get(key)
                    dimension = index.dimension if index else self._default_dimension

                stored = StoredRecord(
                    document_key=document_key,
                    workspace_id=workspace_id,
                    knowledge_base_id=knowledge_base_id,
                    table_name=table_name,
                    metadata=record,
                    vector=vector,
                    text=text,
                    updated_at=datetime.utcnow(),
                )

                key = self._index_key(workspace_id, knowledge_base_id, table_name)
                index = self._indices.setdefault(key, MemoryIndex(table_name, dimension))
                if mode == "overwrite":
                    index.remove_matching({"documentKey": document_key})
                index.upsert(stored)
                results.append(stored.to_dict())
            return results

    async def upsert_graph(
        self,
        relations: List[Dict[str, Any]],
        *,
        mode: str = "append",
    ) -> Dict[str, int]:
        async with self._lock:
            return self._graph.upsert(relations, mode=mode)

    async def query(
        self,
        table_name: str,
        workspace_id: str,
        knowledge_base_id: str,
        query: Any,
        *,
        topk: Optional[int] = None,
        topn: Optional[int] = None,
        uri_list: Optional[List[str]] = None,
        require_access: Optional[str] = None,
        columns_to_select: Optional[List[str]] = None,
        distance_threshold: Optional[float] = None,
        rerank: bool = False,
    ) -> List[Dict[str, Any]]:
        del rerank  # 当前实现不支持额外 rerank
        async with self._lock:
            index = self._indices.get(
                self._index_key(workspace_id, knowledge_base_id, table_name)
            )
            if not index or index.is_empty():
                return []

            if isinstance(query, list):
                queries = [str(q) for q in query]
            else:
                queries = [str(query)]

            query_emb = self._encoder.encode_queries(
                queries,
                dimension=index.dimension,
            )
            if query_emb.size == 0:
                return []
            query_vector = np.mean(query_emb, axis=0)
            norm = np.linalg.norm(query_vector)
            if norm == 0:
                return []
            query_vector = query_vector / norm

            effective_topk = topk or topn or self._settings.default_top_k
            scored: List[Tuple[float, StoredRecord]] = []
            for rec in index.values():
                if rec.vector is None:
                    continue
                if uri_list and rec.metadata.get("uri") not in uri_list:
                    continue
                if require_access == "private":
                    if not bool(_normalize_bool(rec.metadata.get("private"))):
                        continue
                if require_access == "public":
                    if bool(_normalize_bool(rec.metadata.get("private"))):
                        continue
                similarity = float(np.dot(query_vector, rec.vector))
                similarity = max(min(similarity, 1.0), -1.0)
                distance = 1.0 - similarity
                if distance_threshold is not None and distance > distance_threshold:
                    continue
                scored.append((distance, rec))

            scored.sort(key=lambda item: item[0])
            if effective_topk:
                scored = scored[:effective_topk]

            results: List[Dict[str, Any]] = []
            for distance, rec in scored:
                include_vector = bool(columns_to_select and "vector" in columns_to_select)
                payload = rec.to_dict(
                    columns=columns_to_select,
                    include_vector=include_vector,
                    distance=distance,
                )
                results.append(payload)

            if self._settings.log_queries:
                logger.info(
                    "query table=%s workspace=%s kb=%s -> %d 条",
                    table_name,
                    workspace_id,
                    knowledge_base_id,
                    len(results),
                )
            return results

    async def query_by_keys(
        self,
        table_name: str,
        workspace_id: str,
        knowledge_base_id: str,
        key_values: List[str],
        *,
        key_column: str = "documentKey",
        columns_to_select: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        del key_column  # 当前实现仅支持 documentKey
        async with self._lock:
            index = self._indices.get(
                self._index_key(workspace_id, knowledge_base_id, table_name)
            )
            if not index or not key_values:
                return []
            rows = index.select(key_values)
            if limit is not None:
                rows = rows[:limit]
            include_vector = bool(columns_to_select and "vector" in columns_to_select)
            return [
                rec.to_dict(
                    columns=columns_to_select,
                    include_vector=include_vector,
                )
                for rec in rows
            ]

    async def get_existing_keys(
        self,
        table_name: str,
        workspace_id: str,
        knowledge_base_id: str,
        uri: str,
    ) -> List[str]:
        async with self._lock:
            index = self._indices.get(
                self._index_key(workspace_id, knowledge_base_id, table_name)
            )
            if not index:
                return []
            return index.existing_keys(uri)

    async def clean_table(
        self,
        table_name: str,
        where: Dict[str, Any],
    ) -> Dict[str, Any]:
        async with self._lock:
            if table_name == "Graph":
                removed = self._graph.clean(where)
                return {"deleted": removed}

            key = self._index_key(
                str(where.get("workspaceId", "")),
                str(where.get("knowledgeBaseId", "")),
                table_name,
            )
            index = self._indices.get(key)
            if not index:
                return {"deleted": 0}
            removed = index.remove_matching(where)
            if index.is_empty():
                self._indices.pop(key, None)
            return {"deleted": removed}

    async def pagerank(
        self,
        workspace_id: str,
        knowledge_base_id: str,
        reset_weights: Dict[str, float],
        topk: int,
        alpha: float,
    ) -> List[Tuple[str, float]]:
        async with self._lock:
            return self._graph.pagerank(
                workspace_id, knowledge_base_id, reset_weights, topk, alpha
            )

    async def query_node(
        self,
        workspace_id: str,
        knowledge_base_id: str,
        node_id: str,
    ) -> Dict[str, Any]:
        async with self._lock:
            return self._graph.query_node(workspace_id, knowledge_base_id, node_id)
