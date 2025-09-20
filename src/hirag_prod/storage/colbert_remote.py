import asyncio
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import httpx

from hirag_prod.configs.colbert_config import ColbertConfig
from hirag_prod.exceptions import StorageError
from hirag_prod.storage.base_vdb import BaseVDB

logger = logging.getLogger(__name__)


class ColbertRemoteVDB(BaseVDB):
    """Vector database adapter that proxies requests to a remote ColBERT service."""

    def __init__(self, client: httpx.AsyncClient, config: ColbertConfig):
        if not client:
            raise ValueError("client is required for ColbertRemoteVDB")
        if not config or not config.base_url:
            raise ValueError("A valid ColBERT configuration is required")

        self._client = client
        self._config = config
        self.embedding_func = None  # Compat: BaseVDB expects attribute

    @classmethod
    def create(cls, client: httpx.AsyncClient, config: ColbertConfig) -> "ColbertRemoteVDB":
        return cls(client=client, config=config)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_path(self, path: str) -> str:
        return path if path.startswith("/") else f"/{path}"

    def _drop_none(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in payload.items() if v is not None}

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", errors="ignore")
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:  # pragma: no cover - defensive
                return list(value)
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        return value

    def _serialize_record(self, record: Any) -> Dict[str, Any]:
        if record is None:
            return {}
        if isinstance(record, dict):
            data = record
        elif is_dataclass(record):
            data = asdict(record)
        else:
            try:
                data = dict(record)
            except Exception:  # pragma: no cover - defensive
                data = {
                    attr: getattr(record, attr)
                    for attr in dir(record)
                    if not attr.startswith("_")
                }
        return {k: self._serialize_value(v) for k, v in data.items()}

    async def _request(
        self, method: str, path: str, *, json: Optional[Dict[str, Any]] = None
    ) -> Any:
        attempts = 0
        last_exception: Optional[Exception] = None
        path = self._normalize_path(path)

        while attempts <= self._config.max_retries:
            try:
                response = await self._client.request(method, path, json=json)
                if response.status_code >= 400:
                    detail = None
                    try:
                        detail = response.json()
                    except Exception:  # pragma: no cover - defensive
                        detail = response.text
                    raise StorageError(
                        f"ColBERT request failed with status {response.status_code}: {detail}"
                    )
                if response.headers.get("content-type", "").startswith("application/json"):
                    return response.json()
                return None
            except (httpx.HTTPError, StorageError) as exc:
                last_exception = exc
                attempts += 1
                if attempts > self._config.max_retries:
                    break
                backoff = self._config.retry_backoff_seconds * (2 ** (attempts - 1))
                backoff = min(backoff, 30.0)
                logger.warning(
                    "ColBERT request %s %s failed (attempt %s/%s): %s",
                    method,
                    path,
                    attempts,
                    self._config.max_retries,
                    exc,
                )
                await asyncio.sleep(backoff)

        if last_exception:
            raise StorageError("ColBERT request failed") from last_exception
        raise StorageError("ColBERT request failed for unknown reasons")

    def _extract_results(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            for key in ("records", "results", "hits", "data"):
                if key in payload:
                    return payload[key]
        return payload

    # ------------------------------------------------------------------
    # BaseVDB interface
    # ------------------------------------------------------------------
    async def _init_vdb(self, *_, embedding_dimension: int, **__) -> None:
        body: Dict[str, Any] = {
            "embeddingDimension": embedding_dimension,
            "indexPrefix": self._config.index_prefix,
            "defaultIndexName": self._config.default_index_name,
        }
        await self._request("POST", "/vdb/init", json=self._drop_none(body))

    async def upsert_texts(
        self,
        texts_to_upsert: List[str],
        properties_list: List[dict],
        table_name: str,
        with_translation: bool = False,
        mode: Literal["append", "overwrite"] = "append",
    ):
        if len(texts_to_upsert) != len(properties_list):
            raise ValueError("texts_to_upsert and properties_list must be the same length")

        records = [self._serialize_record(p) for p in properties_list]
        body = {
            "tableName": table_name,
            "texts": texts_to_upsert,
            "records": records,
            "withTranslation": with_translation,
            "mode": mode,
        }
        payload = await self._request(
            "POST", f"/vdb/{table_name}/upsert", json=self._drop_none(body)
        )
        return self._extract_results(payload)

    async def upsert_file(
        self,
        file: Any,
        table_name: str = "Files",
        mode: Literal["append", "overwrite"] = "append",
    ):
        body = {
            "tableName": table_name,
            "records": [self._serialize_record(file)],
            "mode": mode,
        }
        payload = await self._request(
            "POST", f"/vdb/{table_name}/upsert", json=self._drop_none(body)
        )
        result = self._extract_results(payload)
        return result[0] if isinstance(result, list) and result else result

    async def upsert_graph(
        self,
        relations: List[Any],
        table_name: str = "Graph",
        mode: Literal["append", "overwrite"] = "append",
    ):
        serialized = [self._serialize_record(rel) for rel in relations]
        body = {
            "tableName": table_name,
            "relations": serialized,
            "mode": mode,
        }
        await self._request("POST", f"/vdb/{table_name}/upsert", json=self._drop_none(body))
        return {"relations": len(serialized)}

    async def query(
        self,
        query: Union[str, List[str]],
        workspace_id: str,
        knowledge_base_id: str,
        table_name: str,
        topk: Optional[int] = None,
        uri_list: Optional[List[str]] = None,
        require_access: Optional[Literal["private", "public"]] = None,
        columns_to_select: Optional[List[str]] = None,
        distance_threshold: Optional[float] = None,
        topn: Optional[int] = None,
        rerank: bool = False,
    ) -> List[dict]:
        body = {
            "query": query,
            "workspaceId": workspace_id,
            "knowledgeBaseId": knowledge_base_id,
            "topk": topk,
            "topn": topn,
            "uriList": uri_list,
            "requireAccess": require_access,
            "columnsToSelect": columns_to_select,
            "distanceThreshold": distance_threshold,
            "rerank": rerank,
        }
        payload = await self._request(
            "POST", f"/vdb/{table_name}/query", json=self._drop_none(body)
        )
        results = self._extract_results(payload)
        return results or []

    async def query_by_keys(
        self,
        key_value: List[str],
        workspace_id: str,
        knowledge_base_id: str,
        table_name: str,
        key_column: str = "documentKey",
        columns_to_select: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        body = {
            "key": key_column,
            "values": key_value,
            "workspaceId": workspace_id,
            "knowledgeBaseId": knowledge_base_id,
            "columnsToSelect": columns_to_select,
            "limit": limit,
        }
        payload = await self._request(
            "POST", f"/vdb/{table_name}/query_by_keys", json=self._drop_none(body)
        )
        results = self._extract_results(payload)
        return results or []

    async def get_existing_document_keys(
        self,
        uri: str,
        workspace_id: str,
        knowledge_base_id: str,
        table_name: str,
    ) -> List[str]:
        body = {
            "uri": uri,
            "workspaceId": workspace_id,
            "knowledgeBaseId": knowledge_base_id,
        }
        payload = await self._request(
            "POST", f"/vdb/{table_name}/keys", json=self._drop_none(body)
        )
        results = self._extract_results(payload)
        if isinstance(results, list):
            return results
        if isinstance(results, dict):
            return list(results.values())
        return []

    async def clean_table(
        self,
        table_name: str,
        where: Dict[str, Any],
    ):
        body = {"where": where or {}}
        await self._request(
            "POST", f"/vdb/{table_name}/delete", json=self._drop_none(body)
        )

    async def pagerank_top_chunks_with_reset(
        self,
        workspace_id: str,
        knowledge_base_id: str,
        reset_weights: Dict[str, float],
        topk: int,
        alpha: float = 0.85,
    ) -> List[Tuple[str, float]]:
        body = {
            "workspaceId": workspace_id,
            "knowledgeBaseId": knowledge_base_id,
            "resetWeights": reset_weights,
            "topk": topk,
            "alpha": alpha,
        }
        payload = await self._request(
            "POST", "/vdb/graph/pagerank", json=self._drop_none(body)
        )
        results = self._extract_results(payload)
        if not results:
            return []
        if isinstance(results, list):
            cleaned: List[Tuple[str, float]] = []
            for item in results:
                if isinstance(item, dict):
                    key = item.get("documentKey") or item.get("id") or item.get("key")
                    score = item.get("score") or item.get("weight")
                    if key is not None and score is not None:
                        cleaned.append((str(key), float(score)))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    cleaned.append((str(item[0]), float(item[1])))
            return cleaned
        return []

    async def query_node(
        self,
        node_id: str,
        workspace_id: str,
        knowledge_base_id: str,
    ) -> Dict[str, Any]:
        body = {
            "nodeId": node_id,
            "workspaceId": workspace_id,
            "knowledgeBaseId": knowledge_base_id,
        }
        payload = await self._request(
            "POST", "/vdb/graph/node", json=self._drop_none(body)
        )
        return payload or {}

    async def health_check(self) -> bool:
        try:
            await self._request("GET", "/healthz")
            return True
        except StorageError:
            return False
