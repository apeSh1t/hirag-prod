import json
import os
from typing import Any, Dict

import httpx
import pytest

os.environ.setdefault("EMBEDDING_DIMENSION", "2")

from hirag_prod.configs.colbert_config import ColbertConfig
from hirag_prod.storage.colbert_remote import ColbertRemoteVDB


def _build_config(overrides: Dict[str, Any] | None = None) -> ColbertConfig:
    data: Dict[str, Any] = {
        "COLBERT_BASE_URL": "https://colbert.example.com",
        "COLBERT_TIMEOUT": 30,
        "COLBERT_INDEX_PREFIX": "hirag",
        "COLBERT_MAX_RETRIES": 0,
        "COLBERT_RETRY_BACKOFF_SECONDS": 0.01,
    }
    if overrides:
        data.update(overrides)
    return ColbertConfig(**data)


@pytest.mark.asyncio
async def test_upsert_texts_routes_request() -> None:
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["payload"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"records": captured["payload"]["records"]})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://colbert.example.com"
    ) as client:
        config = _build_config()
        vdb = ColbertRemoteVDB(client=client, config=config)
        rows = await vdb.upsert_texts(
            ["hello"],
            [{"documentKey": "chunk-1", "workspaceId": "ws", "knowledgeBaseId": "kb"}],
            table_name="Chunks",
        )

    assert captured["method"] == "POST"
    assert captured["path"] == "/vdb/Chunks/upsert"
    assert captured["payload"]["texts"] == ["hello"]
    assert rows == [
        {"documentKey": "chunk-1", "workspaceId": "ws", "knowledgeBaseId": "kb"}
    ]


@pytest.mark.asyncio
async def test_query_by_keys_returns_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["values"] == ["chunk-1"]
        return httpx.Response(
            200,
            json={"results": [{"documentKey": "chunk-1", "vector": [0.1, 0.2]}]},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://colbert.example.com"
    ) as client:
        config = _build_config()
        vdb = ColbertRemoteVDB(client=client, config=config)
        result = await vdb.query_by_keys(
            ["chunk-1"],
            workspace_id="ws",
            knowledge_base_id="kb",
            table_name="Chunks",
        )

    assert result == [{"documentKey": "chunk-1", "vector": [0.1, 0.2]}]


@pytest.mark.asyncio
async def test_clean_table_sends_where_clause() -> None:
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"status": "ok"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://colbert.example.com"
    ) as client:
        config = _build_config()
        vdb = ColbertRemoteVDB(client=client, config=config)
        await vdb.clean_table("Chunks", {"documentKey": "chunk-1"})

    assert captured["path"] == "/vdb/Chunks/delete"
    assert captured["body"] == {"where": {"documentKey": "chunk-1"}}
