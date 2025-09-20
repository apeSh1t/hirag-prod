"""ColBERT 远端服务 FastAPI 入口。"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    DeleteRequest,
    GraphNodeRequest,
    InitRequest,
    KeysRequest,
    PagerankRequest,
    QueryByKeysRequest,
    QueryRequest,
    UpsertRequest,
)
from .settings import ServerSettings, get_settings
from .store import ColbertIndexStore

logger = logging.getLogger(__name__)

settings = get_settings()
store = ColbertIndexStore(settings)
app = FastAPI(title="ColBERT Remote Service", version="0.1.0")

if settings.allow_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


async def authorize(authorization: Optional[str] = Header(default=None)) -> None:
    if not settings.api_key:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )
    token = authorization.split(" ", 1)[1].strip()
    if token != settings.api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")


@app.on_event("startup")
async def _startup() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("ColBERT remote service started with backend=%s", settings.backend)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/vdb/init")
async def init_vdb(payload: InitRequest, _: None = Depends(authorize)) -> dict:
    await store.configure(payload.embeddingDimension)
    return {
        "status": "ok",
        "embeddingDimension": payload.embeddingDimension,
        "indexPrefix": payload.indexPrefix,
        "defaultIndexName": payload.defaultIndexName,
    }


@app.post("/vdb/{table_name}/upsert")
async def upsert_records(
    table_name: str,
    payload: UpsertRequest,
    _: None = Depends(authorize),
) -> JSONResponse:
    if payload.relations:
        result = await store.upsert_graph(payload.relations, mode=payload.mode)
        return JSONResponse({"results": result})

    records = payload.records or []
    results = await store.upsert_records(
        table_name=table_name,
        records=records,
        texts=payload.texts,
        with_translation=payload.withTranslation,
        mode=payload.mode,
    )
    return JSONResponse({"records": results})


@app.post("/vdb/{table_name}/query")
async def query_records(
    table_name: str,
    payload: QueryRequest,
    _: None = Depends(authorize),
) -> JSONResponse:
    results = await store.query(
        table_name=table_name,
        workspace_id=payload.workspaceId,
        knowledge_base_id=payload.knowledgeBaseId,
        query=payload.query,
        topk=payload.topk,
        topn=payload.topn,
        uri_list=payload.uriList,
        require_access=payload.requireAccess,
        columns_to_select=payload.columnsToSelect,
        distance_threshold=payload.distanceThreshold,
        rerank=payload.rerank,
    )
    return JSONResponse({"results": results})


@app.post("/vdb/{table_name}/query_by_keys")
async def query_by_keys(
    table_name: str,
    payload: QueryByKeysRequest,
    _: None = Depends(authorize),
) -> JSONResponse:
    results = await store.query_by_keys(
        table_name=table_name,
        workspace_id=payload.workspaceId,
        knowledge_base_id=payload.knowledgeBaseId,
        key_values=payload.values,
        key_column=payload.key,
        columns_to_select=payload.columnsToSelect,
        limit=payload.limit,
    )
    return JSONResponse({"results": results})


@app.post("/vdb/{table_name}/keys")
async def existing_keys(
    table_name: str,
    payload: KeysRequest,
    _: None = Depends(authorize),
) -> JSONResponse:
    results = await store.get_existing_keys(
        table_name=table_name,
        workspace_id=payload.workspaceId,
        knowledge_base_id=payload.knowledgeBaseId,
        uri=payload.uri,
    )
    return JSONResponse({"results": results})


@app.post("/vdb/{table_name}/delete")
async def delete_records(
    table_name: str,
    payload: DeleteRequest,
    _: None = Depends(authorize),
) -> JSONResponse:
    result = await store.clean_table(table_name=table_name, where=payload.where)
    return JSONResponse({"results": result})


@app.post("/vdb/graph/pagerank")
async def pagerank(
    payload: PagerankRequest,
    _: None = Depends(authorize),
) -> JSONResponse:
    scores = await store.pagerank(
        workspace_id=payload.workspaceId,
        knowledge_base_id=payload.knowledgeBaseId,
        reset_weights=payload.resetWeights,
        topk=payload.topk,
        alpha=payload.alpha,
    )
    results = [
        {"documentKey": doc_id, "score": score}
        for doc_id, score in scores
    ]
    return JSONResponse({"results": results})


@app.post("/vdb/graph/node")
async def query_node(
    payload: GraphNodeRequest,
    _: None = Depends(authorize),
) -> JSONResponse:
    result = await store.query_node(
        workspace_id=payload.workspaceId,
        knowledge_base_id=payload.knowledgeBaseId,
        node_id=payload.nodeId,
    )
    return JSONResponse(result or {})


@app.exception_handler(ValueError)
async def handle_value_error(_: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "Invalid request payload"},
    )


def create_app() -> FastAPI:
    """提供给外部 WSGI/ASGI 服务器的工厂方法。"""

    return app


__all__ = ["app", "create_app", "store", "settings", "ServerSettings"]
