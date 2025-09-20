"""FastAPI 接口使用的请求/响应模型。"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated


class InitRequest(BaseModel):
    embeddingDimension: Optional[int] = Field(default=None, ge=1)
    indexPrefix: Optional[str] = None
    defaultIndexName: Optional[str] = None

    model_config = {
        "extra": "allow",
    }


class UpsertRequest(BaseModel):
    tableName: Optional[str] = None
    texts: Optional[List[str]] = None
    records: Optional[List[Dict[str, Any]]] = None
    relations: Optional[List[Dict[str, Any]]] = None
    withTranslation: bool = False
    mode: Literal["append", "overwrite"] = "append"

    model_config = {
        "extra": "allow",
    }


class QueryRequest(BaseModel):
    query: Union[str, List[str]]
    workspaceId: str
    knowledgeBaseId: str
    topk: Optional[int] = Field(default=None, ge=1)
    topn: Optional[int] = Field(default=None, ge=1)
    uriList: Optional[List[str]] = None
    requireAccess: Optional[Literal["private", "public"]] = None
    columnsToSelect: Optional[List[str]] = None
    distanceThreshold: Optional[float] = Field(default=None, ge=0.0)
    rerank: bool = False

    model_config = {
        "extra": "allow",
    }


class QueryByKeysRequest(BaseModel):
    key: Annotated[str, Field(default="documentKey")]
    values: List[str]
    workspaceId: str
    knowledgeBaseId: str
    columnsToSelect: Optional[List[str]] = None
    limit: Optional[int] = Field(default=None, ge=1)

    model_config = {
        "extra": "allow",
    }


class KeysRequest(BaseModel):
    uri: str
    workspaceId: str
    knowledgeBaseId: str

    model_config = {
        "extra": "allow",
    }


class DeleteRequest(BaseModel):
    where: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "extra": "allow",
    }


class PagerankRequest(BaseModel):
    workspaceId: str
    knowledgeBaseId: str
    resetWeights: Dict[str, float] = Field(default_factory=dict)
    topk: int = Field(default=10, ge=1)
    alpha: float = Field(default=0.85, gt=0, lt=1)

    model_config = {
        "extra": "allow",
    }


class GraphNodeRequest(BaseModel):
    nodeId: str
    workspaceId: str
    knowledgeBaseId: str

    model_config = {
        "extra": "allow",
    }
