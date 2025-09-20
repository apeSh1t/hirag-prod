"""服务器运行时配置。"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class ServerSettings(BaseSettings):
    """通过环境变量控制的服务端配置。"""

    api_key: Optional[str] = Field(
        default=None,
        description="用于 HTTP Bearer 认证的 API Key，留空表示关闭认证",
    )
    backend: Literal["hashing", "sentence-transformer", "colbert"] = Field(
        default="hashing",
        description="向量编码后端，默认为轻量级的哈希编码，可切换为 sentence-transformer 或真正的 ColBERT",
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="当 backend!=hashing 时所使用的模型名称或 checkpoint 路径",
    )
    device: Optional[str] = Field(
        default=None,
        description="模型加载设备，例如 cuda:0；留空时自动检测",
    )
    fallback_dimension: int = Field(
        default=384,
        description="哈希编码的默认维度，同时作为未显式指定维度时的兜底值",
    )
    default_top_k: int = Field(
        default=20,
        description="查询默认返回的候选数量",
    )
    log_queries: bool = Field(
        default=False,
        description="是否在日志中记录查询与得分，仅用于调试环境",
    )
    allow_cors: bool = Field(
        default=False,
        description="是否为调试环境开启宽松的 CORS 设置",
    )
    max_batch_size: int = Field(
        default=128,
        description="单次 upsert 可接受的最大记录数，超过将被拒绝",
    )

    @model_validator(mode="after")
    def _validate_dimension(self) -> "ServerSettings":
        if self.fallback_dimension <= 0:
            raise ValueError("fallback_dimension 必须为正整数")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size 必须为正整数")
        if self.default_top_k <= 0:
            raise ValueError("default_top_k 必须为正整数")
        return self

    class Config:
        env_prefix = "COLBERT_SERVER_"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> ServerSettings:
    """返回带有缓存的全局配置实例。"""

    return ServerSettings()  # type: ignore[call-arg]
