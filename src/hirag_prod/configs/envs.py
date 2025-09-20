from typing import Literal, Optional

from pydantic import ConfigDict, model_validator
from pydantic_settings import BaseSettings


class InitEnvs(BaseSettings):
    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    EMBEDDING_DIMENSION: int
    USE_HALF_VEC: bool = True
    HIRAG_QUERY_TIMEOUT: int = 100  # seconds
    default_query_top_k: int = 10  # Query potential results to rerank
    default_query_top_n: int = 5  # Final results to return after reranking
    default_distance_threshold: float = 0.8  # Similarity threshold for vector search


class Envs(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    ENV: Literal["dev", "prod"] = "dev"
    HI_RAG_LANGUAGE: str = "en"
    POSTGRES_URL_NO_SSL: str
    POSTGRES_URL_NO_SSL_DEV: str
    POSTGRES_TABLE_NAME: str = "KnowledgeBaseJobs"
    POSTGRES_SCHEMA: str = "public"
    REDIS_URL: str = "redis://redis:6379/2"
    REDIS_KEY_PREFIX: str = "hirag"
    REDIS_EXPIRE_TTL: int = 3600 * 24
    EMBEDDING_DIMENSION: int
    USE_HALF_VEC: bool = True

    VDB_TYPE: Literal["lancedb", "pgvector", "colbert_remote"] = "pgvector"

    COLBERT_BASE_URL: Optional[str] = None
    COLBERT_API_KEY: Optional[str] = None
    COLBERT_INDEX_PREFIX: str = "hirag"
    COLBERT_DEFAULT_INDEX_NAME: Optional[str] = None
    COLBERT_TIMEOUT: float = 30.0
    COLBERT_MAX_RETRIES: int = 3
    COLBERT_RETRY_BACKOFF_SECONDS: float = 1.0

    EMBEDDING_SERVICE_TYPE: Literal["openai", "local"] = "openai"
    EMBEDDING_BASE_URL: Optional[str] = None
    EMBEDDING_API_KEY: Optional[str] = None
    OPENAI_EMBEDDING_BASE_URL: Optional[str] = None
    OPENAI_EMBEDDING_API_KEY: Optional[str] = None
    LOCAL_EMBEDDING_BASE_URL: Optional[str] = None
    LOCAL_EMBEDDING_API_KEY: Optional[str] = None

    LLM_SERVICE_TYPE: Literal["openai", "local"] = "openai"
    LLM_BASE_URL: Optional[str] = None
    LLM_API_KEY: Optional[str] = None
    OPENAI_LLM_BASE_URL: Optional[str] = None
    OPENAI_LLM_API_KEY: Optional[str] = None
    LOCAL_LLM_BASE_URL: Optional[str] = None
    LOCAL_LLM_API_KEY: Optional[str] = None

    SEARCH_TRANSLATOR_TYPE: Literal["google", "llm"] = "google"

    RERANKER_TYPE: Literal["api", "local"] = "api"

    # API reranker (Voyage AI) settings
    VOYAGE_API_KEY: Optional[str] = None
    VOYAGE_RERANKER_MODEL_NAME: str = "rerank-2"
    VOYAGE_RERANKER_MODEL_BASE_URL: str = "https://api.voyageai.com/v1/rerank"

    # Local reranker settings
    LOCAL_RERANKER_MODEL_BASE_URL: Optional[str] = None
    LOCAL_RERANKER_MODEL_NAME: str = "Qwen3-Reranker-8B"
    LOCAL_RERANKER_MODEL_ENTRY_POINT: str = "/rerank"
    LOCAL_RERANKER_MODEL_AUTHORIZATION: Optional[str] = None

    @model_validator(mode="after")
    def validate_config_based_on_service_type(self) -> "Envs":
        if self.EMBEDDING_SERVICE_TYPE == "openai":
            if self.OPENAI_EMBEDDING_BASE_URL:
                self.EMBEDDING_BASE_URL = self.OPENAI_EMBEDDING_BASE_URL
            else:
                raise ValueError(
                    "OPENAI_EMBEDDING_BASE_URL is required when EMBEDDING_SERVICE_TYPE is openai"
                )
            if self.OPENAI_EMBEDDING_API_KEY:
                self.EMBEDDING_API_KEY = self.OPENAI_EMBEDDING_API_KEY
            else:
                raise ValueError(
                    "OPENAI_EMBEDDING_API_KEY is required when EMBEDDING_SERVICE_TYPE is openai"
                )
        elif self.EMBEDDING_SERVICE_TYPE == "local":
            if self.LOCAL_EMBEDDING_BASE_URL:
                self.EMBEDDING_BASE_URL = self.LOCAL_EMBEDDING_BASE_URL
            else:
                raise ValueError(
                    "LOCAL_EMBEDDING_BASE_URL is required when EMBEDDING_SERVICE_TYPE is local"
                )
            if self.LOCAL_EMBEDDING_API_KEY:
                self.EMBEDDING_API_KEY = self.LOCAL_EMBEDDING_API_KEY
            else:
                raise ValueError(
                    "LOCAL_EMBEDDING_API_KEY is required when EMBEDDING_SERVICE_TYPE is local"
                )
        if self.LLM_SERVICE_TYPE == "openai":
            if self.OPENAI_LLM_BASE_URL:
                self.LLM_BASE_URL = self.OPENAI_LLM_BASE_URL
            else:
                raise ValueError(
                    "OPENAI_LLM_BASE_URL is required when LLM_SERVICE_TYPE is openai"
                )
            if self.OPENAI_LLM_API_KEY:
                self.LLM_API_KEY = self.OPENAI_LLM_API_KEY
            else:
                raise ValueError(
                    "OPENAI_LLM_API_KEY is required when LLM_SERVICE_TYPE is openai"
                )
        elif self.LLM_SERVICE_TYPE == "local":
            if self.LOCAL_LLM_BASE_URL:
                self.LLM_BASE_URL = self.LOCAL_LLM_BASE_URL
            else:
                raise ValueError(
                    "LOCAL_LLM_BASE_URL is required when LLM_SERVICE_TYPE is local"
                )
            if self.LOCAL_LLM_API_KEY:
                self.LLM_API_KEY = self.LOCAL_LLM_API_KEY
            else:
                raise ValueError(
                    "LOCAL_LLM_API_KEY is required when LLM_SERVICE_TYPE is local"
                )
        if self.VDB_TYPE == "colbert_remote":
            if not self.COLBERT_BASE_URL:
                raise ValueError(
                    "COLBERT_BASE_URL is required when VDB_TYPE is colbert_remote"
                )
            if self.COLBERT_TIMEOUT <= 0:
                raise ValueError("COLBERT_TIMEOUT must be positive")
            if self.COLBERT_MAX_RETRIES < 0:
                raise ValueError("COLBERT_MAX_RETRIES must be non-negative")
            if self.COLBERT_RETRY_BACKOFF_SECONDS <= 0:
                raise ValueError("COLBERT_RETRY_BACKOFF_SECONDS must be positive")

        return self

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
