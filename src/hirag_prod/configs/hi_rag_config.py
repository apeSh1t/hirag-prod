from typing import Literal, Optional

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class HiRAGConfig(BaseSettings):
    """HiRAG system configuration"""

    model_config = ConfigDict(
        alias_generator=lambda x: x.upper(), populate_by_name=True, extra="ignore"
    )

    # Database configuration
    vector_db_path: str = "kb/hirag.db"
    graph_db_path: str = "kb/hirag.gpickle"
    vdb_type: Literal["lancedb", "pgvector", "colbert_remote"] = "pgvector"
    gdb_type: Literal["networkx", "neo4j"] = (
        "networkx"  # TODO: neo4j not implemented yet
    )

    # Chunking configuration
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Batch processing configuration
    embedding_batch_size: int = 1000
    entity_upsert_concurrency: int = 32
    relation_upsert_concurrency: int = 32

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0

    # Vector and Schema Configuration
    embedding_dimension: int

    similarity_threshold: float = 0.5
    similarity_max_difference: float = 0.15
    max_references: int = 3

    # ColBERT remote configuration (used when vdb_type == "colbert_remote")
    colbert_base_url: Optional[str] = None
    colbert_api_key: Optional[str] = None
    colbert_index_prefix: str = "hirag"
    colbert_default_index_name: Optional[str] = None
    colbert_timeout: float = 30.0
    colbert_max_retries: int = 3
    colbert_retry_backoff_seconds: float = 1.0

    max_chunk_ids_per_query: int = 10
    default_query_top_k: int = 10
    default_query_top_n: int = 5
    default_link_top_k: int = 30
    default_passage_node_weight: float = 0.6
    default_pagerank_damping: float = 0.5
