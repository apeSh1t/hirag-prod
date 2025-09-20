from typing import Optional

from pydantic import ConfigDict, field_validator
from pydantic_settings import BaseSettings


class ColbertConfig(BaseSettings):
    """Configuration for ColBERT remote service."""

    model_config = ConfigDict(
        alias_generator=lambda x: f"colbert_{x}".upper(),
        populate_by_name=True,
        extra="ignore",
    )

    base_url: str
    api_key: Optional[str] = None
    index_prefix: str = "hirag"
    default_index_name: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0

    @field_validator("timeout", "retry_backoff_seconds")
    @classmethod
    def _validate_positive(cls, value: float, info):  # type: ignore[override]
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

    @field_validator("max_retries")
    @classmethod
    def _validate_non_negative(cls, value: int):
        if value < 0:
            raise ValueError("max_retries must be non-negative")
        return value
