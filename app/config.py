from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    openai_api_key: str = ""
    openai_base_url: str | None = None
    provider_timeout_seconds: float = 600.0
    provider_max_retries: int = 0
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    top_k: int = 5
    min_similarity: float = 0.25
    chunk_size: int = 1200
    chunk_overlap: int = 150

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
