from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
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
    min_similarity: float = 0.60
    chunk_size: int = 1200
    chunk_overlap: int = 150
    max_pdf_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    tenant_api_keys: dict[str, SecretStr] = Field(default_factory=dict)

    @field_validator("tenant_api_keys")
    @classmethod
    def validate_tenant_api_keys(
        cls, value: dict[str, SecretStr]
    ) -> dict[str, SecretStr]:
        seen_keys = set()
        for tenant_id, api_key_secret in value.items():
            api_key = api_key_secret.get_secret_value()
            if not tenant_id or len(tenant_id) > 100 or not tenant_id.replace("-", "").isalnum():
                raise ValueError("tenant IDs must be 1-100 alphanumeric or hyphen characters")
            if len(api_key) < 16:
                raise ValueError("tenant API keys must contain at least 16 characters")
            if api_key in seen_keys:
                raise ValueError("tenant API keys must be unique")
            seen_keys.add(api_key)
        return value

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
