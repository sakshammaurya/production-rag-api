import pytest
from pydantic import ValidationError

from app.config import Settings


def test_pdf_upload_limit_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(database_url="postgresql://example", max_pdf_upload_bytes=0)


def test_tenant_api_keys_require_strong_unique_values():
    with pytest.raises(ValidationError, match="at least 16 characters"):
        Settings(database_url="postgresql://example", tenant_api_keys={"tenant-a": "short"})

    with pytest.raises(ValidationError, match="must be unique"):
        Settings(
            database_url="postgresql://example",
            tenant_api_keys={
                "tenant-a": "same-api-key-000000",
                "tenant-b": "same-api-key-000000",
            },
        )
