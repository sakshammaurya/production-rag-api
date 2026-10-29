import pytest
from pydantic import ValidationError

from app.config import Settings


def test_pdf_upload_limit_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(database_url="postgresql://example", max_pdf_upload_bytes=0)
