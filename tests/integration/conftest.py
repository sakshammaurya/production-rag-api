from pathlib import Path
from uuid import uuid4

import psycopg
import pytest

from app.config import get_settings


def vector_with(*values: float) -> list[float]:
    return [*values, *([0.0] * (768 - len(values)))]


@pytest.fixture(scope="session", autouse=True)
def initialized_database():
    schema_path = Path(__file__).parents[2] / "sql" / "schema.sql"
    statements = [statement.strip() for statement in schema_path.read_text().split(";")]
    with psycopg.connect(get_settings().database_url) as connection:
        for statement in statements:
            if statement:
                connection.execute(statement)


@pytest.fixture
def document_id():
    value = f"integration-{uuid4()}"
    yield value
    with psycopg.connect(get_settings().database_url) as connection:
        connection.execute("DELETE FROM document_chunks WHERE document_id = %s", (value,))


@pytest.fixture
def embedding_vector():
    return vector_with
