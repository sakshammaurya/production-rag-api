from collections.abc import Sequence

import psycopg
from pgvector.psycopg import register_vector

from app.config import get_settings


def _connect():
    connection = psycopg.connect(get_settings().database_url)
    register_vector(connection)
    return connection


def insert_chunks(
    document_id: str,
    title: str,
    chunks: Sequence,
    embeddings: Sequence[list[float]],
) -> int:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have equal length")

    # The connection context commits on success and rolls back on failure, preserving
    # the previous document version if replacement insertion fails.
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
        cur.executemany(
            """INSERT INTO document_chunks
               (document_id, title, chunk_index, content, embedding)
               VALUES (%s, %s, %s, %s, %s)""",
            [
                (document_id, title, chunk.index, chunk.content, embedding)
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ],
        )
    return len(chunks)


def search_chunks(query_embedding: list[float], top_k: int) -> list[dict]:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT id, document_id, title, content,
                      1 - (embedding <=> %s::vector) AS similarity
               FROM document_chunks
               ORDER BY embedding <=> %s::vector
               LIMIT %s""",
            (query_embedding, query_embedding, top_k),
        )
        rows = cur.fetchall()

    return [
        {
            "chunk_id": row[0],
            "document_id": row[1],
            "title": row[2],
            "content": row[3],
            "similarity": float(row[4]),
        }
        for row in rows
    ]
