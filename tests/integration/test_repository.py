import psycopg
import pytest

from app.config import get_settings
from app.pdf import PDFChunk
from app.repository import insert_chunks, search_chunks

pytestmark = pytest.mark.integration


def test_reingestion_replaces_chunks_and_preserves_page_metadata(
    document_id, embedding_vector
):
    original_chunks = [
        PDFChunk(index=0, page_number=1, content="Original page one"),
        PDFChunk(index=1, page_number=2, content="Original page two"),
    ]
    assert insert_chunks(
        document_id,
        "Original",
        original_chunks,
        [embedding_vector(1.0), embedding_vector(0.0, 1.0)],
    ) == 2

    replacement = [PDFChunk(index=0, page_number=7, content="Replacement content")]
    assert insert_chunks(
        document_id, "Replacement", replacement, [embedding_vector(0.0, 0.0, 1.0)]
    ) == 1

    matches = search_chunks(embedding_vector(0.0, 0.0, 1.0), top_k=100)
    own_matches = [match for match in matches if match["document_id"] == document_id]
    assert len(own_matches) == 1
    assert own_matches[0]["title"] == "Replacement"
    assert own_matches[0]["page_number"] == 7
    assert own_matches[0]["content"] == "Replacement content"
    assert own_matches[0]["similarity"] == pytest.approx(1.0)


def test_failed_reingestion_rolls_back_and_preserves_original_chunks(
    document_id, embedding_vector
):
    original_chunks = [
        PDFChunk(index=0, page_number=1, content="Original page one"),
        PDFChunk(index=1, page_number=2, content="Original page two"),
    ]
    insert_chunks(
        document_id,
        "Original",
        original_chunks,
        [embedding_vector(1.0), embedding_vector(0.0, 1.0)],
    )

    invalid_replacement = [
        PDFChunk(index=0, page_number=7, content="First replacement"),
        PDFChunk(index=0, page_number=8, content="Duplicate chunk index"),
    ]
    with pytest.raises(psycopg.errors.UniqueViolation):
        insert_chunks(
            document_id,
            "Replacement",
            invalid_replacement,
            [embedding_vector(0.0, 0.0, 1.0), embedding_vector(0.0, 0.0, 1.0)],
        )

    with psycopg.connect(get_settings().database_url) as connection:
        rows = connection.execute(
            """SELECT title, page_number, chunk_index, content
               FROM document_chunks
               WHERE document_id = %s
               ORDER BY chunk_index""",
            (document_id,),
        ).fetchall()
    assert rows == [
        ("Original", 1, 0, "Original page one"),
        ("Original", 2, 1, "Original page two"),
    ]


def test_vector_search_orders_results_by_cosine_similarity(document_id, embedding_vector):
    chunks = [
        PDFChunk(index=0, page_number=1, content="Exact match"),
        PDFChunk(index=1, page_number=2, content="Partial match"),
        PDFChunk(index=2, page_number=3, content="Orthogonal match"),
    ]
    insert_chunks(
        document_id,
        "Ordering",
        chunks,
        [
            embedding_vector(1.0),
            embedding_vector(0.8, 0.6),
            embedding_vector(0.0, 1.0),
        ],
    )

    matches = search_chunks(embedding_vector(1.0), top_k=100)
    own_matches = [match for match in matches if match["document_id"] == document_id]
    assert [match["content"] for match in own_matches] == [
        "Exact match",
        "Partial match",
        "Orthogonal match",
    ]
    assert [match["page_number"] for match in own_matches] == [1, 2, 3]
    assert [match["similarity"] for match in own_matches] == pytest.approx([1.0, 0.8, 0.0])


def test_vector_search_respects_top_k_across_documents(
    document_id_factory, embedding_vector
):
    closest_document_id = document_id_factory()
    other_document_id = document_id_factory()
    insert_chunks(
        closest_document_id,
        "Closest",
        [PDFChunk(index=0, page_number=1, content="Closest result")],
        [embedding_vector(1.0)],
    )
    insert_chunks(
        other_document_id,
        "Other",
        [PDFChunk(index=0, page_number=1, content="Other result")],
        [embedding_vector(0.8, 0.6)],
    )

    matches = search_chunks(embedding_vector(1.0), top_k=1)

    assert len(matches) == 1
    assert matches[0]["document_id"] == closest_document_id
    assert matches[0]["content"] == "Closest result"
