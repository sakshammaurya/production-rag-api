import pytest

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
