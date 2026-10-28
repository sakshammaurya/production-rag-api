import pytest

from app.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text(" \n\t ") == []


def test_chunks_have_stable_indexes_and_overlap():
    chunks = chunk_text("abcdefghij", chunk_size=6, overlap=2)
    assert [chunk.index for chunk in chunks] == [0, 1]
    assert [chunk.content for chunk in chunks] == ["abcdef", "efghij"]


@pytest.mark.parametrize("chunk_size,overlap", [(0, 0), (5, -1), (5, 5), (5, 6)])
def test_invalid_chunk_settings_raise(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=chunk_size, overlap=overlap)
