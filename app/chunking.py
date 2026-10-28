from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    content: str


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[TextChunk]:
    """Split normalized text into deterministic character windows with bounded overlap."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[TextChunk] = []
    step = chunk_size - overlap
    for index, start in enumerate(range(0, len(normalized), step)):
        content = normalized[start : start + chunk_size].strip()
        if content:
            chunks.append(TextChunk(index=index, content=content))
        if start + chunk_size >= len(normalized):
            break
    return chunks
