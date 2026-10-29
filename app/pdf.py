from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.chunking import chunk_text


class PDFExtractionError(ValueError):
    """Raised when an uploaded PDF cannot be safely read."""


@dataclass(frozen=True)
class PDFPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class PDFChunk:
    index: int
    page_number: int
    content: str


def extract_pdf_pages(content: bytes) -> list[PDFPage]:
    if not content.startswith(b"%PDF-"):
        raise PDFExtractionError("Uploaded file is not a PDF")

    try:
        reader = PdfReader(BytesIO(content), strict=False)
        if reader.is_encrypted:
            raise PDFExtractionError("Encrypted PDFs are not supported")
        pages = [
            PDFPage(page_number=index, text=page.extract_text() or "")
            for index, page in enumerate(reader.pages, start=1)
        ]
    except PDFExtractionError:
        raise
    except (PdfReadError, OSError, ValueError) as exc:
        raise PDFExtractionError("PDF could not be read") from exc

    if not any(page.text.strip() for page in pages):
        raise PDFExtractionError("PDF contains no extractable text")
    return pages


def chunk_pdf_pages(
    pages: list[PDFPage], chunk_size: int = 1200, overlap: int = 150
) -> list[PDFChunk]:
    chunks: list[PDFChunk] = []
    for page in pages:
        for chunk in chunk_text(page.text, chunk_size, overlap):
            chunks.append(
                PDFChunk(
                    index=len(chunks),
                    page_number=page.page_number,
                    content=chunk.content,
                )
            )
    return chunks
