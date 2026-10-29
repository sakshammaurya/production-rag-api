from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.chunking import chunk_text


class PDFExtractionError(ValueError):
    """Raised when an uploaded PDF cannot be safely read."""


class PDFUploadTooLargeError(ValueError):
    """Raised as soon as an uploaded PDF exceeds the configured byte limit."""


@dataclass(frozen=True)
class PDFPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class PDFChunk:
    index: int
    page_number: int
    content: str


def read_pdf_upload(
    stream: BinaryIO,
    max_bytes: int,
    read_size: int = 64 * 1024,
) -> bytes:
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if read_size < 1:
        raise ValueError("read_size must be positive")

    content = bytearray()
    try:
        while chunk := stream.read(min(read_size, max_bytes - len(content) + 1)):
            content.extend(chunk)
            if len(content) > max_bytes:
                raise PDFUploadTooLargeError(
                    f"PDF exceeds the maximum upload size of {max_bytes} bytes"
                )
    except PDFUploadTooLargeError:
        raise
    except OSError as exc:
        raise PDFExtractionError("PDF upload could not be read") from exc
    return bytes(content)


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
