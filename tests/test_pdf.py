from io import BytesIO

import pytest

from app.pdf import (
    PDFExtractionError,
    PDFPage,
    PDFUploadTooLargeError,
    chunk_pdf_pages,
    extract_pdf_pages,
    read_pdf_upload,
)


def _text_pdf(*page_texts: str) -> bytes:
    """Build a minimal text PDF without adding a test-only PDF dependency."""
    font_object = 3 + (2 * len(page_texts))
    page_objects = " ".join(f"{3 + (2 * index)} 0 R" for index in range(len(page_texts)))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{page_objects}] /Count {len(page_texts)} >>".encode(),
    ]

    for index, text in enumerate(page_texts):
        page_object = 3 + (2 * index)
        content_object = page_object + 1
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
        objects.extend(
            [
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 {font_object} 0 R >> >> "
                    f"/Contents {content_object} 0 R >>"
                ).encode(),
                f"<< /Length {len(stream)} >>\nstream\n".encode()
                + stream
                + b"\nendstream",
            ]
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    result = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(result))
        result.extend(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_offset = len(result)
    result.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    result.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode())
    result.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(result)


def test_extract_pdf_text_preserves_page_numbers():
    pages = extract_pdf_pages(_text_pdf("Alpha policy", "Beta procedure"))

    assert pages == [
        PDFPage(page_number=1, text="Alpha policy"),
        PDFPage(page_number=2, text="Beta procedure"),
    ]


def test_chunk_pdf_pages_assigns_global_indexes_and_pages():
    chunks = chunk_pdf_pages(
        [PDFPage(1, "abcdefgh"), PDFPage(2, "ijklmnop")],
        chunk_size=5,
        overlap=1,
    )

    assert [chunk.index for chunk in chunks] == [0, 1, 2, 3]
    assert [chunk.page_number for chunk in chunks] == [1, 1, 2, 2]


def test_non_pdf_content_is_rejected():
    try:
        extract_pdf_pages(b"not a pdf")
    except ValueError as exc:
        assert str(exc) == "Uploaded file is not a PDF"
    else:
        raise AssertionError("Expected invalid content to be rejected")


def test_upload_at_size_limit_is_accepted():
    assert read_pdf_upload(BytesIO(b"12345"), max_bytes=5, read_size=2) == b"12345"


def test_oversized_upload_is_rejected():
    with pytest.raises(PDFUploadTooLargeError, match="maximum upload size of 5 bytes"):
        read_pdf_upload(BytesIO(b"123456"), max_bytes=5, read_size=2)


def test_upload_read_error_becomes_extraction_error():
    class BrokenStream:
        def read(self, size):
            raise OSError("disk read failed")

    with pytest.raises(PDFExtractionError, match="PDF upload could not be read"):
        read_pdf_upload(BrokenStream(), max_bytes=10)
