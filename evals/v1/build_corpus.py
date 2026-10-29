"""Build the deterministic text PDF used by evaluation dataset v1."""

import json
from pathlib import Path

ROOT = Path(__file__).parent


def build_text_pdf(page_texts: list[str]) -> bytes:
    """Build a minimal PDF with one reviewable line of text per page."""
    font_object = 3 + (2 * len(page_texts))
    page_objects = " ".join(f"{3 + (2 * index)} 0 R" for index in range(len(page_texts)))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{page_objects}] /Count {len(page_texts)} >>".encode(),
    ]

    for index, page_text in enumerate(page_texts):
        page_object = 3 + (2 * index)
        content_object = page_object + 1
        escaped = page_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 11 Tf 54 720 Td ({escaped}) Tj ET".encode()
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


def main() -> None:
    pages_path = ROOT / "corpus" / "pages.json"
    pages = json.loads(pages_path.read_text(encoding="utf-8"))
    (ROOT / "corpus" / "employee_handbook.pdf").write_bytes(build_text_pdf(pages))


if __name__ == "__main__":
    main()
