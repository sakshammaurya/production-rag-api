import re


_PAGE_CITATION = re.compile(
    r"\[?\s*Source\s+(\d+)\s*,\s*Page\s+(\d+)\s*\]?",
    re.IGNORECASE,
)
_CHUNK_CITATION = re.compile(
    r"\[?\s*Source\s+(\d+)\s*\]?(?!\s*,\s*Page)(?!\d)",
    re.IGNORECASE,
)


def normalize_citations(answer: str, evidence: list[dict]) -> str:
    """Canonicalize model citation labels and remove references not in supplied evidence."""
    page_sources = {
        (str(item["chunk_id"]), str(item["page_number"]))
        for item in evidence
        if item.get("page_number") is not None
    }
    chunk_sources = {str(item["chunk_id"]) for item in evidence}

    def replace_page(match: re.Match[str]) -> str:
        citation = (match.group(1), match.group(2))
        if citation not in page_sources:
            return ""
        return f"[Source {citation[0]}, Page {citation[1]}]"

    answer = _PAGE_CITATION.sub(replace_page, answer)

    def replace_chunk(match: re.Match[str]) -> str:
        chunk_id = match.group(1)
        if chunk_id not in chunk_sources:
            return ""
        return f"[Source {chunk_id}]"

    answer = _CHUNK_CITATION.sub(replace_chunk, answer)
    answer = re.sub(r"(?<=[\w.,!?;:])\[Source\b", " [Source", answer)
    answer = re.sub(r"[ \t]{2,}", " ", answer)
    answer = re.sub(r"\s+([,.!?;:])", r"\1", answer)
    answer = re.sub(r"([.!?])(?:\s*[.!?])+", r"\1", answer)
    return answer.strip()
