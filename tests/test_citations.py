from app.providers import normalize_citations


def test_normalize_citations_adds_brackets_to_valid_page_reference():
    evidence = [{"chunk_id": 38, "page_number": 1}]

    result = normalize_citations("Policy says Tuesday. Source 38, Page 1", evidence)

    assert result == "Policy says Tuesday. [Source 38, Page 1]"


def test_normalize_citations_keeps_canonical_valid_reference():
    evidence = [{"chunk_id": 38, "page_number": 1}]

    result = normalize_citations("Policy says Tuesday. [Source 38, Page 1]", evidence)

    assert result == "Policy says Tuesday. [Source 38, Page 1]"


def test_normalize_citations_removes_unknown_source_or_page():
    evidence = [{"chunk_id": 38, "page_number": 1}]

    result = normalize_citations(
        "Policy says Tuesday. Source 999, Page 9. [Source 38, Page 2]",
        evidence,
    )

    assert result == "Policy says Tuesday."


def test_normalize_citations_supports_sources_without_page_numbers():
    evidence = [{"chunk_id": 42, "page_number": None}]

    result = normalize_citations("See Source 42.", evidence)

    assert result == "See [Source 42]."


def test_normalize_citations_removes_unknown_page_less_source():
    evidence = [{"chunk_id": 42, "page_number": None}]

    result = normalize_citations("See [Source 99].", evidence)

    assert result == "See."
