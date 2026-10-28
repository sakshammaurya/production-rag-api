import pytest
from pydantic import ValidationError

from app.schemas import AskRequest, IngestRequest


def test_question_is_trimmed():
    assert AskRequest(question="  What is in the handbook?  ").question == "What is in the handbook?"


def test_blank_question_is_rejected():
    with pytest.raises(ValidationError):
        AskRequest(question="   ")


def test_blank_document_metadata_is_rejected():
    with pytest.raises(ValidationError):
        IngestRequest(document_id=" ", title="Handbook", text="content")
