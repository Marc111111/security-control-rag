from pathlib import Path

import pytest

from secure_rag.chunking import chunk_document
from secure_rag.schema import SourceDocument


def test_chunk_document_preserves_metadata_and_offsets() -> None:
    document = SourceDocument(
        path=Path("policy.md"),
        text="alpha beta gamma delta epsilon",
        metadata={"framework": "ISO27001"},
    )

    chunks = chunk_document(document, chunk_size=16, overlap=4)

    assert len(chunks) == 3
    assert chunks[0].metadata["framework"] == "ISO27001"
    assert chunks[0].metadata["source_path"] == "policy.md"
    assert chunks[1].metadata["char_start"] == 12
    assert chunks[0].id != chunks[1].id


def test_chunk_document_rejects_invalid_overlap() -> None:
    document = SourceDocument(path=Path("x.txt"), text="hello")

    with pytest.raises(ValueError, match="overlap"):
        chunk_document(document, chunk_size=10, overlap=10)

