from __future__ import annotations

from collections.abc import Iterable

from secure_rag.schema import Chunk, SourceDocument, stable_id


def chunk_document(
    document: SourceDocument,
    *,
    chunk_size: int = 1_500,
    overlap: int = 200,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = " ".join(document.text.split())
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        segment = text[start:end].strip()
        metadata = {
            **document.metadata,
            "source_path": str(document.path),
            "chunk_index": index,
            "char_start": start,
            "char_end": end,
        }
        chunks.append(
            Chunk(
                id=stable_id(document.path.as_posix(), document.metadata, index, segment),
                text=segment,
                metadata=metadata,
            )
        )
        if end == len(text):
            break
        start = end - overlap
        index += 1
    return chunks


def chunk_documents(
    documents: Iterable[SourceDocument],
    *,
    chunk_size: int = 1_500,
    overlap: int = 200,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return chunks

