from app.retrieval.vector_store import _qdrant_point_id


def test_qdrant_point_id_uses_full_chunk_id_to_avoid_collisions() -> None:
    prefix = "a" * 32

    first = _qdrant_point_id(f"{prefix}-source-one")
    second = _qdrant_point_id(f"{prefix}-source-two")

    assert first != second
