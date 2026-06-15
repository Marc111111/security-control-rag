from app.graph.store import MemoryGraphStore
from app.planning.planner import RiskQuestionPlanner
from app.retrieval.bm25 import KeywordIndex
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector_store import MemoryDenseStore
from app.schemas import DocumentChunk
from secure_rag.embeddings import HashEmbeddingClient


def test_hybrid_retriever_merges_dense_and_keyword_hits() -> None:
    embedding_client = HashEmbeddingClient(dimensions=32)
    dense_store = MemoryDenseStore()
    keyword_index = KeywordIndex()
    graph_store = MemoryGraphStore()
    retriever = HybridRetriever(
        embedding_client=embedding_client,
        dense_store=dense_store,
        keyword_index=keyword_index,
        graph_store=graph_store,
    )
    chunks = [
        DocumentChunk(
            id="c1",
            text="Anti-malware controls mitigate malware and ransomware threats.",
            metadata={"source_path": "standards/example.md", "control_id": "10.1"},
        ),
        DocumentChunk(
            id="c2",
            text="Physical access reviews apply to facilities.",
            metadata={"source_path": "standards/physical.md"},
        ),
    ]
    embeddings = embedding_client.embed([chunk.text for chunk in chunks])
    retriever.add_chunks(chunks, embeddings)

    plan = RiskQuestionPlanner().plan("No anti-malware solution is in place.")
    hits, _ = retriever.retrieve(plan, top_k=4)

    assert hits
    assert hits[0].chunk.id == "c1"
    assert {hit.chunk.id for hit in hits} == {"c1", "c2"}

