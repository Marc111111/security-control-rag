from secure_rag.retriever import expand_security_query, rerank_hits
from secure_rag.schema import Chunk, RetrievalHit


def test_expand_security_query_maps_playbook_to_incident_response_terms() -> None:
    expanded = expand_security_query("small company has no ransomware playbook")

    assert "incident response plan" in expanded
    assert "data recovery" in expanded
    assert "implementation group 1" in expanded


def test_rerank_hits_rewards_keyword_overlap_and_standards_sources() -> None:
    weaker_vector_but_relevant = RetrievalHit(
        chunk=Chunk(
            id="standard",
            text="Incident response plan tabletop exercise roles responsibilities ransomware",
            metadata={"source_path": "standards/cis.pdf"},
        ),
        score=0.55,
        vector_score=0.55,
    )
    stronger_vector_but_generic = RetrievalHit(
        chunk=Chunk(
            id="generic",
            text="Control framework general overview",
            metadata={"source_path": "notes/general.txt"},
        ),
        score=0.57,
        vector_score=0.57,
    )

    reranked = rerank_hits("ransomware incident response plan", [
        stronger_vector_but_generic,
        weaker_vector_but_relevant,
    ])

    assert reranked[0].chunk.id == "standard"
