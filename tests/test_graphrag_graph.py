from app.graph.extractor import HeuristicGraphExtractor
from app.schemas import DocumentChunk, EntityType, RelationshipType


def test_graph_extractor_finds_security_entities_and_relationships() -> None:
    chunk = DocumentChunk(
        id="chunk-1",
        text=(
            "A gap exists: no anti-malware. Malware exploits unprotected endpoint "
            "conditions and creates business disruption risk. Endpoint protection "
            "control mitigates business disruption."
        ),
        metadata={"framework": "CIS Controls", "control_id": "10.1"},
    )

    extraction = HeuristicGraphExtractor().extract(chunk)

    entity_types = {entity.type for entity in extraction.entities}
    relationship_types = {relationship.type for relationship in extraction.relationships}

    assert EntityType.GAP in entity_types
    assert EntityType.THREAT in entity_types
    assert EntityType.VULNERABILITY in entity_types
    assert EntityType.RISK in entity_types
    assert EntityType.CONTROL in entity_types
    assert RelationshipType.THREAT_EXPLOITS_VULNERABILITY in relationship_types
    assert RelationshipType.CONTROL_MITIGATES_RISK in relationship_types

