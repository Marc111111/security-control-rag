from __future__ import annotations

import re

from app.schemas import (
    DocumentChunk,
    EntityType,
    GraphEntity,
    GraphExtraction,
    GraphRelationship,
    RelationshipType,
)
from secure_rag.schema import stable_id

ENTITY_KEYWORDS = {
    EntityType.GAP: [
        "no anti-malware",
        "missing anti-malware",
        "no malware protection",
        "missing control",
        "gap",
    ],
    EntityType.THREAT: [
        "malware",
        "ransomware",
        "phishing",
        "unauthorized access",
        "data exfiltration",
    ],
    EntityType.VULNERABILITY: [
        "unprotected endpoint",
        "lack of detection",
        "outdated software",
        "misconfiguration",
        "missing patches",
    ],
    EntityType.RISK: [
        "business disruption",
        "data loss",
        "financial loss",
        "confidentiality breach",
        "integrity loss",
    ],
    EntityType.CONTROL: [
        "anti-malware",
        "endpoint protection",
        "malware defense",
        "monitoring",
        "incident response",
        "backup",
    ],
    EntityType.COMPLIANCE_REQUIREMENT: ["requirement", "shall", "must", "control objective"],
    EntityType.ASSET: ["endpoint", "server", "workstation", "laptop", "application", "data"],
}


class HeuristicGraphExtractor:
    """Small deterministic extractor suitable for bootstrapping and tests.

    It intentionally favors transparent candidate extraction over pretending to do deep NLP.
    A future slice can swap this for an LLM extractor with the same output schema.
    """

    def extract(self, chunk: DocumentChunk) -> GraphExtraction:
        text = chunk.text.lower()
        entities = self._entities(chunk, text)
        relationships = self._relationships(chunk.id, entities)
        return GraphExtraction(entities=entities, relationships=relationships)

    def _entities(self, chunk: DocumentChunk, lower_text: str) -> list[GraphEntity]:
        entities: list[GraphEntity] = []
        for entity_type, phrases in ENTITY_KEYWORDS.items():
            for phrase in phrases:
                if phrase in lower_text:
                    entities.append(_entity(entity_type, phrase, chunk.id))
        control_id = chunk.metadata.get("control_id")
        framework = chunk.metadata.get("framework")
        if control_id:
            name = f"{framework or 'Control'} {control_id}"
            entities.append(_entity(EntityType.CONTROL, str(name), chunk.id, chunk.metadata))
        pattern = r"\b(?:threat|risk|control|requirement)\s*[:\-]\s*([^.;\n]{5,90})"
        for match in re.finditer(pattern, chunk.text, re.I):
            label = match.group(1).strip()
            prefix = match.group(0).split(":", 1)[0].split("-", 1)[0].lower()
            entity_type = _prefix_entity_type(prefix)
            entities.append(_entity(entity_type, label, chunk.id))
        return _dedupe_entities(entities)

    def _relationships(
        self,
        chunk_id: str,
        entities: list[GraphEntity],
    ) -> list[GraphRelationship]:
        by_type: dict[EntityType, list[GraphEntity]] = {}
        for entity in entities:
            by_type.setdefault(entity.type, []).append(entity)
        relationships: list[GraphRelationship] = []
        relationships.extend(
            _connect(chunk_id, gaps, threats, RelationshipType.GAP_INCREASES_LIKELIHOOD_OF_THREAT)
            for gaps in by_type.get(EntityType.GAP, [])
            for threats in by_type.get(EntityType.THREAT, [])
        )
        relationships.extend(
            _connect(chunk_id, threats, vulns, RelationshipType.THREAT_EXPLOITS_VULNERABILITY)
            for threats in by_type.get(EntityType.THREAT, [])
            for vulns in by_type.get(EntityType.VULNERABILITY, [])
        )
        relationships.extend(
            _connect(chunk_id, vulns, risks, RelationshipType.VULNERABILITY_CREATES_RISK)
            for vulns in by_type.get(EntityType.VULNERABILITY, [])
            for risks in by_type.get(EntityType.RISK, [])
        )
        relationships.extend(
            _connect(chunk_id, controls, risks, RelationshipType.CONTROL_MITIGATES_RISK)
            for controls in by_type.get(EntityType.CONTROL, [])
            for risks in by_type.get(EntityType.RISK, [])
        )
        relationships.extend(
            _connect(chunk_id, controls, vulns, RelationshipType.CONTROL_ADDRESSES_VULNERABILITY)
            for controls in by_type.get(EntityType.CONTROL, [])
            for vulns in by_type.get(EntityType.VULNERABILITY, [])
        )
        relationships.extend(
            _connect(chunk_id, reqs, controls, RelationshipType.REQUIREMENT_REQUIRES_CONTROL)
            for reqs in by_type.get(EntityType.COMPLIANCE_REQUIREMENT, [])
            for controls in by_type.get(EntityType.CONTROL, [])
        )
        return relationships


def _entity(
    entity_type: EntityType,
    name: str,
    chunk_id: str,
    metadata: dict[str, object] | None = None,
) -> GraphEntity:
    normalized = " ".join(name.lower().split())
    return GraphEntity(
        id=stable_id(entity_type.value, normalized),
        type=entity_type,
        name=name,
        source_chunk_id=chunk_id,
        metadata=dict(metadata or {}),
    )


def _connect(
    chunk_id: str,
    source: GraphEntity,
    target: GraphEntity,
    relationship_type: RelationshipType,
) -> GraphRelationship:
    return GraphRelationship(
        source_id=source.id,
        target_id=target.id,
        type=relationship_type,
        source_chunk_id=chunk_id,
    )


def _dedupe_entities(entities: list[GraphEntity]) -> list[GraphEntity]:
    seen: set[str] = set()
    unique: list[GraphEntity] = []
    for entity in entities:
        if entity.id not in seen:
            unique.append(entity)
            seen.add(entity.id)
    return unique


def _prefix_entity_type(prefix: str) -> EntityType:
    if "threat" in prefix:
        return EntityType.THREAT
    if "risk" in prefix:
        return EntityType.RISK
    if "requirement" in prefix:
        return EntityType.COMPLIANCE_REQUIREMENT
    return EntityType.CONTROL
