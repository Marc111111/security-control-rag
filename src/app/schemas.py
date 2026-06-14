from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EntityType(StrEnum):
    ASSET = "Asset"
    GAP = "Gap"
    THREAT = "Threat"
    VULNERABILITY = "Vulnerability"
    RISK = "Risk"
    CONTROL = "Control"
    COMPLIANCE_REQUIREMENT = "ComplianceRequirement"
    EVIDENCE_SOURCE = "EvidenceSource"


class RelationshipType(StrEnum):
    THREAT_EXPLOITS_VULNERABILITY = "THREAT_EXPLOITS_VULNERABILITY"
    VULNERABILITY_CREATES_RISK = "VULNERABILITY_CREATES_RISK"
    CONTROL_MITIGATES_RISK = "CONTROL_MITIGATES_RISK"
    CONTROL_ADDRESSES_VULNERABILITY = "CONTROL_ADDRESSES_VULNERABILITY"
    GAP_INCREASES_LIKELIHOOD_OF_THREAT = "GAP_INCREASES_LIKELIHOOD_OF_THREAT"
    REQUIREMENT_REQUIRES_CONTROL = "REQUIREMENT_REQUIRES_CONTROL"
    CHUNK_MENTIONS_ENTITY = "CHUNK_MENTIONS_ENTITY"


class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEntity(BaseModel):
    id: str
    type: EntityType
    name: str
    source_chunk_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphRelationship(BaseModel):
    source_id: str
    target_id: str
    type: RelationshipType
    source_chunk_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphExtraction(BaseModel):
    entities: list[GraphEntity] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(default_factory=list)


class SubQuestion(BaseModel):
    label: str
    question: str
    focus: str


class QueryPlan(BaseModel):
    original_question: str
    sub_questions: list[SubQuestion]
    retrieval_queries: list[str]


class RetrievedEvidence(BaseModel):
    chunk: DocumentChunk
    score: float
    source: str
    retrieval_method: str
    sub_question: str | None = None


class MatrixRow(BaseModel):
    gap: str = ""
    threat: str = ""
    vulnerability: str = ""
    risk: str = ""
    likelihood: str = "unknown"
    impact: str = "unknown"
    controls: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class StructuredRiskAnswer(BaseModel):
    executive_summary: str = ""
    assumptions: list[str] = Field(default_factory=list)
    threats: list[str] = Field(default_factory=list)
    vulnerabilities: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommended_controls: list[str] = Field(default_factory=list)
    risk_control_matrix: list[MatrixRow] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    source_citations: list[dict[str, Any]] = Field(default_factory=list)
    from_retrieved_evidence: str = ""
    general_model_reasoning: str = ""


class GraphRagAnswer(BaseModel):
    answer: StructuredRiskAnswer
    insufficient_evidence: bool
    sources: list[dict[str, Any]]
    debug: dict[str, Any] = Field(default_factory=dict)

