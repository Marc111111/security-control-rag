from __future__ import annotations

from typing import Protocol

from app.schemas import GraphEntity, GraphRelationship


class GraphStore(Protocol):
    def upsert(self, entities: list[GraphEntity], relationships: list[GraphRelationship]) -> None:
        ...

    def search_related(self, query: str, *, limit: int = 20) -> list[dict[str, object]]:
        ...


class MemoryGraphStore:
    def __init__(self) -> None:
        self.entities: dict[str, GraphEntity] = {}
        self.relationships: list[GraphRelationship] = []

    def upsert(self, entities: list[GraphEntity], relationships: list[GraphRelationship]) -> None:
        for entity in entities:
            self.entities[entity.id] = entity
        existing = {
            (rel.source_id, rel.target_id, rel.type, rel.source_chunk_id)
            for rel in self.relationships
        }
        for relationship in relationships:
            key = (
                relationship.source_id,
                relationship.target_id,
                relationship.type,
                relationship.source_chunk_id,
            )
            if key not in existing:
                self.relationships.append(relationship)
                existing.add(key)

    def search_related(self, query: str, *, limit: int = 20) -> list[dict[str, object]]:
        terms = {term for term in query.lower().split() if len(term) > 2}
        matched = [
            entity
            for entity in self.entities.values()
            if terms & set(entity.name.lower().replace("-", " ").split())
        ]
        matched_ids = {entity.id for entity in matched}
        rows: list[dict[str, object]] = []
        for rel in self.relationships:
            if rel.source_id in matched_ids or rel.target_id in matched_ids:
                rows.append(
                    {
                        "relationship": rel.type.value,
                        "source": self.entities.get(rel.source_id).model_dump()
                        if rel.source_id in self.entities
                        else rel.source_id,
                        "target": self.entities.get(rel.target_id).model_dump()
                        if rel.target_id in self.entities
                        else rel.target_id,
                        "source_chunk_id": rel.source_chunk_id,
                    }
                )
            if len(rows) >= limit:
                break
        return rows


class Neo4jGraphStore:
    def __init__(self, *, uri: str, user: str, password: str) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Install neo4j to use Neo4jGraphStore") from exc
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def upsert(self, entities: list[GraphEntity], relationships: list[GraphRelationship]) -> None:
        with self.driver.session() as session:
            for entity in entities:
                session.run(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e.name = $name, e.type = $type, e.source_chunk_id = $source_chunk_id
                    """,
                    id=entity.id,
                    name=entity.name,
                    type=entity.type.value,
                    source_chunk_id=entity.source_chunk_id,
                )
            for rel in relationships:
                session.run(
                    """
                    MATCH (s:Entity {id: $source_id})
                    MATCH (t:Entity {id: $target_id})
                    MERGE (s)-[r:RELATED {type: $type, source_chunk_id: $source_chunk_id}]->(t)
                    """,
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    type=rel.type.value,
                    source_chunk_id=rel.source_chunk_id,
                )

    def search_related(self, query: str, *, limit: int = 20) -> list[dict[str, object]]:
        terms = [term for term in query.lower().split() if len(term) > 2]
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Entity)-[r:RELATED]->(t:Entity)
                WHERE any(term IN $terms WHERE
                    toLower(s.name) CONTAINS term OR toLower(t.name) CONTAINS term
                )
                RETURN s, r.type AS relationship, t, r.source_chunk_id AS source_chunk_id
                LIMIT $limit
                """,
                terms=terms,
                limit=limit,
            )
            return [
                {
                    "source": dict(record["s"]),
                    "relationship": record["relationship"],
                    "target": dict(record["t"]),
                    "source_chunk_id": record["source_chunk_id"],
                }
                for record in result
            ]
