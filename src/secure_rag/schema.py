from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeAlias

Metadata: TypeAlias = dict[str, str | int | float | bool | None]


def stable_id(*parts: object) -> str:
    digest = sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceDocument:
    path: Path
    text: str
    metadata: Metadata = field(default_factory=dict)

    @property
    def source_id(self) -> str:
        return stable_id(self.path.as_posix(), self.metadata)


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class ControlAnswer:
    answer: str
    sources: list[Metadata]
    insufficient_evidence: bool = False
    raw: dict[str, Any] = field(default_factory=dict)
