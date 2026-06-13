from dataclasses import dataclass

from secure_rag.engine import ControlRagEngine
from secure_rag.schema import Chunk, RetrievalHit


@dataclass
class FakeRetriever:
    hits: list[RetrievalHit]

    def retrieve(self, query: str, *, top_k: int = 8) -> list[RetrievalHit]:
        return self.hits[:top_k]


class FakeChatClient:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] | None = None

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        return "Use MFA for admins. Sources: [S1]"


def test_engine_returns_insufficient_evidence_without_hits() -> None:
    chat_client = FakeChatClient()
    engine = ControlRagEngine(retriever=FakeRetriever([]), chat_client=chat_client)

    answer = engine.answer("unknown risk")

    assert answer.insufficient_evidence is True
    assert answer.sources == []
    assert chat_client.messages is None


def test_engine_calls_llm_with_supported_hits() -> None:
    hit = RetrievalHit(
        chunk=Chunk(
            id="chunk-1",
            text="Require multi-factor authentication for administrative accounts.",
            metadata={"source_path": "iso.csv"},
        ),
        score=0.5,
    )
    chat_client = FakeChatClient()
    engine = ControlRagEngine(retriever=FakeRetriever([hit]), chat_client=chat_client)

    answer = engine.answer("admin access")

    assert answer.insufficient_evidence is False
    assert "MFA" in answer.answer
    assert answer.sources[0]["chunk_id"] == "chunk-1"
    assert chat_client.messages is not None

