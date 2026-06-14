from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from secure_rag import llm


class _StreamingResponse:
    def __init__(self, cancel_event: threading.Event) -> None:
        self.cancel_event = cancel_event
        self.lines = [
            b'{"message":{"content":"partial"},"done":false}\n',
        ]

    def __enter__(self) -> _StreamingResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def readline(self) -> bytes:
        if not self.lines:
            return b""
        self.cancel_event.set()
        return self.lines.pop(0)


def test_ollama_chat_client_streaming_respects_cancel_event(monkeypatch: Any) -> None:
    cancel_event = threading.Event()
    captured_payload: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _StreamingResponse:
        captured_payload.update(json.loads(request.data.decode("utf-8")))
        return _StreamingResponse(cancel_event)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    client = llm.OllamaChatClient(
        model="qwen3:14b",
        base_url="http://ollama.local",
        keep_alive="0s",
        cancel_event=cancel_event,
    )

    with pytest.raises(llm.ChatCancelled):
        client.chat([{"role": "user", "content": "hello"}])

    assert captured_payload["stream"] is True
    assert captured_payload["keep_alive"] == "0s"
