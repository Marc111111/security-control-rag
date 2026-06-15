from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Protocol


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


class ChatCancelled(RuntimeError):
    pass


class OllamaChatClient:
    def __init__(
        self,
        *,
        model: str = "qwen3:14b",
        base_url: str = "http://localhost:11434",
        timeout: float = 600.0,
        temperature: float = 0,
        keep_alive: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.cancel_event = cancel_event
        self.last_usage: dict[str, int | str] | None = None

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.last_usage = None
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": self.cancel_event is not None,
            "options": {"temperature": self.temperature},
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if self.cancel_event is not None:
            return self._stream_chat(payload)
        data = self._post_json("/api/chat", payload)
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError("Ollama chat response did not contain message.content")
        self.last_usage = _usage_from_ollama_payload(data)
        return message["content"]

    def _stream_chat(self, payload: dict[str, object]) -> str:
        request = self._request("/api/chat", payload)
        parts: list[str] = []
        final_data: dict[str, object] | None = None
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                while True:
                    if self.cancel_event is not None and self.cancel_event.is_set():
                        raise ChatCancelled("Ollama chat was cancelled")
                    line = response.readline()
                    if not line:
                        break
                    data = json.loads(line.decode("utf-8"))
                    if data.get("error"):
                        raise RuntimeError(str(data["error"]))
                    message = data.get("message")
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        parts.append(message["content"])
                    if data.get("done") is True:
                        final_data = data
                        break
        except urllib.error.URLError as exc:
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise ChatCancelled("Ollama chat was cancelled") from exc
            raise RuntimeError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise ChatCancelled("Ollama chat was cancelled")
        if final_data is not None:
            self.last_usage = _usage_from_ollama_payload(final_data)
        return "".join(parts)

    def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        request = self._request(path, payload)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

    def _request(self, path: str, payload: dict[str, object]) -> urllib.request.Request:
        body = json.dumps(payload).encode("utf-8")
        return urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )


def _usage_from_ollama_payload(data: dict[str, object]) -> dict[str, int | str] | None:
    input_tokens = data.get("prompt_eval_count")
    output_tokens = data.get("eval_count")
    if not isinstance(input_tokens, int) and not isinstance(output_tokens, int):
        return None
    safe_input = input_tokens if isinstance(input_tokens, int) else 0
    safe_output = output_tokens if isinstance(output_tokens, int) else 0
    return {
        "provider": "ollama",
        "input_tokens": safe_input,
        "output_tokens": safe_output,
        "total_tokens": safe_input + safe_output,
    }
