from __future__ import annotations

from typing import Protocol

from secure_rag.llm import OllamaChatClient


class ChatModel(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


class OpenAIChatClient:
    def __init__(self, *, api_key: str | None, model: str) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when provider is openai")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to use OpenAIChatClient") from exc
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.1,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI response did not contain content")
        return content


def chat_client_for_provider(
    *,
    provider: str,
    ollama_base_url: str,
    ollama_model: str,
    openai_api_key: str | None,
    openai_model: str,
) -> ChatModel:
    if provider == "openai":
        return OpenAIChatClient(api_key=openai_api_key, model=openai_model)
    return OllamaChatClient(model=ollama_model, base_url=ollama_base_url)

