from __future__ import annotations

from typing import Protocol

from secure_rag.llm import OllamaChatClient


class ChatModel(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


class OpenAIChatClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        max_output_tokens: int | None = None,
    ) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when provider is openai")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to use OpenAIChatClient") from exc
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.last_usage: dict[str, int | str] | None = None

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.last_usage = None
        if hasattr(self.client, "responses"):
            response_kwargs: dict[str, object] = {
                "model": self.model,
                "input": messages,
                "temperature": 0,
            }
            if self.max_output_tokens is not None:
                response_kwargs["max_output_tokens"] = self.max_output_tokens
            response = self.client.responses.create(**response_kwargs)
            self.last_usage = _usage_from_openai_response(response)
            content = getattr(response, "output_text", None)
            if content:
                return str(content)

        kwargs: dict[str, object] = {}
        if self.max_output_tokens is not None:
            kwargs["max_tokens"] = self.max_output_tokens
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0,
            **kwargs,
        )
        self.last_usage = _usage_from_openai_response(response)
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


def _usage_from_openai_response(response: object) -> dict[str, int | str] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = _usage_value(usage, "input_tokens")
    output_tokens = _usage_value(usage, "output_tokens")
    if input_tokens is None:
        input_tokens = _usage_value(usage, "prompt_tokens")
    if output_tokens is None:
        output_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    if total_tokens is None:
        return None
    return {
        "provider": "openai",
        "input_tokens": input_tokens or 0,
        "output_tokens": output_tokens or 0,
        "total_tokens": total_tokens,
    }


def _usage_value(usage: object, key: str) -> int | None:
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    return value if isinstance(value, int) else None
