from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from hashlib import blake2b
from typing import Protocol


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class HashEmbeddingClient:
    """Deterministic embedding client for tests and offline development."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OllamaEmbeddingClient:
    def __init__(
        self,
        *,
        model: str = "mxbai-embed-large",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._embed_batch(texts)
        except RuntimeError:
            return [self._embed_legacy(text) for text in texts]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        data = self._post_json("/api/embed", payload)
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError("Ollama /api/embed response did not contain embeddings")
        return embeddings

    def _embed_legacy(self, text: str) -> list[float]:
        payload = {"model": self.model, "prompt": text}
        data = self._post_json("/api/embeddings", payload)
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Ollama /api/embeddings response did not contain embedding")
        return embedding

    def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach Ollama at {self.base_url}") from exc

