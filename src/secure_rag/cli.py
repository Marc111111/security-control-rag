from __future__ import annotations

import argparse
import json
from pathlib import Path

from secure_rag.embeddings import OllamaEmbeddingClient
from secure_rag.engine import ControlRagEngine
from secure_rag.indexer import RagIndexer
from secure_rag.llm import OllamaChatClient
from secure_rag.retriever import Retriever
from secure_rag.vector_store import ChromaVectorStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="security-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Index a file or directory into ChromaDB")
    ingest.add_argument("--source", required=True, help="File or directory to index")
    ingest.add_argument("--db", required=True, help="ChromaDB persistence directory")
    ingest.add_argument("--embedding-model", default="mxbai-embed-large")
    ingest.add_argument("--chunk-size", type=int, default=1_500)
    ingest.add_argument("--overlap", type=int, default=200)

    query = subparsers.add_parser("query", help="Ask for security controls from the local corpus")
    query.add_argument("--db", required=True, help="ChromaDB persistence directory")
    query.add_argument("--criteria", required=True, help="Risk, vulnerability, tier, or criteria")
    query.add_argument("--embedding-model", default="mxbai-embed-large")
    query.add_argument("--generation-model", default="gemma3:4b")
    query.add_argument("--top-k", type=int, default=8)

    args = parser.parse_args(argv)
    if args.command == "ingest":
        return _ingest(args)
    if args.command == "query":
        return _query(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _ingest(args: argparse.Namespace) -> int:
    embedding_client = OllamaEmbeddingClient(model=args.embedding_model)
    vector_store = ChromaVectorStore(path=Path(args.db))
    indexer = RagIndexer(
        embedding_client=embedding_client,
        vector_store=vector_store,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    chunks = indexer.index_path(args.source)
    print(json.dumps({"indexed_chunks": len(chunks)}, indent=2))
    return 0


def _query(args: argparse.Namespace) -> int:
    embedding_client = OllamaEmbeddingClient(model=args.embedding_model)
    vector_store = ChromaVectorStore(path=Path(args.db))
    retriever = Retriever(embedding_client=embedding_client, vector_store=vector_store)
    engine = ControlRagEngine(
        retriever=retriever,
        chat_client=OllamaChatClient(model=args.generation_model),
    )
    answer = engine.answer(args.criteria, top_k=args.top_k)
    print(
        json.dumps(
            {
                "answer": answer.answer,
                "insufficient_evidence": answer.insufficient_evidence,
                "sources": answer.sources,
                "raw": answer.raw,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

