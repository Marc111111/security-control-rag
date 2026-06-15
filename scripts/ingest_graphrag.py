from __future__ import annotations

import argparse

from app.config import load_settings
from app.pipeline import GraphRagPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the GraphRAG prototype.")
    parser.add_argument("source", help="File or directory to ingest")
    parser.add_argument("--chunk-size", type=int, default=1500)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    pipeline = GraphRagPipeline(load_settings())
    result = pipeline.ingest(
        args.source,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
    print(result)


if __name__ == "__main__":
    main()

