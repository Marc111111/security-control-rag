from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class EvaluationLogger:
    log_dir: Path = Path("storage/evaluation")
    records: list[dict[str, Any]] = field(default_factory=list)

    def log_query(self, payload: dict[str, Any]) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": datetime.now(UTC).isoformat(), **payload}
        self.records.append(record)
        path = self.log_dir / f"query-{len(self.records):04d}.json"
        path.write_text(json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8")
        return path

    def manual_feedback(
        self,
        *,
        query_id: str,
        chunk_id: str,
        relevant: bool,
        notes: str = "",
    ) -> dict[str, Any]:
        feedback = {
            "timestamp": datetime.now(UTC).isoformat(),
            "query_id": query_id,
            "chunk_id": chunk_id,
            "relevant": relevant,
            "notes": notes,
        }
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / "manual-feedback.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(feedback, ensure_ascii=True) + "\n")
        return feedback

