from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class WorkflowRunStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save(self, run: dict[str, Any]) -> Path:
        run_id = str(run["run_id"])
        folder = self.root / run_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "run.json"
        path.write_text(json.dumps(run, ensure_ascii=True, indent=2), encoding="utf-8")
        return path

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        runs: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*/run.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            runs.append(
                {
                    "run_id": data.get("run_id"),
                    "created_at": data.get("created_at"),
                    "model": data.get("model"),
                    "provider": data.get("provider"),
                    "assessment_id": data.get("assessment_id"),
                    "vendor_id": data.get("vendor_id"),
                    "price": data.get("cost_estimate", {}).get("estimated_cost_usd"),
                }
            )
        return runs

    def get(self, run_id: str) -> dict[str, Any]:
        path = self.root / run_id / "run.json"
        if not path.exists():
            raise FileNotFoundError(run_id)
        return json.loads(path.read_text(encoding="utf-8"))

