from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app.workflows.step_audit import audit_workflow_run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and audit the complete assessment workflow through the HTTP API."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--run-id", help="Audit an existing saved run instead of starting a job.")
    parser.add_argument("--latest", action="store_true", help="Audit the latest saved run.")
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--confirm-external-call", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    try:
        if args.run_id:
            run = _get_json(f"{base_url}/api/workflows/complete-assessment/runs/{args.run_id}")
        elif args.latest:
            runs = _get_json(f"{base_url}/api/workflows/complete-assessment/runs")
            if not runs:
                print("No saved workflow runs found.", file=sys.stderr)
                return 2
            run = _get_json(
                f"{base_url}/api/workflows/complete-assessment/runs/{runs[0]['run_id']}"
            )
        else:
            run = _start_and_wait(base_url, args)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP error {exc.code}: {body}", file=sys.stderr)
        return 2

    audit = audit_workflow_run(run)
    print(json.dumps(_summary(run, audit), indent=2, ensure_ascii=True))
    if run.get("failed"):
        print(f"\nWorkflow failed before completion: {run.get('error')}", file=sys.stderr)
        return 1
    if not audit["passed"]:
        print("\nBlocking audit issues:", file=sys.stderr)
        for issue in audit["issues"]:
            if issue["severity"] == "blocking":
                print(
                    f"- Step {issue['step_index']} {issue['step_name']}: "
                    f"{issue['message']} Fix: {issue['fix']}",
                    file=sys.stderr,
                )
        return 1
    return 0


def _start_and_wait(base_url: str, args: argparse.Namespace) -> dict[str, Any]:
    packet = _get_json(f"{base_url}/api/mock/foundation-packet")
    request_body: dict[str, Any] = {
        "input_source": {"adapter": "foundation_packet_v1", "payload": packet},
        "model": {
            "provider": args.provider,
            "model": args.model,
            "confirm_external_call": bool(args.confirm_external_call),
            "openai_api_key": args.openai_api_key,
            "estimated_output_tokens": 1200,
            "token_budget_tolerance_percent": 10,
            "enforce_token_budget": True,
        },
        "top_k": args.top_k,
        "debug": True,
    }
    preflight = _post_json(
        f"{base_url}/api/workflows/complete-assessment/preflight",
        request_body,
    )
    print(
        "Preflight: "
        f"input={preflight['estimated_input_tokens']} "
        f"output={preflight['estimated_output_tokens']} "
        f"cap={preflight['allowed_total_tokens']} "
        f"cost_usd={preflight['estimated_cost_usd']}",
        file=sys.stderr,
    )
    job = _post_json(f"{base_url}/api/workflows/complete-assessment/jobs", request_body)
    job_id = job["job_id"]
    deadline = time.time() + args.timeout_seconds
    while time.time() < deadline:
        snapshot = _get_json(f"{base_url}/api/workflows/complete-assessment/jobs/{job_id}")
        status = snapshot["status"]
        current = snapshot.get("current_step") or ""
        print(f"{status}: {current}", file=sys.stderr)
        if status == "completed":
            return snapshot["result"]
        if status in {"failed", "cancelled"}:
            print(json.dumps(snapshot, indent=2, ensure_ascii=True), file=sys.stderr)
            return {
                "run_id": job_id,
                "steps": snapshot.get("partial_steps") or [],
                "failed": True,
                "error": snapshot.get("error"),
            }
        time.sleep(3)
    raise TimeoutError(f"Workflow job {job_id} did not finish within timeout.")


def _summary(run: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    token_budget = run.get("token_budget") or {}
    return {
        "run_id": run.get("run_id"),
        "status": "failed" if run.get("failed") else "completed",
        "error": run.get("error"),
        "model": run.get("model"),
        "provider": run.get("provider"),
        "step_count": audit["step_count"],
        "audit_passed": audit["passed"],
        "blocking_issue_count": audit["blocking_issue_count"],
        "warning_issue_count": audit["warning_issue_count"],
        "actual_input_tokens": token_budget.get("actual_input_tokens"),
        "actual_output_tokens": token_budget.get("actual_output_tokens"),
        "actual_total_tokens": token_budget.get("actual_total_tokens"),
        "issues": audit["issues"],
    }


def _get_json(url: str) -> Any:
    with urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, body: dict[str, Any]) -> Any:
    data = json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
