from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from app.pipeline import GraphRagPipeline
from app.workflows.complete_assessment import (
    CompleteAssessmentRequest,
    CompleteAssessmentWorkflow,
    WorkflowCancelled,
    estimate_complete_assessment_preflight,
)
from app.workflows.run_store import WorkflowRunStore

JobStatus = Literal["queued", "running", "cancelling", "completed", "failed", "cancelled"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class WorkflowJob:
    job_id: str
    request: CompleteAssessmentRequest
    status: JobStatus = "queued"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "provider": self.request.model.provider,
            "model": self.request.model.model,
            "result": self.result,
            "error": self.error,
            "preflight": estimate_complete_assessment_preflight(self.request),
        }


class WorkflowJobManager:
    def __init__(
        self,
        *,
        pipeline_factory: Callable[[], GraphRagPipeline],
        max_workers: int = 1,
    ) -> None:
        self.pipeline_factory = pipeline_factory
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: dict[str, WorkflowJob] = {}
        self.lock = threading.Lock()

    def preflight(self, request: CompleteAssessmentRequest) -> dict[str, Any]:
        return estimate_complete_assessment_preflight(request)

    def start(self, request: CompleteAssessmentRequest) -> dict[str, Any]:
        job = WorkflowJob(job_id=f"job-{uuid4().hex[:12]}", request=request)
        with self.lock:
            self.jobs[job.job_id] = job
        self.executor.submit(self._run_job, job)
        return job.snapshot()

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job.snapshot()

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        job.cancel_event.set()
        if job.request.model.provider == "ollama":
            _stop_ollama_model(job.request.model.model)
        if job.status == "queued":
            self._update(job, status="cancelled", error="Cancellation requested")
        elif job.status == "running":
            self._update(job, status="cancelling", error="Cancellation requested")
        return job.snapshot()

    def _run_job(self, job: WorkflowJob) -> None:
        if job.cancel_event.is_set():
            self._update(job, status="cancelled", error="Cancelled before start")
            return
        self._update(job, status="running", error=None)
        try:
            pipeline = self.pipeline_factory()
            workflow = CompleteAssessmentWorkflow(
                pipeline=pipeline,
                run_store=WorkflowRunStore(pipeline.settings.run_store_path),
            )
            result = workflow.run(job.request, cancel_event=job.cancel_event)
        except WorkflowCancelled as exc:
            self._update(job, status="cancelled", error=str(exc))
            return
        except Exception as exc:
            if job.cancel_event.is_set():
                self._update(job, status="cancelled", error=str(exc))
            else:
                self._update(job, status="failed", error=str(exc))
            return
        finally:
            if job.request.model.provider == "ollama" and job.cancel_event.is_set():
                _stop_ollama_model(job.request.model.model)
        if job.cancel_event.is_set():
            self._update(job, status="cancelled", error="Cancelled after completion")
            return
        self._update(job, status="completed", result=result, error=None)

    def _update(
        self,
        job: WorkflowJob,
        *,
        status: JobStatus,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        job.status = status
        job.updated_at = _now()
        job.result = result
        job.error = error


def _stop_ollama_model(model: str) -> None:
    try:
        subprocess.run(
            ["ollama", "stop", model],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        pass
