from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable


@dataclass
class JobProgress:
    job_id: str
    path: str
    category: str
    status: str  # running | completed | failed | cancelled
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    embedded_chunks: int = 0
    queued_chunks: int = 0
    error_count: int = 0
    current_files: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def format_progress(self) -> str:
        elapsed = self.elapsed_seconds
        mins, secs = divmod(int(elapsed), 60)
        elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"

        chunk_estimate = (
            f"~{self.total_chunks:,}"
            if self.status == "running" and self.total_chunks > 0
            else f"{self.total_chunks:,}"
        )

        current = ", ".join(self.current_files[:2]) if self.current_files else "—"

        lines = [
            f"Indexing: {self.path} [{self.category}]",
            f"   Files:    {self.completed_files} / {self.total_files}"
            + (f" ({self.failed_files} failed)" if self.failed_files else ""),
            f"   Chunks:   {self.embedded_chunks:,} / {chunk_estimate} embedded",
            f"   Queue:    {self.queued_chunks:,} chunks waiting for embedding",
            f"   Current:  {current}",
            f"   Elapsed:  {elapsed_str}",
        ]
        return "\n".join(lines)


class JobRegistry:
    """In-process job registry. Authoritative for running jobs; DB is the durable record."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobProgress] = {}
        self._lock = asyncio.Lock()

    async def create(self, path: str, category: str) -> JobProgress:
        job_id = str(uuid.uuid4())
        job = JobProgress(job_id=job_id, path=path, category=category, status="running")
        async with self._lock:
            self._jobs[job_id] = job
        return job

    async def get(self, job_id: str) -> JobProgress | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def list(self, status: str | None = None) -> list[JobProgress]:
        async with self._lock:
            jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.started_at, reverse=True)

    async def cancel(self, job_id: str) -> bool:
        job = await self.get(job_id)
        if job is None or job.status != "running":
            return False
        job.cancel_event.set()
        job.status = "cancelled"
        return True

    async def complete(self, job_id: str) -> None:
        job = await self.get(job_id)
        if job is not None:
            job.status = "completed"

    async def fail(self, job_id: str) -> None:
        job = await self.get(job_id)
        if job is not None:
            job.status = "failed"
