import asyncio

import pytest

from doc_search_mcp.jobs import JobRegistry


async def test_create_job():
    registry = JobRegistry()
    job = await registry.create("/some/path", "books")
    assert job.job_id
    assert job.status == "running"
    assert job.path == "/some/path"
    assert job.category == "books"


async def test_get_job():
    registry = JobRegistry()
    job = await registry.create("/p", "c")
    fetched = await registry.get(job.job_id)
    assert fetched is job


async def test_get_nonexistent_job():
    registry = JobRegistry()
    result = await registry.get("does-not-exist")
    assert result is None


async def test_list_jobs():
    registry = JobRegistry()
    j1 = await registry.create("/p1", "c")
    j2 = await registry.create("/p2", "c")
    jobs = await registry.list()
    assert len(jobs) == 2


async def test_list_jobs_filtered_by_status():
    registry = JobRegistry()
    j1 = await registry.create("/p1", "c")
    j2 = await registry.create("/p2", "c")
    await registry.complete(j1.job_id)
    running = await registry.list("running")
    assert len(running) == 1
    assert running[0].job_id == j2.job_id


async def test_cancel_job():
    registry = JobRegistry()
    job = await registry.create("/p", "c")
    result = await registry.cancel(job.job_id)
    assert result is True
    assert job.status == "cancelled"
    assert job.is_cancelled()


async def test_cancel_nonexistent_job():
    registry = JobRegistry()
    result = await registry.cancel("no-such-job")
    assert result is False


async def test_cancel_already_completed_job():
    registry = JobRegistry()
    job = await registry.create("/p", "c")
    await registry.complete(job.job_id)
    result = await registry.cancel(job.job_id)
    assert result is False


async def test_job_progress_format():
    registry = JobRegistry()
    job = await registry.create("/docs/ml-papers", "ml-stuff")
    job.total_files = 47
    job.completed_files = 23
    job.total_chunks = 3800
    job.embedded_chunks = 1840
    job.queued_chunks = 124
    job.current_files = ["attention.pdf", "bert.pdf"]

    progress = job.format_progress()
    assert "ml-papers" in progress
    assert "23 / 47" in progress
    assert "1,840" in progress


async def test_fail_job():
    registry = JobRegistry()
    job = await registry.create("/p", "c")
    await registry.fail(job.job_id)
    assert job.status == "failed"
