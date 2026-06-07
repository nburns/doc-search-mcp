import asyncio
from datetime import datetime
from pathlib import Path

import pytest

from doc_search_mcp.db import (
    ChunkRecord,
    DocumentRecord,
    JobRecord,
    SQLiteBackend,
    WarningRecord,
)


@pytest.fixture
async def backend(tmp_path):
    b = SQLiteBackend(tmp_path / "test.db")
    await b.init_schema()
    return b


async def _make_doc(backend, path="~/docs/test.pdf", category="test") -> int:
    doc = DocumentRecord(
        path=path,
        title="Test Doc",
        file_type="pdf",
        category=category,
        checksum="abc123",
        structure_source="page",
        status="ok",
        chunk_count=0,
    )
    return await backend.upsert_document(doc)


async def test_init_schema(backend):
    # Schema init is idempotent
    await backend.init_schema()


async def test_upsert_document_returns_id(backend):
    doc_id = await _make_doc(backend)
    assert isinstance(doc_id, int)
    assert doc_id > 0


async def test_upsert_document_idempotent(backend):
    id1 = await _make_doc(backend)
    id2 = await _make_doc(backend)
    assert id1 == id2  # same path+category -> same row updated


async def test_get_document_by_path(backend):
    await _make_doc(backend, path="/some/file.pdf", category="books")
    doc = await backend.get_document_by_path("/some/file.pdf", "books")
    assert doc is not None
    assert doc.title == "Test Doc"
    assert doc.checksum == "abc123"


async def test_get_document_by_path_not_found(backend):
    result = await backend.get_document_by_path("/nonexistent.pdf", "test")
    assert result is None


async def test_insert_and_get_chunks(backend):
    doc_id = await _make_doc(backend)
    chunks = [
        ChunkRecord(document_id=doc_id, text="Hello world", page_or_section="Page 1", position=0),
        ChunkRecord(document_id=doc_id, text="Second chunk", page_or_section="Page 2", position=1),
    ]
    ids = await backend.insert_chunks(chunks)
    assert len(ids) == 2
    assert all(isinstance(i, int) for i in ids)

    fetched = await backend.get_chunks_by_ids(ids)
    assert len(fetched) == 2
    texts = {c.text for c in fetched}
    assert texts == {"Hello world", "Second chunk"}


async def test_delete_document_chunks(backend):
    doc_id = await _make_doc(backend)
    chunks = [ChunkRecord(document_id=doc_id, text="To be deleted", page_or_section="p1", position=0)]
    ids = await backend.insert_chunks(chunks)
    await backend.delete_document_chunks(doc_id)
    fetched = await backend.get_chunks_by_ids(ids)
    assert fetched == []


async def test_keyword_search(backend):
    doc_id = await _make_doc(backend)
    chunks = [
        ChunkRecord(document_id=doc_id, text="machine learning is fascinating", page_or_section="p1", position=0),
        ChunkRecord(document_id=doc_id, text="gardening tips for beginners", page_or_section="p2", position=1),
    ]
    await backend.insert_chunks(chunks)

    results = await backend.keyword_search("machine learning", category="test", limit=10)
    assert len(results) >= 1
    assert any("machine learning" in r.text for r in results)


async def test_keyword_search_no_results(backend):
    results = await backend.keyword_search("xyzzy nonexistent", category="test", limit=10)
    assert results == []


async def test_list_documents_empty(backend):
    docs = await backend.list_documents()
    assert docs == []


async def test_list_documents(backend):
    await _make_doc(backend, path="/a.pdf", category="cat1")
    await _make_doc(backend, path="/b.pdf", category="cat2")
    all_docs = await backend.list_documents()
    assert len(all_docs) == 2
    cat1_docs = await backend.list_documents("cat1")
    assert len(cat1_docs) == 1


async def test_list_categories(backend):
    await _make_doc(backend, path="/a.pdf", category="alpha")
    await _make_doc(backend, path="/b.pdf", category="beta")
    cats = await backend.list_categories()
    names = {c.name for c in cats}
    assert names == {"alpha", "beta"}


async def test_get_stats(backend):
    await _make_doc(backend)
    stats = await backend.get_stats()
    assert stats.doc_count == 1


async def test_warnings_lifecycle(backend):
    doc_id = await _make_doc(backend)
    w = WarningRecord(
        document_id=doc_id,
        category="test",
        warning_type="changed",
        detected_at=datetime.utcnow(),
    )
    w_id = await backend.add_warning(w)
    assert isinstance(w_id, int)

    active = await backend.get_active_warnings()
    assert any(x.id == w_id for x in active)

    await backend.acknowledge_warnings([w_id])
    active_after = await backend.get_active_warnings()
    assert not any(x.id == w_id for x in active_after)


async def test_clear_warnings_for_document(backend):
    doc_id = await _make_doc(backend)
    w = WarningRecord(
        document_id=doc_id,
        category="test",
        warning_type="missing",
        detected_at=datetime.utcnow(),
    )
    await backend.add_warning(w)
    await backend.clear_warnings_for_document(doc_id)
    assert await backend.get_active_warnings() == []


async def test_job_upsert_and_get(backend):
    job = JobRecord(
        id="test-job-1",
        path="/some/path",
        category="books",
        status="running",
        total_files=10,
        completed_files=3,
    )
    await backend.upsert_job(job)

    fetched = await backend.get_job("test-job-1")
    assert fetched is not None
    assert fetched.status == "running"
    assert fetched.total_files == 10
    assert fetched.completed_files == 3


async def test_job_update(backend):
    job = JobRecord(id="j2", path="/p", category="c", status="running")
    await backend.upsert_job(job)
    job.status = "completed"
    job.completed_files = 5
    await backend.upsert_job(job)
    fetched = await backend.get_job("j2")
    assert fetched.status == "completed"
    assert fetched.completed_files == 5


async def test_list_jobs_filter(backend):
    await backend.upsert_job(JobRecord(id="j1", path="/p", category="c", status="running"))
    await backend.upsert_job(JobRecord(id="j2", path="/p", category="c", status="completed"))
    running = await backend.list_jobs("running")
    assert len(running) == 1 and running[0].id == "j1"
    all_jobs = await backend.list_jobs()
    assert len(all_jobs) == 2


async def test_server_meta(backend):
    await backend.set_server_meta("embedding_model", "nomic-embed-text")
    val = await backend.get_server_meta("embedding_model")
    assert val == "nomic-embed-text"

    await backend.set_server_meta("embedding_model", "updated-model")
    val2 = await backend.get_server_meta("embedding_model")
    assert val2 == "updated-model"

    missing = await backend.get_server_meta("nonexistent_key")
    assert missing is None
