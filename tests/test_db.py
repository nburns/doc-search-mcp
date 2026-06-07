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


def _doc(paths=None, category="test", checksum="abc123") -> DocumentRecord:
    return DocumentRecord(
        paths=paths or ["/docs/test.pdf"],
        title="Test Doc",
        file_type="pdf",
        category=category,
        checksum=checksum,
        structure_source="page",
        status="ok",
        chunk_count=0,
    )


async def test_init_schema(backend):
    await backend.init_schema()  # idempotent


async def test_upsert_document_returns_id(backend):
    doc_id = await backend.upsert_document(_doc())
    assert isinstance(doc_id, int) and doc_id > 0


async def test_upsert_deduplicates_by_checksum(backend):
    # Same checksum + category -> same document row
    id1 = await backend.upsert_document(_doc(paths=["/a.pdf"], checksum="x1"))
    id2 = await backend.upsert_document(_doc(paths=["/b.pdf"], checksum="x1"))
    assert id1 == id2

    doc = await backend.get_document_by_checksum("x1", "test")
    assert doc is not None
    assert set(doc.paths) == {"/a.pdf", "/b.pdf"}


async def test_upsert_different_checksum_is_different_doc(backend):
    id1 = await backend.upsert_document(_doc(paths=["/a.pdf"], checksum="c1"))
    id2 = await backend.upsert_document(_doc(paths=["/b.pdf"], checksum="c2"))
    assert id1 != id2


async def test_add_path(backend):
    doc_id = await backend.upsert_document(_doc(paths=["/a.pdf"]))
    await backend.add_path(doc_id, "/b.pdf")
    doc = await backend.get_document_by_path("/b.pdf", "test")
    assert doc is not None
    assert "/a.pdf" in doc.paths
    assert "/b.pdf" in doc.paths


async def test_add_path_duplicate_ignored(backend):
    doc_id = await backend.upsert_document(_doc(paths=["/a.pdf"]))
    await backend.add_path(doc_id, "/a.pdf")  # duplicate - should not raise
    doc = await backend.get_document_by_path("/a.pdf", "test")
    assert doc.paths.count("/a.pdf") == 1


async def test_remove_path_last_deletes_document(backend):
    doc_id = await backend.upsert_document(_doc(paths=["/only.pdf"]))
    deleted = await backend.remove_path("/only.pdf", "test")
    assert deleted is True
    assert await backend.get_document_by_path("/only.pdf", "test") is None


async def test_remove_path_retains_document_with_remaining_paths(backend):
    await backend.upsert_document(_doc(paths=["/a.pdf", "/b.pdf"]))
    deleted = await backend.remove_path("/a.pdf", "test")
    assert deleted is False
    doc = await backend.get_document_by_path("/b.pdf", "test")
    assert doc is not None
    assert "/a.pdf" not in doc.paths


async def test_remove_path_not_found(backend):
    result = await backend.remove_path("/nonexistent.pdf", "test")
    assert result is False


async def test_get_document_by_path(backend):
    await backend.upsert_document(_doc(paths=["/some/file.pdf"], category="books"))
    doc = await backend.get_document_by_path("/some/file.pdf", "books")
    assert doc is not None
    assert doc.title == "Test Doc"
    assert doc.checksum == "abc123"


async def test_get_document_by_path_not_found(backend):
    assert await backend.get_document_by_path("/nonexistent.pdf", "test") is None


async def test_get_document_by_checksum(backend):
    await backend.upsert_document(_doc(paths=["/f.pdf"], checksum="sha_xyz"))
    doc = await backend.get_document_by_checksum("sha_xyz", "test")
    assert doc is not None
    assert "/f.pdf" in doc.paths


async def test_get_document_by_checksum_not_found(backend):
    assert await backend.get_document_by_checksum("no_such_hash", "test") is None


async def test_document_record_primary_path(backend):
    await backend.upsert_document(_doc(paths=["/primary.pdf", "/secondary.pdf"]))
    doc = await backend.get_document_by_path("/primary.pdf", "test")
    assert doc.primary_path in doc.paths


async def test_insert_and_get_chunks(backend):
    doc_id = await backend.upsert_document(_doc())
    chunks = [
        ChunkRecord(document_id=doc_id, text="Hello world", page_or_section="Page 1", position=0),
        ChunkRecord(document_id=doc_id, text="Second chunk", page_or_section="Page 2", position=1),
    ]
    ids = await backend.insert_chunks(chunks)
    assert len(ids) == 2

    fetched = await backend.get_chunks_by_ids(ids)
    assert {c.text for c in fetched} == {"Hello world", "Second chunk"}


async def test_delete_document_chunks(backend):
    doc_id = await backend.upsert_document(_doc())
    ids = await backend.insert_chunks([
        ChunkRecord(document_id=doc_id, text="bye", page_or_section="p1", position=0)
    ])
    await backend.delete_document_chunks(doc_id)
    assert await backend.get_chunks_by_ids(ids) == []


async def test_chunks_cascade_delete_with_document(backend):
    doc_id = await backend.upsert_document(_doc(paths=["/solo.pdf"]))
    ids = await backend.insert_chunks([
        ChunkRecord(document_id=doc_id, text="gone", page_or_section="p1", position=0)
    ])
    await backend.remove_path("/solo.pdf", "test")  # deletes document
    assert await backend.get_chunks_by_ids(ids) == []


async def test_keyword_search(backend):
    doc_id = await backend.upsert_document(_doc())
    await backend.insert_chunks([
        ChunkRecord(document_id=doc_id, text="machine learning is fascinating", page_or_section="p1", position=0),
        ChunkRecord(document_id=doc_id, text="gardening tips for beginners", page_or_section="p2", position=1),
    ])
    results = await backend.keyword_search("machine learning", category="test", limit=10)
    assert any("machine learning" in r.text for r in results)


async def test_keyword_search_no_results(backend):
    assert await backend.keyword_search("xyzzy nonexistent", category="test", limit=10) == []


async def test_list_documents_empty(backend):
    assert await backend.list_documents() == []


async def test_list_documents_includes_all_paths(backend):
    await backend.upsert_document(_doc(paths=["/a.pdf", "/b.pdf"]))
    docs = await backend.list_documents()
    assert len(docs) == 1
    assert set(docs[0].paths) == {"/a.pdf", "/b.pdf"}


async def test_list_documents_category_filter(backend):
    await backend.upsert_document(_doc(paths=["/a.pdf"], category="cat1", checksum="c1"))
    await backend.upsert_document(_doc(paths=["/b.pdf"], category="cat2", checksum="c2"))
    cat1_docs = await backend.list_documents("cat1")
    assert len(cat1_docs) == 1 and "/a.pdf" in cat1_docs[0].paths


async def test_list_categories(backend):
    await backend.upsert_document(_doc(paths=["/a.pdf"], category="alpha", checksum="c1"))
    await backend.upsert_document(_doc(paths=["/b.pdf"], category="beta", checksum="c2"))
    cats = await backend.list_categories()
    assert {c.name for c in cats} == {"alpha", "beta"}


async def test_get_stats(backend):
    await backend.upsert_document(_doc())
    stats = await backend.get_stats()
    assert stats.doc_count == 1


async def test_warnings_lifecycle(backend):
    doc_id = await backend.upsert_document(_doc())
    w_id = await backend.add_warning(WarningRecord(
        document_id=doc_id, category="test", warning_type="changed", detected_at=datetime.utcnow()
    ))
    assert any(x.id == w_id for x in await backend.get_active_warnings())
    await backend.acknowledge_warnings([w_id])
    assert not any(x.id == w_id for x in await backend.get_active_warnings())


async def test_clear_warnings_for_document(backend):
    doc_id = await backend.upsert_document(_doc())
    await backend.add_warning(WarningRecord(
        document_id=doc_id, category="test", warning_type="missing", detected_at=datetime.utcnow()
    ))
    await backend.clear_warnings_for_document(doc_id)
    assert await backend.get_active_warnings() == []


async def test_job_upsert_and_get(backend):
    job = JobRecord(id="j1", path="/p", category="c", status="running", total_files=10, completed_files=3)
    await backend.upsert_job(job)
    fetched = await backend.get_job("j1")
    assert fetched.status == "running"
    assert fetched.total_files == 10


async def test_job_update(backend):
    job = JobRecord(id="j2", path="/p", category="c", status="running")
    await backend.upsert_job(job)
    job.status = "completed"
    job.completed_files = 5
    await backend.upsert_job(job)
    assert (await backend.get_job("j2")).status == "completed"


async def test_list_jobs_filter(backend):
    await backend.upsert_job(JobRecord(id="j1", path="/p", category="c", status="running"))
    await backend.upsert_job(JobRecord(id="j2", path="/p", category="c", status="completed"))
    running = await backend.list_jobs("running")
    assert len(running) == 1 and running[0].id == "j1"


async def test_server_meta(backend):
    await backend.set_server_meta("embedding_model", "nomic-embed-text")
    assert await backend.get_server_meta("embedding_model") == "nomic-embed-text"
    await backend.set_server_meta("embedding_model", "updated-model")
    assert await backend.get_server_meta("embedding_model") == "updated-model"
    assert await backend.get_server_meta("nonexistent") is None
