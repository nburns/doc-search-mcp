from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path

from doc_search_mcp.chunker import _Span, chunk_raw
from doc_search_mcp.config import Config
from doc_search_mcp.db import DocumentRecord, SearchBackend
from doc_search_mcp.extractors import SUPPORTED_EXTENSIONS, extract_epub, extract_pdf, extract_text
from doc_search_mcp.jobs import JobProgress, JobRegistry

_HIDDEN_PARTS = {".git", "__pycache__", ".DS_Store", "node_modules"}


async def index_path(
    path: Path,
    category: str,
    config: Config,
    backend: SearchBackend,
    registry: JobRegistry,
    embedder,
) -> str:
    """Start an async indexing job. Returns job_id immediately."""
    job = await registry.create(str(path), category)
    asyncio.create_task(
        _run_job(path, category, config, backend, registry, embedder, job),
        name=f"index-{job.job_id}",
    )
    return job.job_id


async def _run_job(
    path: Path,
    category: str,
    config: Config,
    backend: SearchBackend,
    registry: JobRegistry,
    embedder,
    job: JobProgress,
) -> None:
    try:
        files = _collect_files(path, config.chunking.max_file_size_mb)
        job.total_files = len(files)

        chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=config.performance.embedding_queue_size)
        embed_done = asyncio.Event()

        # Start the embedding + write worker
        writer_task = asyncio.create_task(
            _embed_and_write_worker(chunk_queue, embed_done, job, backend, embedder, config)
        )

        # Extract files concurrently
        sem = asyncio.Semaphore(
            config.performance.extraction_workers or _cpu_count()
        )
        extract_tasks = [
            asyncio.create_task(
                _extract_one(f, category, config, backend, job, chunk_queue, sem)
            )
            for f in files
        ]

        await asyncio.gather(*extract_tasks, return_exceptions=True)
        embed_done.set()
        await writer_task

        await registry.complete(job.job_id)
    except Exception as exc:
        print(f"[doc-search] Job {job.job_id} failed: {exc}", file=sys.stderr)
        job.status = "failed"
        await registry.fail(job.job_id)


async def _extract_one(
    path: Path,
    category: str,
    config: Config,
    backend: SearchBackend,
    job: JobProgress,
    chunk_queue: asyncio.Queue,
    sem: asyncio.Semaphore,
) -> None:
    if job.is_cancelled():
        return

    async with sem:
        try:
            job.current_files.append(path.name)
            checksum = await _checksum_async(path)

            # Dedup: if content already indexed under this category, just register the path
            existing_by_checksum = await backend.get_document_by_checksum(checksum, category)
            if existing_by_checksum:
                await backend.add_path(existing_by_checksum.id, str(path))
                job.completed_files += 1
                return

            # Check if this path was previously indexed with different content
            existing_by_path = await backend.get_document_by_path(str(path), category)
            if existing_by_path:
                # Content changed - disassociate path from old document
                # (leaves old document intact if other paths still reference it)
                await backend.remove_path(str(path), category)
                await backend.clear_warnings_for_document(existing_by_path.id)

            loop = asyncio.get_event_loop()
            raw_chunks, structure_source = await loop.run_in_executor(
                None, lambda: _extract_sync(path, config)
            )

            title = path.stem.replace("-", " ").replace("_", " ").title()
            doc = DocumentRecord(
                paths=[str(path)],
                title=title,
                file_type=path.suffix.lstrip(".").lower(),
                category=category,
                checksum=checksum,
                structure_source=structure_source,
                status="ok",
                chunk_count=0,
            )
            doc_id = await backend.upsert_document(doc)

            spans = [_Span(text=c.text, page_or_section=c.page_or_section) for c in raw_chunks]
            chunks = chunk_raw(
                document_id=doc_id,
                spans=spans,
                target_tokens=config.chunking.target_tokens,
                overlap_tokens=config.chunking.overlap_tokens,
            )
            job.total_chunks += len(chunks)

            for chunk in chunks:
                await chunk_queue.put(chunk)

            job.completed_files += 1
        except Exception as exc:
            job.failed_files += 1
            job.error_count += 1
            print(f"[doc-search] Failed to index {path}: {exc}", file=sys.stderr)
        finally:
            if path.name in job.current_files:
                job.current_files.remove(path.name)


async def _embed_and_write_worker(
    queue: asyncio.Queue,
    done: asyncio.Event,
    job: JobProgress,
    backend: SearchBackend,
    embedder,
    config: Config,
) -> None:
    batch: list = []
    batch_size = config.performance.embedding_batch_size
    loop = asyncio.get_event_loop()

    async def flush():
        if not batch:
            return
        chunks = list(batch)
        batch.clear()
        if embedder is not None:
            texts = [c.text for c in chunks]
            try:
                vectors = await loop.run_in_executor(None, lambda: list(embedder.embed(texts)))
                for chunk, vec in zip(chunks, vectors):
                    chunk.embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            except Exception as exc:
                print(f"[doc-search] Embedding batch failed: {exc}", file=sys.stderr)
        await backend.insert_chunks(chunks)
        job.embedded_chunks += len(chunks)
        job.queued_chunks = max(0, job.queued_chunks - len(chunks))

    while not (done.is_set() and queue.empty()):
        try:
            chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
            batch.append(chunk)
            job.queued_chunks += 1
            if len(batch) >= batch_size:
                await flush()
        except asyncio.TimeoutError:
            if batch:
                await flush()

    await flush()


def _extract_sync(path: Path, config: Config):
    """Run the appropriate extractor synchronously (called in thread pool)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw_chunks, source = extract_pdf(path, config.chunking.max_toc_depth)
        return raw_chunks, source
    elif suffix == ".epub":
        raw_chunks = extract_epub(path)
        return raw_chunks, "chapter"
    elif suffix in (".txt", ".md"):
        raw_chunks = extract_text(path)
        source = "heading" if suffix == ".md" else "paragraph"
        return raw_chunks, source
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _collect_files(path: Path, max_file_size_mb: int) -> list[Path]:
    max_bytes = max_file_size_mb * 1024 * 1024
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    files = []
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            if any(part.startswith(".") or part in _HIDDEN_PARTS for part in p.parts):
                continue
            if p.stat().st_size > max_bytes:
                continue
            files.append(p)
    return sorted(files)


async def _checksum_async(path: Path) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _checksum(path))


def _checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 4
