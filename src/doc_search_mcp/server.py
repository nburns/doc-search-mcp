from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route

from doc_search_mcp import indexer, search
from doc_search_mcp.config import Config, load_config
from doc_search_mcp.db import SearchBackend, make_backend
from doc_search_mcp.embedder import make_embedder
from doc_search_mcp.jobs import JobRegistry
from doc_search_mcp.warnings import run_startup_check, warning_footer

_server = Server("doc-search-mcp")
_config: Config
_backend: SearchBackend
_embedder: Any
_registry: JobRegistry


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TOOLS = [
    types.Tool(
        name="index_path",
        description="Index a file or directory. Use to add documents to the search index. Returns a job_id to track progress.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or ~ path to a file or directory"},
                "category": {"type": "string", "description": "Category name (e.g. 'ml-stuff', 'retro-apple')"},
            },
            "required": ["path", "category"],
        },
    ),
    types.Tool(
        name="reindex_path",
        description="Re-extract and re-index a path, skipping files whose content is unchanged.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "category": {"type": "string"},
            },
            "required": ["path", "category"],
        },
    ),
    types.Tool(
        name="remove_document",
        description="Remove a document from the index.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "category": {"type": "string"},
            },
            "required": ["path", "category"],
        },
    ),
    types.Tool(
        name="search",
        description="Search indexed documents. Use for any question about document contents.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "category": {
                    "type": "string",
                    "description": "Category to search, or 'all' for cross-category search",
                },
                "mode": {
                    "type": "string",
                    "enum": ["auto", "keyword", "semantic", "hybrid"],
                    "default": "auto",
                },
                "limit": {"type": "integer", "default": 10},
                "file_type": {"type": "string", "description": "Filter by file type: pdf, epub, txt, md"},
                "path_prefix": {"type": "string", "description": "Filter by path prefix"},
            },
            "required": ["query", "category"],
        },
    ),
    types.Tool(
        name="search_in_document",
        description="Search within a specific file. Use when you know which document contains the answer.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "path": {"type": "string"},
                "category": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query", "path", "category"],
        },
    ),
    types.Tool(
        name="get_chunks",
        description="Retrieve surrounding context for a search result by chunk ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "integer"}},
                "category": {"type": "string"},
            },
            "required": ["ids", "category"],
        },
    ),
    types.Tool(
        name="list_documents",
        description="List all indexed documents. Use to discover what's available before searching.",
        inputSchema={
            "type": "object",
            "properties": {
                "category": {"type": "string"},
            },
        },
    ),
    types.Tool(
        name="list_categories",
        description="List all document categories with counts. Use before searching to know what's available.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="get_stats",
        description="Get index statistics: DB size, doc count, chunk count.",
        inputSchema={
            "type": "object",
            "properties": {
                "category": {"type": "string"},
            },
        },
    ),
    types.Tool(
        name="get_job_status",
        description="Get progress for an indexing job.",
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
            },
            "required": ["job_id"],
        },
    ),
    types.Tool(
        name="list_jobs",
        description="List indexing jobs.",
        inputSchema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["running", "completed", "failed", "cancelled", "all"],
                },
            },
        },
    ),
    types.Tool(
        name="cancel_job",
        description="Cancel a running indexing job.",
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
            },
            "required": ["job_id"],
        },
    ),
    types.Tool(
        name="check_index",
        description="Check index health. Reports changed, missing, or unindexed files.",
        inputSchema={
            "type": "object",
            "properties": {
                "category": {"type": "string"},
            },
        },
    ),
    types.Tool(
        name="acknowledge_warnings",
        description="Suppress known index warnings until next startup.",
        inputSchema={
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["ids"],
        },
    ),
    types.Tool(
        name="get_config",
        description="Show current configuration and active backend.",
        inputSchema={"type": "object", "properties": {}},
    ),
]


@_server.list_tools()
async def list_tools() -> list[types.Tool]:
    return _TOOLS


@_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    result = await _dispatch(name, arguments)
    return [types.TextContent(type="text", text=result)]


async def _dispatch(name: str, args: dict) -> str:
    if name == "index_path":
        path = Path(args["path"]).expanduser()
        job_id = await indexer.index_path(
            path, args["category"], _config, _backend, _registry, _embedder
        )
        return f"Indexing started. Job ID: `{job_id}`\nUse `get_job_status` to track progress."

    if name == "reindex_path":
        path = Path(args["path"]).expanduser()
        job_id = await indexer.index_path(
            path, args["category"], _config, _backend, _registry, _embedder
        )
        return f"Reindex started. Job ID: `{job_id}`"

    if name == "remove_document":
        doc = await _backend.get_document_by_path(args["path"], args["category"])
        if doc is None:
            return f"No document found at {args['path']!r} in category {args['category']!r}"
        await _backend.delete_document_chunks(doc.id)  # type: ignore[arg-type]
        await _backend.clear_warnings_for_document(doc.id)  # type: ignore[arg-type]
        return f"Removed: {args['path']}"

    if name == "search":
        return await search.search(
            query=args["query"],
            category=args.get("category"),
            mode=args.get("mode", _config.search.default_mode),
            limit=int(args.get("limit", _config.search.default_limit)),
            backend=_backend,
            embedder=_embedder,
            file_type=args.get("file_type"),
            path_prefix=args.get("path_prefix"),
        )

    if name == "search_in_document":
        return await search.search(
            query=args["query"],
            category=args["category"],
            mode=_config.search.default_mode,
            limit=int(args.get("limit", _config.search.default_limit)),
            backend=_backend,
            embedder=_embedder,
            path_prefix=args["path"],
        )

    if name == "get_chunks":
        chunks = await _backend.get_chunks_by_ids(args["ids"])
        if not chunks:
            return "No chunks found for the given IDs."
        lines = []
        for c in chunks:
            lines.append(f"**Chunk {c.id}** — {c.page_or_section}")
            lines.append(c.text)
            lines.append("---")
        return "\n".join(lines)

    if name == "list_documents":
        docs = await _backend.list_documents(args.get("category"))
        if not docs:
            return "No documents indexed."
        lines = [f"| Path | Title | Type | Status | Chunks |",
                 f"|------|-------|------|--------|--------|"]
        for d in docs:
            lines.append(f"| {d.path} | {d.title} | {d.file_type} | {d.status} | {d.chunk_count} |")
        footer = await warning_footer(_backend)
        return "\n".join(lines) + footer

    if name == "list_categories":
        cats = await _backend.list_categories()
        if not cats:
            return "No categories found."
        lines = ["| Category | Docs | Chunks | Last Indexed |",
                 "|----------|------|--------|--------------|"]
        for c in cats:
            last = c.last_indexed.strftime("%Y-%m-%d %H:%M") if c.last_indexed else "—"
            lines.append(f"| {c.name} | {c.doc_count} | {c.chunk_count:,} | {last} |")
        footer = await warning_footer(_backend)
        return "\n".join(lines) + footer

    if name == "get_stats":
        stats = await _backend.get_stats(args.get("category"))
        size_mb = stats.db_size_bytes / 1024 / 1024
        scope = f"category: {stats.category}" if stats.category else "all categories"
        footer = await warning_footer(_backend)
        return (
            f"**Index stats** ({scope})\n"
            f"- Documents: {stats.doc_count:,}\n"
            f"- Chunks: {stats.chunk_count:,}\n"
            f"- DB size: {size_mb:.1f} MB"
        ) + footer

    if name == "get_job_status":
        job = await _registry.get(args["job_id"])
        if job is None:
            # Fall back to DB record
            db_job = await _backend.get_job(args["job_id"])
            if db_job is None:
                return f"Job not found: {args['job_id']}"
            return (
                f"**Job {db_job.id[:8]}**\n"
                f"Status: {db_job.status}\n"
                f"Path: {db_job.path}\n"
                f"Files: {db_job.completed_files}/{db_job.total_files}\n"
                f"Chunks embedded: {db_job.embedded_chunks:,}"
            )
        return job.format_progress()

    if name == "list_jobs":
        status_filter = args.get("status")
        if status_filter == "all":
            status_filter = None
        jobs = await _registry.list(status_filter)
        if not jobs:
            return "No jobs found."
        lines = ["| Job ID | Path | Status | Files | Chunks |",
                 "|--------|------|--------|-------|--------|"]
        for j in jobs:
            lines.append(
                f"| {j.job_id[:8]} | {j.path} | {j.status} | {j.completed_files}/{j.total_files} | {j.embedded_chunks:,} |"
            )
        return "\n".join(lines)

    if name == "cancel_job":
        cancelled = await _registry.cancel(args["job_id"])
        if cancelled:
            return f"Job {args['job_id'][:8]} cancelled."
        return f"Could not cancel job {args['job_id'][:8]} — not found or not running."

    if name == "check_index":
        docs = await _backend.list_documents(args.get("category"))
        issues: list[str] = []
        for doc in docs:
            p = Path(doc.path)
            if not p.exists():
                issues.append(f"❌ Missing: {doc.path}")
            else:
                from doc_search_mcp.warnings import _checksum
                if _checksum(p) != doc.checksum:
                    issues.append(f"⚠️  Changed: {doc.path}")
        if not issues:
            return f"✅ Index is healthy ({len(docs)} documents checked)."
        return "\n".join(issues) + f"\n\n{len(issues)} issue(s) found. Use `reindex_path` or `remove_document` to resolve."

    if name == "acknowledge_warnings":
        await _backend.acknowledge_warnings(args["ids"])
        return f"Acknowledged {len(args['ids'])} warning(s)."

    if name == "get_config":
        embedder_name = type(_embedder).__name__ if _embedder else "none (keyword-only)"
        return (
            f"**doc-search-mcp configuration**\n\n"
            f"- Transport: {_config.server.transport} — {_config.server.host}:{_config.server.port}\n"
            f"- Storage: {_config.storage.backend} — {_config.storage.db_path}\n"
            f"- Embedder: {embedder_name} ({_config.embeddings.model})\n"
            f"- Search mode: {_config.search.default_mode}\n"
            f"- Rerank: {_config.search.rerank}\n"
            f"- Chunking: {_config.chunking.target_tokens} tokens, {_config.chunking.overlap_tokens} overlap\n"
        )

    return f"Unknown tool: {name!r}"


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------


async def _startup(config: Config) -> None:
    global _config, _backend, _embedder, _registry
    _config = config
    _backend = make_backend(config)
    await _backend.init_schema()

    # Verify embedding model consistency
    stored_model = await _backend.get_server_meta("embedding_model")
    if stored_model is None:
        await _backend.set_server_meta("embedding_model", config.embeddings.model)
    elif stored_model != config.embeddings.model:
        raise RuntimeError(
            f"Embedding model mismatch: index was built with '{stored_model}', "
            f"current config is '{config.embeddings.model}'. Run a full reindex to proceed."
        )

    _embedder = make_embedder(config)
    _registry = JobRegistry()

    if config.startup.check_on_startup:
        await run_startup_check(_backend)


def make_app(config: Config) -> Starlette:
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await _server.run(
                streams[0], streams[1], _server.create_initialization_options()
            )

    async def lifespan(app):
        await _startup(config)
        yield

    return Starlette(
        lifespan=lifespan,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="doc-search MCP server")
    parser.add_argument("--config", type=Path, help="Path to config.toml")
    parser.add_argument("--port", type=int, help="Override server port")
    parser.add_argument("--transport", choices=["sse", "stdio"], help="Override transport")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.port:
        config.server.port = args.port
    if args.transport:
        config.server.transport = args.transport

    if config.server.transport == "stdio":
        _run_stdio(config)
    else:
        app = make_app(config)
        uvicorn.run(
            app,
            host=config.server.host,
            port=config.server.port,
            log_level="info",
        )


def _run_stdio(config: Config) -> None:
    from mcp.server.stdio import stdio_server

    async def _main():
        await _startup(config)
        async with stdio_server() as (read_stream, write_stream):
            await _server.run(read_stream, write_stream, _server.create_initialization_options())

    asyncio.run(_main())
