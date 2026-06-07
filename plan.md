# doc-search-mcp — Plan & Decisions

A local MCP server that indexes PDFs, EPUBs, and text files and makes them searchable for LLM coding agents.

---

## Goals

- Index documents into searchable categories (e.g. "retro-apple", "ml-stuff")
- Expose search and retrieval via MCP tools
- Run on Linux, accessible over a local network via SSE
- Fast async indexing with progress tracking
- No cloud dependencies, no ongoing costs

---

## Runtime & Packaging

| Decision | Choice |
|----------|--------|
| Language | Python 3.10+ |
| Package manager | `uv` |
| Package spec | `pyproject.toml` only |
| Entry point | `doc-search-mcp` CLI via `[project.scripts]` |
| Install | `uv sync` — all dependencies installed by default, no extras |

### Install Experience

```bash
git clone ...
cd doc-search-mcp
uv sync
uv run doc-search-mcp
```

### Dev Workflow

```bash
uv sync
uv run pytest
uv run ruff check .
uv run pyright
uv run doc-search-mcp
```

---

## Dependencies

All installed by default:

| Package | Purpose |
|---------|---------|
| `mcp` | MCP server framework (official Anthropic SDK) |
| `pymupdf` | PDF text + structure extraction |
| `ebooklib` | EPUB parsing |
| `beautifulsoup4` | HTML stripping from EPUB chapters |
| `tiktoken` | Token-aware chunking |
| `fastembed` | Local embeddings, CPU-friendly |
| `sqlite-vec` | Vector search extension for SQLite |
| `sentence-transformers` | Reranking cross-encoder + CUDA embedding fallback |
| `asyncpg` | PostgreSQL async driver |
| `tomli` | TOML config parsing |

### Dev Dependencies

```toml
[tool.uv.dev-dependencies]
pytest = "*"
pytest-asyncio = "*"
ruff = "*"
pyright = "*"
```

---

## Transport

**SSE (Server-Sent Events)** — HTTP-based, works over a local network.

| Setting | Value |
|---------|-------|
| Transport | SSE |
| Bind | `0.0.0.0` (all interfaces) |
| Default port | `8080` |
| Auth | None (planned future enhancement) |
| TLS | None — use a reverse proxy (e.g. Caddy) if needed |

> ⚠️ No auth or TLS. Only expose on trusted local networks.

`stdio` transport retained as a config option for local-only use.

### Claude Code Config

```json
{
  "mcpServers": {
    "doc-search": {
      "url": "http://<host>:8080/sse"
    }
  }
}
```

---

## File Layout

```
doc-search-mcp/
├── pyproject.toml
├── README.md
└── src/
    └── doc_search_mcp/
        ├── __init__.py
        ├── server.py         # MCP server, tool definitions, SSE setup
        ├── db.py             # Backend abstraction, schema, migrations
        ├── indexer.py        # Async pipeline orchestration
        ├── search.py         # FTS5 + vector search, RRF merge
        ├── config.py         # Config loading (TOML + env vars)
        ├── jobs.py           # Async job tracking, progress
        ├── warnings.py       # Warning lifecycle management
        └── extractors/
            ├── __init__.py
            ├── pdf.py        # PyMuPDF, TOC detection, page fallback
            ├── epub.py       # ebooklib + bs4, chapter extraction
            └── text.py       # Plain text, Markdown, paragraph splitting
```

---

## Supported Formats

| Format | Extractor | Notes |
|--------|---------|-------|
| PDF | PyMuPDF | TOC detection, page fallback |
| EPUB | ebooklib + bs4 | Chapter-level extraction |
| TXT | Native | Paragraph block splitting |
| MD | Native | Heading-aware splitting |

---

## Document Structure Detection

### PDF

1. Call `doc.get_toc()` — if present and non-empty, chunk by chapter/section using TOC page ranges
2. Fall back to per-page chunking if no TOC

`structure_source` stored per document: `toc` | `page`

TOC chunking respects configurable max depth (default: level 2).

Edge cases handled:
- Nested TOC levels — truncated at max depth
- Duplicate/malformed TOC entries — merged or skipped
- Partial TOC coverage — chapter-chunk covered pages, page-chunk the rest

> **Future enhancement:** Visual chapter detection via font size + margin position heuristics for scanned/non-TOC PDFs.

### EPUB

Chunk per chapter. Subdivide oversized chapters using sentence-boundary splitting.

### Text / Markdown

Split on blank lines into paragraph blocks. Merge small blocks, subdivide large ones. Markdown splits on `##` headings — each section becomes a chunk.

---

## Chunking Strategy

| Setting | Value |
|---------|-------|
| Target chunk size | 400 tokens |
| Overlap | 50 tokens |
| Split boundary | Sentence first, word fallback |
| Token counter | `tiktoken` |

Subdivision: when a page/chapter/block exceeds 400 tokens, split on sentence boundaries. Copy the last 50 tokens of the prior chunk as the head of the next.

### `page_or_section` Format

| Situation | Value |
|-----------|-------|
| PDF with TOC | `"Chapter 3: Getting Started (p.42)"` |
| PDF without TOC | `"Page 42"` |
| EPUB | `"Chapter 3: Getting Started"` |
| MD with headings | `"## Installation"` |
| Plain text | `"Lines 120–180"` |

---

## Database

### Backends

| Backend | Use case |
|---------|---------|
| SQLite (default) | Local, single user, zero config |
| PostgreSQL | Shared/networked, multi-user |

Pluggable via a `SearchBackend` protocol — both backends implement the same interface.

SQLite runs in WAL mode for concurrent reads during writes.

### Schema

```sql
-- metadata
documents (
    id, path, title, file_type, category,
    checksum, structure_source, status,
    indexed_at, chunk_count
)

-- warnings
warnings (
    id, document_id, category,
    warning_type, detected_at, acknowledged_at
)

-- job tracking
indexing_jobs (
    id, path, category, status,
    total_files, completed_files,
    total_chunks, embedded_chunks,
    started_at, updated_at, error_count
)

-- config/metadata
server_meta (key, value)   -- stores embedding model name, schema version

-- text chunks
chunks (id, document_id, text, page_or_section, position)

-- FTS5 virtual table (keyword search, BM25)
chunks_fts USING fts5(text, content=chunks)

-- vector virtual table (semantic search)
chunks_vec USING vec0(embedding float[768])
```

### Portability

Embedding model name stored in `server_meta` at index creation. On startup, verify current backend model matches stored model — hard error with clear message if not, requiring reindex.

This ensures vectors are always consistent regardless of which backend runtime (fastembed/sentence-transformers) produced them.

---

## Embeddings

### Model

`nomic-embed-text` — available across all backends, same output vectors regardless of runtime. Cross-platform portable DB.

### Backend Detection Order

1. Check `DOC_SEARCH_EMBEDDER` env var — use explicitly if set
2. Check for CUDA → `sentence-transformers` with CUDA
3. Default → `fastembed` (CPU, no PyTorch required)
4. Ollama → opt-in via config
5. MLX → dormant in code, enabled when running on Apple Silicon in future

| Platform | Default backend |
|----------|----------------|
| Linux (no GPU) | fastembed |
| Linux (NVIDIA GPU) | sentence-transformers + CUDA |
| Mac M2 (future) | mlx-embeddings |
| Any (opt-in) | Ollama |

---

## Search & Retrieval

### Retrieval Mode

**Default: Hybrid (BM25 + Vector)**

| Mode | Behavior |
|------|---------|
| `keyword` | FTS5/BM25 only |
| `semantic` | Vector only |
| `hybrid` | Both, merged via RRF |
| `auto` | Hybrid if embeddings configured, keyword if not |

### Hybrid Search — RRF Merge

Reciprocal Rank Fusion formula: `1 / (k + rank)` where `k = 60` (standard default, hardcoded for now).

Each side fetches `limit * 3` candidates (minimum 50), results merged and re-ranked by RRF score, truncated to `limit`.

When a document has no vector index (embedding failed), hybrid silently falls back to keyword-only for that document.

### Reranking (opt-in)

Cross-encoder reranking via `sentence-transformers` (`cross-encoder/ms-marco-MiniLM`) as an optional third pass on top of hybrid results. Configured via `search.rerank = true`.

### Result Format

All results returned as a single markdown string — no JSON, token-efficient:

```
**Source:** Deep Learning with Python — Chapter 3: Getting Started (p.42)
**Score:** 0.87

Chunk text goes here...

---
**More context:** call `get_chunks` with ids [14, 15, 17]

---

⚠️ Index issues detected: 2 files changed, 1 missing. Call `check_index` for details.
```

File paths not included in results — agent gets title + section. Paths available via `list_documents` if needed.

---

## Categories

Documents are organized into named categories (e.g. `retro-apple`, `ml-stuff`).

| Decision | Choice |
|----------|--------|
| Storage | Single DB, `category` column on documents table |
| Creation | Implicit — created on first `index_path` call |
| Required | Yes — always specify a category |
| Cross-category search | `category="all"` explicit opt-in |

---

## MCP Tools

### Indexing

| Tool | Args | Description |
|------|------|-------------|
| `index_path` | `path, category` | File or directory (recursive, async). Returns `job_id` immediately. |
| `reindex_path` | `path, category` | Re-extract if checksum changed |
| `remove_document` | `path, category` | Remove from index |

### Search

| Tool | Args | Description |
|------|------|-------------|
| `search` | `query, category, mode="auto", limit=10, file_type=None, path_prefix=None` | Search documents |
| `search_in_document` | `query, path, category, limit=10` | Scoped to one file |
| `get_chunks` | `ids, category` | Retrieve chunks by ID for context expansion |

### Documents

| Tool | Args | Description |
|------|------|-------------|
| `list_documents` | `category=None` | All indexed docs + metadata |
| `list_categories` | — | Categories with doc/chunk counts, last indexed |
| `get_stats` | `category=None` | DB size, doc count, chunk count, index health |

### Jobs

| Tool | Args | Description |
|------|------|-------------|
| `get_job_status` | `job_id` | Progress report for an indexing job |
| `list_jobs` | `status=None` | All jobs — `running\|completed\|failed\|all` |
| `cancel_job` | `job_id` | Cancel a running job |

### Maintenance

| Tool | Args | Description |
|------|------|-------------|
| `check_index` | `category=None` | Full health report — changed, missing, unindexed files |
| `acknowledge_warnings` | `ids` | Suppress known warnings until next startup |
| `get_config` | — | Current config + active backend as markdown |
| `set_config` | `key, value` | Runtime override, persists to config.toml |

### Tool Descriptions (as agent sees them)

- `search` — "Search indexed documents. Use for any question about document contents."
- `search_in_document` — "Search within a specific file. Use when you know which document contains the answer."
- `get_chunks` — "Retrieve surrounding context for a search result by chunk ID."
- `list_documents` — "List all indexed documents. Use to discover what's available before searching."
- `list_categories` — "List all document categories with counts. Use before searching to know what's available."
- `check_index` — "Check index health. Reports changed, missing, or unindexed files."

---

## Async Indexing Pipeline

```
index_path called
      ↓
directory scan (recursive, single pass)
      ↓
job created → job_id returned immediately to agent
      ↓
┌─────────────────────────────────────────┐
│  Extraction workers (asyncio + threads) │  ← concurrent, GIL-released via PyMuPDF
│  PDF / EPUB / text parsing              │
└─────────────────────────────────────────┘
              ↓ chunks queued
┌─────────────────────────────────────────┐
│  Embedding workers (batched)            │  ← batch size 32, saturates available hardware
│  nomic-embed-text                       │
└─────────────────────────────────────────┘
              ↓ vectors + text
┌─────────────────────────────────────────┐
│  DB writer (single async writer)        │  ← WAL mode, queue-drained
└─────────────────────────────────────────┘
```

### Performance Config

```toml
[performance]
extraction_workers = 0       # 0 = auto (CPU core count)
embedding_batch_size = 32
embedding_queue_size = 256   # chunks buffered between stages
max_concurrent_jobs = 3
```

### Directory Scanning

- Recursive walk, all supported extensions
- Skip hidden files/dirs (`.git`, `__pycache__`, etc.)
- Skip files exceeding `max_file_size` (default 500MB)
- Checksum check per file — skip unchanged if already indexed
- Unsupported files silently skipped by default; pass `report_skipped=True` to list them

### Progress Format

```
📚 Indexing: ~/docs/ml-papers [ml-stuff]
   Files:    23 / 47 (2 failed)
   Chunks:   1,840 / ~3,800 embedded
   Queue:    124 chunks waiting for embedding
   Workers:  4 extracting | 1 embedding batch in flight
   Current:  attention-is-all-you-need.pdf, transformers-survey.pdf
   Elapsed:  2m 14s | ETA: ~2m 20s
```

---

## Change Detection & Warnings

### Startup Check

On server startup:
1. Scan all indexed paths across all categories
2. Compare SHA256 checksums against stored values
3. Check file existence
4. Log summary to stderr
5. Add warnings to `warnings` table for changed/missing files

Read-only — never auto-reindexes or auto-deletes.

```
[doc-search] Startup check: 3 categories, 47 documents
[doc-search] ⚠️  2 files changed since last index
[doc-search] ⚠️  1 file missing
[doc-search] Run `check_index` for details
```

### Warning Lifecycle

```
startup check → warnings table populated
      ↓
warning footer embedded in EVERY tool response
      ↓
agent calls check_index → full details
      ↓
reindex_path / remove_document → warning auto-cleared
  OR
acknowledge_warnings(ids) → suppressed until next startup
      ↓
next startup → acknowledged warnings re-evaluated
```

Acknowledgement is "I know about this" not "never tell me again." Resurfaces if issue persists.

### File Change Behavior

| Scenario | Behavior |
|----------|---------|
| Checksum unchanged | Skip on reindex, log "unchanged" |
| Checksum changed | Re-extract, delete old chunks, insert new, update checksum |
| File missing at search | Warn in results, don't auto-delete |
| File missing at reindex | Hard error, don't delete silently |

---

## Error Handling

### Error Types

| Error | Type | Behavior |
|-------|------|---------|
| File not found | Hard | Clear message, nothing indexed |
| Permission denied | Hard | Clear message, nothing indexed |
| Corrupt PDF | Hard | Clear message, nothing indexed |
| PDF has no extractable text | Soft | Warn, index metadata only |
| EPUB missing chapters | Soft | Warn, index what was found |
| Partial chunk extraction | Soft | Warn, index successful chunks |
| Embedding failure | Soft | Index text/FTS only, warn semantic unavailable |
| DB write failure | Hard | Rollback, clear message |
| Unknown file type | Hard | Clear message, list supported types |

### Document Status

| Status | Meaning |
|--------|---------|
| `ok` | Fully indexed |
| `partial` | Indexed with warnings |
| `failed` | Not indexed |

### Agent-Facing Messages

Hard error:
```
❌ Could not index ~/docs/broken.pdf
   Reason: PDF is corrupted or password protected
   Supported formats: pdf, epub, txt, md
```

Soft error:
```
⚠️ ~/docs/scanned-book.pdf indexed with warnings
   No extractable text found — stored metadata only
   Keyword and semantic search unavailable for this document
   Consider running OCR before indexing
```

Success:
```
✅ Indexed ~/docs/deep-learning.pdf
   Category: ml-stuff
   Chunks: 847 | Structure: toc (12 chapters) | Embeddings: ok
```

---

## Configuration

### File Location

```
~/.doc-search/
├── index.db
└── config.toml
```

Overridable via `DOC_SEARCH_DB` env var.

### config.toml

```toml
[server]
transport = "sse"           # sse | stdio
host = "0.0.0.0"
port = 8080

[storage]
backend = "sqlite"          # sqlite | postgres
db_path = "~/.doc-search/index.db"
postgres_url = ""

[embeddings]
backend = "auto"            # auto | fastembed | sentence-transformers | ollama
model = "nomic-embed-text"
ollama_url = "http://localhost:11434"

[search]
default_mode = "auto"       # auto | keyword | semantic | hybrid
default_limit = 10
rerank = false

[chunking]
target_tokens = 400
overlap_tokens = 50
max_toc_depth = 2
max_file_size_mb = 500

[performance]
extraction_workers = 0      # 0 = auto
embedding_batch_size = 32
embedding_queue_size = 256
max_concurrent_jobs = 3

[startup]
check_on_startup = true
```

### Environment Variable Overrides

| Var | Overrides |
|-----|-----------|
| `DOC_SEARCH_DB` | `storage.db_path` |
| `DOC_SEARCH_BACKEND` | `storage.backend` |
| `DOC_SEARCH_POSTGRES_URL` | `storage.postgres_url` |
| `DOC_SEARCH_EMBEDDER` | `embeddings.backend` |
| `DOC_SEARCH_OLLAMA_URL` | `embeddings.ollama_url` |
| `DOC_SEARCH_DEFAULT_MODE` | `search.default_mode` |
| `DOC_SEARCH_PORT` | `server.port` |

---

## Future Enhancements

- Auth (bearer token, named API keys)
- TLS support
- Visual chapter detection for PDFs without TOC (font size + margin position heuristics)
- MLX embeddings on Apple Silicon (M2+)
- Code file indexing with AST-aware chunking (tree-sitter)
- OCR pipeline for scanned PDFs
- Web UI for browsing/managing the index

