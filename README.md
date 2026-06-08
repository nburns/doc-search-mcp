# doc-search-mcp

A local MCP server that indexes PDFs, EPUBs, and text files and makes them searchable via MCP tools. Designed for LLM coding agents that need to query large local document collections without uploading them to the cloud.

## Features

- Indexes PDFs (with TOC detection), EPUBs, Markdown, and plain text
- Keyword search via FTS5/BM25 and semantic search via vector embeddings
- Hybrid search merging both with Reciprocal Rank Fusion
- Documents organized into named categories
- Same content at multiple paths shares one index record (deduplication by checksum)
- Async indexing with progress tracking via job IDs
- Startup health checks with per-file change detection
- Accessible over a local network via SSE transport

## Install

```bash
git clone ...
cd doc-search-mcp
uv sync
uv run doc-search-mcp
```

Requires Python 3.10+. All dependencies are installed by default — no extras needed.

## Connect to Claude Code

Start the server, then add an entry to `~/.claude/.mcp.json` (create it if it doesn't exist):

```json
{
  "mcpServers": {
    "doc-search": {
      "type": "sse",
      "url": "http://localhost:8080/sse"
    }
  }
}
```

Use the machine's hostname or IP instead of `localhost` if connecting from another machine.

In a new Claude Code session, run `/mcp` to confirm the server is connected and its tools are listed.

> No auth or TLS. Only expose on trusted local networks.

## Configuration

Config file lives at `~/.doc-search/config.toml`. All settings are optional — defaults work out of the box.

```toml
[server]
transport = "sse"           # sse | stdio
host = "0.0.0.0"
port = 8080

[storage]
backend = "sqlite"          # sqlite | postgres
db_path = "~/.doc-search/index.db"

[embeddings]
backend = "auto"            # auto | fastembed | sentence-transformers | ollama | none
model = "nomic-ai/nomic-embed-text-v1.5"
ollama_url = "http://localhost:11434"

[search]
default_mode = "auto"       # auto | keyword | semantic | hybrid
default_limit = 10

[chunking]
target_tokens = 400
overlap_tokens = 50
max_toc_depth = 2
max_file_size_mb = 500

[performance]
extraction_workers = 0      # 0 = auto (CPU count)
embedding_batch_size = 32
embedding_queue_size = 256
```

Environment variable overrides: `DOC_SEARCH_DB`, `DOC_SEARCH_BACKEND`, `DOC_SEARCH_POSTGRES_URL`, `DOC_SEARCH_EMBEDDER`, `DOC_SEARCH_OLLAMA_URL`, `DOC_SEARCH_DEFAULT_MODE`, `DOC_SEARCH_PORT`.

### Embeddings

Backend auto-detection order:
1. `DOC_SEARCH_EMBEDDER` env var if set
2. CUDA available → `sentence-transformers`
3. Default → `fastembed` (CPU, no PyTorch required)
4. `ollama` — opt-in via config
5. `none` — keyword-only search, no vectors

Set `backend = "none"` to skip embedding entirely and use keyword search only. This makes indexing much faster and is a good default for technical documentation with precise terminology.

## MCP Tools

### Indexing

| Tool | Args | Description |
|------|------|-------------|
| `index_path` | `path, category` | Index a file or directory. Returns a `job_id` immediately. |
| `reindex_path` | `path, category` | Re-extract files whose content has changed. |
| `remove_document` | `path, category` | Remove a path from the index. |

### Search

| Tool | Args | Description |
|------|------|-------------|
| `search` | `query, category, mode, limit, file_type, path_prefix` | Search documents. |
| `search_in_document` | `query, path, category, limit` | Search within one file. |
| `get_chunks` | `ids, category` | Retrieve chunks by ID to expand context around a result. |

### Documents

| Tool | Args | Description |
|------|------|-------------|
| `list_documents` | `category` | All indexed docs with metadata. |
| `list_categories` | — | Categories with doc/chunk counts and last indexed timestamp. |
| `get_stats` | `category` | DB size, document count, chunk count. |

### Jobs

| Tool | Args | Description |
|------|------|-------------|
| `get_job_status` | `job_id` | Progress report for a running indexing job. |
| `list_jobs` | `status` | List jobs filtered by status. |
| `cancel_job` | `job_id` | Cancel a running job. |

### Maintenance

| Tool | Args | Description |
|------|------|-------------|
| `check_index` | `category` | Full health report — changed, missing, and unindexed files. |
| `acknowledge_warnings` | `ids` | Suppress known warnings until next startup. |
| `get_config` | — | Current config and active backend. |
| `set_config` | `key, value` | Runtime override, persisted to config.toml. |

## Supported Formats

| Format | Notes |
|--------|-------|
| PDF | TOC-aware chunking; falls back to per-page if no TOC. Image-only PDFs are automatically OCR'd via tesseract if available. |
| EPUB | Per-chapter extraction |
| Markdown | Heading-aware splitting on `##` |
| Plain text | Paragraph block splitting |

### OCR for image-only PDFs

If a PDF has no text layer, the extractor automatically falls back to OCR using [tesseract](https://github.com/tesseract-ocr/tesseract). Install the system binary to enable it:

```bash
sudo apt install tesseract-ocr        # Debian/Ubuntu
brew install tesseract                 # macOS
```

Without tesseract, image-only PDFs fail with a clear error rather than silently producing an empty index entry.

## Dev

```bash
uv run pytest
uv run ruff check .
uv run pyright
```
