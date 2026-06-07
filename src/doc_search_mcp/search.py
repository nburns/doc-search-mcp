from __future__ import annotations

from doc_search_mcp.db import SearchBackend, SearchResult
from doc_search_mcp.warnings import warning_footer

_RRF_K = 60  # standard RRF constant


async def search(
    query: str,
    category: str | None,
    mode: str,
    limit: int,
    backend: SearchBackend,
    embedder,
    file_type: str | None = None,
    path_prefix: str | None = None,
) -> str:
    """Run search and return a markdown-formatted result string."""
    results = await _retrieve(
        query=query,
        category=category,
        mode=mode,
        limit=limit,
        backend=backend,
        embedder=embedder,
        file_type=file_type,
        path_prefix=path_prefix,
    )

    if not results:
        footer = await warning_footer(backend)
        return f"No results found for: {query!r}{footer}"

    ids_by_doc: dict[int, list[int]] = {}
    for r in results:
        ids_by_doc.setdefault(r.document_id, []).append(r.chunk_id)

    lines: list[str] = []
    for r in results:
        nearby = [i for i in ids_by_doc[r.document_id] if i != r.chunk_id][:2]
        lines.append(f"**Source:** {r.title} — {r.page_or_section}")
        lines.append(f"**Score:** {r.score:.4f}")
        lines.append("")
        lines.append(r.text)
        if nearby:
            lines.append("")
            lines.append(f"**More context:** call `get_chunks` with ids {[r.chunk_id, *nearby]}")
        lines.append("\n---")

    footer = await warning_footer(backend)
    return "\n".join(lines) + footer


async def _retrieve(
    query: str,
    category: str | None,
    mode: str,
    limit: int,
    backend: SearchBackend,
    embedder,
    file_type: str | None,
    path_prefix: str | None,
) -> list[SearchResult]:
    fetch_n = max(limit * 3, 50)

    # Resolve "auto" mode: use hybrid if embedder is available
    if mode == "auto":
        mode = "hybrid" if embedder is not None else "keyword"

    if mode == "keyword":
        return await backend.keyword_search(query, category, limit, file_type, path_prefix)

    if mode == "semantic":
        if embedder is None:
            raise RuntimeError("Semantic search requested but no embedder is configured")
        embedding = await _embed(query, embedder)
        return await backend.vector_search(embedding, category, limit, file_type, path_prefix)

    if mode == "hybrid":
        kw_results = await backend.keyword_search(query, category, fetch_n, file_type, path_prefix)
        if embedder is not None:
            embedding = await _embed(query, embedder)
            vec_results = await backend.vector_search(
                embedding, category, fetch_n, file_type, path_prefix
            )
        else:
            vec_results = []
        merged = _rrf_merge(kw_results, vec_results, limit)
        return merged

    raise ValueError(f"Unknown search mode: {mode!r}")


def _rrf_merge(
    kw: list[SearchResult],
    vec: list[SearchResult],
    limit: int,
) -> list[SearchResult]:
    scores: dict[int, float] = {}
    by_id: dict[int, SearchResult] = {}

    for rank, result in enumerate(kw):
        scores[result.chunk_id] = scores.get(result.chunk_id, 0) + 1.0 / (_RRF_K + rank + 1)
        by_id[result.chunk_id] = result

    for rank, result in enumerate(vec):
        scores[result.chunk_id] = scores.get(result.chunk_id, 0) + 1.0 / (_RRF_K + rank + 1)
        if result.chunk_id not in by_id:
            by_id[result.chunk_id] = result

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    results = []
    for chunk_id, rrf_score in ranked:
        r = by_id[chunk_id]
        results.append(
            SearchResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                text=r.text,
                page_or_section=r.page_or_section,
                title=r.title,
                category=r.category,
                score=rrf_score,
            )
        )
    return results


async def _embed(text: str, embedder) -> list[float]:
    import asyncio

    loop = asyncio.get_event_loop()
    # Embedders are synchronous; run in thread pool
    result = await loop.run_in_executor(None, lambda: embedder.embed([text]))
    vectors = list(result)
    return vectors[0].tolist()
