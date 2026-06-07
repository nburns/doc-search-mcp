from unittest.mock import AsyncMock, MagicMock

import pytest

from doc_search_mcp.db import SearchResult
from doc_search_mcp.search import _rrf_merge, search


def _result(chunk_id, score=1.0, text="sample text"):
    return SearchResult(
        chunk_id=chunk_id,
        document_id=1,
        text=text,
        page_or_section="Page 1",
        title="Test Doc",
        category="test",
        score=score,
    )


def test_rrf_merge_deduplicates():
    kw = [_result(1), _result(2), _result(3)]
    vec = [_result(1), _result(4), _result(5)]
    merged = _rrf_merge(kw, vec, limit=10)
    ids = [r.chunk_id for r in merged]
    assert len(ids) == len(set(ids))  # no duplicates


def test_rrf_merge_boosts_shared_results():
    # chunk 1 appears in both lists, should rank higher than chunk 3 (kw only) and chunk 4 (vec only)
    kw = [_result(1), _result(3)]
    vec = [_result(1), _result(4)]
    merged = _rrf_merge(kw, vec, limit=10)
    assert merged[0].chunk_id == 1  # boosted by appearing in both


def test_rrf_merge_respects_limit():
    kw = [_result(i) for i in range(20)]
    vec = [_result(i + 10) for i in range(20)]
    merged = _rrf_merge(kw, vec, limit=5)
    assert len(merged) == 5


def test_rrf_merge_empty_inputs():
    assert _rrf_merge([], [], limit=10) == []
    assert len(_rrf_merge([_result(1)], [], limit=10)) == 1
    assert len(_rrf_merge([], [_result(1)], limit=10)) == 1


async def test_search_keyword_mode():
    backend = MagicMock()
    backend.keyword_search = AsyncMock(return_value=[_result(1, text="neural networks are cool")])
    backend.get_active_warnings = AsyncMock(return_value=[])

    result = await search(
        query="neural networks",
        category="ml",
        mode="keyword",
        limit=10,
        backend=backend,
        embedder=None,
    )
    assert "neural networks are cool" in result
    backend.keyword_search.assert_called_once()


async def test_search_no_results_message():
    backend = MagicMock()
    backend.keyword_search = AsyncMock(return_value=[])
    backend.get_active_warnings = AsyncMock(return_value=[])

    result = await search(
        query="xyzzy",
        category="test",
        mode="keyword",
        limit=10,
        backend=backend,
        embedder=None,
    )
    assert "No results" in result


async def test_search_auto_falls_back_to_keyword_without_embedder():
    backend = MagicMock()
    backend.keyword_search = AsyncMock(return_value=[_result(1)])
    backend.get_active_warnings = AsyncMock(return_value=[])

    await search(
        query="test",
        category="test",
        mode="auto",
        limit=10,
        backend=backend,
        embedder=None,
    )
    backend.keyword_search.assert_called_once()


async def test_search_hybrid_uses_both_when_embedder_present():
    backend = MagicMock()
    backend.keyword_search = AsyncMock(return_value=[_result(1)])
    backend.vector_search = AsyncMock(return_value=[_result(2)])
    backend.get_active_warnings = AsyncMock(return_value=[])

    import numpy as np

    embedder = MagicMock()
    embedder.embed = MagicMock(return_value=iter([np.zeros(768)]))

    await search(
        query="test query",
        category="test",
        mode="hybrid",
        limit=10,
        backend=backend,
        embedder=embedder,
    )
    backend.keyword_search.assert_called_once()
    backend.vector_search.assert_called_once()


async def test_search_appends_warning_footer():
    from doc_search_mcp.db import WarningRecord
    from datetime import datetime

    backend = MagicMock()
    backend.keyword_search = AsyncMock(return_value=[_result(1)])
    backend.get_active_warnings = AsyncMock(
        return_value=[
            WarningRecord(id=1, document_id=1, category="test", warning_type="changed", detected_at=datetime.utcnow())
        ]
    )

    result = await search(
        query="test",
        category="test",
        mode="keyword",
        limit=10,
        backend=backend,
        embedder=None,
    )
    assert "Index issues" in result
