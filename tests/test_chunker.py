import pytest

from doc_search_mcp.chunker import _Span, _overlap_head, _split_sentences, _subdivide, chunk_raw
from doc_search_mcp.db import ChunkRecord


def test_subdivide_short_text():
    text = "Hello world. This is a short sentence."
    parts = _subdivide(text, target_tokens=400)
    assert parts == [text]


def test_subdivide_splits_long_text():
    # Build text that's definitively over 400 tokens
    sentence = "The quick brown fox jumps over the lazy dog. "
    text = sentence * 50  # ~450+ tokens
    parts = _subdivide(text, target_tokens=400)
    assert len(parts) > 1
    # Reassembled text should contain all original content (modulo whitespace)
    combined = " ".join(parts)
    assert "quick brown fox" in combined


def test_subdivide_preserves_text_content():
    sentence = "Word " * 100  # definitely over 400 tokens
    parts = _subdivide(sentence.strip(), target_tokens=50)
    assert len(parts) > 1
    # Every part should be non-empty
    assert all(p.strip() for p in parts)


def test_overlap_head():
    text = "one two three four five six seven eight nine ten"
    head = _overlap_head(text, overlap_tokens=3)
    # Should contain the last few tokens
    assert head.strip()
    assert len(head) < len(text)


def test_split_sentences_basic():
    text = "Hello world. How are you? Fine, thanks!"
    sentences = _split_sentences(text)
    assert len(sentences) == 3


def test_chunk_raw_single_short_span():
    spans = [_Span(text="Short text.", page_or_section="Page 1")]
    records = chunk_raw(document_id=1, spans=spans, target_tokens=400, overlap_tokens=50)
    assert len(records) == 1
    assert records[0].text == "Short text."
    assert records[0].page_or_section == "Page 1"
    assert records[0].document_id == 1
    assert records[0].position == 0


def test_chunk_raw_preserves_positions():
    spans = [
        _Span(text="First span.", page_or_section="Page 1"),
        _Span(text="Second span.", page_or_section="Page 2"),
    ]
    records = chunk_raw(document_id=5, spans=spans, target_tokens=400, overlap_tokens=0)
    positions = [r.position for r in records]
    assert positions == sorted(positions)


def test_chunk_raw_no_embedding_initially():
    spans = [_Span(text="Test.", page_or_section="Section 1")]
    records = chunk_raw(document_id=1, spans=spans)
    assert all(r.embedding is None for r in records)


def test_overlap_carries_across_spans():
    # Two spans, each short enough to be a single chunk.
    # The second chunk should start with the tail of the first.
    first = "The cat sat on the mat. " * 5
    second = "Dogs are also great pets. " * 5
    spans = [
        _Span(text=first.strip(), page_or_section="Page 1"),
        _Span(text=second.strip(), page_or_section="Page 2"),
    ]
    records = chunk_raw(document_id=1, spans=spans, target_tokens=400, overlap_tokens=10)
    assert len(records) == 2
    # The second chunk must begin with content from the first span
    assert records[1].text != second.strip(), "overlap should prepend tail of first chunk"
    # The first span's content should appear at the start of the second chunk
    assert records[0].text[:20] in records[1].text


def test_overlap_carries_across_multi_subtext_span():
    # A span long enough to be split into multiple sub-texts.
    # The NEXT span should still receive overlap from the last sub-text.
    long_sentence = "The fox jumps. " * 60  # well over 400 tokens
    short_next = "Next section content."
    spans = [
        _Span(text=long_sentence.strip(), page_or_section="Page 1"),
        _Span(text=short_next, page_or_section="Page 2"),
    ]
    records = chunk_raw(document_id=1, spans=spans, target_tokens=400, overlap_tokens=10)
    # The last record (short_next) should have overlap prepended
    last = records[-1]
    assert last.text != short_next, "overlap from last sub-text of prior span should be prepended"
    assert last.page_or_section == "Page 2"


def test_overlap_zero_no_prepend():
    spans = [
        _Span(text="First span text here.", page_or_section="p1"),
        _Span(text="Second span text here.", page_or_section="p2"),
    ]
    records = chunk_raw(document_id=1, spans=spans, target_tokens=400, overlap_tokens=0)
    assert records[1].text == "Second span text here."


def test_overlap_head_zero_returns_empty():
    result = _overlap_head("some text here", overlap_tokens=0)
    assert result == ""
