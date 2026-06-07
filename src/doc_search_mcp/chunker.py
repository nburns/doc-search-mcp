from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from doc_search_mcp.db import ChunkRecord

# Shared encoder — cl100k_base works for all current models
_enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class _Span:
    text: str
    page_or_section: str


def chunk_raw(
    document_id: int,
    spans: list[_Span],
    target_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[ChunkRecord]:
    """
    Convert raw extractor spans into DB-ready ChunkRecords.
    Spans that exceed target_tokens are subdivided on sentence boundaries.
    Overlap is applied by prepending the tail of the prior chunk.
    """
    records: list[ChunkRecord] = []
    position = 0
    prev_tail: str = ""

    for span in spans:
        sub_texts = _subdivide(span.text, target_tokens)
        for i, text in enumerate(sub_texts):
            if i == 0:
                full_text = prev_tail + text
            else:
                full_text = _overlap_head(sub_texts[i - 1], overlap_tokens) + text
            records.append(
                ChunkRecord(
                    document_id=document_id,
                    text=full_text.strip(),
                    page_or_section=span.page_or_section,
                    position=position,
                )
            )
            position += 1
        if sub_texts:
            prev_tail = _overlap_head(sub_texts[-1], overlap_tokens)

    return records


def _subdivide(text: str, target_tokens: int) -> list[str]:
    """Split text into chunks of at most target_tokens."""
    tokens = _enc.encode(text)
    if len(tokens) <= target_tokens:
        return [text]

    # Split on sentence boundaries first
    sentences = _split_sentences(text)
    parts: list[str] = []
    current: list[str] = []
    current_count = 0

    for sentence in sentences:
        n = len(_enc.encode(sentence))
        if current and current_count + n > target_tokens:
            parts.append(" ".join(current))
            current = [sentence]
            current_count = n
        else:
            current.append(sentence)
            current_count += n

    if current:
        parts.append(" ".join(current))

    # Any sentence that is still oversized gets word-split
    final: list[str] = []
    for part in parts:
        if len(_enc.encode(part)) > target_tokens:
            final.extend(_word_split(part, target_tokens))
        else:
            final.append(part)

    return final


def _overlap_head(text: str, overlap_tokens: int) -> str:
    """Return the last overlap_tokens tokens of text as a string prefix."""
    if overlap_tokens == 0:
        return ""
    tokens = _enc.encode(text)
    tail_tokens = tokens[-overlap_tokens:]
    return _enc.decode(tail_tokens) + " "


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _word_split(text: str, target_tokens: int) -> list[str]:
    words = text.split()
    parts: list[str] = []
    current: list[str] = []
    current_count = 0

    for word in words:
        n = len(_enc.encode(word))
        if current and current_count + n > target_tokens:
            parts.append(" ".join(current))
            current = [word]
            current_count = n
        else:
            current.append(word)
            current_count += n

    if current:
        parts.append(" ".join(current))

    return parts
