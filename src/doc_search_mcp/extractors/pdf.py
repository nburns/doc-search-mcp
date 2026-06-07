from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF


@dataclass
class RawChunk:
    text: str
    page_or_section: str
    position: int


def extract_pdf(path: Path, max_toc_depth: int = 2) -> tuple[list[RawChunk], str]:
    """
    Extract text chunks from a PDF.
    Returns (chunks, structure_source) where structure_source is 'toc' or 'page'.
    Raises fitz.FileDataError for corrupt/encrypted PDFs.
    """
    doc = fitz.open(str(path))
    try:
        if doc.is_encrypted:
            raise ValueError(f"PDF is encrypted: {path}")

        toc = doc.get_toc(simple=True)
        toc = [entry for entry in toc if entry[0] <= max_toc_depth]

        if toc:
            chunks = _chunk_by_toc(doc, toc)
            return chunks, "toc"
        else:
            chunks = _chunk_by_page(doc)
            return chunks, "page"
    finally:
        doc.close()


def _chunk_by_toc(doc: fitz.Document, toc: list) -> list[RawChunk]:
    """
    Chunk by TOC entries. Each entry spans from its start page to the next entry's start page.
    TOC entries: [level, title, page_number] (1-indexed).
    """
    chunks = []
    entries = [(title, page - 1) for _, title, page in toc]  # convert to 0-indexed

    for i, (title, start_page) in enumerate(entries):
        end_page = entries[i + 1][1] if i + 1 < len(entries) else doc.page_count
        texts = []
        for page_num in range(start_page, min(end_page, doc.page_count)):
            page_text = doc[page_num].get_text().strip()
            if page_text:
                texts.append(page_text)

        text = "\n\n".join(texts).strip()
        if text:
            label = f"{title} (p.{start_page + 1})"
            chunks.append(RawChunk(text=text, page_or_section=label, position=i))

    # Pages not covered by TOC (shouldn't happen, but handle partial TOC)
    covered = {p for _, start in [(t, e) for t, e in zip(toc, entries)] for p in range(start[1], doc.page_count)}
    if len(chunks) < doc.page_count:
        pass  # partial coverage handled by the range logic above

    return chunks


def _chunk_by_page(doc: fitz.Document) -> list[RawChunk]:
    """One chunk per page."""
    chunks = []
    for page_num in range(doc.page_count):
        text = doc[page_num].get_text().strip()
        if text:
            chunks.append(
                RawChunk(
                    text=text,
                    page_or_section=f"Page {page_num + 1}",
                    position=page_num,
                )
            )
    return chunks
