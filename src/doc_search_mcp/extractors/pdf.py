from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class RawChunk:
    text: str
    page_or_section: str
    position: int


@dataclass
class PageStat:
    page: int  # 1-indexed
    char_count: int
    word_count: int
    has_text: bool


@dataclass
class PdfInspection:
    path: str
    total_pages: int
    pages_with_text: int
    empty_page_numbers: list[int]
    toc_entries: int
    structure_source: str  # 'toc', 'page', or 'ocr' (would need OCR)
    per_page: list[PageStat] = field(default_factory=list)


def extract_pdf(path: Path, max_toc_depth: int = 2) -> tuple[list[RawChunk], str]:
    """
    Extract text chunks from a PDF.
    Returns (chunks, structure_source) where structure_source is 'toc', 'page', or 'ocr'.
    Raises fitz.FileDataError for corrupt/encrypted PDFs.
    Prints a stderr warning if any pages yield no text.
    """
    doc = fitz.open(str(path))
    try:
        if doc.is_encrypted:
            raise ValueError(f"PDF is encrypted: {path}")

        toc = doc.get_toc(simple=True)
        toc = [entry for entry in toc if entry[0] <= max_toc_depth]

        if toc:
            chunks, empty_pages = _chunk_by_toc(doc, toc)
            source = "toc"
        else:
            chunks, empty_pages = _chunk_by_page(doc)
            source = "page"

        if not chunks:
            print(f"[doc-search] No text layer in {path.name}, attempting OCR", file=sys.stderr)
            chunks = _ocr_pages(doc, path)
            source = "ocr"
            empty_pages = []

        if empty_pages and chunks:
            pct = len(empty_pages) / doc.page_count * 100
            shown = empty_pages[:20]
            tail = "..." if len(empty_pages) > 20 else ""
            print(
                f"[doc-search] {path.name}: {len(empty_pages)}/{doc.page_count} pages ({pct:.0f}%)"
                f" have no text layer — pages: {shown}{tail}",
                file=sys.stderr,
            )

        if not chunks:
            raise ValueError(
                f"No text extracted from {path.name} ({doc.page_count} pages) - "
                "PDF appears to be image-only with no recognisable text"
            )
        return chunks, source
    finally:
        doc.close()


def inspect_pdf(path: Path, max_toc_depth: int = 2) -> PdfInspection:
    """Return per-page extraction diagnostics without modifying the index."""
    doc = fitz.open(str(path))
    try:
        if doc.is_encrypted:
            raise ValueError(f"PDF is encrypted: {path}")

        per_page: list[PageStat] = []
        for page_num in range(doc.page_count):
            text = doc[page_num].get_text().strip()
            per_page.append(PageStat(
                page=page_num + 1,
                char_count=len(text),
                word_count=len(text.split()) if text else 0,
                has_text=bool(text),
            ))

        toc = doc.get_toc(simple=True)
        toc = [e for e in toc if e[0] <= max_toc_depth]

        empty = [p.page for p in per_page if not p.has_text]
        pages_with_text = sum(1 for p in per_page if p.has_text)

        if toc:
            structure_source = "toc"
        elif pages_with_text > 0:
            structure_source = "page"
        else:
            structure_source = "ocr"

        return PdfInspection(
            path=str(path),
            total_pages=doc.page_count,
            pages_with_text=pages_with_text,
            empty_page_numbers=empty,
            toc_entries=len(toc),
            structure_source=structure_source,
            per_page=per_page,
        )
    finally:
        doc.close()


def _chunk_by_toc(doc: fitz.Document, toc: list) -> tuple[list[RawChunk], list[int]]:
    """
    Chunk by TOC entries. Each entry spans from its start page to the next entry's start page.
    TOC entries: [level, title, page_number] (1-indexed).
    Returns (chunks, empty_page_numbers).
    """
    chunks = []
    empty_pages: list[int] = []
    entries = [(title, page - 1) for _, title, page in toc]  # convert to 0-indexed

    for i, (title, start_page) in enumerate(entries):
        end_page = entries[i + 1][1] if i + 1 < len(entries) else doc.page_count
        texts = []
        for page_num in range(start_page, min(end_page, doc.page_count)):
            page_text = doc[page_num].get_text().strip()
            if page_text:
                texts.append(page_text)
            else:
                empty_pages.append(page_num + 1)

        text = "\n\n".join(texts).strip()
        if text:
            label = f"{title} (p.{start_page + 1})"
            chunks.append(RawChunk(text=text, page_or_section=label, position=i))

    return chunks, empty_pages


def _ocr_pages(doc: fitz.Document, path: Path) -> list[RawChunk]:
    """Render each page to an image and OCR it with tesseract."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise ValueError(
            f"pytesseract and Pillow are required to OCR {path.name} - "
            "install them with: uv add pytesseract Pillow"
        )

    chunks = []
    for page_num in range(doc.page_count):
        pix = doc[page_num].get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        try:
            text = pytesseract.image_to_string(img).strip()
        except pytesseract.TesseractNotFoundError:
            raise ValueError(
                f"tesseract binary not found - install it with: sudo apt install tesseract-ocr"
            )
        if text:
            chunks.append(RawChunk(
                text=text,
                page_or_section=f"Page {page_num + 1}",
                position=page_num,
            ))
    return chunks


def _chunk_by_page(doc: fitz.Document) -> tuple[list[RawChunk], list[int]]:
    """One chunk per page. Returns (chunks, empty_page_numbers)."""
    chunks = []
    empty_pages: list[int] = []
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
        else:
            empty_pages.append(page_num + 1)
    return chunks, empty_pages
