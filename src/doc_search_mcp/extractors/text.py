from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RawChunk:
    text: str
    page_or_section: str
    position: int


def extract_text(path: Path) -> list[RawChunk]:
    content = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".md":
        return _split_markdown(content)
    return _split_plain(content)


def _split_markdown(content: str) -> list[RawChunk]:
    """Split on ## headings; each section is one raw chunk."""
    # Split on lines that start with one or more # characters
    heading_re = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
    parts = heading_re.split(content)

    chunks: list[RawChunk] = []
    position = 0
    current_heading = "Introduction"
    current_lines: list[str] = []

    for part in parts:
        if heading_re.fullmatch(part.strip()):
            if current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    chunks.append(RawChunk(text=text, page_or_section=current_heading, position=position))
                    position += 1
            current_heading = part.strip()
            current_lines = []
        else:
            current_lines.extend(part.splitlines())

    if current_lines:
        text = "\n".join(current_lines).strip()
        if text:
            chunks.append(RawChunk(text=text, page_or_section=current_heading, position=position))

    return chunks or [RawChunk(text=content.strip(), page_or_section="Document", position=0)]


def _split_plain(content: str) -> list[RawChunk]:
    """Split on blank lines into paragraph blocks."""
    # Split on two or more consecutive newlines
    blocks = re.split(r"\n{2,}", content)
    chunks: list[RawChunk] = []
    position = 0
    line_offset = 0

    for block in blocks:
        text = block.strip()
        line_count = block.count("\n") + 1
        if text:
            start_line = line_offset + 1
            end_line = line_offset + line_count
            label = f"Lines {start_line}-{end_line}" if start_line != end_line else f"Line {start_line}"
            chunks.append(RawChunk(text=text, page_or_section=label, position=position))
            position += 1
        line_offset += line_count

    return chunks or [RawChunk(text=content.strip(), page_or_section="Document", position=0)]
