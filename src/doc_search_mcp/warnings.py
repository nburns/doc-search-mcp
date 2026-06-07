from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

from doc_search_mcp.db import DocumentRecord, SearchBackend, WarningRecord


async def run_startup_check(backend: SearchBackend) -> int:
    """
    Scan all indexed paths, compare checksums, and populate the warnings table.
    Returns the number of issues found.
    Returns a count of changed + missing files.
    Read-only: never reindexes or deletes anything.
    """
    docs = await backend.list_documents()
    if not docs:
        return 0

    changed = 0
    missing = 0
    categories = {d.category for d in docs}

    print(
        f"[doc-search] Startup check: {len(categories)} categories, {len(docs)} documents",
        file=sys.stderr,
    )

    for doc in docs:
        # A document may have multiple paths (duplicate files). Check all of them.
        # Warn if every path is missing, or if any readable path has a changed checksum.
        all_missing = all(not Path(p).exists() for p in doc.paths)
        any_changed = any(
            Path(p).exists() and _checksum(Path(p)) != doc.checksum
            for p in doc.paths
        )

        if all_missing and doc.paths:
            missing += 1
            await backend.add_warning(
                WarningRecord(
                    document_id=doc.id,  # type: ignore[arg-type]
                    category=doc.category,
                    warning_type="missing",
                    detected_at=datetime.utcnow(),
                )
            )
        elif any_changed:
            changed += 1
            await backend.add_warning(
                WarningRecord(
                    document_id=doc.id,  # type: ignore[arg-type]
                    category=doc.category,
                    warning_type="changed",
                    detected_at=datetime.utcnow(),
                )
            )

    if changed:
        print(f"[doc-search] ⚠️  {changed} files changed since last index", file=sys.stderr)
    if missing:
        print(f"[doc-search] ⚠️  {missing} files missing", file=sys.stderr)
    if changed or missing:
        print("[doc-search] Run `check_index` for details", file=sys.stderr)

    return changed + missing


async def warning_footer(backend: SearchBackend) -> str:
    """Return a warning footer string to append to tool responses, or empty string."""
    warnings = await backend.get_active_warnings()
    if not warnings:
        return ""
    changed = sum(1 for w in warnings if w.warning_type == "changed")
    missing = sum(1 for w in warnings if w.warning_type == "missing")
    parts = []
    if changed:
        parts.append(f"{changed} files changed")
    if missing:
        parts.append(f"{missing} missing")
    summary = ", ".join(parts)
    return f"\n\n---\n⚠️ Index issues detected: {summary}. Call `check_index` for details."


def _checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
