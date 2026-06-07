from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------



@dataclass
class DocumentRecord:
    # paths is a list: same content at multiple locations shares one record
    paths: list[str]
    title: str
    file_type: str  # pdf | epub | txt | md
    category: str
    checksum: str
    structure_source: str  # toc | page | chapter | heading | paragraph
    status: str  # ok | partial | failed
    chunk_count: int = 0
    id: int | None = None
    indexed_at: datetime | None = None

    @property
    def primary_path(self) -> str:
        return self.paths[0] if self.paths else ""


@dataclass
class ChunkRecord:
    document_id: int
    text: str
    page_or_section: str
    position: int
    embedding: list[float] | None = None
    id: int | None = None


@dataclass
class SearchResult:
    chunk_id: int
    document_id: int
    text: str
    page_or_section: str
    title: str
    category: str
    score: float


@dataclass
class CategoryInfo:
    name: str
    doc_count: int
    chunk_count: int
    last_indexed: datetime | None


@dataclass
class StatsInfo:
    db_size_bytes: int
    doc_count: int
    chunk_count: int
    category: str | None


@dataclass
class WarningRecord:
    document_id: int
    category: str
    warning_type: str
    detected_at: datetime
    id: int | None = None
    acknowledged_at: datetime | None = None


@dataclass
class JobRecord:
    id: str  # UUID
    path: str
    category: str
    status: str  # running | completed | failed | cancelled
    total_files: int = 0
    completed_files: int = 0
    total_chunks: int = 0
    embedded_chunks: int = 0
    error_count: int = 0
    started_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SearchBackend(Protocol):
    async def init_schema(self) -> None: ...

    async def upsert_document(self, doc: DocumentRecord) -> int:
        """Insert or update a document record, deduplicating by (checksum, category).
        Adds doc.paths to document_paths. Returns the document id."""
        ...

    async def add_path(self, document_id: int, path: str) -> None: ...

    async def remove_path(self, path: str, category: str) -> bool:
        """Remove a path. Deletes the document if it was the last path. Returns True if document deleted."""
        ...

    async def delete_document_chunks(self, document_id: int) -> None: ...

    async def insert_chunks(self, chunks: list[ChunkRecord]) -> list[int]:
        """Insert chunks and return their assigned ids."""
        ...

    async def keyword_search(
        self,
        query: str,
        category: str | None,
        limit: int,
        file_type: str | None = None,
        path_prefix: str | None = None,
    ) -> list[SearchResult]: ...

    async def vector_search(
        self,
        embedding: list[float],
        category: str | None,
        limit: int,
        file_type: str | None = None,
        path_prefix: str | None = None,
    ) -> list[SearchResult]: ...

    async def get_chunks_by_ids(self, ids: list[int]) -> list[ChunkRecord]: ...

    async def get_document_by_path(self, path: str, category: str) -> DocumentRecord | None: ...

    async def get_document_by_checksum(self, checksum: str, category: str) -> DocumentRecord | None: ...

    async def list_documents(self, category: str | None = None) -> list[DocumentRecord]: ...

    async def list_categories(self) -> list[CategoryInfo]: ...

    async def get_stats(self, category: str | None = None) -> StatsInfo: ...

    async def add_warning(self, warning: WarningRecord) -> int: ...

    async def acknowledge_warnings(self, ids: list[int]) -> None: ...

    async def get_active_warnings(self) -> list[WarningRecord]: ...

    async def clear_warnings_for_document(self, document_id: int) -> None: ...

    async def upsert_job(self, job: JobRecord) -> None: ...

    async def get_job(self, job_id: str) -> JobRecord | None: ...

    async def list_jobs(self, status: str | None = None) -> list[JobRecord]: ...

    async def get_server_meta(self, key: str) -> str | None: ...

    async def set_server_meta(self, key: str, value: str) -> None: ...


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    title            TEXT NOT NULL,
    file_type        TEXT NOT NULL,
    category         TEXT NOT NULL,
    checksum         TEXT NOT NULL,
    structure_source TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'ok',
    indexed_at       TEXT,
    chunk_count      INTEGER NOT NULL DEFAULT 0,
    UNIQUE(checksum, category)
);

-- One document can live at multiple paths (duplicate files)
CREATE TABLE IF NOT EXISTS document_paths (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    path        TEXT NOT NULL,
    UNIQUE(document_id, path)
);

CREATE INDEX IF NOT EXISTS idx_document_paths_path ON document_paths(path);

CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text            TEXT NOT NULL,
    page_or_section TEXT NOT NULL,
    position        INTEGER NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content=chunks,
    content_rowid=id
);

CREATE TABLE IF NOT EXISTS warnings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    category        TEXT NOT NULL,
    warning_type    TEXT NOT NULL,
    detected_at     TEXT NOT NULL,
    acknowledged_at TEXT
);

CREATE TABLE IF NOT EXISTS indexing_jobs (
    id               TEXT PRIMARY KEY,
    path             TEXT NOT NULL,
    category         TEXT NOT NULL,
    status           TEXT NOT NULL,
    total_files      INTEGER NOT NULL DEFAULT 0,
    completed_files  INTEGER NOT NULL DEFAULT 0,
    total_chunks     INTEGER NOT NULL DEFAULT 0,
    embedded_chunks  INTEGER NOT NULL DEFAULT 0,
    error_count      INTEGER NOT NULL DEFAULT 0,
    started_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS server_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- FTS triggers to keep chunks_fts in sync
CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
"""




class SQLiteBackend:
    """SQLite backend using sqlite-vec for vector search and FTS5 for keyword search."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = asyncio.Lock()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception as exc:
            import sys

            print(f"[doc-search] Warning: sqlite-vec unavailable: {exc}", file=sys.stderr)
        return conn

    async def _run(self, fn):
        loop = asyncio.get_event_loop()
        async with self._lock:
            return await loop.run_in_executor(None, fn)

    async def init_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        def _init():
            conn = self._connect()
            conn.executescript(_SCHEMA)
            try:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(embedding float[768])"
                )
                conn.commit()
            except sqlite3.OperationalError:
                pass
            conn.commit()
            conn.close()

        await self._run(_init)

    # --- documents ---

    async def upsert_document(self, doc: DocumentRecord) -> int:
        def _upsert():
            conn = self._connect()
            now = datetime.utcnow().isoformat()

            existing = conn.execute(
                "SELECT id FROM documents WHERE checksum = ? AND category = ?",
                (doc.checksum, doc.category),
            ).fetchone()

            if existing:
                doc_id = existing["id"]
                conn.execute(
                    """UPDATE documents SET title=?, file_type=?, structure_source=?,
                       status=?, indexed_at=?, chunk_count=? WHERE id=?""",
                    (doc.title, doc.file_type, doc.structure_source,
                     doc.status, now, doc.chunk_count, doc_id),
                )
            else:
                cur = conn.execute(
                    """INSERT INTO documents
                           (title, file_type, category, checksum, structure_source, status, indexed_at, chunk_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (doc.title, doc.file_type, doc.category, doc.checksum,
                     doc.structure_source, doc.status, now, doc.chunk_count),
                )
                doc_id = cur.lastrowid

            for path in doc.paths:
                conn.execute(
                    "INSERT OR IGNORE INTO document_paths (document_id, path) VALUES (?, ?)",
                    (doc_id, path),
                )

            conn.commit()
            conn.close()
            return doc_id

        return await self._run(_upsert)

    async def add_path(self, document_id: int, path: str) -> None:
        def _add():
            conn = self._connect()
            conn.execute(
                "INSERT OR IGNORE INTO document_paths (document_id, path) VALUES (?, ?)",
                (document_id, path),
            )
            conn.commit()
            conn.close()

        await self._run(_add)

    async def remove_path(self, path: str, category: str) -> bool:
        """Remove a path. Deletes the document if it was the last path. Returns True if document deleted."""
        def _remove():
            conn = self._connect()
            row = conn.execute(
                """SELECT dp.document_id FROM document_paths dp
                   JOIN documents d ON d.id = dp.document_id
                   WHERE dp.path = ? AND d.category = ?""",
                (path, category),
            ).fetchone()
            if row is None:
                conn.close()
                return False
            doc_id = row["document_id"]
            conn.execute(
                "DELETE FROM document_paths WHERE document_id = ? AND path = ?",
                (doc_id, path),
            )
            remaining = conn.execute(
                "SELECT COUNT(*) FROM document_paths WHERE document_id = ?", (doc_id,)
            ).fetchone()[0]
            deleted = False
            if remaining == 0:
                conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                deleted = True
            conn.commit()
            conn.close()
            return deleted

        return await self._run(_remove)

    async def delete_document_chunks(self, document_id: int) -> None:
        def _delete():
            conn = self._connect()
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.commit()
            conn.close()

        await self._run(_delete)

    async def insert_chunks(self, chunks: list[ChunkRecord]) -> list[int]:
        def _insert():
            conn = self._connect()
            ids = []
            for chunk in chunks:
                cur = conn.execute(
                    "INSERT INTO chunks (document_id, text, page_or_section, position) VALUES (?, ?, ?, ?)",
                    (chunk.document_id, chunk.text, chunk.page_or_section, chunk.position),
                )
                chunk_id = cur.lastrowid
                ids.append(chunk_id)
                if chunk.embedding is not None:
                    try:
                        import struct

                        blob = struct.pack(f"{len(chunk.embedding)}f", *chunk.embedding)
                        conn.execute(
                            "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                            (chunk_id, blob),
                        )
                    except Exception:
                        pass
            conn.commit()
            conn.close()
            return ids

        return await self._run(_insert)

    async def get_document_by_path(self, path: str, category: str) -> DocumentRecord | None:
        def _get():
            conn = self._connect()
            row = conn.execute(
                """
                SELECT d.* FROM documents d
                JOIN document_paths dp ON dp.document_id = d.id
                WHERE dp.path = ? AND d.category = ?
                """,
                (path, category),
            ).fetchone()
            if row is None:
                conn.close()
                return None
            paths = _fetch_paths(conn, row["id"])
            conn.close()
            return row, paths

        result = await self._run(_get)
        return _row_to_document(*result) if result else None

    async def get_document_by_checksum(self, checksum: str, category: str) -> DocumentRecord | None:
        def _get():
            conn = self._connect()
            row = conn.execute(
                "SELECT * FROM documents WHERE checksum = ? AND category = ?",
                (checksum, category),
            ).fetchone()
            if row is None:
                conn.close()
                return None
            paths = _fetch_paths(conn, row["id"])
            conn.close()
            return row, paths

        result = await self._run(_get)
        return _row_to_document(*result) if result else None

    async def list_documents(self, category: str | None = None) -> list[DocumentRecord]:
        def _list():
            conn = self._connect()
            if category is None:
                rows = conn.execute(
                    "SELECT * FROM documents ORDER BY category, title"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE category = ? ORDER BY title",
                    (category,),
                ).fetchall()
            results = [(row, _fetch_paths(conn, row["id"])) for row in rows]
            conn.close()
            return results

        results = await self._run(_list)
        return [_row_to_document(row, paths) for row, paths in results]

    async def list_categories(self) -> list[CategoryInfo]:
        def _list():
            conn = self._connect()
            rows = conn.execute(
                """
                SELECT d.category,
                       COUNT(DISTINCT d.id) AS doc_count,
                       SUM(d.chunk_count)   AS chunk_count,
                       MAX(d.indexed_at)    AS last_indexed
                FROM documents d
                GROUP BY d.category
                ORDER BY d.category
                """
            ).fetchall()
            conn.close()
            return rows

        rows = await self._run(_list)
        return [
            CategoryInfo(
                name=r["category"],
                doc_count=r["doc_count"],
                chunk_count=r["chunk_count"] or 0,
                last_indexed=datetime.fromisoformat(r["last_indexed"]) if r["last_indexed"] else None,
            )
            for r in rows
        ]

    async def get_stats(self, category: str | None = None) -> StatsInfo:
        def _stats():
            conn = self._connect()
            if category:
                row = conn.execute(
                    "SELECT COUNT(*) AS doc_count, SUM(chunk_count) AS chunk_count FROM documents WHERE category = ?",
                    (category,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS doc_count, SUM(chunk_count) AS chunk_count FROM documents"
                ).fetchone()
            size = self._db_path.stat().st_size if self._db_path.exists() else 0
            conn.close()
            return row, size

        row, size = await self._run(_stats)
        return StatsInfo(
            db_size_bytes=size,
            doc_count=row["doc_count"] or 0,
            chunk_count=row["chunk_count"] or 0,
            category=category,
        )

    # --- search ---

    async def keyword_search(
        self,
        query: str,
        category: str | None,
        limit: int,
        file_type: str | None = None,
        path_prefix: str | None = None,
    ) -> list[SearchResult]:
        def _search():
            conn = self._connect()
            conditions = ["1=1"]
            params: list = [query]
            if category and category != "all":
                conditions.append("d.category = ?")
                params.append(category)
            if file_type:
                conditions.append("d.file_type = ?")
                params.append(file_type)
            if path_prefix:
                conditions.append(
                    "EXISTS (SELECT 1 FROM document_paths dp2 WHERE dp2.document_id = d.id AND dp2.path LIKE ?)"
                )
                params.append(f"{path_prefix}%")
            where = " AND ".join(conditions)
            params.append(limit)
            rows = conn.execute(
                f"""
                SELECT c.id AS chunk_id, c.document_id, c.text, c.page_or_section,
                       d.title, d.category,
                       bm25(chunks_fts) AS score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                JOIN documents d ON d.id = c.document_id
                WHERE chunks_fts MATCH ? AND {where}
                ORDER BY score
                LIMIT ?
                """,
                params,
            ).fetchall()
            conn.close()
            return rows

        rows = await self._run(_search)
        return [
            SearchResult(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                text=r["text"],
                page_or_section=r["page_or_section"],
                title=r["title"],
                category=r["category"],
                score=abs(r["score"]),
            )
            for r in rows
        ]

    async def vector_search(
        self,
        embedding: list[float],
        category: str | None,
        limit: int,
        file_type: str | None = None,
        path_prefix: str | None = None,
    ) -> list[SearchResult]:
        def _search():
            import struct

            conn = self._connect()
            blob = struct.pack(f"{len(embedding)}f", *embedding)
            conditions = []
            params: list = [blob, limit * 3]
            if category and category != "all":
                conditions.append("d.category = ?")
                params.append(category)
            if file_type:
                conditions.append("d.file_type = ?")
                params.append(file_type)
            if path_prefix:
                conditions.append(
                    "EXISTS (SELECT 1 FROM document_paths dp2 WHERE dp2.document_id = d.id AND dp2.path LIKE ?)"
                )
                params.append(f"{path_prefix}%")
            where = ("AND " + " AND ".join(conditions)) if conditions else ""
            params.append(limit)
            rows = conn.execute(
                f"""
                SELECT c.id AS chunk_id, c.document_id, c.text, c.page_or_section,
                       d.title, d.category,
                       v.distance AS score
                FROM chunks_vec v
                JOIN chunks c ON c.id = v.rowid
                JOIN documents d ON d.id = c.document_id
                WHERE v.embedding MATCH ? AND k = ? {where}
                ORDER BY v.distance
                LIMIT ?
                """,
                params,
            ).fetchall()
            conn.close()
            return rows

        rows = await self._run(_search)
        return [
            SearchResult(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                text=r["text"],
                page_or_section=r["page_or_section"],
                title=r["title"],
                category=r["category"],
                score=r["score"],
            )
            for r in rows
        ]

    async def get_chunks_by_ids(self, ids: list[int]) -> list[ChunkRecord]:
        def _get():
            conn = self._connect()
            placeholders = ",".join("?" * len(ids))
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})", ids
            ).fetchall()
            conn.close()
            return rows

        rows = await self._run(_get)
        return [
            ChunkRecord(
                id=r["id"],
                document_id=r["document_id"],
                text=r["text"],
                page_or_section=r["page_or_section"],
                position=r["position"],
            )
            for r in rows
        ]

    # --- warnings ---

    async def add_warning(self, warning: WarningRecord) -> int:
        def _add():
            conn = self._connect()
            cur = conn.execute(
                "INSERT INTO warnings (document_id, category, warning_type, detected_at) VALUES (?, ?, ?, ?)",
                (warning.document_id, warning.category, warning.warning_type, warning.detected_at.isoformat()),
            )
            conn.commit()
            conn.close()
            return cur.lastrowid

        return await self._run(_add)

    async def acknowledge_warnings(self, ids: list[int]) -> None:
        def _ack():
            conn = self._connect()
            now = datetime.utcnow().isoformat()
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                f"UPDATE warnings SET acknowledged_at = ? WHERE id IN ({placeholders})",
                [now, *ids],
            )
            conn.commit()
            conn.close()

        await self._run(_ack)

    async def get_active_warnings(self) -> list[WarningRecord]:
        def _get():
            conn = self._connect()
            rows = conn.execute(
                "SELECT * FROM warnings WHERE acknowledged_at IS NULL ORDER BY detected_at"
            ).fetchall()
            conn.close()
            return rows

        rows = await self._run(_get)
        return [_row_to_warning(r) for r in rows]

    async def clear_warnings_for_document(self, document_id: int) -> None:
        def _clear():
            conn = self._connect()
            conn.execute("DELETE FROM warnings WHERE document_id = ?", (document_id,))
            conn.commit()
            conn.close()

        await self._run(_clear)

    # --- jobs ---

    async def upsert_job(self, job: JobRecord) -> None:
        def _upsert():
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO indexing_jobs
                    (id, path, category, status, total_files, completed_files,
                     total_chunks, embedded_chunks, error_count, started_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    total_files=excluded.total_files,
                    completed_files=excluded.completed_files,
                    total_chunks=excluded.total_chunks,
                    embedded_chunks=excluded.embedded_chunks,
                    error_count=excluded.error_count,
                    updated_at=excluded.updated_at
                """,
                (
                    job.id, job.path, job.category, job.status,
                    job.total_files, job.completed_files, job.total_chunks,
                    job.embedded_chunks, job.error_count,
                    job.started_at.isoformat(), job.updated_at.isoformat(),
                ),
            )
            conn.commit()
            conn.close()

        await self._run(_upsert)

    async def get_job(self, job_id: str) -> JobRecord | None:
        def _get():
            conn = self._connect()
            row = conn.execute("SELECT * FROM indexing_jobs WHERE id = ?", (job_id,)).fetchone()
            conn.close()
            return row

        row = await self._run(_get)
        return _row_to_job(row) if row else None

    async def list_jobs(self, status: str | None = None) -> list[JobRecord]:
        def _list():
            conn = self._connect()
            if status:
                rows = conn.execute(
                    "SELECT * FROM indexing_jobs WHERE status = ? ORDER BY started_at DESC", (status,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM indexing_jobs ORDER BY started_at DESC"
                ).fetchall()
            conn.close()
            return rows

        rows = await self._run(_list)
        return [_row_to_job(r) for r in rows]

    # --- server meta ---

    async def get_server_meta(self, key: str) -> str | None:
        def _get():
            conn = self._connect()
            row = conn.execute("SELECT value FROM server_meta WHERE key = ?", (key,)).fetchone()
            conn.close()
            return row

        row = await self._run(_get)
        return row["value"] if row else None

    async def set_server_meta(self, key: str, value: str) -> None:
        def _set():
            conn = self._connect()
            conn.execute(
                "INSERT INTO server_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            conn.commit()
            conn.close()

        await self._run(_set)


# ---------------------------------------------------------------------------
# Row -> dataclass helpers
# ---------------------------------------------------------------------------


def _fetch_paths(conn: sqlite3.Connection, document_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT path FROM document_paths WHERE document_id = ? ORDER BY id",
        (document_id,),
    ).fetchall()
    return [r["path"] for r in rows]


def _row_to_document(row: sqlite3.Row, paths: list[str]) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        paths=paths,
        title=row["title"],
        file_type=row["file_type"],
        category=row["category"],
        checksum=row["checksum"],
        structure_source=row["structure_source"],
        status=row["status"],
        chunk_count=row["chunk_count"],
        indexed_at=datetime.fromisoformat(row["indexed_at"]) if row["indexed_at"] else None,
    )


def _row_to_warning(row: sqlite3.Row) -> WarningRecord:
    return WarningRecord(
        id=row["id"],
        document_id=row["document_id"],
        category=row["category"],
        warning_type=row["warning_type"],
        detected_at=datetime.fromisoformat(row["detected_at"]),
        acknowledged_at=datetime.fromisoformat(row["acknowledged_at"]) if row["acknowledged_at"] else None,
    )


def _row_to_job(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        id=row["id"],
        path=row["path"],
        category=row["category"],
        status=row["status"],
        total_files=row["total_files"],
        completed_files=row["completed_files"],
        total_chunks=row["total_chunks"],
        embedded_chunks=row["embedded_chunks"],
        error_count=row["error_count"],
        started_at=datetime.fromisoformat(row["started_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def make_backend(config) -> SearchBackend:
    from doc_search_mcp.config import Config

    cfg: Config = config
    if cfg.storage.backend == "sqlite":
        return SQLiteBackend(cfg.storage.db_path)
    raise ValueError(f"Unsupported storage backend: {cfg.storage.backend!r} (only 'sqlite' implemented)")
