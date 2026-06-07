from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-reuse-declared-module]

DEFAULT_CONFIG_DIR = Path.home() / ".doc-search"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"


@dataclass
class ServerConfig:
    transport: str = "sse"
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass
class StorageConfig:
    backend: str = "sqlite"
    db_path: Path = field(default_factory=lambda: DEFAULT_CONFIG_DIR / "index.db")
    postgres_url: str = ""


@dataclass
class EmbeddingsConfig:
    backend: str = "auto"
    model: str = "nomic-embed-text"
    ollama_url: str = "http://localhost:11434"


@dataclass
class SearchConfig:
    default_mode: str = "auto"
    default_limit: int = 10
    rerank: bool = False


@dataclass
class ChunkingConfig:
    target_tokens: int = 400
    overlap_tokens: int = 50
    max_toc_depth: int = 2
    max_file_size_mb: int = 500


@dataclass
class PerformanceConfig:
    extraction_workers: int = 0  # 0 = auto (CPU count)
    embedding_batch_size: int = 32
    embedding_queue_size: int = 256
    max_concurrent_jobs: int = 3


@dataclass
class StartupConfig:
    check_on_startup: bool = True


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    startup: StartupConfig = field(default_factory=StartupConfig)


def load_config(path: Path | None = None) -> Config:
    config = Config()
    toml_path = path or DEFAULT_CONFIG_PATH
    if toml_path.exists():
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        _apply_toml(config, data)
    _apply_env(config)
    return config


def _apply_toml(config: Config, data: dict) -> None:
    if server := data.get("server"):
        config.server.transport = server.get("transport", config.server.transport)
        config.server.host = server.get("host", config.server.host)
        config.server.port = int(server.get("port", config.server.port))

    if storage := data.get("storage"):
        config.storage.backend = storage.get("backend", config.storage.backend)
        if db_path := storage.get("db_path"):
            config.storage.db_path = Path(db_path).expanduser()
        config.storage.postgres_url = storage.get("postgres_url", config.storage.postgres_url)

    if embeddings := data.get("embeddings"):
        config.embeddings.backend = embeddings.get("backend", config.embeddings.backend)
        config.embeddings.model = embeddings.get("model", config.embeddings.model)
        config.embeddings.ollama_url = embeddings.get("ollama_url", config.embeddings.ollama_url)

    if search := data.get("search"):
        config.search.default_mode = search.get("default_mode", config.search.default_mode)
        config.search.default_limit = int(search.get("default_limit", config.search.default_limit))
        config.search.rerank = bool(search.get("rerank", config.search.rerank))

    if chunking := data.get("chunking"):
        config.chunking.target_tokens = int(chunking.get("target_tokens", config.chunking.target_tokens))
        config.chunking.overlap_tokens = int(chunking.get("overlap_tokens", config.chunking.overlap_tokens))
        config.chunking.max_toc_depth = int(chunking.get("max_toc_depth", config.chunking.max_toc_depth))
        config.chunking.max_file_size_mb = int(chunking.get("max_file_size_mb", config.chunking.max_file_size_mb))

    if perf := data.get("performance"):
        config.performance.extraction_workers = int(
            perf.get("extraction_workers", config.performance.extraction_workers)
        )
        config.performance.embedding_batch_size = int(
            perf.get("embedding_batch_size", config.performance.embedding_batch_size)
        )
        config.performance.embedding_queue_size = int(
            perf.get("embedding_queue_size", config.performance.embedding_queue_size)
        )
        config.performance.max_concurrent_jobs = int(
            perf.get("max_concurrent_jobs", config.performance.max_concurrent_jobs)
        )

    if startup := data.get("startup"):
        config.startup.check_on_startup = bool(
            startup.get("check_on_startup", config.startup.check_on_startup)
        )


def _apply_env(config: Config) -> None:
    if v := os.environ.get("DOC_SEARCH_DB"):
        config.storage.db_path = Path(v).expanduser()
    if v := os.environ.get("DOC_SEARCH_BACKEND"):
        config.storage.backend = v
    if v := os.environ.get("DOC_SEARCH_POSTGRES_URL"):
        config.storage.postgres_url = v
    if v := os.environ.get("DOC_SEARCH_EMBEDDER"):
        config.embeddings.backend = v
    if v := os.environ.get("DOC_SEARCH_OLLAMA_URL"):
        config.embeddings.ollama_url = v
    if v := os.environ.get("DOC_SEARCH_DEFAULT_MODE"):
        config.search.default_mode = v
    if v := os.environ.get("DOC_SEARCH_PORT"):
        config.server.port = int(v)
