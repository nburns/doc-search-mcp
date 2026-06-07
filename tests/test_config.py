import os
from pathlib import Path

import pytest

from doc_search_mcp.config import Config, _apply_env, _apply_toml, load_config


def test_defaults():
    config = Config()
    assert config.server.port == 8080
    assert config.server.transport == "sse"
    assert config.storage.backend == "sqlite"
    assert config.embeddings.model == "nomic-embed-text"
    assert config.chunking.target_tokens == 400
    assert config.chunking.overlap_tokens == 50


def test_load_config_no_file(tmp_path):
    # Points at a nonexistent file - should return defaults
    config = load_config(tmp_path / "does_not_exist.toml")
    assert config.server.port == 8080


def test_apply_toml_partial(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text('[server]\nport = 9090\n')
    config = load_config(toml)
    assert config.server.port == 9090
    assert config.server.transport == "sse"  # default preserved


def test_apply_toml_storage(tmp_path):
    db = tmp_path / "mydb.db"
    toml = tmp_path / "config.toml"
    toml.write_text(f'[storage]\ndb_path = "{db}"\nbackend = "sqlite"\n')
    config = load_config(toml)
    assert config.storage.db_path == db


def test_apply_toml_chunking(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text('[chunking]\ntarget_tokens = 200\noverlap_tokens = 25\n')
    config = load_config(toml)
    assert config.chunking.target_tokens == 200
    assert config.chunking.overlap_tokens == 25


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("DOC_SEARCH_PORT", "7777")
    monkeypatch.setenv("DOC_SEARCH_BACKEND", "postgres")
    monkeypatch.setenv("DOC_SEARCH_DEFAULT_MODE", "keyword")
    config = load_config(tmp_path / "nofile.toml")
    assert config.server.port == 7777
    assert config.storage.backend == "postgres"
    assert config.search.default_mode == "keyword"


def test_env_db_path(monkeypatch, tmp_path):
    db = tmp_path / "custom.db"
    monkeypatch.setenv("DOC_SEARCH_DB", str(db))
    config = load_config(tmp_path / "nofile.toml")
    assert config.storage.db_path == db


def test_env_takes_precedence_over_toml(monkeypatch, tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text("[server]\nport = 9000\n")
    monkeypatch.setenv("DOC_SEARCH_PORT", "1234")
    config = load_config(toml)
    assert config.server.port == 1234
