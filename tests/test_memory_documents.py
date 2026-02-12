"""Tests for markdown memory document storage."""

from pathlib import Path

from memory.documents import MemoryDocumentStore


def test_ensure_initialized_creates_default_documents(tmp_path: Path):
    """Document store should create the expected markdown files."""
    store = MemoryDocumentStore(str(tmp_path / "memory"))
    store.ensure_initialized()

    assert store.shiyi_path.exists()
    assert store.identity_state_path.exists()
    assert store.user_path.exists()
    assert store.project_path.exists()
    assert store.insights_path.exists()


def test_write_initial_identity_overwrites_shiyi_and_user(tmp_path: Path):
    """Onboarding write should persist custom identity texts."""
    store = MemoryDocumentStore(str(tmp_path / "memory"))
    store.ensure_initialized()

    shiyi_identity = "你是十一，负责高效协作。"
    user_identity = "用户是腿哥，偏好 Python。"
    store.write_initial_identity(shiyi_identity, user_identity)

    assert shiyi_identity in store.shiyi_path.read_text(encoding="utf-8")
    assert user_identity in store.user_path.read_text(encoding="utf-8")


def test_upsert_user_fact_adds_and_updates_key(tmp_path: Path):
    """User fact upsert should update existing key without duplication."""
    store = MemoryDocumentStore(str(tmp_path / "memory"))
    store.ensure_initialized()

    store.upsert_user_fact("preferred_tech", "Python")
    store.upsert_user_fact("preferred_tech", "Go")

    text = store.user_path.read_text(encoding="utf-8")
    assert "- preferred_tech: Go" in text
    assert text.count("preferred_tech") == 1


def test_identity_state_read_and_write(tmp_path: Path):
    """Identity state should use explicit confirmation marker file."""
    store = MemoryDocumentStore(str(tmp_path / "memory"))
    store.ensure_initialized()

    initial_state = store.read_identity_state()
    assert initial_state["identity_confirmed"] is None

    store.write_identity_state(True, "腿哥")
    state = store.read_identity_state()
    assert state["identity_confirmed"] is True
    assert state["display_name"] == "腿哥"
