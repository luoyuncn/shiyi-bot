"""Tests for session manager"""
import pytest
from core.session_manager import SessionManager
from pydantic import BaseModel


class MemoryConfig(BaseModel):
    """Memory config for testing"""
    sqlite_path: str = ":memory:"
    cache_size: int = 10
    auto_flush_interval: int = 60
    memory_root: str = "data/memory"


@pytest.mark.asyncio
async def test_create_and_get_session():
    """Test session creation and retrieval"""
    config = MemoryConfig()
    manager = SessionManager(config)
    await manager.initialize()

    # Create session
    context = await manager.create_session({"channel": "test"})
    assert context.session_id is not None

    # Get session from cache
    retrieved = await manager.get_session(context.session_id)
    assert retrieved is not None
    assert retrieved.session_id == context.session_id

    await manager.cleanup()


@pytest.mark.asyncio
async def test_save_and_load_messages():
    """Test message persistence"""
    config = MemoryConfig()
    manager = SessionManager(config)
    await manager.initialize()

    context = await manager.create_session({})
    session_id = context.session_id

    # Save messages
    await manager.save_message(session_id, "user", "你好")
    await manager.save_message(session_id, "assistant", "你好！")

    # Clear cache to force database load
    manager.cache.clear()

    # Load from database
    loaded = await manager.get_session(session_id)
    assert len(loaded.messages) == 2

    await manager.cleanup()


@pytest.mark.asyncio
async def test_complete_identity_onboarding_updates_state_and_documents(tmp_path):
    """First-time onboarding should persist state and markdown identity docs."""
    config = MemoryConfig(memory_root=str(tmp_path / "memory"))
    manager = SessionManager(config)
    await manager.initialize()

    state = await manager.get_global_user_state()
    assert state["identity_confirmed"] is False

    await manager.complete_identity_onboarding(
        shiyi_identity="你是十一，负责代码协作。",
        user_identity="我是腿哥，偏好 Python。",
        display_name="腿哥",
    )

    state = await manager.get_global_user_state()
    assert state["identity_confirmed"] is True
    assert state["display_name"] == "腿哥"
    assert manager.documents.shiyi_path.exists()
    assert manager.documents.user_path.exists()

    shiyi_text = manager.documents.shiyi_path.read_text(encoding="utf-8")
    user_text = manager.documents.user_path.read_text(encoding="utf-8")
    assert "你是十一，负责代码协作。" in shiyi_text
    assert "我是腿哥，偏好 Python。" in user_text

    await manager.cleanup()


@pytest.mark.asyncio
async def test_session_manager_memory_pending_proxy():
    """Session manager should expose pending-memory CRUD wrappers."""
    config = MemoryConfig()
    manager = SessionManager(config)
    await manager.initialize()

    pending_id = await manager.save_memory_pending(
        candidate_fact={"scope": "user", "fact_key": "language"},
        confidence=0.8,
        source_message_id="msg-2",
    )
    pending_items = await manager.list_memory_pending(status="pending")
    assert len(pending_items) == 1
    assert pending_items[0].id == pending_id

    await manager.update_memory_pending_status(pending_id, "confirmed")
    confirmed_items = await manager.list_memory_pending(status="confirmed")
    assert len(confirmed_items) == 1
    assert confirmed_items[0].id == pending_id

    await manager.cleanup()


@pytest.mark.asyncio
async def test_prepare_messages_for_agent_uses_onboarding_prompt_when_unconfirmed(tmp_path):
    """When identity is not confirmed, system should prepend onboarding guidance."""
    config = MemoryConfig(memory_root=str(tmp_path / "memory"))
    manager = SessionManager(config)
    await manager.initialize()

    prepared = await manager.prepare_messages_for_agent([{"role": "user", "content": "你好"}])

    assert prepared[0]["role"] == "system"
    assert "身份初始化" in prepared[0]["content"]
    assert prepared[1]["role"] == "user"

    await manager.cleanup()


@pytest.mark.asyncio
async def test_prepare_messages_for_agent_onboarding_prompt_only_once(tmp_path):
    """When unconfirmed, onboarding prompt should be injected only once globally."""
    config = MemoryConfig(memory_root=str(tmp_path / "memory"))
    manager = SessionManager(config)
    await manager.initialize()

    prepared_first = await manager.prepare_messages_for_agent([{"role": "user", "content": "你好"}])
    assert prepared_first[0]["role"] == "system"
    assert "身份初始化" in prepared_first[0]["content"]

    prepared_second = await manager.prepare_messages_for_agent([{"role": "user", "content": "继续聊"}])
    assert prepared_second[0]["role"] == "user"
    assert all("身份初始化" not in item["content"] for item in prepared_second if item["role"] == "system")

    state = await manager.get_global_user_state()
    assert state["identity_confirmed"] is False
    assert state["onboarding_prompted"] is True

    await manager.cleanup()


@pytest.mark.asyncio
async def test_onboarding_prompt_once_persists_across_restart(tmp_path):
    """Onboarding first-prompt flag should persist in DB across restarts."""
    sqlite_path = str(tmp_path / "sessions.db")
    memory_root = str(tmp_path / "memory")
    config = MemoryConfig(memory_root=memory_root, sqlite_path=sqlite_path)

    manager = SessionManager(config)
    await manager.initialize()
    first = await manager.prepare_messages_for_agent([{"role": "user", "content": "你好"}])
    assert first[0]["role"] == "system"
    assert "身份初始化" in first[0]["content"]
    await manager.cleanup()

    manager = SessionManager(config)
    await manager.initialize()
    second = await manager.prepare_messages_for_agent([{"role": "user", "content": "再来一次"}])
    assert second[0]["role"] == "user"
    await manager.cleanup()


@pytest.mark.asyncio
async def test_user_inline_onboarding_confirmation_writes_db_and_md(tmp_path):
    """Inline onboarding confirmation message should persist DB state and identity markdown."""
    config = MemoryConfig(memory_root=str(tmp_path / "memory"))
    manager = SessionManager(config)
    await manager.initialize()
    session = await manager.create_session({"channel": "test"})

    await manager.save_message(
        session.session_id,
        "user",
        "十一人设：你是我的长期工程伙伴。 用户身份：我是腿哥，专注 Python 后端。 称呼：腿哥。 确认：是",
    )

    state = await manager.get_global_user_state()
    assert state["identity_confirmed"] is True
    assert state["display_name"] == "腿哥"

    shiyi_text = manager.documents.shiyi_path.read_text(encoding="utf-8")
    user_text = manager.documents.user_path.read_text(encoding="utf-8")
    assert "长期工程伙伴" in shiyi_text
    assert "我是腿哥，专注 Python 后端" in user_text

    await manager.cleanup()


@pytest.mark.asyncio
async def test_preseeded_shiyi_md_should_not_auto_confirm_identity(tmp_path):
    """Pre-existing ShiYi.md content must not auto-flip identity_confirmed."""
    memory_root = tmp_path / "memory"
    system_dir = memory_root / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "ShiYi.md").write_text(
        "# ShiYi\n\n## 核心身份\n\n你是一个初始化过的人设模板。\n",
        encoding="utf-8",
    )

    config = MemoryConfig(memory_root=str(memory_root))
    manager = SessionManager(config)
    await manager.initialize()

    state = await manager.get_global_user_state()
    assert state["identity_confirmed"] is False

    prepared = await manager.prepare_messages_for_agent([{"role": "user", "content": "你好"}])
    assert "身份初始化" in prepared[0]["content"]

    await manager.cleanup()


@pytest.mark.asyncio
async def test_identity_state_file_true_should_not_auto_confirm(tmp_path):
    """IdentityState.md must not auto-confirm DB state before onboarding."""
    memory_root = tmp_path / "memory"
    system_dir = memory_root / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "IdentityState.md").write_text(
        "# IdentityState\n\nidentity_confirmed: true\n",
        encoding="utf-8",
    )

    config = MemoryConfig(memory_root=str(memory_root))
    manager = SessionManager(config)
    await manager.initialize()

    state = await manager.get_global_user_state()
    assert state["identity_confirmed"] is False

    await manager.cleanup()


@pytest.mark.asyncio
async def test_identity_state_file_false_should_not_force_reonboarding(tmp_path):
    """IdentityState.md must not reset a confirmed DB state."""
    memory_root = tmp_path / "memory"
    sqlite_path = str(tmp_path / "sessions.db")
    config = MemoryConfig(memory_root=str(memory_root), sqlite_path=sqlite_path)

    manager = SessionManager(config)
    await manager.initialize()
    await manager.complete_identity_onboarding(
        shiyi_identity="你是十一。",
        user_identity="我是腿哥。",
        display_name="腿哥",
    )
    await manager.cleanup()

    store = manager.documents
    store.write_identity_state(False, None)

    manager = SessionManager(config)
    await manager.initialize()
    state = await manager.get_global_user_state()
    assert state["identity_confirmed"] is True
    await manager.cleanup()


@pytest.mark.asyncio
async def test_prepare_messages_for_agent_injects_memory_card_after_onboarding(tmp_path):
    """After onboarding completion, system should inject condensed memory card."""
    config = MemoryConfig(memory_root=str(tmp_path / "memory"))
    manager = SessionManager(config)
    await manager.initialize()

    await manager.complete_identity_onboarding(
        shiyi_identity="你是十一，编程协作助手。",
        user_identity="我是腿哥，偏好 Python。",
        display_name="腿哥",
    )
    prepared = await manager.prepare_messages_for_agent([{"role": "user", "content": "继续"}])

    assert prepared[0]["role"] == "system"
    assert "记忆卡片" in prepared[0]["content"]
    assert "编程协作助手" in prepared[0]["content"]
    assert "偏好 Python" in prepared[0]["content"]

    await manager.cleanup()


@pytest.mark.asyncio
async def test_confirmed_pending_fact_is_applied_to_user_memory(tmp_path):
    """Confirmed pending fact should persist into facts storage and User.md."""
    config = MemoryConfig(memory_root=str(tmp_path / "memory"))
    manager = SessionManager(config)
    await manager.initialize()
    await manager.complete_identity_onboarding(
        shiyi_identity="你是十一。",
        user_identity="我是腿哥。",
        display_name="腿哥",
    )

    pending_id = await manager.save_memory_pending(
        candidate_fact={
            "scope": "user",
            "fact_type": "preference",
            "fact_key": "preferred_tech",
            "fact_value": "Python",
        },
        confidence=0.75,
        source_message_id="msg-pending",
    )
    await manager.update_memory_pending_status(pending_id, "confirmed")

    facts = await manager.list_memory_facts(scope="user")
    assert any(f.fact_key == "preferred_tech" and f.fact_value == "Python" for f in facts)
    user_text = manager.documents.user_path.read_text(encoding="utf-8")
    assert "- preferred_tech: Python" in user_text
    events = await manager.list_memory_events()
    assert any(e.event_type == "memory_fact_applied" for e in events)
    assert any(e.event_type == "memory_pending_status_updated" for e in events)

    await manager.cleanup()


@pytest.mark.asyncio
async def test_user_message_auto_extracts_high_confidence_preference(tmp_path):
    """Explicit preference statement should auto-write high-confidence memory fact."""
    config = MemoryConfig(memory_root=str(tmp_path / "memory"))
    manager = SessionManager(config)
    await manager.initialize()
    await manager.complete_identity_onboarding(
        shiyi_identity="你是十一。",
        user_identity="我是腿哥。",
        display_name="腿哥",
    )
    session = await manager.create_session({"channel": "test"})

    await manager.save_message(session.session_id, "user", "我更喜欢Python")

    facts = await manager.list_memory_facts(scope="user")
    assert any(f.fact_key == "preferred_tech" and f.fact_value == "Python" for f in facts)
    user_text = manager.documents.user_path.read_text(encoding="utf-8")
    assert "- preferred_tech: Python" in user_text

    await manager.cleanup()


@pytest.mark.asyncio
async def test_user_message_extracts_medium_confidence_preference_to_pending(tmp_path):
    """Medium-confidence preference signal should go to pending queue."""
    config = MemoryConfig(memory_root=str(tmp_path / "memory"))
    manager = SessionManager(config)
    await manager.initialize()
    await manager.complete_identity_onboarding(
        shiyi_identity="你是十一。",
        user_identity="我是腿哥。",
        display_name="腿哥",
    )
    session = await manager.create_session({"channel": "test"})

    await manager.save_message(session.session_id, "user", "我最近在用Rust")

    pending_items = await manager.list_memory_pending(status="pending")
    assert any(
        p.candidate_fact.get("fact_key") == "preferred_tech"
        and p.candidate_fact.get("fact_value") == "Rust"
        for p in pending_items
    )

    facts = await manager.list_memory_facts(scope="user")
    assert not any(f.fact_key == "preferred_tech" and f.fact_value == "Rust" for f in facts)

    await manager.cleanup()


@pytest.mark.asyncio
async def test_prepare_messages_for_agent_injects_keyword_recall(tmp_path):
    """When user asks historical question, system should inject keyword recall snippets."""
    config = MemoryConfig(memory_root=str(tmp_path / "memory"))
    manager = SessionManager(config)
    await manager.initialize()
    await manager.complete_identity_onboarding(
        shiyi_identity="你是十一。",
        user_identity="我是腿哥。",
        display_name="腿哥",
    )
    session = await manager.create_session({"channel": "test"})
    await manager.save_message(session.session_id, "user", "上次 Redis timeout 的问题是连接池设置不当")

    prepared = await manager.prepare_messages_for_agent(
        [{"role": "user", "content": "你还记得之前 Redis timeout 的问题吗"}]
    )

    assert any(
        item["role"] == "system" and "历史记忆检索" in item["content"]
        for item in prepared
    )
    assert any("Redis timeout" in item["content"] for item in prepared if item["role"] == "system")

    await manager.cleanup()
