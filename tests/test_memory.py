from app import memory


def test_in_session_memory_keeps_a_bounded_history():
    store = memory.InSessionMemory()
    for index in range(memory.MAX_MESSAGES_PER_SESSION + 5):
        store.add_message("session", "user", str(index))

    history = store.get_history("session")

    assert len(history) == memory.MAX_MESSAGES_PER_SESSION
    assert history[0]["content"] == "5"


def test_persistent_memory_is_namespaced_by_session(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_FILE", tmp_path / "memory.json")
    store = memory.PersistentMemory()

    store.add_message("alice:session", "user", "private")

    assert store.get_history("alice:session")[0]["content"] == "private"
    assert store.get_history("bob:session") == []
