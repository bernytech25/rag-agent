"""Per-conversation coordination and transactionally persisted turns."""
from datetime import datetime, timezone
from app.corpus import MEMORY_FILE, get_corpus_profile
from app.concurrency import KeyedLocks
from app.sql_memory import SQLMemory, ConversationConflict, MAX_MESSAGES_PER_SESSION


class InSessionMemory:
    def __init__(self):
        self._sessions = {}
        self._revisions = {}
        self._locks = KeyedLocks()

    def turn_scope(self, session_id, corpus_id=None):
        return self._locks.hold(session_id)

    def get_revision(self, session_id, corpus_id=None):
        return self._revisions.get(session_id, 0)

    def add_message(self, session_id, role, content, corpus_id=None):
        with self.turn_scope(session_id):
            messages = self._sessions.setdefault(session_id, [])
            messages.append({"role": role, "content": content, "timestamp": datetime.now(timezone.utc).isoformat()})
            self._sessions[session_id] = messages[-MAX_MESSAGES_PER_SESSION:]
            self._revisions[session_id] = self.get_revision(session_id) + 1

    def append_turn(self, session_id, question, answer, corpus_id=None, expected_revision=None):
        with self.turn_scope(session_id):
            if expected_revision is not None and self.get_revision(session_id) != expected_revision:
                raise ConversationConflict("La conversación cambió; repetí la consulta.")
            self.add_message(session_id, "user", question)
            self.add_message(session_id, "assistant", answer)

    def get_history(self, session_id, corpus_id=None):
        with self.turn_scope(session_id):
            return [{"role": m["role"], "content": m["content"]} for m in self._sessions.get(session_id, [])]

    def get_history_with_timestamps(self, session_id, corpus_id=None):
        with self.turn_scope(session_id):
            return [dict(m) for m in self._sessions.get(session_id, [])]

    def clear(self, session_id, corpus_id=None):
        with self.turn_scope(session_id):
            self._sessions.pop(session_id, None)
            self._revisions[session_id] = self.get_revision(session_id) + 1


class PersistentMemory(SQLMemory):
    def __init__(self, database_url=None):
        super().__init__(database_url, lambda cid: get_corpus_profile(cid).memory_file if cid else MEMORY_FILE)


in_session_memory = InSessionMemory()
persistent_memory = PersistentMemory()
