import json
from pathlib import Path
from datetime import datetime

MEMORY_FILE = Path(__file__).parent.parent / "data" / "memory.json"

class InSessionMemory:
    def __init__(self):
        self._sessions = {}

    def add_message(self, session_id, role, content):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})

    def get_history(self, session_id):
        return [{"role": m["role"], "content": m["content"]} for m in self._sessions.get(session_id, [])]

    def clear(self, session_id):
        self._sessions.pop(session_id, None)

class PersistentMemory:
    def _load(self):
        if not MEMORY_FILE.exists():
            return {}
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data):
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_message(self, session_id, role, content):
        data = self._load()
        if session_id not in data:
            data[session_id] = []
        data[session_id].append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})
        self._save(data)

    def get_history(self, session_id):
        data = self._load()
        return [{"role": m["role"], "content": m["content"]} for m in data.get(session_id, [])]

    def get_history_with_timestamps(self, session_id):
        return self._load().get(session_id, [])

    def clear(self, session_id):
        data = self._load()
        data.pop(session_id, None)
        self._save(data)

in_session_memory = InSessionMemory()
persistent_memory = PersistentMemory()
