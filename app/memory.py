import json
import os
import threading
import tempfile
from pathlib import Path
from datetime import datetime

MEMORY_FILE = Path(__file__).parent.parent / "data" / "memory.json"
MAX_MESSAGES_PER_SESSION = 50
_memory_lock = threading.RLock()

class InSessionMemory:
    def __init__(self):
        self._sessions = {}

    def add_message(self, session_id, role, content):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})
        self._sessions[session_id] = self._sessions[session_id][-MAX_MESSAGES_PER_SESSION:]

    def get_history(self, session_id):
        return [{"role": m["role"], "content": m["content"]} for m in self._sessions.get(session_id, [])]

    def clear(self, session_id):
        self._sessions.pop(session_id, None)

class PersistentMemory:
    def _load(self):
        with _memory_lock:
            if not MEMORY_FILE.exists():
                return {}
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)

    def _save(self, data):
        with _memory_lock:
            MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_path = tempfile.mkstemp(dir=MEMORY_FILE.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(temporary_path, MEMORY_FILE)
            except Exception:
                if os.path.exists(temporary_path):
                    os.unlink(temporary_path)
                raise

    def add_message(self, session_id, role, content):
        with _memory_lock:
            data = self._load()
            if session_id not in data:
                data[session_id] = []
            data[session_id].append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})
            data[session_id] = data[session_id][-MAX_MESSAGES_PER_SESSION:]
            self._save(data)

    def get_history(self, session_id):
        data = self._load()
        return [{"role": m["role"], "content": m["content"]} for m in data.get(session_id, [])]

    def get_history_with_timestamps(self, session_id):
        return self._load().get(session_id, [])

    def clear(self, session_id):
        with _memory_lock:
            data = self._load()
            data.pop(session_id, None)
            self._save(data)

in_session_memory = InSessionMemory()
persistent_memory = PersistentMemory()
