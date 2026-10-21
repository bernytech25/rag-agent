"""Transactional session memory for SQLite locally and PostgreSQL deployments."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from uuid import uuid4
from contextlib import nullcontext
from filelock import FileLock
from sqlalchemy import (Column, Integer, MetaData, String, Table, Text, and_, create_engine,
                        delete, event, insert, select, update)
from sqlalchemy.exc import IntegrityError
from app.concurrency import KeyedLocks
from app.corpus import PROJECT_ROOT, CORPUS_ID

MAX_MESSAGES_PER_SESSION = 50


class ConversationConflict(RuntimeError):
    pass


class SQLMemory:
    def __init__(self, database_url=None, legacy_resolver=None):
        self.database_url = database_url or os.getenv("RAG_DATABASE_URL") or "sqlite:///" + (PROJECT_ROOT / "data" / "conversations.sqlite3").as_posix()
        self.legacy_resolver = legacy_resolver
        self._engine = None
        self._init_lock = threading.Lock()
        self._locks = KeyedLocks()
        self.metadata = MetaData()
        self.sessions = Table("rag_sessions", self.metadata,
                              Column("corpus_id", String(128), primary_key=True),
                              Column("namespace", String(512), primary_key=True),
                              Column("revision", Integer, nullable=False))
        self.messages = Table("rag_messages", self.metadata,
                              Column("id", Integer, primary_key=True, autoincrement=True),
                              Column("corpus_id", String(128), nullable=False, index=True),
                              Column("namespace", String(512), nullable=False, index=True),
                              Column("turn_id", String(64), nullable=False),
                              Column("role", String(32), nullable=False),
                              Column("content", Text, nullable=False),
                              Column("timestamp", String(64), nullable=False))

    def _db(self):
        with self._init_lock:
            if self._engine is None:
                url = self.database_url
                if url.startswith("postgresql://"):
                    url = url.replace("postgresql://", "postgresql+psycopg://", 1)
                if not url.startswith(("sqlite:///", "postgresql+psycopg://")):
                    raise ValueError("RAG_DATABASE_URL debe usar SQLite o PostgreSQL/psycopg.")
                if url.startswith("sqlite:///"):
                    path = url[len("sqlite:///"):]
                    if path != ":memory:":
                        Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)
                engine = create_engine(url, hide_parameters=True, pool_pre_ping=True,
                                       connect_args={"check_same_thread": False, "timeout": 30} if url.startswith("sqlite") else {})
                if engine.dialect.name == "sqlite":
                    @event.listens_for(engine, "connect")
                    def configure_sqlite(connection, record):
                        connection.execute("PRAGMA journal_mode=WAL")
                        connection.execute("PRAGMA busy_timeout=30000")
                if engine.dialect.name == "sqlite":
                    lock = FileLock(path + ".schema.lock", timeout=30) if path != ":memory:" else nullcontext()
                    with lock:
                        self.metadata.create_all(engine)
                else:
                    with engine.begin() as connection:
                        connection.exec_driver_sql("SELECT pg_advisory_xact_lock(731940201)")
                        self.metadata.create_all(connection)
                self._engine = engine
        return self._engine

    def _where(self, table, namespace, corpus_id):
        return and_(table.c.corpus_id == (corpus_id or CORPUS_ID), table.c.namespace == namespace)

    def turn_scope(self, namespace, corpus_id=None):
        return self._locks.hold((corpus_id or CORPUS_ID, namespace))

    def _ensure_session(self, namespace, corpus_id=None):
        db = self._db()
        where = self._where(self.sessions, namespace, corpus_id)
        with db.connect() as connection:
            if connection.execute(select(self.sessions.c.revision).where(where)).first() is not None:
                return
        imported = []
        if self.legacy_resolver:
            path = self.legacy_resolver(corpus_id)
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("El JSON heredado no contiene un mapa de sesiones.")
                imported = data.get(namespace)
                if imported is None and corpus_id and namespace.startswith(corpus_id + ":"):
                    # Older corpus-local JSONs used <user>:<session>, before the
                    # corpus prefix was introduced. Preserve that user's history.
                    imported = data.get(namespace[len(corpus_id) + 1:])
                imported = [] if imported is None else imported
                if not isinstance(imported, list) or any(not isinstance(m, dict) or not isinstance(m.get("role"), str) or not isinstance(m.get("content"), str) for m in imported):
                    raise ValueError("Historial heredado inválido; no se modificó el JSON original.")
        try:
            with db.begin() as connection:
                connection.execute(insert(self.sessions).values(corpus_id=corpus_id or CORPUS_ID, namespace=namespace, revision=0))
                for message in imported[-MAX_MESSAGES_PER_SESSION:]:
                    connection.execute(insert(self.messages).values(
                        corpus_id=corpus_id or CORPUS_ID, namespace=namespace, turn_id="legacy-" + uuid4().hex,
                        role=message["role"], content=message["content"],
                        timestamp=message.get("timestamp") or datetime.now(timezone.utc).isoformat()))
        except IntegrityError:
            # Another worker already imported this session. Never import it twice.
            with db.connect() as connection:
                if connection.execute(select(self.sessions.c.revision).where(where)).first() is None:
                    raise

    def get_revision(self, namespace, corpus_id=None):
        self._ensure_session(namespace, corpus_id)
        with self._db().connect() as connection:
            return connection.execute(select(self.sessions.c.revision).where(self._where(self.sessions, namespace, corpus_id))).scalar_one()

    def get_history_with_timestamps(self, namespace, corpus_id=None):
        self._ensure_session(namespace, corpus_id)
        with self._db().connect() as connection:
            rows = connection.execute(select(self.messages.c.role, self.messages.c.content, self.messages.c.timestamp)
                                      .where(self._where(self.messages, namespace, corpus_id)).order_by(self.messages.c.id)).mappings()
            return [dict(row) for row in rows]

    def get_history(self, namespace, corpus_id=None):
        return [{"role": m["role"], "content": m["content"]} for m in self.get_history_with_timestamps(namespace, corpus_id)]

    def _append(self, namespace, messages, corpus_id=None, expected_revision=None):
        self._ensure_session(namespace, corpus_id)
        turn_id = uuid4().hex
        with self._db().begin() as connection:
            where = self._where(self.sessions, namespace, corpus_id)
            if expected_revision is not None:
                where = and_(where, self.sessions.c.revision == expected_revision)
            changed = connection.execute(update(self.sessions).where(where).values(revision=self.sessions.c.revision + 1))
            if changed.rowcount != 1:
                raise ConversationConflict("La conversación cambió durante la respuesta; repetí la consulta.")
            for role, content in messages:
                connection.execute(insert(self.messages).values(
                    corpus_id=corpus_id or CORPUS_ID, namespace=namespace, turn_id=turn_id,
                    role=role, content=content, timestamp=datetime.now(timezone.utc).isoformat()))
            old = select(self.messages.c.id).where(self._where(self.messages, namespace, corpus_id)).order_by(self.messages.c.id.desc()).offset(MAX_MESSAGES_PER_SESSION)
            connection.execute(delete(self.messages).where(self.messages.c.id.in_(old)))

    def append_turn(self, namespace, question, answer, corpus_id=None, expected_revision=None):
        self._append(namespace, [("user", question), ("assistant", answer)], corpus_id, expected_revision)

    def add_message(self, namespace, role, content, corpus_id=None):
        self._append(namespace, [(role, content)], corpus_id)

    def clear(self, namespace, corpus_id=None):
        with self.turn_scope(namespace, corpus_id):
            self._ensure_session(namespace, corpus_id)
            with self._db().begin() as connection:
                connection.execute(update(self.sessions).where(self._where(self.sessions, namespace, corpus_id)).values(revision=self.sessions.c.revision + 1))
                connection.execute(delete(self.messages).where(self._where(self.messages, namespace, corpus_id)))
                # Keep the session marker: an old JSON must not resurrect cleared history.
