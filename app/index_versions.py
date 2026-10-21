"""Immutable hybrid bundles with an atomic active-version pointer."""
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
import re
import sqlite3
import tempfile
from uuid import uuid4
import faiss
from filelock import FileLock
from app.corpus import get_corpus_profile
from app.hybrid import build_bm25_index
from app.ingestion import sha256


def corpus_write_lock(profile):
    return FileLock(str(profile.root / ".index-build.lock"), timeout=60)


def candidate_profile(profile, version):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", version):
        raise ValueError("Versión inválida.")
    root = (profile.root / "indexes" / version).resolve()
    if not root.is_relative_to(profile.root.resolve()):
        raise ValueError("Índice fuera del corpus.")
    return replace(profile, index_version=version, faiss_index_dir=root / "faiss_index", bm25_index_path=root / "bm25.sqlite3")


def write_candidate(profile, vectorstore, origin):
    version = "v-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid4().hex[:12]
    candidate = candidate_profile(profile, version)
    root = candidate.faiss_index_dir.parent
    root.mkdir(parents=True, exist_ok=False)
    documents = list(vectorstore.docstore._dict.values())
    if (vectorstore.index.ntotal != len(documents)
            or len(vectorstore.index_to_docstore_id) != len(documents)
            or any(key not in vectorstore.docstore._dict for key in vectorstore.index_to_docstore_id.values())):
        raise ValueError("El docstore y las referencias FAISS son inconsistentes.")
    vectorstore.save_local(str(candidate.faiss_index_dir))
    build_bm25_index(candidate.bm25_index_path, documents)
    metadata = {
        "version": version, "corpus_id": profile.corpus_id, "origin": origin,
        "created_at": datetime.now(timezone.utc).isoformat(), "chunks": len(documents),
        "dimension": vectorstore.index.d, "ingestion_signature": profile.ingestion_signature(),
        "embedding_model": profile.embedding_model,
        "ingestion_verified": origin != "legacy-adoption-unverified-ingestion",
        "files": {name: sha256(root / name) for name in ("faiss_index/index.faiss", "faiss_index/index.pkl", "bm25.sqlite3")}}
    (root / "bundle.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    validate_candidate(profile, version)
    return metadata


def validate_candidate(profile, version):
    candidate = candidate_profile(profile, version)
    root = candidate.faiss_index_dir.parent
    metadata = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if metadata["corpus_id"] != profile.corpus_id or metadata["version"] != version:
        raise ValueError("Corpus o versión del índice inconsistente.")
    if metadata["ingestion_signature"] != profile.ingestion_signature():
        raise ValueError("El perfil de ingesta cambió; reconstruir antes de activar.")
    for name in ("faiss_index/index.faiss", "faiss_index/index.pkl", "bm25.sqlite3"):
        if sha256(root / name) != metadata["files"][name]:
            raise ValueError(f"Archivo de índice alterado: {name}")
    index = faiss.read_index(str(candidate.faiss_index_dir / "index.faiss"))
    with sqlite3.connect(candidate.bm25_index_path) as connection:
        count = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
    if index.ntotal != metadata["chunks"] or count != index.ntotal or index.d != metadata["dimension"]:
        raise ValueError("FAISS y BM25 inconsistentes.")
    return metadata


def activate_index(corpus_id, version, evaluation_reference):
    profile = get_corpus_profile(corpus_id)
    if not evaluation_reference.strip():
        raise ValueError("Falta la referencia de evaluación o migración técnica.")
    with corpus_write_lock(profile):
        metadata = validate_candidate(profile, version)
        pointer = {"version": version, "evaluation_reference": evaluation_reference,
                   "activated_at": datetime.now(timezone.utc).isoformat()}
        fd, temporary = tempfile.mkstemp(prefix="active-", suffix=".json", dir=profile.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as target:
                json.dump(pointer, target)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, profile.root / "active-index.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return {**metadata, "active": True, "evaluation_reference": evaluation_reference}


def list_versions(corpus_id):
    profile = get_corpus_profile(corpus_id)
    return [{**json.loads(path.read_text(encoding="utf-8")), "active": path.parent.name == profile.index_version}
            for path in sorted((profile.root / "indexes").glob("*/bundle.json"))]
