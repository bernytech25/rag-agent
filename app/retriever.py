"""Per-corpus hybrid retrieval; versioned candidates and bounded index cache."""
from collections import OrderedDict
from dataclasses import replace
import json
import logging
import os
import threading
from app import runtime_config  # Also covers CLI imports bypassing the API.
import huggingface_hub
from app.vectorstore import FAISS
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from app.concurrency import KeyedLocks
from app.corpus import get_corpus_profile
from app.hybrid import bm25_search, rrf_fuse, retain_explicit_references
from app.index_versions import corpus_write_lock, write_candidate
from app.ingestion import ingest, validated_pdf_paths, sha256
from app.grounding import chunk_id

logger = logging.getLogger(__name__)
TOP_K_FINAL = 5
_embeddings_cache = {}
_ranker_cache = {}
_index_cache = OrderedDict()
_cache_guard = threading.Lock()
_resource_locks = KeyedLocks()


def embeddings_offline_enabled():
    """Whether embedding model resolution must be limited to the local cache."""
    return os.getenv("EMBEDDINGS_OFFLINE", "1") == "1"


def verify_embeddings_available(model_name):
    """Fail early when an offline embedding model is absent from the local cache."""
    if not embeddings_offline_enabled():
        return
    try:
        huggingface_hub.snapshot_download(repo_id=model_name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"El modelo de embeddings '{model_name}' no está disponible en la caché local. "
            "EMBEDDINGS_OFFLINE=1 no intenta descargar modelos. "
            "Descargá y validá el modelo durante el aprovisionamiento, o configurá "
            "EMBEDDINGS_OFFLINE=0 temporalmente en un entorno autorizado con red."
        ) from exc


def _get_embeddings(model_name):
    with _resource_locks.hold(("embeddings", model_name)):
        if model_name not in _embeddings_cache:
            kwargs = {"model_kwargs": {"local_files_only": True}} if embeddings_offline_enabled() else {}
            _embeddings_cache[model_name] = HuggingFaceEmbeddings(model_name=model_name, **kwargs)
        return _embeddings_cache[model_name]


class _LazyEmbeddings(Embeddings):
    """Opening an index does not download its model until an embedding is needed."""
    def __init__(self, model_name):
        self.model_name = model_name

    def embed_documents(self, texts):
        return _get_embeddings(self.model_name).embed_documents(texts)

    def embed_query(self, text):
        return _get_embeddings(self.model_name).embed_query(text)


def _get_ranker(model_name="ms-marco-TinyBERT-L-2-v2"):
    with _resource_locks.hold(("ranker-init", model_name)):
        if model_name not in _ranker_cache:
            from flashrank import Ranker
            _ranker_cache[model_name] = Ranker(model_name=model_name)
        return _ranker_cache[model_name]


def index_exists(corpus_id=None):
    profile = get_corpus_profile(corpus_id)
    return all((profile.faiss_index_dir / name).is_file() for name in ("index.faiss", "index.pkl"))


def _load_raw(profile):
    return FAISS.load_local(str(profile.faiss_index_dir), _LazyEmbeddings(profile.embedding_model), allow_dangerous_deserialization=True)


def _load_profile_index(profile):
    paths = [profile.faiss_index_dir / name for name in ("index.faiss", "index.pkl")]
    fingerprint = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
    key = (profile.corpus_id, profile.index_version, profile.embedding_model, profile.ingestion_signature(), fingerprint)
    with _cache_guard:
        cached = _index_cache.get(key)
        if cached is not None:
            _index_cache.move_to_end(key)
            return cached
    with _resource_locks.hold(("index-load", key)):
        with _cache_guard:
            if key in _index_cache:
                return _index_cache[key]
        metadata_path = profile.faiss_index_dir.parent / "bundle.json"
        if profile.index_version != "legacy" and not metadata_path.is_file():
            raise ValueError("Faltan los metadatos del índice activo; no se puede verificar su configuración.")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["ingestion_signature"] != profile.ingestion_signature():
                raise ValueError("El índice activo no corresponde al perfil de ingesta; reconstruir y activar otra versión.")
        vectorstore = _load_raw(profile)
        limit = max(1, int(os.getenv("RAG_INDEX_CACHE_SIZE", "4")))
        with _cache_guard:
            _index_cache[key] = vectorstore
            while len(_index_cache) > limit:
                _index_cache.popitem(last=False)
        return vectorstore


def load_index(corpus_id=None):
    return _load_profile_index(get_corpus_profile(corpus_id))


def clear_index_cache():
    with _cache_guard:
        _index_cache.clear()


def create_index(pdf_paths, corpus_id=None):
    profile = get_corpus_profile(corpus_id)
    with corpus_write_lock(profile):
        chunks = ingest(pdf_paths, profile)
        vectorstore = FAISS.from_documents(chunks, _get_embeddings(profile.embedding_model))
        return write_candidate(profile, vectorstore, "rebuild")


def add_documents(pdf_paths, corpus_id=None):
    """Create a candidate without modifying the active or cached index."""
    profile = get_corpus_profile(corpus_id)
    with corpus_write_lock(profile):
        paths = validated_pdf_paths(pdf_paths, profile)
        if not all((profile.faiss_index_dir / name).is_file() for name in ("index.faiss", "index.pkl")):
            vectorstore = FAISS.from_documents(ingest(pdf_paths, profile), _get_embeddings(profile.embedding_model))
        else:
            metadata_path = profile.faiss_index_dir.parent / "bundle.json"
            if not metadata_path.is_file():
                raise ValueError("Preparar el índice heredado antes de agregar documentos, o usar rebuild.")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not metadata.get("ingestion_verified"):
                raise ValueError("La configuración de ingesta heredada no está comprobada: usar rebuild antes de agregar fragmentos.")
            if metadata["ingestion_signature"] != profile.ingestion_signature():
                raise ValueError("Cambió la ingesta: usar rebuild para no mezclar embeddings o fragmentaciones.")
            vectorstore = _load_raw(profile)
            existing = {doc.metadata.get("source_file") for doc in vectorstore.docstore._dict.values()}
            if any(path.name in existing for path in paths):
                raise ValueError("Documento ya indexado: usar rebuild para reemplazar su contenido.")
            vectorstore.add_documents(ingest(pdf_paths, profile))
        return write_candidate(profile, vectorstore, "append")


def prepare_legacy_index(corpus_id):
    """Preserve original FAISS, copy its chunks into a hybrid versioned candidate."""
    profile = get_corpus_profile(corpus_id)
    legacy = replace(profile, faiss_index_dir=profile.root / "faiss_index",
                     bm25_index_path=profile.root / "bm25.sqlite3", index_version="legacy")
    with corpus_write_lock(profile):
        vectorstore = _load_raw(legacy)
        verified = set()
        for doc in vectorstore.docstore._dict.values():
            name, digest = doc.metadata.get("source_file"), doc.metadata.get("source_sha256")
            if not name or not digest:
                raise ValueError("Índice heredado sin hashes de fuente; requiere rebuild.")
            if (name, digest) not in verified:
                path = validated_pdf_paths([name], profile)[0]
                if sha256(path) != digest:
                    raise ValueError(f"El PDF {name} cambió respecto del índice heredado.")
                verified.add((name, digest))
        return write_candidate(profile, vectorstore, "legacy-adoption-unverified-ingestion")


def rerank_documents(query, docs, top_k=TOP_K_FINAL, model_name="ms-marco-TinyBERT-L-2-v2"):
    if not docs:
        return []
    try:
        from flashrank import RerankRequest
        ranker = _get_ranker(model_name)
        passages = [{"id": i, "text": doc.page_content, "meta": dict(doc.metadata)} for i, doc in enumerate(docs)]
        with _resource_locks.hold(("ranker-use", model_name)):
            results = ranker.rerank(RerankRequest(query=query, passages=passages))
        output = []
        for result in results[:top_k]:
            doc = docs[result["id"]].model_copy(deep=True)
            doc.metadata["rerank_score"] = float(result.get("score", 0))
            output.append(doc)
        return output
    except Exception:
        if os.getenv("RAG_STRICT", "0") == "1":
            raise
        logger.exception("Reranking falló; se conserva el orden RRF", extra={"model": model_name})
        return docs[:top_k]


def expand_neighbors(seeds, vectorstore, radius):
    """Add nearby indexed text chunks, bounded by source/hash and adjacent pages.

    Index order preserves ingestion order. Tables are never treated as text
    continuations. Added chunks do not inherit the seed's relevance score.
    """
    if not radius:
        return seeds
    ordered = [vectorstore.docstore.search(cid) for _, cid in sorted(vectorstore.index_to_docstore_id.items())]
    positions = {chunk_id(doc.metadata, doc.page_content): i for i, doc in enumerate(ordered)}
    output, seen = [], set()
    def add(doc, role):
        cid = chunk_id(doc.metadata, doc.page_content)
        if cid not in seen:
            seen.add(cid)
            copy = doc.model_copy(deep=True)
            copy.metadata["retrieval_role"] = role
            output.append(copy)
    # Preserve all ranked seeds first, including their own scores.
    for seed in seeds:
        add(seed, seed.metadata.get("retrieval_role", "ranked"))
    for seed in seeds:
        meta = seed.metadata
        pos = positions.get(chunk_id(meta, seed.page_content))
        if pos is None or meta.get("type", "text") != "text" or not meta.get("source_sha256") or type(meta.get("page")) is not int:
            continue
        # An explicitly requested article often starts near the end of a chunk
        # and continues across one page boundary. Preserve one additional local
        # continuation only for that structural anchor, not for every result.
        seed_radius = radius + 1 if meta.get("retrieval_role") == "explicit_reference" else radius
        for doc in ordered[max(0, pos - seed_radius):pos + seed_radius + 1]:
            other = doc.metadata
            if (other.get("source_file") == meta.get("source_file")
                    and other.get("source_sha256") == meta["source_sha256"]
                    and other.get("type", "text") == "text"
                    and type(other.get("page")) is int
                    and abs(other["page"] - meta["page"]) <= 1):
                add(doc, "neighbor")
    return output


def retrieve_profile(query, profile, top_k=None):
    if not profile.bm25_index_path.is_file():
        raise FileNotFoundError("Preparar y activar un índice híbrido desde /index/prepare o construirlo desde /index.")
    vectorstore = _load_profile_index(profile)
    dense = vectorstore.similarity_search(query, k=profile.faiss_candidates)
    sparse = bm25_search(profile.bm25_index_path, query, profile.bm25_candidates)
    candidates = rrf_fuse(dense, sparse, profile.rrf_k)
    k = profile.top_k if top_k is None else top_k
    seeds = rerank_documents(query, candidates, top_k=k, model_name=profile.rerank_model)
    if profile.preserve_explicit_references:
        seeds = retain_explicit_references(query, candidates, seeds, k)
    return expand_neighbors(seeds, vectorstore, profile.neighbor_chunks)


def get_retriever_with_reranking(query, top_k=None, corpus_id=None):
    return retrieve_profile(query, get_corpus_profile(corpus_id), top_k)


def get_retriever(top_k=TOP_K_FINAL, corpus_id=None):
    return load_index(corpus_id).as_retriever(search_kwargs={"k": top_k})
