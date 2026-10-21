"""Configuration for a versioned document corpus used by the RAG service."""

import os
import re
import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
_CORPUS_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


def _corpora_root() -> Path:
    configured = os.getenv("RAG_CORPORA_ROOT")
    return Path(configured).resolve() if configured else (PROJECT_ROOT / "corpora").resolve()


def _corpus_id() -> str:
    corpus_id = os.getenv("RAG_CORPUS_ID", "rotulado-v1")
    if not _CORPUS_ID_PATTERN.fullmatch(corpus_id):
        raise RuntimeError("RAG_CORPUS_ID debe usar minúsculas, números y guiones.")
    return corpus_id


CORPORA_ROOT = _corpora_root()
CORPUS_ID = _corpus_id()
CORPUS_ROOT = (CORPORA_ROOT / CORPUS_ID).resolve()

try:
    CORPUS_ROOT.relative_to(CORPORA_ROOT)
except ValueError as exc:
    raise RuntimeError("El corpus activo debe estar dentro de RAG_CORPORA_ROOT.") from exc

DOCUMENTS_DIR = CORPUS_ROOT / "documents"
FAISS_INDEX_DIR = CORPUS_ROOT / "faiss_index"
MEMORY_FILE = CORPUS_ROOT / "memory.json"


@dataclass(frozen=True)
class GenerationProfile:
    prompt_path: Path
    prompt_version: int
    model: str
    temperature: float
    max_history: int
    max_output_tokens: int = 2048

    def read_prompt(self) -> str:
        text = self.prompt_path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Prompt vacío: {self.prompt_path.name}")
        return text


@dataclass(frozen=True)
class CorpusProfile:
    corpus_id: str
    description: str
    profile_version: int
    root: Path
    documents_dir: Path
    faiss_index_dir: Path
    bm25_index_path: Path
    memory_file: Path
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    faiss_candidates: int
    bm25_candidates: int
    rrf_k: int
    rerank_model: str
    top_k: int
    max_context_chars: int
    parser: str
    ingestion_version: int
    generation: GenerationProfile
    index_version: str = "legacy"
    neighbor_chunks: int = 0
    preserve_explicit_references: bool = False

    def ingestion_signature(self) -> str:
        settings = {name: getattr(self, name) for name in (
            "parser", "ingestion_version", "chunk_size", "chunk_overlap", "embedding_model",
        )}
        return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()

    def retrieval_signature(self) -> str:
        settings = {name: getattr(self, name) for name in (
            "profile_version", "faiss_candidates", "bm25_candidates", "rrf_k", "rerank_model", "top_k", "max_context_chars", "neighbor_chunks", "preserve_explicit_references",
        )}
        return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def get_corpus_profile(corpus_id: str | None = None) -> CorpusProfile:
    """Resolve a registered corpus; never accept an arbitrary filesystem path."""
    selected = CORPUS_ID if corpus_id is None else corpus_id
    if not _CORPUS_ID_PATTERN.fullmatch(selected):
        raise ValueError("Identificador de corpus inválido.")
    root = (CORPORA_ROOT / selected).resolve()
    if not root.is_relative_to(CORPORA_ROOT):
        raise ValueError("El corpus debe estar dentro de RAG_CORPORA_ROOT.")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Corpus no registrado: {selected}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("corpus_id") != selected:
        raise ValueError(f"Identificador inconsistente en el manifiesto: {selected}")
    retrieval = manifest.get("retrieval", {})
    if not isinstance(retrieval, dict):
        raise ValueError("La configuración de retrieval debe ser un objeto.")

    ingestion = manifest.get("ingestion")
    generation = manifest.get("generation")
    if not isinstance(ingestion, dict) or not isinstance(generation, dict):
        raise ValueError(f"El corpus {selected} requiere perfiles explícitos de ingestion y generation.")
    parser = ingestion.get("parser")
    if parser not in {"pdf_text", "pdf_text_tables"}:
        raise ValueError(f"Parser no registrado: {parser}")
    ingestion_version = ingestion.get("version")
    if type(ingestion_version) is not int or ingestion_version < 1:
        raise ValueError("La versión de ingesta debe ser un entero positivo.")
    prompt = generation.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"Falta el prompt propio de {selected}.")
    prompt_path = (root / prompt).resolve()
    if Path(prompt).is_absolute() or not prompt_path.is_relative_to(root) or not prompt_path.is_file():
        raise ValueError(f"Prompt inexistente o fuera del corpus: {selected}.")
    prompt_version = generation.get("prompt_version")
    max_history = generation.get("max_history", 10)
    if any(type(value) is not int or value < 1 for value in (prompt_version, max_history)):
        raise ValueError("prompt_version y max_history deben ser enteros positivos.")
    temperature = generation.get("temperature")
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature debe ser un número entre 0 y 2.")
    model = generation.get("model") or os.getenv(generation.get("model_env", ""), "")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"Falta el modelo de generación configurado para {selected}.")
    max_output_tokens = generation.get("max_output_tokens", 2048)
    if type(max_output_tokens) is not int or max_output_tokens < 256:
        raise ValueError("max_output_tokens debe ser un entero de al menos 256.")
    generation_profile = GenerationProfile(prompt_path, prompt_version, model.strip(), float(temperature), max_history, max_output_tokens)
    generation_profile.read_prompt()  # Fail explicitly; never inherit another corpus's prompt.

    index_version = "legacy"
    index_root = root
    pointer = root / "active-index.json"
    if pointer.is_file():
        index_version = json.loads(pointer.read_text(encoding="utf-8"))["version"]
        if not isinstance(index_version, str) or not _CORPUS_ID_PATTERN.fullmatch(index_version):
            raise ValueError("Versión de índice inválida.")
        index_root = (root / "indexes" / index_version).resolve()
        if not index_root.is_relative_to(root):
            raise ValueError("Índice fuera del corpus.")

    def positive(name: str, fallback: int) -> int:
        value = retrieval.get(name, fallback)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} debe ser un entero positivo.")
        return value

    chunk_size = manifest.get("chunk_size", 800)
    chunk_overlap = manifest.get("chunk_overlap", 200)
    if type(chunk_size) is not int or chunk_size <= 0 or type(chunk_overlap) is not int or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_size y chunk_overlap inválidos en el manifiesto.")
    embedding_model = manifest.get("embedding_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    rerank_model = retrieval.get("rerank_model", "ms-marco-TinyBERT-L-2-v2")
    if not isinstance(embedding_model, str) or not embedding_model or not isinstance(rerank_model, str) or not rerank_model:
        raise ValueError("Modelos de embedding o reranking inválidos.")
    neighbor_chunks = retrieval.get("neighbor_chunks", 0)
    if type(neighbor_chunks) is not int or not 0 <= neighbor_chunks <= 2:
        raise ValueError("neighbor_chunks debe ser un entero entre 0 y 2.")
    preserve_refs = retrieval.get("preserve_explicit_references", False)
    if type(preserve_refs) is not bool:
        raise ValueError("preserve_explicit_references debe ser booleano.")
    return CorpusProfile(
        corpus_id=selected,
        description=str(manifest.get("description", "")),
        profile_version=positive("profile_version", 1),
        root=root,
        documents_dir=root / "documents",
        faiss_index_dir=index_root / "faiss_index",
        bm25_index_path=index_root / "bm25.sqlite3",
        memory_file=root / "memory.json",
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        faiss_candidates=positive("faiss_candidates", 15),
        bm25_candidates=positive("bm25_candidates", 15),
        rrf_k=positive("rrf_k", 60),
        rerank_model=rerank_model,
        top_k=positive("top_k", 3),
        max_context_chars=positive("max_context_chars", 20000),
        parser=parser,
        ingestion_version=ingestion_version,
        generation=generation_profile,
        index_version=index_version,
        neighbor_chunks=neighbor_chunks,
        preserve_explicit_references=preserve_refs,
    )


def list_corpus_ids() -> list[str]:
    """List registered corpus identifiers, without exposing host paths."""
    if not CORPORA_ROOT.is_dir():
        return []
    return sorted(
        directory.name
        for directory in CORPORA_ROOT.iterdir()
        if directory.is_dir()
        and _CORPUS_ID_PATTERN.fullmatch(directory.name)
        and (directory / "manifest.json").is_file()
    )


def corpus_details() -> dict[str, str]:
    """Safe runtime metadata for health checks and logs."""
    return {"corpus_id": CORPUS_ID}
