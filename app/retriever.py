"""
Capa de recuperación con Reranking.
Pipeline: Pregunta -> FAISS (15 candidatos) -> Flashrank reordena -> top_k -> LLM

Los embeddings se mantienen 100% locales (HuggingFace + FAISS en disco) a propósito:
no consumen tokens ni créditos de la nube. Solo la generación final (agent_rag.py)
llama a un modelo pago (Vertex AI Gemini).

NOTA SOBRE RERANKING:
Flashrank usa por defecto ms-marco-TinyBERT-L-2-v2 (~4MB). Este modelo comprime
los scores cerca de 1.0, por lo que NO es confiable para filtrar por umbral absoluto.
Su valor real está en REORDENAR los candidatos de FAISS, no en descartar por score.
Se probó ms-marco-MiniLM-L-12-v2 (modelo más grande, ~23MB) pero empeoró los
resultados porque está entrenado en inglés (MS MARCO) y no discrimina bien en español.
"""

import os
import hashlib
import threading
from pathlib import Path
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FAISS_INDEX_PATH = str(PROJECT_ROOT / "faiss_index")
EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 800          # Punto medio entre 600 (muy chico) y 1000 (muy grande)
CHUNK_OVERLAP = 200       # Overlap un poco mayor para mantener coherencia
FAISS_CANDIDATES = 15     # Recupera 15 candidatos con FAISS para dar margen al reranker
TOP_K_FINAL = 5           # Máximo de chunks a devolver tras reranking

_embeddings_cache = None
_ranker_cache = None
_index_lock = threading.RLock()


def _table_to_markdown(table):
    """Convierte una tabla extraída por pdfplumber a formato markdown."""
    if not table:
        return None

    lines = []
    for i, row in enumerate(table):
        # Convertir None a string vacío
        row = [str(cell) if cell is not None else "" for cell in row]
        lines.append("| " + " | ".join(row) + " |")

        # Agregar separador después del encabezado (primera fila)
        if i == 0:
            lines.append("|" + "|".join(["---"] * len(row)) + "|")

    return "\n".join(lines)


def _extract_tables_from_pdf(pdf_path, source_name, source_sha256):
    """
    Extrae tablas de un PDF usando pdfplumber y devuelve chunks de tabla.
    Cada tabla se guarda como un chunk separado para no cortarla.
    """
    table_chunks = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables):
                        markdown_table = _table_to_markdown(table)
                        if markdown_table:
                            doc = Document(
                                page_content=f"Tabla {table_idx + 1} (página {page_num}):\n\n{markdown_table}",
                                metadata={
                                    "source_file": source_name,
                                    "page": page_num,
                                    "type": "table",
                                    "source_sha256": source_sha256,
                                }
                            )
                            table_chunks.append(doc)
    except Exception as e:
        print(f"Error extrayendo tablas de {pdf_path}: {e}")

    return table_chunks


def _get_embeddings():
    # Cacheado para no recargar el modelo local en cada llamada
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    return _embeddings_cache


def _get_ranker():
    """Cachea el ranker de Flashrank para no recargarlo en cada consulta."""
    global _ranker_cache
    if _ranker_cache is None:
        from flashrank import Ranker
        _ranker_cache = Ranker()  # Default: ms-marco-TinyBERT-L-2-v2
    return _ranker_cache


def index_exists():
    index_path = Path(FAISS_INDEX_PATH)
    return (index_path / "index.faiss").is_file() and (index_path / "index.pkl").is_file()


def load_index():
    # Index writes are serialized in-process; loading under the same lock avoids
    # reading one of FAISS's two persistence files while it is being replaced.
    with _index_lock:
        return FAISS.load_local(FAISS_INDEX_PATH, _get_embeddings(), allow_dangerous_deserialization=True)


def _validated_pdf_paths(pdf_paths: list) -> list[Path]:
    """Only ingest PDFs already placed under data/, never arbitrary filesystem paths."""
    data_root = DATA_DIR.resolve()
    validated = []
    for raw_path in pdf_paths:
        candidate = Path(raw_path).resolve()
        try:
            candidate.relative_to(data_root)
        except ValueError as exc:
            raise ValueError("Los documentos deben estar dentro de data/.") from exc
        if candidate.suffix.lower() != ".pdf":
            raise ValueError(f"Solo se permiten PDF: {candidate.name}")
        if not candidate.is_file():
            raise FileNotFoundError(f"No encontrado: {candidate.name}")
        validated.append(candidate)
    if not validated:
        raise ValueError("No se recibieron PDFs válidos.")
    return validated


def _sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pdf_documents(pdf_path: Path) -> tuple[list, list]:
    source_name = pdf_path.name
    source_sha256 = _sha256(pdf_path)
    print(f"Indexando: {source_name}")
    documents = PyPDFLoader(str(pdf_path)).load()
    for doc in documents:
        # PyPDFLoader stores its page index from zero; API citations use human page numbers.
        page = doc.metadata.get("page")
        if isinstance(page, int):
            doc.metadata["page"] = page + 1
        doc.metadata.update({"source_file": source_name, "source_sha256": source_sha256, "type": "text"})
    table_docs = _extract_tables_from_pdf(str(pdf_path), source_name, source_sha256)
    print(f"  → Texto: {len(documents)} páginas, Tablas: {len(table_docs)}")
    return documents, table_docs


def create_index(pdf_paths: list):
    pdf_paths = _validated_pdf_paths(pdf_paths)
    all_text_docs = []
    all_table_docs = []

    for pdf_path in pdf_paths:
        documents, table_docs = _load_pdf_documents(pdf_path)
        all_text_docs.extend(documents)
        all_table_docs.extend(table_docs)

    if not all_text_docs and not all_table_docs:
        raise ValueError("No se encontraron documentos.")

    # Chunking: solo aplica a texto; las tablas permanecen intactas como chunks únicos
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    text_chunks = splitter.split_documents(all_text_docs)

    # Unir chunks de texto + tablas (tablas sin modificar)
    chunks = text_chunks + all_table_docs
    print(f"{len(chunks)} fragmentos totales (texto: {len(text_chunks)}, tablas: {len(all_table_docs)})")

    vectorstore = FAISS.from_documents(chunks, _get_embeddings())
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Indice guardado en: {FAISS_INDEX_PATH}")
    return vectorstore


def _add_documents_unlocked(pdf_paths: list):
    pdf_paths = _validated_pdf_paths(pdf_paths)
    if not index_exists():
        return create_index([str(path) for path in pdf_paths])

    existing = load_index()
    existing_hashes = {
        document.metadata.get("source_sha256")
        for document in existing.docstore._dict.values()
        if document.metadata.get("source_sha256")
    }
    existing_names = {
        document.metadata.get("source_file")
        for document in existing.docstore._dict.values()
        if document.metadata.get("source_file")
    }
    all_text_docs = []
    all_table_docs = []

    for pdf_path in pdf_paths:
        source_hash = _sha256(pdf_path)
        if source_hash in existing_hashes or pdf_path.name in existing_names:
            print(f"Omitido (ya indexado): {pdf_path.name}")
            continue
        documents, table_docs = _load_pdf_documents(pdf_path)
        all_text_docs.extend(documents)
        all_table_docs.extend(table_docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    text_chunks = splitter.split_documents(all_text_docs)

    chunks = text_chunks + all_table_docs
    if not chunks:
        raise ValueError("Todos los documentos recibidos ya estaban indexados.")
    existing.add_documents(chunks)
    existing.save_local(FAISS_INDEX_PATH)
    print(f"{len(chunks)} fragmentos agregados (texto: {len(text_chunks)}, tablas: {len(all_table_docs)})")
    return existing


def add_documents(pdf_paths: list):
    """Serialize index mutations within this API process."""
    with _index_lock:
        return _add_documents_unlocked(pdf_paths)


def rerank_documents(query: str, docs: list, top_k: int = TOP_K_FINAL) -> list:
    """
    Reordena chunks con Flashrank (100% local, no consume tokens de LLM).

    FAISS usa similitud vectorial (rápido, aproximado).
    Flashrank usa relevancia semántica real (más preciso) para REORDENAR.

    NOTA: El modelo default (TinyBERT) comprime scores cerca de 1.0.
    Por eso NO se aplica umbral absoluto de score. El valor del reranking
    está en poner los chunks más relevantes PRIMERO, no en filtrar.
    """
    try:
        from flashrank import RerankRequest

        ranker = _get_ranker()
        passages = [{"id": i, "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(docs)]
        results = ranker.rerank(RerankRequest(query=query, passages=passages))

        reranked = []
        for result in results[:top_k]:
            original_doc = docs[result["id"]]
            original_doc.metadata["rerank_score"] = round(result.get("score", 0), 4)
            reranked.append(original_doc)

        return reranked

    except Exception as e:
        print(f"Reranking fallo, usando FAISS directo: {e}")
        return docs[:top_k]


def get_retriever_with_reranking(query: str, top_k: int = TOP_K_FINAL) -> list:
    """
    Pipeline completo con reranking:
    1. FAISS recupera FAISS_CANDIDATES (15) candidatos
    2. Flashrank reordena por relevancia real
    3. Retorna los mejores top_k chunks
    """
    if not index_exists():
        pdf_files = list(DATA_DIR.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError("No hay indice ni PDFs en data/.")
        with _index_lock:
            if not index_exists():
                create_index([str(p) for p in pdf_files])

    vectorstore = load_index()
    candidates = vectorstore.similarity_search(query, k=FAISS_CANDIDATES)
    return rerank_documents(query, candidates, top_k=top_k)


def get_retriever(top_k: int = TOP_K_FINAL):
    """Retriever sin reranking - mantenido para compatibilidad."""
    if not index_exists():
        pdf_files = list(DATA_DIR.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError("No hay indice ni PDFs en data/.")
        with _index_lock:
            if not index_exists():
                create_index([str(p) for p in pdf_files])

    vectorstore = load_index()
    return vectorstore.as_retriever(search_kwargs={"k": top_k})
