"""
Capa de recuperación con Reranking.
Pipeline: Pregunta -> FAISS (10 candidatos) -> Flashrank reordena -> top 3 -> LLM

Los embeddings se mantienen 100% locales (HuggingFace + FAISS en disco) a propósito:
no consumen tokens ni créditos de la nube. Solo la generación final (agent_rag.py)
llama a un modelo pago (Vertex AI Gemini).
"""

import os
from pathlib import Path
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

FAISS_INDEX_PATH = "faiss_index"
EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 800          # Punto medio entre 600 (muy chico) y 1000 (muy grande)
CHUNK_OVERLAP = 200       # Overlap un poco mayor para mantener coherencia
FAISS_CANDIDATES = 15     # Mantener: recupera 15, Flashrank elige top 4
TOP_K_FINAL = 5           # Aumentado a 5 para compensar chunks más chicos

_embeddings_cache = None


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


def _extract_tables_from_pdf(pdf_path, source_name):
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
                                    "type": "table"
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


def index_exists():
    return os.path.exists(FAISS_INDEX_PATH)


def load_index():
    return FAISS.load_local(FAISS_INDEX_PATH, _get_embeddings(), allow_dangerous_deserialization=True)


def create_index(pdf_paths: list):
    all_docs = []
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"No encontrado: {pdf_path}")
            continue
        
        source_name = Path(pdf_path).name
        print(f"Indexando: {source_name}")
        
        # 1. Extraer texto con PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source_file"] = source_name
        all_docs.extend(documents)
        
        # 2. Extraer tablas con pdfplumber (como chunks separados)
        table_docs = _extract_tables_from_pdf(pdf_path, source_name)
        all_docs.extend(table_docs)
        print(f"  → Texto: {len(documents)} páginas, Tablas: {len(table_docs)}")

    if not all_docs:
        raise ValueError("No se encontraron documentos.")

    # Chunking: solo aplica a texto, las tablas ya son chunks completos
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(all_docs)
    print(f"{len(chunks)} fragmentos totales (texto + tablas)")

    vectorstore = FAISS.from_documents(chunks, _get_embeddings())
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Indice guardado en: {FAISS_INDEX_PATH}")
    return vectorstore


def add_documents(pdf_paths: list):
    if not index_exists():
        return create_index(pdf_paths)

    existing = load_index()
    all_docs = []
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            continue
        
        source_name = Path(pdf_path).name
        print(f"Agregando: {source_name}")
        
        # 1. Extraer texto
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source_file"] = source_name
        all_docs.extend(documents)
        
        # 2. Extraer tablas
        table_docs = _extract_tables_from_pdf(pdf_path, source_name)
        all_docs.extend(table_docs)
        print(f"  → Texto: {len(documents)} páginas, Tablas: {len(table_docs)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(all_docs)
    existing.add_documents(chunks)
    existing.save_local(FAISS_INDEX_PATH)
    print(f"{len(chunks)} fragmentos agregados (texto + tablas)")
    return existing


def rerank_documents(query: str, docs: list, top_k: int = TOP_K_FINAL) -> list:
    """
    Reordena chunks con Flashrank (100% local, no consume tokens de LLM).
    FAISS usa similitud vectorial (rapido, aproximado).
    Flashrank usa relevancia semantica real (mas preciso).
    """
    try:
        from flashrank import Ranker, RerankRequest

        ranker = Ranker()
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
    1. FAISS recupera 10 candidatos
    2. Flashrank reordena por relevancia real
    3. Retorna los mejores top_k
    """
    if not index_exists():
        pdf_files = list(Path("data").glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError("No hay indice ni PDFs en data/.")
        create_index([str(p) for p in pdf_files])

    vectorstore = load_index()
    candidates = vectorstore.similarity_search(query, k=FAISS_CANDIDATES)
    return rerank_documents(query, candidates, top_k=top_k)


def get_retriever(top_k: int = TOP_K_FINAL):
    """Retriever sin reranking - mantenido para compatibilidad."""
    if not index_exists():
        pdf_files = list(Path("data").glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError("No hay indice ni PDFs en data/.")
        create_index([str(p) for p in pdf_files])

    vectorstore = load_index()
    return vectorstore.as_retriever(search_kwargs={"k": top_k})