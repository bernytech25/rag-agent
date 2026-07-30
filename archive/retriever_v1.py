"""
Capa de recuperación de documentos (Retrieval) con Reranking.

Pipeline mejorado:
  Antes:  Pregunta → FAISS recupera 3 chunks → LLM
  Ahora:  Pregunta → FAISS recupera 10 chunks → Flashrank reordena → top 3 → LLM

El reranking mejora la precisión porque FAISS usa similitud vectorial (rápido pero
aproximado), mientras Flashrank usa un modelo de relevancia semántica más preciso
para ordenar los candidatos y seleccionar los mejores.
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

FAISS_INDEX_PATH = "faiss_index"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Reranking config
FAISS_CANDIDATES = 10  # FAISS recupera más candidatos
TOP_K_FINAL = 3        # Flashrank elige los mejores


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)


def index_exists() -> bool:
    return os.path.exists(FAISS_INDEX_PATH)


def load_index() -> FAISS:
    embeddings = _get_embeddings()
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def create_index(pdf_paths: list) -> FAISS:
    all_docs = []
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"Archivo no encontrado: {pdf_path}")
            continue
        print(f"Indexando: {Path(pdf_path).name}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source_file"] = Path(pdf_path).name
        all_docs.extend(documents)

    if not all_docs:
        raise ValueError("No se encontraron documentos para indexar.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)
    print(f"{len(chunks)} fragmentos generados de {len(pdf_paths)} documentos")

    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Indice guardado en: {FAISS_INDEX_PATH}")
    return vectorstore


def add_documents(pdf_paths: list) -> FAISS:
    if not index_exists():
        return create_index(pdf_paths)

    existing = load_index()
    all_docs = []
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            continue
        print(f"Agregando: {Path(pdf_path).name}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source_file"] = Path(pdf_path).name
        all_docs.extend(documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)
    existing.add_documents(chunks)
    existing.save_local(FAISS_INDEX_PATH)
    print(f"{len(chunks)} fragmentos agregados al indice")
    return existing


def rerank_documents(query: str, docs: list[Document], top_k: int = TOP_K_FINAL) -> list[Document]:
    """
    Reordena los documentos usando Flashrank.
    Recibe los candidatos de FAISS y retorna los top_k más relevantes.

    Args:
        query: Pregunta del usuario
        docs: Chunks candidatos de FAISS
        top_k: Cuántos chunks retornar después del reranking

    Returns:
        Lista de documentos reordenados por relevancia real
    """
    try:
        from flashrank import Ranker, RerankRequest

        ranker = Ranker()

        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(docs)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_request)

        reranked = []
        for result in results[:top_k]:
            original_doc = docs[result["id"]]
            original_doc.metadata["rerank_score"] = result.get("score", 0)
            reranked.append(original_doc)

        return reranked

    except Exception as e:
        print(f"Reranking falló, usando orden original de FAISS: {e}")
        return docs[:top_k]


def get_retriever_with_reranking(query: str, top_k: int = TOP_K_FINAL) -> list[Document]:
    """
    Recupera documentos relevantes con pipeline FAISS + Flashrank reranking.

    Pipeline:
    1. FAISS recupera FAISS_CANDIDATES (10) chunks por similitud vectorial
    2. Flashrank reordena por relevancia semántica real
    3. Retorna los top_k (3) mejores

    Args:
        query: Pregunta del usuario
        top_k: Cantidad de chunks finales a retornar

    Returns:
        Lista de documentos rerankeados
    """
    if not index_exists():
        pdf_files = list(Path("data").glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                "No hay indice FAISS ni PDFs en data/. "
                "Copia al menos un PDF a la carpeta data/ y reinicia el servidor."
            )
        create_index([str(p) for p in pdf_files])

    vectorstore = load_index()

    # Paso 1: FAISS recupera más candidatos de los necesarios
    candidates = vectorstore.similarity_search(query, k=FAISS_CANDIDATES)

    # Paso 2: Flashrank reordena y selecciona los mejores
    reranked = rerank_documents(query, candidates, top_k=top_k)

    return reranked


def get_retriever(top_k: int = TOP_K_FINAL):
    """Retriever legacy sin reranking — mantenido para compatibilidad."""
    if not index_exists():
        pdf_files = list(Path("data").glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError("No hay indice FAISS ni PDFs en data/.")
        create_index([str(p) for p in pdf_files])

    vectorstore = load_index()
    return vectorstore.as_retriever(search_kwargs={"k": top_k})