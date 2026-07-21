"""
Capa de recuperación de documentos (Retrieval).
Maneja ingesta de múltiples PDFs, embeddings HuggingFace, FAISS persistente.
Desacoplado del LLM y de FastAPI.
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_INDEX_PATH = "faiss_index"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


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


def get_retriever(top_k: int = 3):
    if not index_exists():
        pdf_files = list(Path("data").glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                "No hay indice FAISS ni PDFs en data/. "
                "Copia al menos un PDF a la carpeta data/ y reinicia el servidor."
            )
        create_index([str(p) for p in pdf_files])

    vectorstore = load_index()
    return vectorstore.as_retriever(search_kwargs={"k": top_k})
