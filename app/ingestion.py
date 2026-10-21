"""Registered ingestion strategies; profiles choose the parser and chunking."""
import hashlib
from datetime import datetime
from pathlib import Path
import pdfplumber
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validated_pdf_paths(names, profile):
    root = profile.documents_dir.resolve()
    paths = []
    for name in names:
        supplied = Path(name)
        path = (root / name).resolve()
        if supplied.is_absolute() or len(supplied.parts) != 1 or not path.is_relative_to(root) or path.suffix.lower() != ".pdf":
            raise ValueError("Enviar solo nombres de PDF ubicados en documents/ del corpus.")
        if not path.is_file():
            raise FileNotFoundError(f"No encontrado: {supplied.name}")
        paths.append(path)
    if not paths:
        raise ValueError("No se recibieron PDFs válidos.")
    return paths


def pdf_text(path):
    digest = sha256(path)
    reader = PdfReader(path)
    raw = {"producer": "PyPDF", "creator": "PyPDF", "creationdate": ""}
    raw.update(reader.metadata or {})
    raw.update(source=str(path), total_pages=len(reader.pages))
    metadata = {}
    for key, value in raw.items():
        key = key.lstrip("/").lower()
        value = value if type(value) in (str, int) else str(value)
        if key in {"creationdate", "moddate"}:
            try:
                value = datetime.strptime(value.replace("'", ""), "D:%Y%m%d%H%M%S%z").isoformat("T")
            except ValueError:
                pass
        elif isinstance(value, str):
            value = value.strip()
        metadata[key] = value
        if key in {"page_count", "file_path"}:
            metadata[{"page_count": "total_pages", "file_path": "source"}[key]] = value
    labels = reader.page_labels
    return [Document(page_content=(page.extract_text(extraction_mode="plain") or "").strip(),
                     metadata={**metadata, "page": number + 1, "page_label": labels[number],
                               "source_file": path.name, "source_sha256": digest, "type": "text"})
            for number, page in enumerate(reader.pages)]


def pdf_text_tables(path):
    documents = pdf_text(path)
    digest = sha256(path)
    # Abort a candidate on extraction failure instead of silently losing tables.
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for table_index, table in enumerate(page.extract_tables(), start=1):
                if not table:
                    continue
                rows = [[str(cell) if cell is not None else "" for cell in row] for row in table]
                lines = ["| " + " | ".join(row) + " |" for row in rows]
                lines.insert(1, "|" + "|".join(["---"] * len(rows[0])) + "|")
                documents.append(Document(
                    page_content=f"Tabla {table_index} (página {page_number}):\n\n" + "\n".join(lines),
                    metadata={"source_file": path.name, "source_sha256": digest, "page": page_number, "type": "table"}))
    return documents


PARSERS = {"pdf_text": pdf_text, "pdf_text_tables": pdf_text_tables}


def ingest(names, profile):
    documents = [doc for path in validated_pdf_paths(names, profile) for doc in PARSERS[profile.parser](path)]
    text = [doc for doc in documents if doc.metadata["type"] == "text"]
    tables = [doc for doc in documents if doc.metadata["type"] == "table"]
    splitter = RecursiveCharacterTextSplitter(chunk_size=profile.chunk_size, chunk_overlap=profile.chunk_overlap)
    chunks = splitter.split_documents(text) + tables
    if not chunks:
        raise ValueError("El parser no produjo fragmentos.")
    return chunks
