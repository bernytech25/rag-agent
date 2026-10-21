"""Per-corpus SQLite FTS5/BM25 index and rank-based fusion."""

import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path

from langchain_core.documents import Document

from app.grounding import chunk_id


def build_bm25_index(path: Path, documents: list[Document]) -> None:
    """Atomically replace a generated sparse index from the indexed FAISS documents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="bm25-", suffix=".sqlite3", dir=path.parent)
    os.close(fd)
    try:
        connection = sqlite3.connect(temporary)
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE chunks USING fts5("
                "chunk_id UNINDEXED, content, metadata UNINDEXED, "
                "tokenize='unicode61 remove_diacritics 2')"
            )
            connection.executemany(
                "INSERT INTO chunks(chunk_id, content, metadata) VALUES (?, ?, ?)",
                (
                    (
                        chunk_id(document.metadata, document.page_content),
                        document.page_content,
                        json.dumps(document.metadata, ensure_ascii=False, default=str),
                    )
                    for document in documents
                ),
            )
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bm25_search(path: Path, query: str, limit: int) -> list[Document]:
    """Search literal Unicode terms; no user-controlled FTS query syntax."""
    terms = list(dict.fromkeys(re.findall(r"[^\W_]+", query, flags=re.UNICODE)))[:64]
    if not terms or not path.is_file():
        return []
    expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT content, metadata, bm25(chunks) FROM chunks "
            "WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?",
            (expression, limit),
        ).fetchall()
    finally:
        connection.close()
    return [
        Document(page_content=content, metadata={**json.loads(metadata), "bm25_score": score})
        for content, metadata, score in rows
    ]


def rrf_fuse(dense: list[Document], sparse: list[Document], rrf_k: int) -> list[Document]:
    """Fuse ranks, not incomparable FAISS and BM25 raw scores."""
    merged: dict[str, Document] = {}
    scores: dict[str, float] = {}
    for method, ranking in (("dense", dense), ("bm25", sparse)):
        for rank, document in enumerate(ranking, start=1):
            key = chunk_id(document.metadata, document.page_content)
            if key not in merged:
                merged[key] = Document(page_content=document.page_content, metadata=dict(document.metadata))
                scores[key] = 0.0
            scores[key] += 1.0 / (rrf_k + rank)
            merged[key].metadata[f"{method}_rank"] = rank
    for key, document in merged.items():
        document.metadata["rrf_score"] = scores[key]
    return sorted(merged.values(), key=lambda document: document.metadata["rrf_score"], reverse=True)


def retain_explicit_references(query, candidates, ranked, top_k):
    """Reserve candidates for explicitly requested section or article headings.

    Only structural references are supported here, not arbitrary numbers or
    inferred references. Candidates must contain the exact heading at line start.
    This does not create evidence or increase the configured result budget.
    """
    section_reference = re.compile(
        r"\b(?:apartado|secci[oó]n|section|clause)\s+(\d+(?:\.\d+)+)(?!\w|\.\d)",
        flags=re.IGNORECASE)
    article_reference = re.compile(
        r"\bart[ií]culo\s+(\d+)(?:\s+(bis|ter|quater))?(?!\w)",
        flags=re.IGNORECASE)
    previous, separator, follow_up = query.rpartition("\nSeguimiento:")
    # A new explicit reference supersedes references in the previous question.
    has_current_reference = section_reference.search(follow_up) or article_reference.search(follow_up)
    scope = follow_up if separator and has_current_reference else previous if separator else query
    refs = [("section", value) for value in section_reference.findall(scope)]
    refs.extend(("article", number, suffix.lower()) for number, suffix in article_reference.findall(scope))
    refs = list(dict.fromkeys(refs))
    chosen, seen = [], set()
    for ref in refs:
        if ref[0] == "section":
            heading = re.compile(
                r"^\s*(?:(?:apartado|secci[oó]n|section|clause)\s+)?"
                + re.escape(ref[1]) + r"(?!\w|\.\d)(?=[\s.:)\-–—]|$)",
                flags=re.MULTILINE | re.IGNORECASE)
        else:
            number, suffix = ref[1], ref[2]
            suffix_pattern = (r"\s+" + re.escape(suffix) + r"(?!\w)") if suffix else r"(?!\s+(?:bis|ter|quater)\b)"
            heading = re.compile(
                r"^\s*art[ií]culo\s+" + re.escape(number) + suffix_pattern
                + r"(?=[\s.(\-–—]|$)", flags=re.MULTILINE | re.IGNORECASE)
        match = next((doc for doc in candidates if heading.search(doc.page_content)), None)
        if match is not None:
            cid = chunk_id(match.metadata, match.page_content)
            if cid not in seen:
                # A lexical anchor does not inherit a cross-encoder score.
                copy = match.model_copy(deep=True)
                copy.metadata["retrieval_role"] = "explicit_reference"
                chosen.append(copy)
                seen.add(cid)
        if len(chosen) == top_k:
            return chosen
    for doc in ranked:
        cid = chunk_id(doc.metadata, doc.page_content)
        if cid not in seen:
            chosen.append(doc)
            seen.add(cid)
        if len(chosen) == top_k:
            break
    return chosen
