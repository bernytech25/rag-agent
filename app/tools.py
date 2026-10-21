"""Specialized retrieval tools sharing a traceable evidence contract."""
from dataclasses import dataclass
from app.corpus import get_corpus_profile, list_corpus_ids
from app.grounding import chunk_id
from app.retriever import retrieve_profile


@dataclass(frozen=True)
class CorpusRetrievalTool:
    profile: object

    def retrieve(self, query):
        profile = self.profile
        documents = retrieve_profile(query, profile)
        return [{
            "chunk_id": f"{profile.corpus_id}:{profile.index_version}:{chunk_id(doc.metadata, doc.page_content)}",
            "corpus_id": profile.corpus_id, "profile_version": profile.profile_version,
            "index_version": profile.index_version, "source_file": doc.metadata.get("source_file", "Documento desconocido"),
            "source_sha256": doc.metadata.get("source_sha256"), "page": doc.metadata.get("page"),
            "content_type": doc.metadata.get("type", "text"), "content": doc.page_content,
            "relevance": doc.metadata.get("rerank_score"), "dense_rank": doc.metadata.get("dense_rank"),
            "bm25_rank": doc.metadata.get("bm25_rank"), "rrf_score": doc.metadata.get("rrf_score"),
            "retrieval_role": doc.metadata.get("retrieval_role", "ranked"),
        } for doc in documents]


def tool_catalog():
    return [{"corpus_id": cid, "description": get_corpus_profile(cid).description,
             "documents": sorted(p.name for p in get_corpus_profile(cid).documents_dir.glob("*.pdf"))}
            for cid in list_corpus_ids()]
