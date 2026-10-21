"""Small L2 FAISS adapter with read compatibility for trusted legacy bundles.

New index.pkl files contain plain data, not Community classes. The legacy
filename is retained so existing bundle/hash and activation contracts survive.
"""
from pathlib import Path
import pickle
from uuid import uuid4

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class DocumentStore:
    def __init__(self, documents=None):
        self._dict = documents or {}

    def search(self, identifier):
        return self._dict[identifier]


class _LegacyReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("langchain_community.docstore.in_memory", "InMemoryDocstore"): DocumentStore,
            ("langchain_core.documents.base", "Document"): Document,
        }
        try:
            return allowed[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"Clase no permitida en el índice: {module}.{name}") from None


class FAISS(VectorStore):
    """The project's used subset: L2 search, append and local persistence."""

    def __init__(self, embeddings, index, documents, mapping):
        self._embeddings = embeddings
        self.index = index
        self.docstore = DocumentStore(documents)
        self.index_to_docstore_id = mapping
        if index.metric_type != faiss.METRIC_L2:
            raise ValueError("Solo se admiten índices FAISS de distancia L2.")
        if (set(mapping) != set(range(index.ntotal))
                or len(set(mapping.values())) != index.ntotal
                or set(mapping.values()) != set(documents)
                or any(not isinstance(doc, Document) for doc in documents.values())):
            raise ValueError("El docstore y las referencias FAISS son inconsistentes.")

    @property
    def embeddings(self):
        return self._embeddings

    @classmethod
    def from_documents(cls, documents, embedding, *, ids=None, **kwargs):
        documents = list(documents)
        if not documents:
            raise ValueError("No se recibieron documentos.")
        vectors = cls._vectors(embedding.embed_documents([d.page_content for d in documents]))
        if len(vectors) != len(documents):
            raise ValueError("Cantidad de embeddings inconsistente.")
        identifiers = cls._ids(documents, ids)
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        copies = {cid: Document(id=cid, page_content=doc.page_content, metadata=dict(doc.metadata))
                  for cid, doc in zip(identifiers, documents)}
        return cls(embedding, index, copies, dict(enumerate(identifiers)))

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, *, ids=None, **kwargs):
        texts = list(texts)
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError("Cantidad de metadatos inconsistente.")
        return cls.from_documents([Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)],
                                  embedding, ids=ids)

    @staticmethod
    def _ids(documents, ids):
        identifiers = list(ids) if ids is not None else [doc.id or str(uuid4()) for doc in documents]
        if (len(identifiers) != len(documents) or len(set(identifiers)) != len(identifiers)
                or any(not isinstance(cid, str) or not cid for cid in identifiers)):
            raise ValueError("IDs de documentos inválidos o duplicados.")
        return identifiers

    @staticmethod
    def _vectors(values):
        vectors = np.asarray(values, dtype=np.float32)
        if vectors.ndim != 2 or not vectors.shape[0] or not vectors.shape[1] or not np.isfinite(vectors).all():
            raise ValueError("Embeddings vacíos, no finitos o de dimensiones inconsistentes.")
        return np.ascontiguousarray(vectors)

    def add_documents(self, documents, *, ids=None, **kwargs):
        documents = list(documents)
        if not documents:
            return []
        identifiers = self._ids(documents, ids)
        if set(identifiers) & self.docstore._dict.keys():
            raise ValueError("IDs ya indexados.")
        vectors = self._vectors(self.embeddings.embed_documents([d.page_content for d in documents]))
        if len(vectors) != len(documents) or vectors.shape[1] != self.index.d:
            raise ValueError("Dimensiones o cantidad de embeddings inconsistentes.")
        copies = {cid: Document(id=cid, page_content=doc.page_content, metadata=dict(doc.metadata))
                  for cid, doc in zip(identifiers, documents)}
        offset = self.index.ntotal
        self.index.add(vectors)
        self.docstore._dict.update(copies)
        self.index_to_docstore_id.update({offset + i: cid for i, cid in enumerate(identifiers)})
        return identifiers

    def similarity_search_by_vector(self, embedding, k=4):
        if type(k) is not int or k < 1:
            raise ValueError("k debe ser un entero positivo.")
        vector = self._vectors([embedding])
        if vector.shape[1] != self.index.d:
            raise ValueError("Dimensión de consulta incompatible con el índice.")
        _, positions = self.index.search(vector, k)
        return [self.docstore.search(self.index_to_docstore_id[int(pos)]) for pos in positions[0] if pos != -1]

    def similarity_search(self, query, k=4, **kwargs):
        if kwargs:
            raise ValueError("Este adaptador solo admite búsqueda L2 sin filtros adicionales.")
        return self.similarity_search_by_vector(self.embeddings.embed_query(query), k)

    def save_local(self, folder_path):
        path = Path(folder_path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        payload = {"format_version": 1,
                   "documents": {cid: doc.model_dump() for cid, doc in self.docstore._dict.items()},
                   "index_to_docstore_id": self.index_to_docstore_id}
        with (path / "index.pkl").open("wb") as target:
            pickle.dump(payload, target, protocol=4)

    @classmethod
    def load_local(cls, folder_path, embeddings, *, allow_dangerous_deserialization=False):
        if not allow_dangerous_deserialization:
            raise ValueError("La carga de un índice local requiere autorización explícita.")
        path = Path(folder_path)
        with (path / "index.pkl").open("rb") as source:
            payload = _LegacyReader(source).load()
        if isinstance(payload, dict) and payload.get("format_version") == 1:
            documents = {cid: Document.model_validate(doc) for cid, doc in payload["documents"].items()}
            mapping = payload["index_to_docstore_id"]
        elif isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], DocumentStore):
            documents, mapping = payload[0]._dict, payload[1]
        else:
            raise ValueError("Formato de almacén documental no reconocido.")
        return cls(embeddings, faiss.read_index(str(path / "index.faiss")), documents, mapping)
