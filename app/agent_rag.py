"""Generation uses the selected corpus's retrieval tool and own generation profile."""
import hashlib
import os
from copy import deepcopy
from google import genai
from google.genai import types
from app.corpus import get_corpus_profile
from app.grounding import build_retrieval_query
from app.answer_contract import ANSWER_SCHEMA, render_structured_answer
from app.tools import CorpusRetrievalTool

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("Falta GEMINI_API_KEY en .env")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def limit_context(evidence: list[dict], max_context_chars: int) -> list[dict]:
    """Keep whole evidence fragments, in retrieval order, within a corpus budget."""
    selected, used = [], 0
    for item in evidence:
        size = len(item["content"])
        if selected and used + size > max_context_chars:
            continue
        if not selected and size > max_context_chars:
            raise ValueError("El primer fragmento supera max_context_chars del perfil.")
        selected.append(item)
        used += size
    return selected


def run_rag(question: str, history: list = None, corpus_id: str | None = None) -> dict:
    profile = get_corpus_profile(corpus_id)
    generation = profile.generation
    system_prompt = generation.read_prompt()
    history = (history or [])[-generation.max_history:]
    query = build_retrieval_query(question, history)
    evidence = limit_context(CorpusRetrievalTool(profile).retrieve(query), profile.max_context_chars)
    contexts = [item["content"] for item in evidence]
    context = "\n\n---\n\n".join(
        f"[EVIDENCIA id={item['chunk_id']} | Fuente: {item['source_file']} | Página: {item['page']}]\n{item['content']}"
        for item in evidence)
    contents = [types.Content(role="user" if message["role"] == "user" else "model",
                              parts=[types.Part.from_text(text=message["content"])]) for message in history]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=f"Contexto:\n{context}\n\nPregunta: {question}")]))
    response = client.models.generate_content(
        model=generation.model, contents=contents,
        config={"system_instruction": system_prompt, "temperature": generation.temperature,
                "max_output_tokens": generation.max_output_tokens,
                "response_mime_type": "application/json", "response_schema": deepcopy(ANSWER_SCHEMA)})
    if any(str(getattr(candidate, "finish_reason", "")).endswith("MAX_TOKENS")
           for candidate in (getattr(response, "candidates", None) or [])):
        raise RuntimeError("La respuesta alcanzó el límite de salida; no se entrega una respuesta truncada.")
    if not response.text or not response.text.strip():
        raise RuntimeError("El proveedor devolvió una respuesta vacía.")
    answer, citations, status = render_structured_answer(response.text, evidence)
    return {
        "corpus_id": profile.corpus_id, "answer": answer,
        "sources": list(dict.fromkeys(f"{item['source_file']} (pág. {item['page']})" for item in citations)),
        "contexts": contexts, "citations": citations, "retrieval_query": query,
        "answer_status": status,
        "execution": {"profile_version": profile.profile_version, "index_version": profile.index_version,
                      "retrieval_signature": profile.retrieval_signature(),
                      "ingestion_signature": profile.ingestion_signature(),
                      "prompt_version": generation.prompt_version,
                      "prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
                      "answer_contract_version": 1,
                      "model": generation.model, "temperature": generation.temperature,
                      "max_output_tokens": generation.max_output_tokens},
    }
