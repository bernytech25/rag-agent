"""
Agente RAG con memoria conversacional y reranking.
Pipeline: Pregunta -> FAISS (10 candidatos, embeddings locales) -> Flashrank reordena -> top 3 -> Gemini (API key)

LLM: Gemini vía API key (google.generativeai), la misma GEMINI_API_KEY que usa evaluate_ragas.py.
No requiere proyecto de GCP ni gcloud auth. Los embeddings de retriever.py siguen locales
(HuggingFace) para no gastar tokens/creditos en la etapa de recuperacion.
"""

import os
from dotenv import load_dotenv
from google import genai
from app.retriever import get_retriever_with_reranking

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

if not GEMINI_API_KEY:
    raise RuntimeError("Falta GEMINI_API_KEY en el .env")
if not GEMINI_MODEL:
    raise RuntimeError("Falta GEMINI_MODEL en el .env (ej: gemini-3.1-flash-lite)")

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = r"""Sos un asistente experto que responde preguntas basándote ÚNICAMENTE
en los documentos proporcionados como contexto.

REGLAS:
- Usá SOLO la información del contexto para responder
- Si la información no está en el contexto, decí exactamente:
  "La información no está disponible en los documentos."
- Citá siempre el archivo fuente y la página cuando estén disponibles
- Si hay preguntas de seguimiento, usá el historial de conversación
- Respondé en el mismo idioma en que se hace la pregunta
- Para umbrales numéricos, citá textualmente el valor del documento
  sin agregar interpretaciones propias (ej: si dice ≥ 10%, no digas
  10% de algo diferente)
"""



MAX_HISTORY = 10


def _truncate_history(history: list) -> list:
    if len(history) <= MAX_HISTORY:
        return history
    return history[-MAX_HISTORY:]


def _build_gemini_prompt(system_prompt: str, history: list, question_block: str) -> str:
    """
    google.generativeai no usa el formato de mensajes de LangChain; arma un solo
    prompt de texto con el system prompt, el historial y la pregunta+contexto.
    """
    parts = [system_prompt]
    for msg in history:
        role = "Usuario" if msg["role"] == "user" else "Asistente"
        parts.append(f"{role}: {msg['content']}")
    parts.append(question_block)
    return "\n\n".join(parts)


def run_rag(question: str, history: list = None) -> dict:
    """
    Pipeline RAG completo con reranking.

    Args:
        question: Pregunta del usuario
        history: Historial de conversación

    Returns:
        dict con answer, sources y contexts (para evaluación)
    """
    history = history or []
    history = _truncate_history(history)

    # 1. Recuperar y rerankear documentos (local, no consume tokens)
    docs = get_retriever_with_reranking(question, top_k=3)

    # 2. Construir contexto con metadata y rerank score
    context_parts = []
    sources = []
    raw_contexts = []
    for doc in docs:
        source_file = doc.metadata.get("source_file", "Documento desconocido")
        page = doc.metadata.get("page", "N/A")
        score = doc.metadata.get("rerank_score", "N/A")
        context_parts.append(
            f"[Fuente: {source_file} | Página: {page} | Relevancia: {score}]\n{doc.page_content}"
        )
        raw_contexts.append(doc.page_content)
        source_key = f"{source_file} (pág. {page})"
        if source_key not in sources:
            sources.append(source_key)

    context = "\n\n---\n\n".join(context_parts)

    # 3. Armar el prompt (pregunta + contexto + historial)
    question_block = f"Contexto:\n{context}\n\nPregunta: {question}"
    prompt = _build_gemini_prompt(SYSTEM_PROMPT, history, question_block)

    # 4. Llamar a Gemini (una sola llamada por pregunta)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"temperature": 0},
    )

    return {
        "answer": response.text,
        "sources": sources,
        "contexts": raw_contexts,
    }