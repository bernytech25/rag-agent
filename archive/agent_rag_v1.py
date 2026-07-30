"""
Agente RAG con memoria conversacional.
Combina recuperación de documentos (FAISS) con generación (Groq).
Mantiene historial de conversación para preguntas de seguimiento.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from app.retriever import get_retriever

load_dotenv()

SYSTEM_PROMPT = """Sos un asistente experto que responde preguntas basándote ÚNICAMENTE
en los documentos proporcionados como contexto.

REGLAS:
- Usá SOLO la información del contexto para responder
- Si la información no está en el contexto, decí exactamente:
  "La información no está disponible en los documentos."
- Citá siempre el archivo fuente y la página cuando estén disponibles
- Si hay preguntas de seguimiento, usá el historial de conversación para entender el contexto
- Respondé en el mismo idioma en que se hace la pregunta
"""

MAX_HISTORY = 10


def _truncate_history(history: list) -> list:
    if len(history) <= MAX_HISTORY:
        return history
    return history[-MAX_HISTORY:]


def run_rag(question: str, history: list = None) -> dict:
    """
    Ejecuta el pipeline RAG completo.

    Args:
        question: Pregunta del usuario
        history: Historial de conversación [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        dict con "answer" y "sources" (lista de fuentes citadas)
    """
    history = history or []
    history = _truncate_history(history)

    # 1. Recuperar documentos relevantes
    retriever = get_retriever(top_k=3)
    docs = retriever.invoke(question)

    # 2. Construir contexto con metadata
    context_parts = []
    sources = []
    for doc in docs:
        source_file = doc.metadata.get("source_file", "Documento desconocido")
        page = doc.metadata.get("page", "N/A")
        context_parts.append(
            f"[Fuente: {source_file} | Página: {page}]\n{doc.page_content}"
        )
        source_key = f"{source_file} (pág. {page})"
        if source_key not in sources:
            sources.append(source_key)

    context = "\n\n---\n\n".join(context_parts)

    # 3. Construir historial para el LLM
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # 4. Agregar pregunta con contexto
    user_message = f"""Contexto de los documentos:
{context}

Pregunta: {question}"""
    messages.append({"role": "user", "content": user_message})

    # 5. Llamar al LLM
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": sources,
    }
