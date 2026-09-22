"""
FastAPI - RAG Agent API

Endpoints públicos:
  GET  /              → health check
  POST /auth/token    → obtener JWT

Endpoints protegidos:
  POST /ask           → pregunta al RAG con memoria in-session
  POST /ask/persistent → pregunta al RAG con memoria persistente
  POST /index         → indexar nuevos PDFs
  GET  /memory/{id}   → ver historial
  DELETE /memory/{id} → limpiar historial
"""

import logging
from typing import Annotated
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from app.agent_rag import run_rag
from app.memory import in_session_memory, persistent_memory
from app.retriever import add_documents, index_exists
from app.auth import Token, User, authenticate_user, create_access_token, get_current_user, require_admin, ACCESS_TOKEN_EXPIRE_MINUTES

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Agent API",
    description="Agente RAG sobre múltiples PDFs con FAISS + Groq + Memoria conversacional",
    version="1.0.0",
)

class AskRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    question: str = Field(min_length=1, max_length=4000)
    model_config = {"json_schema_extra": {"example": {"session_id": "user-123", "question": "What is the passing score?"}}}

class AskResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    sources: list[str]

class IndexRequest(BaseModel):
    pdf_paths: list[str] = Field(min_length=1, max_length=20)


def _memory_key(user: User, session_id: str) -> str:
    """Namespace client-chosen IDs so one user cannot access another's history."""
    return f"{user.username}:{session_id}"

@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "service": "rag-agent", "index_ready": index_exists()}

@app.post("/auth/token", response_model=Token, tags=["Auth"])
def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos")
    return Token(access_token=create_access_token({"sub": user.username}), token_type="bearer", expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60)

@app.post("/ask", response_model=AskResponse, tags=["RAG - In-Session"])
def ask(request: AskRequest, current_user: Annotated[User, Depends(get_current_user)]):
    memory_key = _memory_key(current_user, request.session_id)
    history = in_session_memory.get_history(memory_key)
    try:
        result = run_rag(question=request.question, history=history)
    except Exception:
        logger.exception("RAG request failed", extra={"user": current_user.username})
        raise HTTPException(status_code=503, detail="El servicio de respuestas no está disponible temporalmente.")
    in_session_memory.add_message(memory_key, "user", request.question)
    in_session_memory.add_message(memory_key, "assistant", result["answer"])
    return AskResponse(session_id=request.session_id, question=request.question, answer=result["answer"], sources=result["sources"])

@app.post("/ask/persistent", response_model=AskResponse, tags=["RAG - Persistente"])
def ask_persistent(request: AskRequest, current_user: Annotated[User, Depends(get_current_user)]):
    memory_key = _memory_key(current_user, request.session_id)
    history = persistent_memory.get_history(memory_key)
    try:
        result = run_rag(question=request.question, history=history)
    except Exception:
        logger.exception("Persistent RAG request failed", extra={"user": current_user.username})
        raise HTTPException(status_code=503, detail="El servicio de respuestas no está disponible temporalmente.")
    persistent_memory.add_message(memory_key, "user", request.question)
    persistent_memory.add_message(memory_key, "assistant", result["answer"])
    return AskResponse(session_id=request.session_id, question=request.question, answer=result["answer"], sources=result["sources"])

@app.post("/index", tags=["Admin"])
def index_documents(request: IndexRequest, current_user: Annotated[User, Depends(require_admin)]):
    try:
        add_documents(request.pdf_paths)
        return {"status": "ok", "requested": len(request.pdf_paths)}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Indexing failed", extra={"user": current_user.username})
        raise HTTPException(status_code=500, detail="No se pudo actualizar el índice.")

@app.get("/memory/{session_id}", tags=["Memoria"])
def get_memory(session_id: str, current_user: Annotated[User, Depends(get_current_user)], persistent: bool = False):
    memory_key = _memory_key(current_user, session_id)
    history = persistent_memory.get_history_with_timestamps(memory_key) if persistent else in_session_memory.get_history(memory_key)
    return {"session_id": session_id, "messages": history, "total": len(history)}

@app.delete("/memory/{session_id}", tags=["Memoria"])
def clear_memory(session_id: str, current_user: Annotated[User, Depends(get_current_user)], persistent: bool = False):
    memory_key = _memory_key(current_user, session_id)
    if persistent:
        persistent_memory.clear(memory_key)
    else:
        in_session_memory.clear(memory_key)
    return {"status": "cleared", "session_id": session_id}
