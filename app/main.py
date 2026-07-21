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

import os
from typing import Annotated
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from app.agent_rag import run_rag
from app.memory import in_session_memory, persistent_memory
from app.retriever import add_documents, index_exists
from app.auth import Token, User, authenticate_user, create_access_token, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES

app = FastAPI(
    title="RAG Agent API",
    description="Agente RAG sobre múltiples PDFs con FAISS + Groq + Memoria conversacional",
    version="1.0.0",
)

class AskRequest(BaseModel):
    session_id: str
    question: str
    model_config = {"json_schema_extra": {"example": {"session_id": "user-123", "question": "What is the passing score?"}}}

class AskResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    sources: list[str]

class IndexRequest(BaseModel):
    pdf_paths: list[str]

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
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    history = in_session_memory.get_history(request.session_id)
    try:
        result = run_rag(question=request.question, history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    in_session_memory.add_message(request.session_id, "user", request.question)
    in_session_memory.add_message(request.session_id, "assistant", result["answer"])
    return AskResponse(session_id=request.session_id, question=request.question, answer=result["answer"], sources=result["sources"])

@app.post("/ask/persistent", response_model=AskResponse, tags=["RAG - Persistente"])
def ask_persistent(request: AskRequest, current_user: Annotated[User, Depends(get_current_user)]):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    history = persistent_memory.get_history(request.session_id)
    try:
        result = run_rag(question=request.question, history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    persistent_memory.add_message(request.session_id, "user", request.question)
    persistent_memory.add_message(request.session_id, "assistant", result["answer"])
    return AskResponse(session_id=request.session_id, question=request.question, answer=result["answer"], sources=result["sources"])

@app.post("/index", tags=["Admin"])
def index_documents(request: IndexRequest, current_user: Annotated[User, Depends(get_current_user)]):
    try:
        add_documents(request.pdf_paths)
        return {"status": "ok", "indexed": len(request.pdf_paths)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/{session_id}", tags=["Memoria"])
def get_memory(session_id: str, current_user: Annotated[User, Depends(get_current_user)], persistent: bool = False):
    history = persistent_memory.get_history_with_timestamps(session_id) if persistent else in_session_memory.get_history(session_id)
    return {"session_id": session_id, "messages": history, "total": len(history)}

@app.delete("/memory/{session_id}", tags=["Memoria"])
def clear_memory(session_id: str, current_user: Annotated[User, Depends(get_current_user)], persistent: bool = False):
    if persistent:
        persistent_memory.clear(session_id)
    else:
        in_session_memory.clear(session_id)
    return {"status": "cleared", "session_id": session_id}
