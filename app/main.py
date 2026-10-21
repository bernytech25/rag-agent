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

from app import runtime_config  # Load .env/offline flags before model imports.

from contextlib import asynccontextmanager
import logging
from enum import Enum
from typing import Annotated, Literal
from fastapi import FastAPI, HTTPException, Depends, Path, Query
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from app.agent_rag import run_rag, client
from app.answer_contract import AnswerContractError
from app.memory import in_session_memory, persistent_memory
from app.sql_memory import ConversationConflict
from app.retriever import add_documents, create_index, index_exists, prepare_legacy_index, verify_embeddings_available
from app.index_versions import activate_index, list_versions
from app.routing import RoutingClarification, select_corpus
from app.corpus import CORPUS_ID, get_corpus_profile, list_corpus_ids
from app.auth import Token, User, authenticate_user, create_access_token, get_current_user, require_admin, ACCESS_TOKEN_EXPIRE_MINUTES

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app):
    """Verify offline embedding dependencies before accepting API requests."""
    for corpus_id in list_corpus_ids():
        verify_embeddings_available(get_corpus_profile(corpus_id).embedding_model)
    yield


app = FastAPI(
    title="RAG Agent API",
    description="RAG conversacional híbrido por corpus con FAISS, BM25, FlashRank y Gemini",
    version="1.0.0",
    lifespan=lifespan,
)

# The enum is a startup snapshot of the registry. Restart the API after adding
# a corpus so Swagger and request validation expose the same set of choices.
CorpusChoice = Enum(
    "CorpusChoice",
    {f"corpus_{index}": corpus_id for index, corpus_id in enumerate(list_corpus_ids())},
    type=str,
)
CorpusSelector = Annotated[
    CorpusChoice | None,
    Query(description="Elegí el documento. Esta selección tiene prioridad sobre corpus_id del JSON. En modo manual, omitir ambos usa el corpus predeterminado."),
]
SessionPath = Annotated[str, Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")]

class RoutingMode(str, Enum):
    manual = "manual"
    auto = "auto"

RoutingSelector = Annotated[RoutingMode, Query(description="manual conserva tu selección. auto elige una herramienta solo si no fijaste corpus; requiere habilitar el enrutador experimental en el servidor.")]

class IndexMode(str, Enum):
    append = "append"
    rebuild = "rebuild"

class AskRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    question: str = Field(min_length=1, max_length=4000)
    corpus_id: str | None = Field(default=None, description="Compatibilidad con clientes JSON. En Swagger, usá el selector corpus_id de Parameters.")
    model_config = {"json_schema_extra": {"example": {"session_id": "consulta-001", "question": "¿Qué indica el documento?"}}}

class AskResponse(BaseModel):
    corpus_id: str
    session_id: str
    question: str
    answer: str
    sources: list[str]
    citations: list[dict]
    answer_status: Literal["cited", "partial", "insufficient_evidence"] = Field(description="cited: bloques con referencias validadas; partial: bloques citados y faltantes declarados; insufficient_evidence: abstención. La suficiencia la declara el modelo; no certifica exactitud semántica.")
    execution: dict | None = None

class IndexRequest(BaseModel):
    pdf_paths: list[str] = Field(min_length=1, max_length=20)
    corpus_id: str | None = None
    model_config = {"json_schema_extra": {"example": {"pdf_paths": ["documento.pdf"]}}}


class HealthResponse(BaseModel):
    status: str
    service: str
    corpus_id: str
    index_ready: bool
    hybrid_ready: bool
    index_version: str


class CorpusSummary(BaseModel):
    corpus_id: str
    index_ready: bool = Field(description="Indica si existen ambos archivos FAISS; no evalúa su contenido ni la calidad del retrieval.")
    hybrid_ready: bool = Field(description="Existen los archivos FAISS y BM25; no es una evaluación de relevancia.")
    index_version: str


class CorporaResponse(BaseModel):
    corpora: list[CorpusSummary]


class IndexResponse(BaseModel):
    status: str
    corpus_id: str
    requested: int
    index_version: str
    chunks: int
    active: bool = False

class ActivateRequest(BaseModel):
    version: str = Field(min_length=1, max_length=63, pattern=r"^[a-z0-9][a-z0-9-]*$")
    evaluation_reference: str = Field(min_length=1, max_length=500)

class IndexVersionResponse(BaseModel):
    corpus_id: str
    version: str
    chunks: int
    origin: str
    active: bool = False
    evaluation_reference: str | None = None

class IndexVersionsResponse(BaseModel):
    corpus_id: str
    versions: list[IndexVersionResponse]


class MemoryMessage(BaseModel):
    role: str
    content: str
    timestamp: str | None = None


class MemoryResponse(BaseModel):
    corpus_id: str
    session_id: str
    messages: list[MemoryMessage]
    total: int


class ClearMemoryResponse(BaseModel):
    status: str
    corpus_id: str
    session_id: str

class ErrorResponse(BaseModel):
    detail: str

QUERY_ERRORS = {
    400: {"model": ErrorResponse, "description": "Corpus o perfil inválido"},
    401: {"model": ErrorResponse, "description": "Autenticación requerida"},
    409: {"model": ErrorResponse, "description": "Se requiere aclarar la fuente o repetir un turno concurrente"},
    502: {"model": ErrorResponse, "description": "Salida del modelo inválida o referencias no verificables; el turno no se guarda"},
    503: {"model": ErrorResponse, "description": "Herramienta, almacenamiento o generación no disponible"},
}


def _memory_key(user: User, session_id: str, corpus_id: str) -> str:
    """Separate histories by corpus, user and client-chosen session ID."""
    return f"{corpus_id}:{user.username}:{session_id}"


def _selected_corpus(corpus_id: str | CorpusChoice | None, body_corpus_id: str | None = None) -> str:
    selected = corpus_id.value if isinstance(corpus_id, CorpusChoice) else corpus_id
    try:
        return get_corpus_profile(selected if selected is not None else body_corpus_id).corpus_id
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/", response_model=HealthResponse, tags=["Health"], summary="Estado del servicio")
def health():
    profile = get_corpus_profile()
    ready = index_exists()
    return {"status": "ok", "service": "rag-agent", "corpus_id": CORPUS_ID, "index_ready": ready,
            "hybrid_ready": ready and profile.bm25_index_path.is_file(), "index_version": profile.index_version}


@app.get(
    "/corpora", response_model=CorporaResponse, tags=["RAG"],
    summary="Listar documentos disponibles",
    description="Lista los corpus registrados. Para elegir uno, abrí POST /ask o POST /ask/persistent y usá el selector corpus_id en Parameters.",
)
def corpora(current_user: Annotated[User, Depends(get_current_user)]):
    items = []
    for corpus_id in list_corpus_ids():
        profile = get_corpus_profile(corpus_id)
        ready = index_exists(corpus_id)
        items.append({"corpus_id": corpus_id, "index_ready": ready,
                      "hybrid_ready": ready and profile.bm25_index_path.is_file(), "index_version": profile.index_version})
    return {"corpora": items}

@app.post("/auth/token", response_model=Token, tags=["Auth"])
def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos")
    return Token(access_token=create_access_token({"sub": user.username}), token_type="bearer", expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60)

def _answer(request, current_user, corpus_id, routing_mode, store):
    explicit = corpus_id is not None or request.corpus_id is not None
    if not explicit and routing_mode == RoutingMode.auto:
        try:
            corpus_id = select_corpus(request.question, client)
        except RoutingClarification as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Routing failed")
            raise HTTPException(status_code=503, detail="No se pudo seleccionar una herramienta.") from exc
    corpus_id = _selected_corpus(corpus_id, request.corpus_id)
    memory_key = _memory_key(current_user, request.session_id, corpus_id)
    try:
        with store.turn_scope(memory_key, corpus_id):
            revision = store.get_revision(memory_key, corpus_id)
            history = store.get_history(memory_key, corpus_id)
            result = run_rag(question=request.question, history=history, corpus_id=corpus_id)
            response = AskResponse(corpus_id=corpus_id, session_id=request.session_id, question=request.question,
                                   answer=result["answer"], sources=result["sources"], citations=result["citations"],
                                   answer_status=result["answer_status"], execution=result.get("execution"))
            store.append_turn(memory_key, request.question, result["answer"], corpus_id, expected_revision=revision)
            return response
    except ConversationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except AnswerContractError as exc:
        logger.warning("Invalid answer contract", extra={"corpus_id": corpus_id})
        raise HTTPException(status_code=502, detail="La respuesta del modelo no cumplió el formato o las referencias requeridas. No se guardó el turno.") from exc
    except Exception:
        logger.exception("RAG request failed", extra={"user": current_user.username})
        raise HTTPException(status_code=503, detail="El servicio de respuestas no está disponible temporalmente.")

@app.post("/ask", response_model=AskResponse, responses=QUERY_ERRORS, tags=["RAG - In-Session"], summary="Preguntar con memoria temporal")
def ask(request: AskRequest, current_user: Annotated[User, Depends(get_current_user)], corpus_id: CorpusSelector = None,
        routing_mode: RoutingSelector = RoutingMode.manual):
    return _answer(request, current_user, corpus_id, routing_mode, in_session_memory)

@app.post("/ask/persistent", response_model=AskResponse, responses=QUERY_ERRORS, tags=["RAG - Persistente"], summary="Preguntar con memoria guardada")
def ask_persistent(request: AskRequest, current_user: Annotated[User, Depends(get_current_user)], corpus_id: CorpusSelector = None,
                   routing_mode: RoutingSelector = RoutingMode.manual):
    return _answer(request, current_user, corpus_id, routing_mode, persistent_memory)

@app.post("/index", response_model=IndexResponse, tags=["Admin"], summary="Construir un índice candidato sin activar")
def index_documents(request: IndexRequest, current_user: Annotated[User, Depends(require_admin)], corpus_id: CorpusSelector = None,
                    mode: IndexMode = IndexMode.append):
    corpus_id = _selected_corpus(corpus_id, request.corpus_id)
    try:
        builder = create_index if mode == IndexMode.rebuild else add_documents
        candidate = builder(request.pdf_paths, corpus_id=corpus_id)
        return {"status": "candidate", "corpus_id": corpus_id, "requested": len(request.pdf_paths),
                "index_version": candidate["version"], "chunks": candidate["chunks"], "active": False}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Indexing failed", extra={"user": current_user.username})
        raise HTTPException(status_code=500, detail="No se pudo actualizar el índice.")

@app.post("/index/prepare", response_model=IndexVersionResponse, tags=["Admin"],
          summary="Preparar versión híbrida desde FAISS heredado, sin reingerir ni activar")
def prepare_index(current_user: Annotated[User, Depends(require_admin)], corpus_id: CorpusSelector = None):
    selected = _selected_corpus(corpus_id)
    try:
        return prepare_legacy_index(selected)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/indexes", response_model=IndexVersionsResponse, tags=["RAG"], summary="Listar versiones de índice del corpus")
def indexes(current_user: Annotated[User, Depends(get_current_user)], corpus_id: CorpusSelector = None):
    selected = _selected_corpus(corpus_id)
    return {"corpus_id": selected, "versions": list_versions(selected)}

@app.post("/index/activate", response_model=IndexVersionResponse, tags=["Admin"],
          summary="Activar o recuperar una versión con referencia de evaluación")
def activate(request: ActivateRequest, current_user: Annotated[User, Depends(require_admin)], corpus_id: CorpusSelector = None):
    selected = _selected_corpus(corpus_id)
    try:
        return activate_index(selected, request.version, request.evaluation_reference)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/memory/{session_id}", response_model=MemoryResponse, response_model_exclude_none=True, tags=["Memoria"], summary="Consultar historial del documento seleccionado")
def get_memory(session_id: SessionPath, current_user: Annotated[User, Depends(get_current_user)], persistent: bool = False, corpus_id: CorpusSelector = None):
    selected = _selected_corpus(corpus_id)
    memory_key = _memory_key(current_user, session_id, selected)
    history = persistent_memory.get_history_with_timestamps(memory_key, corpus_id=selected) if persistent else in_session_memory.get_history(memory_key)
    return {"corpus_id": selected, "session_id": session_id, "messages": history, "total": len(history)}

@app.delete("/memory/{session_id}", response_model=ClearMemoryResponse, tags=["Memoria"], summary="Borrar historial del documento seleccionado")
def clear_memory(session_id: SessionPath, current_user: Annotated[User, Depends(get_current_user)], persistent: bool = False, corpus_id: CorpusSelector = None):
    selected = _selected_corpus(corpus_id)
    memory_key = _memory_key(current_user, session_id, selected)
    if persistent:
        persistent_memory.clear(memory_key, corpus_id=selected)
    else:
        in_session_memory.clear(memory_key)
    return {"status": "cleared", "corpus_id": selected, "session_id": session_id}
