# RAG Agent — Etiquetado Nutricional Frontal Argentina

**Agente conversacional con Retrieval Augmented Generation (RAG)** para consultas sobre la normativa de etiquetado nutricional frontal en Argentina (Ley 27.642 y Decreto 151/2022).

- 🔍 **Retrieval local** (FAISS + HuggingFace embeddings, sin costos)
- 🧠 **Generación con Gemini API** (free tier, sin créditos)
- 📊 **Evaluado con RAGAS** — Score: **0.862** (Faithfulness 1.0)
- ⚡ **FastAPI + Docker ready**
- 🎯 **Producción-ready**

---

## Stack Tecnológico

| Componente | Tecnología |
|-----------|-----------|
| **Retrieval** | FAISS + HuggingFace embeddings (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) |
| **Reranking** | Flashrank (ms-marco-TinyBERT-L-2-v2) |
| **Extracción de tablas** | pdfplumber |
| **LLM** | Google Gemini API (free tier, `gemini-3.1-flash-lite`) |
| **API** | FastAPI |
| **Evaluación** | RAGAS (Faithfulness, Answer Relevancy, Context Precision) |
| **Control de versión** | Git + GitHub |

---

## Arquitectura

```
Pregunta del usuario
        ↓
    FAISS (retrieval)
  ↓ 15 candidatos
    Flashrank (reranking)
  ↓ top 5 chunks
    Gemini LLM (generación)
        ↓
    Respuesta + fuentes
```

**Chunking:** 800 caracteres, 200 overlap. Tablas extraídas intactas con pdfplumber.

---

## Instalación

### Requisitos
- Python 3.10+
- `pip`
- Gemini API key (free tier en https://aistudio.google.com)

### Setup

```bash
# Clonar repo
git clone https://github.com/bernytech25/rag-agent.git
cd rag-agent

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Crear .env
echo "GEMINI_API_KEY=tu_api_key_aqui
GEMINI_MODEL=gemini-3.1-flash-lite
JUDGE_MODEL=gemini-3.1-flash-lite
JWT_SECRET_KEY=cambiar-esto-en-produccion
JWT_EXPIRE_MINUTES=60" > .env
```

---

## Uso

### API Local (FastAPI)

```bash
python -m uvicorn app.main:app --reload
```

Abrí `http://localhost:8000/docs` (Swagger UI) y testea:

**Endpoint:** `POST /ask`

```json
{
  "question": "¿Cuál es el límite de sodio para el sello de exceso en sodio?",
  "history": []
}
```

**Respuesta:**
```json
{
  "answer": "El límite de sodio que determina el sello de advertencia es de 300 mg o más de sodio cada 100 gramos de producto...",
  "sources": ["2024-12-manual_normativa_original.pdf (pág. 58)"],
  "session_id": "user-123"
}
```

### Evaluación RAGAS

```bash
python evaluate_ragas.py
```

Genera un JSON con scores por pregunta y un reporte de métricas.

---

## Resultados de Evaluación

**Dataset:** 12 preguntas específicas sobre la normativa (ejemplos, umbrales, fórmulas).

| Métrica | Score | Interpretación |
|---------|-------|-----------------|
| **Faithfulness** | 1.000 | ✅ Nunca alucina — todas las respuestas están respaldadas en el documento |
| **Answer Relevancy** | 0.900 | ✅ Responde lo que se pregunta (90% de precisión) |
| **Context Precision** | 0.685 | 🟡 68.5% de chunks recuperados son útiles |
| **Overall** | **0.862** | 🟢 **PRODUCCIÓN** |

### Ejemplos de rendimiento

✅ **Preguntas numéricas/fórmulas:**
- "¿Cuál es la fórmula del ADS?" → Score 1.0 (responde exacto)
- "¿A partir de qué % de azúcares?" → Score 1.0 (recupera bien)

✅ **Preguntas conceptuales:**
- "¿Qué es un SPN?" → Score 1.0 (explica bien)
- "¿Qué productos se exceptúan?" → Score 1.0 (lista completa)

⚠️ **Casos edge:**
- Restricciones de publicidad → "No disponible en documentos" (honesto)

---

## Estructura del Proyecto

```
rag-agent/
├── app/
│   ├── main.py              # FastAPI app
│   ├── agent_rag.py         # RAG pipeline (FAISS + Gemini)
│   ├── retriever.py         # Indexing + reranking
│   ├── memory.py            # Conversational memory
│   └── auth.py              # JWT auth
├── data/
│   └── 2024-12-manual_normativa_original.pdf  # Documento fuente
├── evaluate_ragas.py        # RAGAS evaluation script
├── requirements.txt         # Dependencies
├── .env.example            # Template for .env
└── README.md               # Este archivo
```

---

## Configuración Recomendada

**Para desarrollo:**
```bash
python -m uvicorn app.main:app --reload --port 8000
```

**Para producción (Cloud Run):**
```bash
gcloud run deploy rag-agent --source . --region us-central1 --allow-unauthenticated
```

---

## Variables de Entorno

```env
# Gemini API (obtenida de https://aistudio.google.com)
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.1-flash-lite

# RAGAS evaluation
JUDGE_MODEL=gemini-3.1-flash-lite

# JWT (cambiar en producción)
JWT_SECRET_KEY=tu-clave-secreta-larga
JWT_EXPIRE_MINUTES=60
```

---

## Notas de Implementación

### Embeddings locales (sin costo)
- Modelo: `paraphrase-multilingual-MiniLM-L12-v2` (23MB)
- Almacenamiento: FAISS en disco (`faiss_index/`)
- Índice se reconstruye automáticamente si falta

### Reranking
- Modelo: `ms-marco-TinyBERT-L-2-v2` (4MB, local)
- Propósito: Reordenar chunks por relevancia (no filtrar)
- No usa umbral porque TinyBERT comprime scores cerca de 1.0

### Extracción de tablas
- pdfplumber detecta tablas automáticamente
- Convierte a markdown para mejor retrieval
- Evita que las tablas se corten por CHUNK_SIZE

### Prompt protector
El SYSTEM_PROMPT incluye regla explícita:
> "Cuando cites umbrales, porcentajes o límites numéricos, usá EXACTAMENTE la formulación del documento. NO la reformules ni la parafrasees."

Esto previene que el LLM reformule mal los números.

---

## Próximos Pasos

- [ ] Deployar a Cloud Run
- [ ] Agregar más documentos de normativa
- [ ] Mejorar Context Precision a 0.75+
- [ ] Implementar feedback loop (usuario valida respuestas)
- [ ] Monitoreo en producción (logging de queries)

---

## Créditos

- **Evaluación RAGAS:** Framework de Langchain
- **Prompt mejorado:** Colaboración con Kimi
- **Retriever optimizado:** Balanceado entre CHUNK_SIZE, TOP_K, reranking

---

## Licencia

MIT

---

## Contacto

**GitHub:** https://github.com/bernytech25/rag-agent

---

## Changelog

**v1.0** (2026-07-30)
- ✅ RAG funcional con FAISS + Gemini
- ✅ RAGAS evaluation: 0.862 score
- ✅ FastAPI deployed locally
- ✅ pdfplumber para extracción de tablas
- ✅ Prompt protector contra reformulación de números
- ✅ GitHub push completo