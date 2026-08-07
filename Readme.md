# RAG Agent — Multidocument Nutritional Labeling (Argentina)

**Multidocument conversational agent with Retrieval Augmented Generation (RAG)** for querying Argentine front-of-pack nutritional labeling regulations (Law 27.642 and Decree 151/2022). Designed to scale across multiple regulatory documents.

- 🔍 **Local retrieval** (FAISS + HuggingFace embeddings, zero cloud costs)
- 🧠 **Generation via Gemini API** (free tier, no credits required)
- 📄 **Multidocument ingestion** — index and query across multiple PDFs simultaneously
- 📊 **Evaluated with RAGAS** — Score: **0.862** (Faithfulness 1.0)
- ⚡ **FastAPI + Docker ready**
- 🎯 **Production-ready**

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Retrieval** | FAISS + HuggingFace embeddings (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) |
| **Reranking** | Flashrank (`ms-marco-TinyBERT-L-2-v2`) |
| **Table Extraction** | pdfplumber |
| **LLM** | Google Gemini API (free tier, `gemini-3.1-flash-lite`) |
| **API** | FastAPI |
| **Evaluation** | RAGAS (Faithfulness, Answer Relevancy, Context Precision) |
| **Version Control** | Git + GitHub |

---

## Architecture

```
User query
    ↓
FAISS (multidocument retrieval)
  ↓ 15 candidates
Flashrank (reranking)
  ↓ top 5 chunks
Gemini LLM (generation)
    ↓
Answer + sources
```

**Chunking:** 800 characters, 200 overlap. Tables extracted intact via pdfplumber.  
**Multidocument:** Drop multiple PDFs into `data/` — all are indexed and searchable through a single query endpoint.

---

## Installation

### Requirements

- Python 3.10+
- `pip`
- Gemini API key (free tier at https://aistudio.google.com)

### Setup

```bash
# Clone repo
git clone https://github.com/bernytech25/rag-agent.git
cd rag-agent

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Create .env
echo "GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
JUDGE_MODEL=gemini-3.1-flash-lite
JWT_SECRET_KEY=change-this-in-production
JWT_EXPIRE_MINUTES=60" > .env
```

---

## Usage

### Local API (FastAPI)

```bash
python -m uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` (Swagger UI) and test:

**Endpoint:** `POST /ask`

```json
{
  "question": "What is the sodium limit for the excess sodium warning label?",
  "history": []
}
```

**Response:**

```json
{
  "answer": "The sodium limit that triggers the warning label is 300 mg or more of sodium per 100 grams of product...",
  "sources": ["2024-12-manual_normativa_original.pdf (p. 58)"],
  "session_id": "user-123"
}
```

### RAGAS Evaluation

```bash
python evaluate_ragas.py
```

Generates a JSON with per-question scores and a metrics report.

---

## Evaluation Results

**Dataset:** 12 specific questions about the regulation (examples, thresholds, formulas).

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Faithfulness** | 1.000 | ✅ Never hallucinates — all answers are grounded in the document |
| **Answer Relevancy** | 0.900 | ✅ Answers what is asked (90% precision) |
| **Context Precision** | 0.685 | 🟡 68.5% of retrieved chunks are useful |
| **Overall** | **0.862** | 🟢 **PRODUCTION** |

### Performance Examples

✅ **Numeric/formula questions:**

- "What is the ADS formula?" → Score 1.0 (exact answer)
- "What percentage of added sugars triggers the label?" → Score 1.0 (retrieves correctly)

✅ **Conceptual questions:**

- "What is an SPN?" → Score 1.0 (explains well)
- "Which products are exempt?" → Score 1.0 (complete list)

⚠️ **Edge cases:**

- Advertising restrictions → "Not available in documents" (honest)

---

## Project Structure

```
rag-agent/
├── app/
│   ├── main.py              # FastAPI app
│   ├── agent_rag.py         # RAG pipeline (FAISS + Gemini)
│   ├── retriever.py         # Multidocument indexing + reranking
│   ├── memory.py            # Conversational memory
│   └── auth.py              # JWT auth
├── data/                    # Drop multiple PDFs here
│   └── 2024-12-manual_normativa_original.pdf
├── evaluate_ragas.py        # RAGAS evaluation script
├── requirements.txt         # Dependencies
├── .env.example             # Environment template
└── README.md                # This file
```

---

## Recommended Configuration

**Development:**

```bash
python -m uvicorn app.main:app --reload --port 8000
```

**Production (Cloud Run):**

```bash
gcloud run deploy rag-agent --source . --region us-central1 --allow-unauthenticated
```

---

## Environment Variables

```bash
# Gemini API (get at https://aistudio.google.com)
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.1-flash-lite

# RAGAS evaluation
JUDGE_MODEL=gemini-3.1-flash-lite

# JWT (change in production)
JWT_SECRET_KEY=your-long-secret-key
JWT_EXPIRE_MINUTES=60
```

---

## Implementation Notes

### Local Embeddings (zero cost)

- Model: `paraphrase-multilingual-MiniLM-L12-v2` (23MB)
- Storage: FAISS on disk (`faiss_index/`)
- Index rebuilds automatically if missing

### Multidocument Retrieval

- Drop any number of PDFs into `data/`
- `retriever.py` indexes all documents into a single FAISS vector store
- Metadata tracks `source_file` per chunk so answers cite the correct document
- Scales from 1 to N documents without code changes

### Reranking

- Model: `ms-marco-TinyBERT-L-2-v2` (4MB, local)
- Purpose: Reorder chunks by relevance (not filter)
- No threshold applied because TinyBERT compresses scores near 1.0

### Table Extraction

- pdfplumber detects tables automatically
- Converts to markdown for better retrieval
- Prevents tables from being split by CHUNK_SIZE

### Prompt Guard

The SYSTEM_PROMPT includes an explicit rule:

> "When citing thresholds, percentages, or numeric limits, use EXACTLY the document's wording. DO NOT rephrase or paraphrase."

This prevents the LLM from misstating numbers.

---

## Roadmap

- [ ] Deploy to Cloud Run
- [ ] Add more regulatory documents (multidocument expansion)
- [ ] Improve Context Precision to 0.75+
- [ ] Implement feedback loop (user validates answers)
- [ ] Production monitoring (query logging)

---

## Credits

- **RAGAS Evaluation:** LangChain framework
- **Prompt engineering:** Collaboration with Kimi
- **Retriever optimization:** Balanced CHUNK_SIZE, TOP_K, and reranking

---

## License

MIT

---

## Contact

**GitHub:** https://github.com/bernytech25/rag-agent

---

## Changelog

**v1.0** (2026-07-30)

- ✅ Functional RAG with FAISS + Gemini
- ✅ RAGAS evaluation: 0.862 score
- ✅ FastAPI deployed locally
- ✅ pdfplumber for table extraction
- ✅ Guard prompt against number rephrasing
- ✅ Full GitHub push
- ✅ Multidocument ingestion support
