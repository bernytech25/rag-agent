FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements-postgres.txt ./
ARG INSTALL_POSTGRES=false
RUN if [ "$INSTALL_POSTGRES" = "true" ]; then pip install --no-cache-dir -r requirements-postgres.txt; else pip install --no-cache-dir -r requirements.txt; fi

COPY app/ ./app/

# Documents, FAISS/BM25 indexes and persistent memory are corpus data, not image data.
# Mount /app/corpora from durable storage before serving traffic.
ENV RAG_CORPORA_ROOT=/app/corpora
RUN mkdir -p /app/corpora /app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
