from app.retriever import get_retriever_with_reranking

# Pregunta que NO existe → debería devolver 0 chunks
chunks = get_retriever_with_reranking("¿Cuántos sellos debe llevar el alfajor del Ejemplo 5?")
print(f"Chunks: {len(chunks)}")  # Esperado: 0

# Pregunta que SÍ existe → debería devolver 1-3 chunks con scores reales
chunks2 = get_retriever_with_reranking("¿Cuál es la fórmula ADS?")
for c in chunks2:
    print(f"Score: {c.metadata['rerank_score']}, Max: {c.metadata['rerank_max_score']}")