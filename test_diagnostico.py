"""Manual retrieval diagnostic. It is intentionally not a pytest test suite."""

def diagnose(question: str) -> None:
    """Show ranked candidates; retrieval always returns its best available evidence."""
    from app.retriever import get_retriever_with_reranking

    chunks = get_retriever_with_reranking(question)
    print(f"Chunks recuperados: {len(chunks)}")
    for chunk in chunks:
        print(
            f"Score: {chunk.metadata.get('rerank_score', 'N/A')} | "
            f"Fuente: {chunk.metadata.get('source_file')} | "
            f"Página: {chunk.metadata.get('page')}"
        )


if __name__ == "__main__":
    diagnose("¿Cuál es la fórmula ADS?")
