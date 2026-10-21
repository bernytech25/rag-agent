"""Pure helpers for contextual retrieval and verifiable evidence identifiers."""

import hashlib

FOLLOW_UP_MARKERS = (
    "y ", "eso", "esa", "ese", "esto", "esta", "este", "aquello",
    "lo anterior", "también", "tambien", "en ese caso", "qué más", "que mas",
)


def build_retrieval_query(question: str, history: list[dict]) -> str:
    """Add the most recent user turn only when the new question looks contextual."""
    normalized = question.strip()
    lowered = normalized.lower()
    is_follow_up = len(normalized) <= 140 and any(marker in lowered for marker in FOLLOW_UP_MARKERS)
    if not is_follow_up:
        return normalized

    previous_questions = [message["content"] for message in history if message.get("role") == "user"]
    if not previous_questions:
        return normalized
    return f"Pregunta anterior: {previous_questions[-1]}\nSeguimiento: {normalized}"


def chunk_id(metadata: dict, content: str) -> str:
    """Stable evidence id, including for indexes generated before chunk ids existed."""
    existing = metadata.get("chunk_id")
    if existing:
        return str(existing)
    source = metadata.get("source_file", "unknown")
    page = metadata.get("page", "na")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"{source}:{page}:{digest}"
