"""Opt-in single-tool agent routing; explicit user selections bypass it."""
import json
import os
from pathlib import Path
from app.tools import tool_catalog


class RoutingClarification(ValueError):
    pass


def select_corpus(question, client):
    if os.getenv("RAG_ROUTER_ENABLED", "false").lower() != "true":
        raise RoutingClarification("El enrutamiento automático no está habilitado. Elegí un corpus en el selector.")
    catalog = tool_catalog()
    prompt = (Path(__file__).parent / "prompts" / "router.v1.txt").read_text(encoding="utf-8")
    response = client.models.generate_content(
        model=os.getenv("RAG_ROUTER_MODEL") or os.environ["GEMINI_MODEL"],
        contents=json.dumps({"catalog": catalog, "question": question}, ensure_ascii=False),
        config={"system_instruction": prompt, "temperature": 0, "response_mime_type": "application/json",
                "response_schema": {"type": "OBJECT", "properties": {
                    "corpus_id": {"type": "STRING"}, "clarification": {"type": "STRING"}},
                    "required": ["corpus_id", "clarification"]}})
    try:
        decision = json.loads(response.text)
        selected = decision["corpus_id"]
        if selected not in {item["corpus_id"] for item in catalog} or decision.get("clarification"):
            raise RoutingClarification(decision.get("clarification") or "¿Qué documento querés consultar?")
    except (TypeError, KeyError, json.JSONDecodeError) as exc:
        raise RoutingClarification("No se pudo determinar una fuente. Elegí el documento en el selector.") from exc
    return selected
