"""Provider output contract and deterministic citation rendering."""
from typing import Literal
import re
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class AnswerContractError(RuntimeError):
    """The provider output cannot be delivered as a verifiable response."""


class EvidenceBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=6000)
    evidence_ids: list[str] = Field(min_length=1, max_length=30)


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    evidence_status: Literal["supported", "partial", "insufficient"]
    blocks: list[EvidenceBlock] = Field(max_length=30)
    missing_evidence: str = Field(max_length=1500)


# Explicit schema avoids provider-specific support for additionalProperties;
# local Pydantic validation still rejects extra fields and enforces constraints.
ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "evidence_status": {"type": "STRING", "enum": ["supported", "partial", "insufficient"]},
        "blocks": {"type": "ARRAY", "items": {
            "type": "OBJECT", "properties": {
                "text": {"type": "STRING"},
                "evidence_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
            }, "required": ["text", "evidence_ids"]}},
        "missing_evidence": {"type": "STRING"},
    },
    "required": ["evidence_status", "blocks", "missing_evidence"],
}


def render_structured_answer(raw: str, evidence: list[dict]) -> tuple[str, list[dict], str]:
    try:
        result = GeneratedAnswer.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        raise AnswerContractError("La respuesta no cumple el esquema de generación.") from exc
    missing = result.missing_evidence.strip()
    if result.evidence_status == "supported":
        if not result.blocks or missing:
            raise AnswerContractError("Una respuesta sustentada requiere bloques y no debe declarar faltantes.")
    elif result.evidence_status == "partial":
        if not result.blocks or not missing:
            raise AnswerContractError("Una respuesta parcial requiere bloques y el dato faltante.")
    elif result.blocks or not missing:
        raise AnswerContractError("Una abstención requiere el dato faltante y ningún bloque afirmativo.")

    by_id = {item["chunk_id"]: item for item in evidence}
    labels, citations, paragraphs = {}, [], []
    for block in result.blocks:
        text = block.text.strip()
        if not text or re.search(r"\[CITA:|\[\d+(?:\s*,\s*\d+)*\]", text):
            raise AnswerContractError("El texto debe ser no vacío y no debe contener marcas de cita del modelo.")
        ids = list(dict.fromkeys(block.evidence_ids))
        if any(cid not in by_id for cid in ids):
            raise AnswerContractError("La respuesta cita evidencia ajena al turno.")
        for cid in ids:
            if cid not in labels:
                labels[cid] = str(len(labels) + 1)
                citations.append({**by_id[cid], "label": labels[cid]})
        references = ", ".join(labels[cid] for cid in ids)
        paragraphs.append(f"{text} [{references}]")
    if missing:
        # Scope the limitation to evidence retrieved in this turn, never the PDF.
        paragraphs.append(f"No puedo confirmarlo con la evidencia disponible. Falta en los fragmentos recuperados: {missing}")
    status = {"supported": "cited", "partial": "partial", "insufficient": "insufficient_evidence"}[result.evidence_status]
    return "\n\n".join(paragraphs), citations, status
