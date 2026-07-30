"""
Evaluación de calidad del RAG con LLM-as-a-Judge (Gemini API).
Set de preguntas balanceado: cubre las 9 secciones principales del manual
de rotulado nutricional frontal (no solo el Anexo C de ejemplos).
"""

import os
import json
import re
import time
from datetime import datetime
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
JUDGE_MODEL = os.getenv("JUDGE_MODEL")

if not GEMINI_API_KEY:
    raise RuntimeError("Falta GEMINI_API_KEY en el .env")
if not JUDGE_MODEL:
    raise RuntimeError("Falta JUDGE_MODEL en el .env (ej: gemini-3.1-flash-lite)")

client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# SET DE PREGUNTAS BALANCEADO
# Cubre: Introducción, Conceptos, Rotulado Frontal, Criterios
# (azúcares, grasas, sodio, calorías, edulcorantes), Características,
# Publicidad, y Ejemplos. Sin preguntas "trampa".
# ============================================================
EVAL_QUESTIONS = [
    # IV. Rotulado Nutricional Frontal - Conceptos
    "¿Qué es un sistema de perfil de nutrientes (SPN) y para qué se utiliza en el rotulado nutricional frontal?",

    # IV. Rotulado Nutricional Frontal - Productos alcanzados
    "¿Cuáles son los nutrientes críticos que deben declararse en el rotulado nutricional frontal?",

    # V. Criterios - Azúcares añadidos
    "¿A partir de qué porcentaje de calorías provenientes de azúcares añadidos corresponde declarar el sello de exceso en azúcares?",

    # V. Criterios - Grasas saturadas
    "¿Cuál es el límite de grasas saturadas para declarar el sello de exceso en grasas saturadas?",

    # V. Criterios - Sodio
    "¿Cuál es el límite de sodio en miligramos por 100 gramos de producto que determina el sello de exceso en sodio?",

    # IV. Rotulado Frontal - Edulcorantes (regla condicional)
    "¿Cuándo se entiende que un alimento contiene edulcorantes y debe declararlo en el rotulado nutricional frontal?",

    # IV. Rotulado Frontal - Productos no alcanzados
    "¿Qué productos no se encuentran alcanzados por la normativa de rotulado nutricional frontal?",

    # VI. Características - Área ADS
    "¿Cuál es la fórmula para calcular el área de la cara principal disponible para los sellos (ADS) cuando un producto lleva dos o más sellos?",

    # VI. Características - Prohibiciones
    "¿Cuáles son las prohibiciones establecidas por la normativa para el rotulado de los productos con sellos de advertencia?",

    # VII. Publicidad, Promoción y Patrocinio
    "¿Qué restricciones tiene la publicidad de productos que llevan sellos de advertencia de exceso de nutrientes críticos?",

    # V. Criterios - Cálculo de azúcares añadidos (proceso)
    "¿Cómo se obtiene el contenido de azúcares añadidos de un alimento para evaluar si corresponde el sello de exceso?",

    # IX-C. Ejemplos (aplicación de reglas)
    "¿El jugo de naranja exprimido del Ejemplo 1 debe llevar sellos de advertencia? ¿Por qué?",
]

PROMPT_TEMPLATE = """Eres un evaluador experto de sistemas RAG (Retrieval Augmented Generation).
Evalúa la siguiente interacción y devuelve ÚNICAMENTE un objeto JSON válido.

PREGUNTA DEL USUARIO:
{question}

RESPUESTA GENERADA:
{answer}

CONTEXTO RECUPERADO:
{contexts}

Instrucciones:
1. Faithfulness (0.0 a 1.0): ¿La respuesta está fundamentada en el contexto? 1.0 = todos los hechos vienen del contexto. 0.0 = alucina completamente.
2. Answer Relevancy (0.0 a 1.0): ¿La respuesta responde directamente lo que se preguntó? 1.0 = responde exactamente. 0.0 = no tiene relación.
3. Context Precision (0.0 a 1.0): ¿El contexto recuperado es útil para responder? 1.0 = todo el contexto es relevante. 0.0 = nada sirve.

Devuelve ÚNICAMENTE este JSON (sin markdown, sin explicaciones previas):
{{"faithfulness": <float>, "answer_relevancy": <float>, "context_precision": <float>, "justificacion": "<string breve>"}}
"""


def parse_json_from_response(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return json.loads(text)


def run_evaluation():
    from app.agent_rag import run_rag

    print("\n" + "=" * 60)
    print("RAG Evaluation — LLM-as-a-Judge (Gemini API)")
    print("Set de preguntas BALANCEADO (12 preguntas)")
    print("=" * 60)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Preguntas: {len(EVAL_QUESTIONS)}")
    print(f"Modelo juez: {JUDGE_MODEL}")
    print("=" * 60 + "\n")

    results = []
    for i, question in enumerate(EVAL_QUESTIONS, 1):
        print(f"[{i}/{len(EVAL_QUESTIONS)}] {question[:55]}...")

        # Obtener respuesta y contextos del RAG (una sola pasada, sin repetir FAISS)
        result = run_rag(question=question, history=[])
        answer = result["answer"]
        contexts = result["contexts"]
        context_text = "\n\n---\n\n".join(
            f"CHUNK {idx+1}:\n{ctx}" for idx, ctx in enumerate(contexts)
        )

        # Construir prompt
        prompt = PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
            contexts=context_text,
        )

        # Llamada a Gemini (juez)
        print("    ⏳ Evaluando con Gemini...")
        response = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config={"temperature": 0},
        )
        raw = response.text

        try:
            scores = parse_json_from_response(raw)
            scores["pregunta"] = question
            scores["respuesta"] = answer
            scores["chunks"] = len(contexts)
            results.append(scores)
            print(f"    ✅ Faithfulness: {scores['faithfulness']:.2f} | Relevancy: {scores['answer_relevancy']:.2f} | Precision: {scores['context_precision']:.2f}")
        except Exception as e:
            print(f"    ⚠️ Error parseando respuesta: {e}")
            print(f"    Respuesta cruda: {raw[:200]}...")
            results.append({
                "pregunta": question,
                "respuesta": answer,
                "error": str(e),
                "raw": raw,
            })

        time.sleep(5)

    # Calcular promedios
    valid = [r for r in results if "error" not in r]
    if valid:
        avg_faith = sum(r["faithfulness"] for r in valid) / len(valid)
        avg_rel = sum(r["answer_relevancy"] for r in valid) / len(valid)
        avg_prec = sum(r["context_precision"] for r in valid) / len(valid)
        overall = (avg_faith + avg_rel + avg_prec) / 3
    else:
        avg_faith = avg_rel = avg_prec = overall = 0.0

    # Mostrar resultados
    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    print(f"\n{'Métrica':<25} {'Promedio':<10}")
    print("-" * 60)

    def emoji(score):
        return "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"

    print(f"{'Faithfulness':<25} {avg_faith:.3f}     {emoji(avg_faith)} ¿Basada en contexto?")
    print(f"{'Answer Relevancy':<25} {avg_rel:.3f}     {emoji(avg_rel)} ¿Responde la pregunta?")
    print(f"{'Context Precision':<25} {avg_prec:.3f}     {emoji(avg_prec)} ¿Chunks útiles?")
    print("-" * 60)
    print(f"{'Score general':<25} {overall:.3f}")

    # Mostrar detalle por pregunta
    print("\n" + "=" * 60)
    print("DETALLE POR PREGUNTA")
    print("=" * 60)
    for i, r in enumerate(valid, 1):
        cp = r["context_precision"]
        cp_emoji = "🟢" if cp >= 0.8 else "🟡" if cp >= 0.6 else "🔴"
        print(f"\nP{i}: {r['pregunta'][:60]}...")
        print(f"    Faithfulness: {r['faithfulness']:.2f} | Relevancy: {r['answer_relevancy']:.2f} | Precision: {r['context_precision']:.2f} {cp_emoji}")
        print(f"    Chunks: {r['chunks']} | {r['justificacion'][:100]}...")

    output = {
        "fecha": datetime.now().isoformat(),
        "preguntas": len(EVAL_QUESTIONS),
        "scores": {
            "faithfulness": round(avg_faith, 4),
            "answer_relevancy": round(avg_rel, 4),
            "context_precision": round(avg_prec, 4),
            "overall": round(overall, 4),
        },
        "detalle": results,
    }

    output_file = f"rag_eval_balanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Guardado en: {output_file}")

    print("\n" + "=" * 60)
    print("VEREDICTO")
    print("=" * 60)
    if overall >= 0.8:
        print("🟢 RAG de alta calidad — listo para producción")
    elif overall >= 0.6:
        print("🟡 RAG aceptable — considerar ajustar chunks o top_k")
    else:
        print("🔴 RAG necesita mejoras — revisar embeddings y prompts")

    print("\n📊 Cobertura del documento evaluada:")
    print("   ✅ Conceptos generales (SPN, nutrientes críticos)")
    print("   ✅ Criterios de exceso (azúcares, grasas, sodio)")
    print("   ✅ Reglas condicionales (edulcorantes, no alcanzados)")
    print("   ✅ Características técnicas (ADS, prohibiciones)")
    print("   ✅ Publicidad y promoción")
    print("   ✅ Cálculos y procesos")
    print("   ✅ Aplicación práctica (ejemplos)")


if __name__ == "__main__":
    run_evaluation()