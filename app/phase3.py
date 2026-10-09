"""Execução reproduzível das cinco perguntas reais exigidas na fase 3."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from app.config import get_settings
from app.rag_chain import RAGService

ROOT = Path(__file__).parents[1]
REPORT_PATH = ROOT / "artifacts" / "rag" / "fase3.json"

QUESTIONS = (
    "Qual é o prazo mínimo de inscrição indicado para o processo seletivo?",
    "Em quais situações um processo seletivo pode ser cancelado por ausência ou inadequação de candidatos?",
    "Quais informações devem constar no formulário de demanda de recrutamento e seleção?",
    "Quais práticas ajudam a estruturar uma entrevista por competências?",
    "Quais princípios devem orientar um processo seletivo público?",
)


def executar_fase3(report_path: Path = REPORT_PATH) -> dict[str, object]:
    """Executa retrieve → generate cinco vezes e preserva respostas e fontes reais."""

    service = RAGService()
    resultados: list[dict[str, object]] = []
    inicio_total = perf_counter()
    for pergunta in QUESTIONS:
        inicio = perf_counter()
        resposta = service.answer(pergunta)
        registro = resposta.as_dict()
        registro["elapsed_seconds"] = round(perf_counter() - inicio, 3)
        resultados.append(registro)

    settings = get_settings()
    relatorio: dict[str, object] = {
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "chat_model": settings.ollama_model,
        "embedding_model": settings.ollama_embedding_model,
        "temperature": 0,
        "top_k": service.top_k,
        "minimum_score": service.minimum_score,
        "question_count": len(resultados),
        "results": resultados,
        "elapsed_seconds": round(perf_counter() - inicio_total, 3),
        "limitations": [
            "As respostas refletem uma execução real e podem variar em nova chamada.",
            "A validação garante que toda citação corresponda a um chunk recuperado.",
            "A qualidade semântica será medida com RAGAS somente na fase 4.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return relatorio


def main() -> None:
    print(json.dumps(executar_fase3(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
