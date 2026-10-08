"""Comparação executável dos diferenciais de recuperação da fase 5."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from app.reranker import CrossEncoderReranker, expand_selected_neighbors
from app.retriever import DocumentoRecuperado, buscar

ROOT = Path(__file__).parents[1]
DATASET_PATH = ROOT / "artifacts" / "evaluation" / "dataset.json"
MANIFEST_PATH = ROOT / "data" / "manifest.json"
OUTPUT_DIR = ROOT / "artifacts" / "phase5"


def _document_id(item: DocumentoRecuperado) -> str:
    return item.chunk_id.rsplit("-C", 1)[0]


def _recall(items: list[DocumentoRecuperado], expected: set[str]) -> float:
    if not expected:
        return 1.0
    found = {_document_id(item) for item in items}
    return len(found & expected) / len(expected)


def _timed_search(question: str, *, top_k: int, category: str | None = None):
    started = perf_counter()
    filters = {"category": category} if category else None
    items = buscar(question, top_k=top_k, filtros=filters)
    return items, perf_counter() - started


def run() -> list[dict[str, object]]:
    """Executa busca base, filtro e reranking sobre as mesmas perguntas."""

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["items"]
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["documents"]
    category_by_document = {
        item["document_id"]: item["category"] for item in manifest
    }
    reranker = CrossEncoderReranker()
    rows: list[dict[str, object]] = []

    # O aquecimento retira abertura do Chroma e carga do modelo das latências,
    # evitando favorecer o método executado por último em cada comparação.
    warmup = buscar(dataset[0]["question"], top_k=2)
    reranker.rerank(dataset[0]["question"], warmup, top_n=1)

    for item in dataset:
        if not item["answerable"]:
            continue
        expected = set(item["expected_document_ids"])
        categories = {category_by_document[document_id] for document_id in expected}
        category = categories.pop() if len(categories) == 1 else None

        baseline, baseline_seconds = _timed_search(item["question"], top_k=5)
        candidates, retrieval_seconds = _timed_search(item["question"], top_k=10)
        started = perf_counter()
        selected = reranker.rerank(item["question"], candidates, top_n=5)
        reranked = expand_selected_neighbors(selected, candidates)
        rerank_seconds = perf_counter() - started

        filtered: list[DocumentoRecuperado] = []
        filter_seconds: float | None = None
        if category:
            filtered, filter_seconds = _timed_search(
                item["question"], top_k=5, category=category
            )

        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_documents": ",".join(sorted(expected)),
                "filter_category": category or "",
                "baseline_recall": _recall(baseline, expected),
                "filtered_recall": _recall(filtered, expected) if category else "",
                "reranked_recall": _recall(reranked, expected),
                "baseline_latency_seconds": round(baseline_seconds, 4),
                "filtered_latency_seconds": (
                    round(filter_seconds, 4) if filter_seconds is not None else ""
                ),
                "reranked_latency_seconds": round(
                    retrieval_seconds + rerank_seconds, 4
                ),
                "baseline_ids": ",".join(doc.chunk_id for doc in baseline),
                "filtered_ids": ",".join(doc.chunk_id for doc in filtered),
                "reranked_ids": ",".join(doc.chunk_id for doc in reranked),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "comparacao_recuperacao.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "executed_at": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
        "real_execution": True,
        "questions": len(rows),
        "cross_encoder": reranker.model_name,
        "baseline_top_k": 5,
        "reranking_candidates": 10,
        "reranking_top_n": 5,
        "comparison_metric": "recall de documentos esperados e latência",
    }
    (OUTPUT_DIR / "configuracao_execucao.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    baseline_recall = sum(float(row["baseline_recall"]) for row in rows) / len(rows)
    reranked_recall = sum(float(row["reranked_recall"]) for row in rows) / len(rows)
    baseline_latency = sum(
        float(row["baseline_latency_seconds"]) for row in rows
    ) / len(rows)
    reranked_latency = sum(
        float(row["reranked_latency_seconds"]) for row in rows
    ) / len(rows)
    filtered_rows = [row for row in rows if row["filtered_recall"] != ""]
    filtered_recall = sum(
        float(row["filtered_recall"]) for row in filtered_rows
    ) / len(filtered_rows)
    report = (
        "# Fase 5 — comparação dos diferenciais\n\n"
        f"Execução real em `{metadata['executed_at']}` sobre {len(rows)} perguntas "
        "respondíveis. O índice e o cross-encoder foram aquecidos antes da medição.\n\n"
        "| Estratégia | Perguntas | Recall médio de fontes | Latência média |\n"
        "|---|---:|---:|---:|\n"
        f"| Busca vetorial | {len(rows)} | {baseline_recall:.3f} | "
        f"{baseline_latency:.3f} s |\n"
        f"| Cross-encoder | {len(rows)} | {reranked_recall:.3f} | "
        f"{reranked_latency:.3f} s |\n"
        f"| Filtro por metadata | {len(filtered_rows)} | {filtered_recall:.3f} | "
        "ver CSV |\n\n"
        "O filtro foi avaliado apenas quando todos os documentos esperados tinham a "
        "mesma categoria. O cross-encoder usa 10 candidatos, seleciona 5 hits e "
        "reanexa seus vizinhos disponíveis. "
        "O CSV preserva IDs, recalls e latências por pergunta. Recall não mede "
        "faithfulness da resposta; ele compara somente a etapa de recuperação.\n"
    )
    (OUTPUT_DIR / "relatorio.md").write_text(report, encoding="utf-8")
    return rows


if __name__ == "__main__":
    results = run()
    print(f"Fase 5 concluída para {len(results)} perguntas.")
