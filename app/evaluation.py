"""Avaliação quantitativa reproduzível do pipeline RAG do CKP02."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import statistics
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.run_config import RunConfig

from app.chain import create_chat_llm
from app.chunking import BALANCED_512, GRANULAR_256, ChunkingConfig, gerar_chunks
from app.config import get_settings
from app.document_loader import carregar_documentos
from app.embeddings import create_embeddings, validar_embeddings_reais
from app.rag_chain import INSUFFICIENT_EVIDENCE_MESSAGE, RAGService
from app.retriever import DocumentoRecuperado
from app.vector_store import (
    DEFAULT_INDEX_DIR,
    abrir_indice,
    buscar_contexto_no_indice,
    criar_indice,
)

ROOT = Path(__file__).parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "evaluation"
DATASET_PATH = ARTIFACT_DIR / "dataset.json"
INDEX_ROOT = ROOT / "artifacts" / "indexes" / "evaluation"
CONFIGS = (GRANULAR_256, BALANCED_512)


class EvaluationError(RuntimeError):
    """Falha explícita que impede resultados parciais de parecerem completos."""


def carregar_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """Carrega e valida o dataset de referência revisado por leitura humana."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Não foi possível carregar o dataset em {path}.") from exc
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list) or len(items) < 5:
        raise EvaluationError("O dataset deve conter pelo menos cinco perguntas.")
    ids: set[str] = set()
    for item in items:
        obrigatorios = {
            "id",
            "question",
            "reference_answer",
            "expected_document_ids",
            "category",
            "answerable",
        }
        if not isinstance(item, dict) or not obrigatorios.issubset(item):
            raise EvaluationError("Há um item incompleto no dataset de avaliação.")
        if item["id"] in ids or not str(item["question"]).strip():
            raise EvaluationError("IDs devem ser únicos e perguntas não podem ser vazias.")
        ids.add(str(item["id"]))
    return items


def _document_id(chunk_id: str) -> str:
    return chunk_id.rsplit("-C", 1)[0]


def _search_for_store(store: Any):
    def search(
        question: str,
        *,
        top_k: int,
        filtros: dict[str, Any] | None = None,
    ) -> Sequence[DocumentoRecuperado]:
        return buscar_contexto_no_indice(
            store,
            question,
            top_k,
            filtros,
            vizinhos=1,
        )

    return search


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metricas_ragas(
    records: list[dict[str, Any]],
    llm: Any,
    embeddings: Any,
) -> list[dict[str, float | None]]:
    samples = [
        SingleTurnSample(
            user_input=record["question"],
            response=record["answer"],
            retrieved_contexts=record["contexts"],
            reference=record["reference_answer"],
        )
        for record in records
    ]
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[Faithfulness(), ResponseRelevancy(strictness=3)],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        run_config=RunConfig(
            timeout=240,
            max_retries=3,
            max_wait=30,
            max_workers=2,
            seed=42,
        ),
        raise_exceptions=False,
        show_progress=True,
    )
    frame = result.to_pandas()
    return [
        {
            "faithfulness": _safe_float(row.get("faithfulness")),
            "answer_relevancy": _safe_float(row.get("answer_relevancy")),
        }
        for row in frame.to_dict(orient="records")
    ]


def _media_desvio(rows: list[dict[str, Any]], field: str) -> tuple[float | None, float | None]:
    values = [value for row in rows if (value := _safe_float(row.get(field))) is not None]
    if not values:
        return None, None
    return statistics.fmean(values), statistics.pstdev(values)


def _fmt(value: Any, digits: int = 3) -> str:
    number = _safe_float(value)
    return "N/D" if number is None else f"{number:.{digits}f}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _gerar_grafico(summaries: list[dict[str, Any]], path: Path) -> None:
    """Gera PNG sem depender de backend gráfico do sistema operacional."""

    width, height = 1200, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(font_path, size=20)
        small = ImageFont.truetype(font_path, size=16)
    except OSError:
        font = ImageFont.load_default(size=20)
        small = ImageFont.load_default(size=16)
    draw.text((60, 35), "Comparação de chunking - RAGAS", fill="#172554", font=font)
    left, top, right, bottom = 100, 120, 1140, 610
    draw.line((left, top, left, bottom), fill="#334155", width=2)
    draw.line((left, bottom, right, bottom), fill="#334155", width=2)
    for step in range(6):
        value = step / 5
        y = bottom - int(value * (bottom - top))
        draw.line((left, y, right, y), fill="#e2e8f0", width=1)
        draw.text((45, y - 9), f"{value:.1f}", fill="#475569", font=small)
    colors = {"faithfulness_mean": "#2563eb", "answer_relevancy_mean": "#16a34a"}
    group_width = (right - left) / max(1, len(summaries))
    bar_width = min(110, group_width / 3)
    for index, summary in enumerate(summaries):
        center = left + group_width * (index + 0.5)
        for offset, field in ((-bar_width / 2, "faithfulness_mean"), (bar_width / 2, "answer_relevancy_mean")):
            value = _safe_float(summary.get(field)) or 0.0
            x0 = center + offset - bar_width / 2
            x1 = x0 + bar_width
            y0 = bottom - value * (bottom - top)
            draw.rectangle((x0, y0, x1, bottom), fill=colors[field])
            draw.text((x0 + 8, y0 - 25), f"{value:.3f}", fill="#0f172a", font=small)
        label = f"{summary['chunk_size']} / {summary['chunk_overlap']}"
        draw.text((center - 55, bottom + 18), label, fill="#0f172a", font=small)
    draw.text((370, 665), "Azul: faithfulness", fill=colors["faithfulness_mean"], font=small)
    draw.text((650, 665), "Verde: answer relevancy", fill=colors["answer_relevancy_mean"], font=small)
    image.save(path)


def _escolher_vencedora(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aplica a regra documentada sem esconder ausência de métricas."""

    valid = [row for row in summaries if row["faithfulness_mean"] is not None]
    if not valid:
        raise EvaluationError("Nenhuma configuração produziu faithfulness válida.")
    return max(
        valid,
        key=lambda row: (
            row["faithfulness_mean"],
            row["answer_relevancy_mean"] or -1,
            row["expected_source_recall_mean"],
            -row["answer_latency_mean_seconds"],
        ),
    )


def executar_avaliacao(
    *,
    configs: Sequence[ChunkingConfig] = CONFIGS,
    artifact_dir: Path = ARTIFACT_DIR,
) -> dict[str, Any]:
    """Executa indexação, respostas e métricas reais para todas as configurações."""

    if len(configs) < 2:
        raise ValueError("A fase 4 exige pelo menos duas configurações de chunking.")
    dataset = carregar_dataset(artifact_dir / "dataset.json")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    response_dir = artifact_dir / "respostas"
    response_dir.mkdir(parents=True, exist_ok=True)

    documents = carregar_documentos()
    embeddings = create_embeddings()
    embedding_dimension = validar_embeddings_reais(embeddings)
    llm = create_chat_llm(temperature=0)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for config in configs:
        chunks = gerar_chunks(documents, config)
        # A configuração granular já foi indexada e validada na fase 2. Reutilizar
        # exatamente esse índice evita repetir milhares de embeddings sem alterar
        # qualquer variável do experimento; as demais estratégias têm índice próprio.
        index_dir = (
            DEFAULT_INDEX_DIR if config == GRANULAR_256 else INDEX_ROOT / config.name
        )
        start_index = perf_counter()
        if index_dir.exists() and any(index_dir.iterdir()):
            store = abrir_indice(
                embeddings,
                config=config,
                persist_directory=index_dir,
            )
            stored_count = store._collection.count()
            if stored_count != len(chunks):
                raise EvaluationError(
                    f"O índice existente de {config.name} tem {stored_count} registros, "
                    f"mas a configuração atual gera {len(chunks)}. Reconstrua esse índice."
                )
        else:
            store = criar_indice(
                chunks,
                embeddings,
                config=config,
                persist_directory=index_dir,
            )
        index_seconds = perf_counter() - start_index
        service = RAGService(llm=llm, search=_search_for_store(store), top_k=5)
        generated: list[dict[str, Any]] = []
        for item in dataset:
            start_answer = perf_counter()
            try:
                result = service.answer(item["question"])
            except Exception as exc:
                failures.append(
                    {
                        "config": config.name,
                        "question_id": item["id"],
                        "stage": "retrieve_generate",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                raise EvaluationError(
                    f"Falha em {config.name}/{item['id']}; resultados incompletos não foram consolidados."
                ) from exc
            elapsed = perf_counter() - start_answer
            retrieved_ids = sorted({_document_id(doc.chunk_id) for doc in result.contextos_recuperados})
            expected = set(item["expected_document_ids"])
            recall = len(expected.intersection(retrieved_ids)) / len(expected) if expected else 1.0
            generated.append(
                {
                    **item,
                    "answer": result.resposta,
                    "formatted_answer": result.texto_formatado,
                    "contexts": [doc.conteudo for doc in result.contextos_recuperados],
                    "retrieved_chunks": [doc.as_dict() for doc in result.contextos_recuperados],
                    "cited_chunks": [doc.as_dict() for doc in result.fontes],
                    "retrieved_document_ids": retrieved_ids,
                    "expected_source_recall": recall,
                    "answer_latency_seconds": elapsed,
                }
            )
        response_path = response_dir / f"{config.name}.json"
        response_path.write_text(
            json.dumps(generated, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        metric_rows = _metricas_ragas(generated, llm, embeddings)
        if len(metric_rows) != len(generated):
            raise EvaluationError("RAGAS retornou quantidade inesperada de resultados.")
        config_rows: list[dict[str, Any]] = []
        for record, metrics in zip(generated, metric_rows):
            row = {
                "question_id": record["id"],
                "question": record["question"],
                "category": record["category"],
                "answerable": record["answerable"],
                "chunking_config": config.name,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "faithfulness": metrics["faithfulness"],
                "answer_relevancy": metrics["answer_relevancy"],
                "expected_source_recall": record["expected_source_recall"],
                "expected_document_ids": "|".join(record["expected_document_ids"]),
                "retrieved_document_ids": "|".join(record["retrieved_document_ids"]),
                "answer_latency_seconds": record["answer_latency_seconds"],
                "insufficient_evidence": (
                    record["answer"] == INSUFFICIENT_EVIDENCE_MESSAGE
                ),
                "cited_chunk_ids": "|".join(
                    chunk["chunk_id"] for chunk in record["cited_chunks"]
                ),
            }
            config_rows.append(row)
            all_rows.append(row)
        # RAGAS mede a qualidade de respostas fundamentadas. Uma recusa correta em uma
        # pergunta deliberadamente fora da base recebe zero de relevância/faithfulness,
        # por isso perguntas não respondíveis são avaliadas separadamente por recusa.
        answerable_rows = [row for row in config_rows if row["answerable"]]
        unanswerable_rows = [row for row in config_rows if not row["answerable"]]
        faith_mean, faith_std = _media_desvio(answerable_rows, "faithfulness")
        relevancy_mean, relevancy_std = _media_desvio(answerable_rows, "answer_relevancy")
        summaries.append(
            {
                "chunking_config": config.name,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "indexed_chunks": len(chunks),
                "mean_chunk_characters": statistics.fmean(len(c.page_content) for c in chunks),
                "indexing_seconds": index_seconds,
                "faithfulness_mean": faith_mean,
                "faithfulness_std": faith_std,
                "answer_relevancy_mean": relevancy_mean,
                "answer_relevancy_std": relevancy_std,
                "expected_source_recall_mean": statistics.fmean(
                    row["expected_source_recall"] for row in answerable_rows
                ),
                "answer_latency_mean_seconds": statistics.fmean(
                    row["answer_latency_seconds"] for row in config_rows
                ),
                "valid_faithfulness_scores": sum(
                    row["faithfulness"] is not None for row in answerable_rows
                ),
                "valid_answer_relevancy_scores": sum(
                    row["answer_relevancy"] is not None for row in answerable_rows
                ),
                "unanswerable_refusal_rate": (
                    statistics.fmean(
                        float(row["insufficient_evidence"])
                        for row in unanswerable_rows
                    )
                    if unanswerable_rows
                    else None
                ),
            }
        )

    winner = _escolher_vencedora(summaries)
    result_fields = [
        "question_id", "question", "category", "answerable", "chunking_config",
        "chunk_size", "chunk_overlap", "faithfulness", "answer_relevancy",
        "expected_source_recall", "expected_document_ids", "retrieved_document_ids",
        "answer_latency_seconds", "insufficient_evidence", "cited_chunk_ids",
    ]
    summary_fields = list(summaries[0])
    _write_csv(artifact_dir / "resultados_por_pergunta.csv", all_rows, result_fields)
    _write_csv(artifact_dir / "resumo_por_configuracao.csv", summaries, summary_fields)
    _gerar_grafico(summaries, artifact_dir / "comparacao_chunking.png")

    settings = get_settings()
    manifest_bytes = (ROOT / "data" / "manifest.json").read_bytes()
    prompt_bytes = b"\n".join(
        (ROOT / "prompts" / name).read_bytes()
        for name in ("rag_system_v1.md", "rag_human_v1.md")
    )
    execution = {
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "real_execution": True,
        "question_count": len(dataset),
        "chat_model": settings.ollama_model,
        "embedding_model": settings.ollama_embedding_model,
        "embedding_mode": "local",
        "embedding_dimension": embedding_dimension,
        "temperature": 0,
        "top_k": 5,
        "neighbor_chunks": 1,
        "minimum_score": 0.2,
        "chunking_configs": [
            {"name": c.name, "chunk_size": c.chunk_size, "chunk_overlap": c.chunk_overlap}
            for c in configs
        ],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "rag_prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "dependency_versions": {
            name: importlib.metadata.version(name)
            for name in ("ragas", "langchain", "langchain-ollama", "langchain-chroma", "chromadb")
        },
        "failures": failures,
        "limitations": [
            "As métricas dependem de julgamentos do gemma4:cloud e podem variar em nova execução.",
            "Embeddings locais são a adaptação documentada ao 401 do endpoint cloud.",
            "Valores N/D são preservados e excluídos das médias; não são convertidos em zero.",
        ],
    }
    (artifact_dir / "configuracao_execucao.json").write_text(
        json.dumps(execution, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# Relatório da fase 4 - avaliação quantitativa",
        "",
        f"Execução real em {execution['executed_at']}, com {len(dataset)} perguntas.",
        "",
        "| Configuração | Chunks | Faithfulness | Answer relevancy | Recall de fontes | Latência média (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        report_lines.append(
            f"| {row['chunking_config']} | {row['indexed_chunks']} | "
            f"{_fmt(row['faithfulness_mean'])} ± {_fmt(row['faithfulness_std'])} | "
            f"{_fmt(row['answer_relevancy_mean'])} ± {_fmt(row['answer_relevancy_std'])} | "
            f"{_fmt(row['expected_source_recall_mean'])} | "
            f"{_fmt(row['answer_latency_mean_seconds'])} |"
        )
    threshold_note = (
        "atingiu" if (winner["faithfulness_mean"] or 0) >= 0.7 else "não atingiu"
    )
    report_lines.extend(
        [
            "",
            "## Escolha",
            "",
            (
                f"A configuração **{winner['chunking_config']}** venceu pela regra ordenada "
                "(faithfulness, answer relevancy, recuperação de fontes e latência). "
                f"Seu faithfulness médio {threshold_note} a meta de 0,7."
            ),
            "",
            (
                "A escolha considera somente métricas válidas. Respostas, contextos, citações e "
                "eventuais valores N/D permanecem nos artefatos para auditoria."
            ),
            "",
        ]
    )
    report_lines.extend(
        [
            "## Critério das médias",
            "",
            (
                "As médias RAGAS usam somente as sete perguntas respondíveis. A pergunta "
                "deliberadamente fora da base permanece na tabela por pergunta e é medida "
                "separadamente pela taxa de recusa, pois uma recusa correta recebe zero no "
                "RAGAS e distorceria a qualidade das respostas fundamentadas."
            ),
            "",
        ]
    )
    (artifact_dir / "relatorio.md").write_text("\n".join(report_lines), encoding="utf-8")
    return {"summaries": summaries, "winner": winner, "execution": execution}


def main() -> None:
    try:
        result = executar_avaliacao()
    except Exception as exc:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        failure = {
            "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        (ARTIFACT_DIR / "falha.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc
    else:
        (ARTIFACT_DIR / "falha.json").unlink(missing_ok=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
