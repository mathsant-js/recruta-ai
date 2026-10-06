"""Comando reproduzivel da fase 2: load, split, embed, store e retrieve."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from app.chunking import GRANULAR_256, gerar_chunks
from app.config import get_settings
from app.document_loader import carregar_documentos
from app.embeddings import create_embeddings, validar_embeddings_reais
from app.vector_store import DEFAULT_INDEX_DIR, buscar_no_indice, criar_indice

ROOT = Path(__file__).parents[1]
REPORT_PATH = ROOT / "artifacts" / "ingestion" / "fase2.json"
FAILURE_REPORT_PATH = ROOT / "artifacts" / "ingestion" / "fase2_falha.json"
SMOKE_QUERY = "Como deve ser feita a protecao de dados pessoais de candidatos?"


def executar_ingestao(
    *,
    index_dir: Path = DEFAULT_INDEX_DIR,
    report_path: Path = REPORT_PATH,
) -> dict[str, object]:
    """Executa chamadas reais e registra somente metadata segura e resultados verificaveis."""

    inicio = perf_counter()
    documentos = carregar_documentos()
    chunks = gerar_chunks(documentos, GRANULAR_256)
    embeddings = create_embeddings()
    dimensao = validar_embeddings_reais(embeddings)
    store = criar_indice(chunks, embeddings, config=GRANULAR_256, persist_directory=index_dir)
    resultados = buscar_no_indice(store, SMOKE_QUERY, top_k=5)
    if not resultados:
        raise RuntimeError("A busca semantica real nao retornou resultados.")

    settings = get_settings()
    relatorio: dict[str, object] = {
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "embedding_model": settings.ollama_embedding_model,
        "embedding_host": settings.ollama_embedding_host,
        "execution_mode": "local_embeddings",
        "embedding_dimension": dimensao,
        "chunking": {
            "name": GRANULAR_256.name,
            "chunk_size": GRANULAR_256.chunk_size,
            "chunk_overlap": GRANULAR_256.chunk_overlap,
        },
        "loaded_pages": len(documentos),
        "indexed_chunks": len(chunks),
        "collection": store._collection.name,
        "query": SMOKE_QUERY,
        "results": [resultado.as_dict() for resultado in resultados],
        "elapsed_seconds": round(perf_counter() - inicio, 3),
        "limitations": [
            "A dimensao e os resultados refletem uma unica execucao real.",
            "O indice local e ignorado pelo Git e deve ser reconstruido no ambiente de avaliacao.",
            "Embeddings locais sao uma adaptacao ao 401 recebido pelo endpoint cloud.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return relatorio


def main() -> None:
    try:
        relatorio = executar_ingestao()
    except Exception as exc:
        # A evidencia de falha evita que indisponibilidade cloud pareca execucao omitida.
        falha = {
            "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "failed",
            "stage": "embedding_validation",
            "embedding_model": get_settings().ollama_embedding_model,
            "embedding_host": get_settings().ollama_embedding_host,
            "execution_mode": "local_embeddings",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "documents_sent": False,
            "note": (
                "A validacao curta ocorre antes da indexacao; nenhum PDF foi enviado "
                "quando esta etapa falha."
            ),
        }
        FAILURE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        FAILURE_REPORT_PATH.write_text(
            json.dumps(falha, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(falha, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc
    else:
        FAILURE_REPORT_PATH.unlink(missing_ok=True)
        print(json.dumps(relatorio, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
