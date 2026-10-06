"""Persistencia local e busca semantica da primeira configuracao do CKP02."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.chunking import GRANULAR_256, ChunkingConfig
from app.retriever import DocumentoRecuperado, FiltroMetadata

ROOT = Path(__file__).parents[1]
DEFAULT_INDEX_DIR = ROOT / "artifacts" / "indexes" / GRANULAR_256.name


def collection_name(config: ChunkingConfig) -> str:
    """Mantem nomes validos, explicitos e relacionados ao dominio."""

    return f"recruta_ai_{config.chunk_size}"


def criar_indice(
    chunks: list[Document],
    embeddings: Embeddings,
    *,
    config: ChunkingConfig = GRANULAR_256,
    persist_directory: Path = DEFAULT_INDEX_DIR,
    batch_size: int = 64,
) -> Chroma:
    """Cria uma colecao Chroma persistente com distancia cosseno."""

    if not chunks:
        raise ValueError("Nao e possivel indexar uma lista vazia de chunks.")
    persist_directory.mkdir(parents=True, exist_ok=True)
    ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Os chunk_ids devem ser unicos.")
    if batch_size < 1:
        raise ValueError("batch_size deve ser positivo.")
    store = Chroma(
        collection_name=collection_name(config),
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
        collection_metadata={"hnsw:space": "cosine"},
    )
    # Lotes pequenos evitam um payload unico com milhares de chunks na API cloud.
    for inicio in range(0, len(chunks), batch_size):
        fim = inicio + batch_size
        store.add_documents(documents=chunks[inicio:fim], ids=ids[inicio:fim])
    return store


def abrir_indice(
    embeddings: Embeddings,
    *,
    config: ChunkingConfig = GRANULAR_256,
    persist_directory: Path = DEFAULT_INDEX_DIR,
) -> Chroma:
    """Abre o indice ja construido e falha claramente quando ele nao existe."""

    if not persist_directory.exists() or not any(persist_directory.iterdir()):
        raise FileNotFoundError(
            f"Indice ausente em {persist_directory}. Execute python -m app.ingestion."
        )
    return Chroma(
        collection_name=collection_name(config),
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def buscar_no_indice(
    store: Chroma,
    consulta: str,
    top_k: int,
    filtros: FiltroMetadata | None = None,
) -> list[DocumentoRecuperado]:
    """Adapta resultados do Chroma ao contrato publico do futuro agente."""

    filtro: dict[str, Any] | None = dict(filtros) if filtros else None
    pares = store.similarity_search_with_relevance_scores(
        consulta,
        k=top_k,
        filter=filtro,
    )
    return [
        DocumentoRecuperado(
            conteudo=documento.page_content,
            score=float(score),
            titulo=str(documento.metadata["title"]),
            fonte=str(documento.metadata["source"]),
            pagina=int(documento.metadata["page"]) if documento.metadata.get("page") else None,
            categoria=str(documento.metadata["category"]),
            chunk_id=str(documento.metadata["chunk_id"]),
        )
        for documento, score in pares
    ]
