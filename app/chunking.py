"""Divisao reproduzivel dos documentos da base em chunks inspecionaveis."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Parametros de uma estrategia de chunking aprovada pelo enunciado."""

    name: str
    chunk_size: int
    chunk_overlap: int

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size deve ser positivo.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap deve estar entre zero e chunk_size - 1.")
        percentual = self.chunk_overlap / self.chunk_size
        if not 0.10 <= percentual <= 0.15:
            raise ValueError("chunk_overlap deve representar de 10% a 15% do chunk_size.")


GRANULAR_256 = ChunkingConfig("granular_256", 256, 32)
BALANCED_512 = ChunkingConfig("equilibrada_512", 512, 64)
WIDE_1024 = ChunkingConfig("ampla_1024", 1024, 128)


def gerar_chunks(
    documentos: list[Document],
    config: ChunkingConfig = GRANULAR_256,
) -> list[Document]:
    """Aplica o splitter literal do PDF e atribui IDs estaveis por documento."""

    if not documentos:
        raise ValueError("A lista de documentos nao pode estar vazia.")
    splitter = RecursiveCharacterTextSplitter(
        separators=SEPARATORS,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    contadores: dict[str, int] = {}
    chunks: list[Document] = []
    for chunk in splitter.split_documents(documentos):
        document_id = str(chunk.metadata["document_id"])
        contadores[document_id] = contadores.get(document_id, 0) + 1
        chunk_id = f"{document_id}-C{contadores[document_id]:04d}"
        chunk.metadata.update(
            {
                "chunk_id": chunk_id,
                "chunking_config": config.name,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
            }
        )
        chunks.append(chunk)
    return chunks
