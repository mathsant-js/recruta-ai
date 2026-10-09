"""Testes do splitter exigido pelo CKP02."""

import pytest
from langchain_core.documents import Document

from app.chunking import GRANULAR_256, ChunkingConfig, gerar_chunks


def test_gera_chunks_inspecionaveis_com_ids_unicos() -> None:
    documento = Document(
        page_content=("recrutamento inclusivo e protecao de dados. " * 30).strip(),
        metadata={"document_id": "TESTE", "page": 1, "title": "Teste"},
    )

    chunks = gerar_chunks([documento], GRANULAR_256)

    assert len(chunks) > 1
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)
    assert all(len(chunk.page_content) <= 256 for chunk in chunks)
    assert all(chunk.metadata["chunking_config"] == "granular_256" for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (100, 9), (100, 16), (100, 100)],
)
def test_rejeita_configuracao_fora_do_enunciado(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig("invalida", size, overlap)
