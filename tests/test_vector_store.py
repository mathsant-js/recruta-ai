"""Testes locais do Chroma sem chamadas ao Ollama Cloud."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.chunking import GRANULAR_256
from app.vector_store import buscar_contexto_no_indice, buscar_no_indice, criar_indice


class EmbeddingsDeterministicos(Embeddings):
    """Vetores controlados validam persistencia e adaptacao, nao relevancia real."""

    def embed_documents(self, textos: list[str]) -> list[list[float]]:
        return [self.embed_query(texto) for texto in textos]

    def embed_query(self, texto: str) -> list[float]:
        texto = texto.lower()
        return [
            float("dados" in texto or "privacidade" in texto),
            float("entrevista" in texto),
            0.1,
        ]


def _chunk(chunk_id: str, texto: str, categoria: str) -> Document:
    return Document(
        page_content=texto,
        metadata={
            "document_id": "TESTE",
            "source": "https://example.invalid/fonte",
            "title": "Documento ficticio de teste",
            "organization": "Organizacao ficticia",
            "document_type": "manual",
            "category": categoria,
            "publication_date": "2026",
            "language": "pt-BR",
            "page": 1,
            "chunk_id": chunk_id,
            "chunking_config": GRANULAR_256.name,
            "chunk_size": 256,
            "chunk_overlap": 32,
            "start_index": 0,
        },
    )


def test_indexa_busca_filtra_e_adapta_contrato(tmp_path: Path) -> None:
    chunks = [
        _chunk("TESTE-C0001", "Minimizacao de dados pessoais.", "privacidade"),
        _chunk("TESTE-C0002", "Entrevista estruturada.", "entrevista"),
    ]
    store = criar_indice(
        chunks,
        EmbeddingsDeterministicos(),
        persist_directory=tmp_path / "indice",
        batch_size=1,
    )

    resultados = buscar_no_indice(
        store,
        "Como proteger dados?",
        top_k=2,
        filtros={"category": "privacidade"},
    )

    assert len(resultados) == 1
    assert resultados[0].chunk_id == "TESTE-C0001"
    assert resultados[0].pagina == 1
    assert 0 <= resultados[0].score <= 1


def test_busca_de_contexto_inclui_vizinho_com_citacao_propria(tmp_path: Path) -> None:
    chunks = [
        _chunk("TESTE-C0001", "Diretrizes de privacidade devem adotar uma abordagem", "privacidade"),
        _chunk("TESTE-C0002", "estruturada e alinhada aos riscos.", "privacidade"),
        _chunk("TESTE-C0003", "Assunto sem relação.", "privacidade"),
    ]
    store = criar_indice(
        chunks,
        EmbeddingsDeterministicos(),
        persist_directory=tmp_path / "indice-vizinhos",
    )

    resultados = buscar_contexto_no_indice(
        store,
        "privacidade de dados",
        top_k=1,
        vizinhos=1,
    )

    assert [item.chunk_id for item in resultados] == ["TESTE-C0001", "TESTE-C0002"]
    assert resultados[1].conteudo == "estruturada e alinhada aos riscos."
