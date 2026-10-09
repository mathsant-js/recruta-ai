"""Testes do reranking sem download do cross-encoder real."""

import pytest

from app.reranker import CrossEncoderReranker, expand_selected_neighbors
from app.retriever import DocumentoRecuperado


class FakeCrossEncoder:
    def predict(self, sentences: list[list[str]]) -> list[float]:
        assert sentences[0][0] == "privacidade"
        return [0.1, 0.9, 0.4]


def document(chunk_id: str) -> DocumentoRecuperado:
    return DocumentoRecuperado(
        conteudo=f"conteúdo {chunk_id}",
        score=0.5,
        titulo="Documento de teste",
        fonte="https://example.invalid",
        pagina=1,
        categoria="privacidade",
        chunk_id=chunk_id,
    )


def test_cross_encoder_reordena_e_limita_resultados() -> None:
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    result = reranker.rerank(
        "privacidade", [document("C1"), document("C2"), document("C3")], top_n=2
    )
    assert [item.chunk_id for item in result] == ["C2", "C3"]


def test_reranker_valida_entradas() -> None:
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    with pytest.raises(ValueError, match="consulta"):
        reranker.rerank(" ", [document("C1")])
    with pytest.raises(ValueError, match="top_n"):
        reranker.rerank("pergunta", [document("C1")], top_n=0)


def test_expansao_recompoe_vizinhos_disponiveis() -> None:
    candidates = [
        document("DOC-01-C0001"),
        document("DOC-01-C0002"),
        document("DOC-01-C0003"),
        document("DOC-02-C0001"),
    ]
    result = expand_selected_neighbors([candidates[1]], candidates)
    assert [item.chunk_id for item in result] == [
        "DOC-01-C0001",
        "DOC-01-C0002",
        "DOC-01-C0003",
    ]
