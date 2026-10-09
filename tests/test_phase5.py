"""Testes das métricas auxiliares da fase 5."""

from app.phase5 import _document_id, _recall
from app.retriever import DocumentoRecuperado


def document(chunk_id: str) -> DocumentoRecuperado:
    return DocumentoRecuperado(
        conteudo="conteúdo",
        score=0.8,
        titulo="título",
        fonte="https://example.invalid",
        pagina=1,
        categoria="teste",
        chunk_id=chunk_id,
    )


def test_document_id_e_recall() -> None:
    items = [document("DOC-01-C0001"), document("DOC-03-C0012")]
    assert _document_id(items[0]) == "DOC-01"
    assert _recall(items, {"DOC-01", "DOC-02"}) == 0.5
    assert _recall([], set()) == 1.0
