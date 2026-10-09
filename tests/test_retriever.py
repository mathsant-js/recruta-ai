"""Testes determinísticos do contrato público de recuperação do CKP02."""

import pytest
from types import ModuleType
import sys

import app.retriever as retriever
from app.retriever import DocumentoRecuperado, RecuperadorNaoConfiguradoError


@pytest.fixture(autouse=True)
def restaurar_backend() -> None:
    """Evita que a injeção de um teste vaze para os demais."""

    backend_original = retriever._backend_busca
    yield
    retriever._backend_busca = backend_original


def test_importacao_nao_inicializa_indice() -> None:
    assert retriever._backend_busca is retriever._backend_padrao


def test_backend_padrao_expoe_erro_claro_sem_indice(monkeypatch) -> None:
    embeddings = ModuleType("app.embeddings")
    embeddings.create_embeddings = lambda: object()  # type: ignore[attr-defined]
    vector_store = ModuleType("app.vector_store")
    vector_store.index_directory = lambda _: object()  # type: ignore[attr-defined]
    vector_store.abrir_indice = (  # type: ignore[attr-defined]
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("índice ausente"))
    )
    vector_store.buscar_contexto_no_indice = lambda *args: []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app.embeddings", embeddings)
    monkeypatch.setitem(sys.modules, "app.vector_store", vector_store)
    with pytest.raises(RecuperadorNaoConfiguradoError, match="app.ingestion"):
        retriever.buscar("Como proteger dados de candidatos?")


def test_buscar_normaliza_entrada_e_retorna_tipo_publico() -> None:
    chamadas: list[tuple[str, int, object]] = []
    documento = DocumentoRecuperado(
        conteudo="Colete apenas os dados necessários para a finalidade informada.",
        score=0.91,
        titulo="Guia fictício para teste unitário",
        fonte="https://example.invalid/guia",
        pagina=3,
        categoria="privacidade",
        chunk_id="TESTE-DOC-C001",
    )

    def backend(consulta: str, top_k: int, filtros: object) -> list[DocumentoRecuperado]:
        chamadas.append((consulta, top_k, filtros))
        return [documento]

    retriever.configurar_backend_busca(backend)
    resultado = retriever.buscar(
        "  minimização de dados  ",
        top_k=3,
        filtros={"category": "privacidade"},
    )

    assert resultado == [documento]
    assert chamadas == [
        ("minimização de dados", 3, {"category": "privacidade"})
    ]


@pytest.mark.parametrize("consulta", ["", "   "])
def test_buscar_rejeita_consulta_vazia(consulta: str) -> None:
    with pytest.raises(ValueError, match="não pode estar vazia"):
        retriever.buscar(consulta)


def test_buscar_rejeita_top_k_invalido() -> None:
    with pytest.raises(ValueError, match="maior ou igual a 1"):
        retriever.buscar("consulta", top_k=0)


def test_documento_rejeita_metadata_invalida() -> None:
    with pytest.raises(ValueError, match="página deve ser positiva"):
        DocumentoRecuperado(
            conteudo="conteúdo",
            score=0.5,
            titulo="título",
            fonte="fonte",
            pagina=0,
            categoria="categoria",
            chunk_id="chunk",
        )
