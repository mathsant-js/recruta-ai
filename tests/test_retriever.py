"""Testes determinísticos do contrato público de recuperação do CKP02."""

import pytest

import app.retriever as retriever
from app.retriever import DocumentoRecuperado, RecuperadorNaoConfiguradoError


@pytest.fixture(autouse=True)
def restaurar_backend() -> None:
    """Evita que a injeção de um teste vaze para os demais."""

    backend_original = retriever._backend_busca
    yield
    retriever._backend_busca = backend_original


def test_importacao_nao_inicializa_indice() -> None:
    with pytest.raises(RecuperadorNaoConfiguradoError, match="ainda não foi construído"):
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
