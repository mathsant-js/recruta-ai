"""Testes determinísticos dos fluxos da interface RAG."""

from typing import Any

import pytest

import app.main as main_module
from app.config import Settings
from app.main import _answer_question, _clear_interface, _format_sources
from app.rag_chain import RespostaRAG
from app.retriever import DocumentoRecuperado


def document() -> DocumentoRecuperado:
    return DocumentoRecuperado(
        conteudo="Colete apenas dados necessários.",
        score=0.93,
        titulo="Guia fictício",
        fonte="https://example.invalid/guia",
        pagina=7,
        categoria="privacidade",
        chunk_id="DOC-02-C014",
    )


class FakeRAGService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def answer(self, question: str, *, filters=None) -> RespostaRAG:
        self.calls.append((question, filters))
        source = document()
        return RespostaRAG(question, "Resposta [DOC-02-C014].", (source,))


class FakeDemo:
    def __init__(self) -> None:
        self.launch_kwargs: dict[str, Any] = {}

    def launch(self, **kwargs: Any) -> None:
        self.launch_kwargs = kwargs


def test_consulta_com_filtro_e_reranking() -> None:
    base = FakeRAGService()
    reranked = FakeRAGService()
    question, answer, sources, status = _answer_question(
        "  Como proteger dados? ",
        "Privacidade e proteção de dados",
        True,
        base,  # type: ignore[arg-type]
        reranked,  # type: ignore[arg-type]
    )
    assert question == "Como proteger dados?"
    assert "DOC-02-C014" in answer
    assert "p. 7" in sources
    assert "https://example.invalid/guia" in sources
    assert "cross-encoder" in status
    assert base.calls == []
    assert reranked.calls == [
        ("Como proteger dados?", {"category": "privacidade"})
    ]


def test_consulta_sem_filtro_usa_busca_base() -> None:
    base = FakeRAGService()
    reranked = FakeRAGService()
    _answer_question(
        "Pergunta", "Todas as categorias", False, base, reranked  # type: ignore[arg-type]
    )
    assert base.calls == [("Pergunta", None)]
    assert reranked.calls == []


def test_pergunta_vazia_nao_chama_servico() -> None:
    base = FakeRAGService()
    reranked = FakeRAGService()
    result = _answer_question(
        " ", "Todas as categorias", True, base, reranked  # type: ignore[arg-type]
    )
    assert "Informe uma pergunta" in result[-1]
    assert base.calls == reranked.calls == []


def test_limpeza_remove_resultados_visuais() -> None:
    assert _clear_interface() == ("", "", "", "Consulta limpa.")


def test_formatacao_sem_fontes_e_explicita() -> None:
    result = RespostaRAG("pergunta", "sem evidência", ())
    assert "Nenhuma fonte" in _format_sources(result)


def test_ponto_de_entrada_inicia_na_porta_7860(monkeypatch) -> None:
    demo = FakeDemo()
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: Settings(ollama_api_key="chave-ficticia-de-teste"),
    )
    monkeypatch.setattr(main_module, "build_interface", lambda: demo)
    main_module.main()
    assert demo.launch_kwargs == {"server_name": "0.0.0.0", "server_port": 7860}


def test_ponto_de_entrada_falha_sem_chave(monkeypatch) -> None:
    monkeypatch.setattr(
        main_module, "get_settings", lambda: Settings(ollama_api_key="")
    )
    monkeypatch.setattr(
        main_module,
        "build_interface",
        lambda: pytest.fail("A interface não deve ser construída sem credencial."),
    )
    with pytest.raises(RuntimeError, match="OLLAMA_API_KEY ausente"):
        main_module.main()
