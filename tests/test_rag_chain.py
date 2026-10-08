"""Testes determinísticos do pipeline retrieve → generate do CKP02."""

from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.rag_chain import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    CitacaoInvalidaError,
    RAGService,
)
from app.retriever import DocumentoRecuperado


class StaticModel(BaseChatModel):
    """Modelo sem rede que permite inspecionar o prompt final."""

    response: str
    received_messages: list[BaseMessage] = []

    @property
    def _llm_type(self) -> str:
        return "static-rag-test-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        self.received_messages = messages
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.response))]
        )


def documento(*, score: float = 0.91) -> DocumentoRecuperado:
    return DocumentoRecuperado(
        conteudo="Colete apenas dados necessários e informe a finalidade.",
        score=score,
        titulo="Guia fictício de teste",
        fonte="https://example.invalid/guia",
        pagina=7,
        categoria="privacidade",
        chunk_id="DOC-02-C014",
    )


def busca_com(resultados: list[DocumentoRecuperado]):
    def search(pergunta: str, *, top_k: int, filtros=None):
        assert pergunta
        assert top_k == 5
        assert filtros is None
        return resultados

    return search


def test_pipeline_retrieve_generate_cita_apenas_chunk_recuperado() -> None:
    llm = StaticModel(response="Use minimização de dados [DOC-02-C014].")
    service = RAGService(llm=llm, search=busca_com([documento()]))

    result = service.answer("Como proteger dados de candidatos?")

    assert result.fontes[0].chunk_id == "DOC-02-C014"
    assert "[DOC-02-C014, p. 7]" in result.texto_formatado
    assert "https://example.invalid/guia" in result.texto_formatado
    assert "exclusivamente" in llm.received_messages[0].content
    assert "<document_context>" in llm.received_messages[1].content


def test_pipeline_recusa_citacao_que_nao_foi_recuperada() -> None:
    llm = StaticModel(response="Afirmação sem suporte [DOC-99-C999].")
    service = RAGService(llm=llm, search=busca_com([documento()]))

    with pytest.raises(CitacaoInvalidaError, match="não recuperados"):
        service.answer("Pergunta")


def test_pipeline_valida_todos_os_ids_de_citacao_agrupada() -> None:
    llm = StaticModel(
        response="Afirmação [DOC-02-C014, DOC-99-C999]."
    )
    service = RAGService(llm=llm, search=busca_com([documento()]))

    with pytest.raises(CitacaoInvalidaError, match="DOC-99-C999"):
        service.answer("Pergunta")


def test_pipeline_exige_ao_menos_uma_citacao_na_resposta() -> None:
    llm = StaticModel(response="Afirmação sem marcador de evidência.")
    service = RAGService(llm=llm, search=busca_com([documento()]))

    with pytest.raises(CitacaoInvalidaError, match="não citou"):
        service.answer("Pergunta")


def test_ausencia_de_evidencia_nao_chama_modelo() -> None:
    llm = StaticModel(response="não deveria ser chamada")
    service = RAGService(llm=llm, search=busca_com([documento(score=0.1)]))

    result = service.answer("Pergunta fora da base")

    assert result.resposta == INSUFFICIENT_EVIDENCE_MESSAGE
    assert result.fontes == ()
    assert llm.received_messages == []
