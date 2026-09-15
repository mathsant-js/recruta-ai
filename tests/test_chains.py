"""Testes determinísticos do pipeline estruturado LCEL."""

import json
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.chain import (
    StructuredAnalysisError,
    StructuredAnalysisService,
    create_structured_chain,
)
from app.schemas import AnaliseRecrutamento


VALID_OUTPUT = {
    "intencao": "organizar_requisitos",
    "resumo": "Vaga backend pleno com Python.",
    "competencias_tecnicas": ["Python"],
    "competencias_comportamentais": ["Colaboração"],
    "perguntas_sugeridas": ["Conte uma entrega relevante em Python."],
    "pontos_de_atencao": ["Localidade não informada"],
    "proximo_passo": "Confirmar localidade e modalidade.",
    "confianca": 0.9,
}


class SequenceModel(BaseChatModel):
    """Modelo local que devolve respostas predefinidas sem acessar a rede."""

    responses: list[str]
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "structured-sequence-test-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        index = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.responses[index]))]
        )


def test_pipeline_lcel_retorna_instancia_pydantic() -> None:
    llm = SequenceModel(responses=[json.dumps(VALID_OUTPUT)])
    chain, parser = create_structured_chain(llm)

    result = chain.invoke(
        {
            "conversation": "Usuário: vaga backend pleno com Python.",
            "analysis_request": "Organize os requisitos.",
            "format_instructions": parser.get_format_instructions(),
        }
    )

    assert isinstance(result, AnaliseRecrutamento)
    assert result.confianca == 0.9
    assert "intencao" in parser.get_format_instructions()


def test_servico_corrige_uma_saida_invalida_uma_unica_vez() -> None:
    llm = SequenceModel(responses=["não é JSON", json.dumps(VALID_OUTPUT)])
    service = StructuredAnalysisService(llm=llm)

    result = service.analyze("Usuário: vaga backend pleno com Python.")

    assert isinstance(result, AnaliseRecrutamento)
    assert llm.call_count == 2


def test_servico_expoe_erro_compreensivel_apos_retry() -> None:
    llm = SequenceModel(responses=["inválido", "ainda inválido"])
    service = StructuredAnalysisService(llm=llm)

    with pytest.raises(StructuredAnalysisError, match="tentativa de correção"):
        service.analyze("Usuário: vaga backend.")
    assert llm.call_count == 2


def test_servico_rejeita_conversa_vazia_sem_chamar_modelo() -> None:
    llm = SequenceModel(responses=[json.dumps(VALID_OUTPUT)])
    service = StructuredAnalysisService(llm=llm)

    with pytest.raises(ValueError, match="não pode estar vazia"):
        service.analyze("   ")
    assert llm.call_count == 0
