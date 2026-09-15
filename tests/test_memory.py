"""Testes da memoria por sessao e do roteiro de cinco ou mais turnos."""

from typing import Any

from langchain.chains import ConversationChain
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.chain import ChatService, create_chat_llm, create_conversation_chain
from app.config import Settings
from app.memory_manager import DEFAULT_MEMORY_TOKEN_LIMIT, SessionMemoryManager


class RecruitingTestModel(BaseChatModel):
    """Modelo local deterministico que permite inspecionar o prompt recebido."""

    received_prompts: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "recruiting-test-model"

    def get_num_tokens(self, text: str) -> int:
        return len(text.split())

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        prompt = "\n".join(str(message.content) for message in messages)
        self.received_prompts.append(prompt)

        if "Crie cinco perguntas" in prompt:
            content = (
                "1. Como voce aplica Python no backend?\n"
                "2. Como modela dados no PostgreSQL?\n"
                "3. Conte uma entrega compativel com nivel pleno.\n"
                "4. Como colabora em trabalho hibrido?\n"
                "5. Como investiga problemas de desempenho?"
            )
        elif "Quais requisitos obrigatorios" in prompt:
            content = (
                "Obrigatorios informados: senioridade pleno, Python, PostgreSQL "
                "e trabalho hibrido em Sao Paulo."
            )
        else:
            content = "Informacao registrada."

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )


def test_chain_usa_conversation_chain_e_limite_de_1200_tokens() -> None:
    llm = RecruitingTestModel()
    manager = SessionMemoryManager()

    chain = create_conversation_chain(llm, manager, "sessao-a")

    assert isinstance(chain, ConversationChain)
    assert chain.memory is manager.get_or_create("sessao-a", llm)
    assert chain.memory.max_token_limit == DEFAULT_MEMORY_TOKEN_LIMIT == 1_200


def test_llm_cloud_tem_contador_local_sem_realizar_chamada() -> None:
    llm = create_chat_llm(Settings(ollama_api_key="chave-ficticia-de-teste"))

    assert llm.model == "gemma4:cloud"
    assert llm.get_num_tokens("Teste simples de contagem local.") > 0


def test_roteiro_recupera_requisitos_e_embasa_perguntas() -> None:
    llm = RecruitingTestModel()
    service = ChatService(llm=llm)
    session_id = "roteiro-etapa-3"
    messages = [
        "Estou abrindo uma vaga de desenvolvedor backend.",
        "A senioridade e pleno.",
        "Precisamos de Python e PostgreSQL.",
        "O trabalho sera hibrido em Sao Paulo.",
        "Quais requisitos obrigatorios eu ja informei?",
        "Crie cinco perguntas de entrevista com base neles.",
    ]

    responses = [service.chat(session_id, message) for message in messages]

    summary = responses[4].lower()
    questions = responses[5].lower()
    final_prompt = llm.received_prompts[-1].lower()
    for expected in ("pleno", "python", "postgresql", "hibrido", "sao paulo"):
        assert expected in summary
        assert expected in final_prompt
    assert questions.count("\n") == 4
    for expected in ("pleno", "python", "postgresql", "hibrido"):
        assert expected in questions


def test_sessoes_sao_isoladas_e_podem_ser_limpas() -> None:
    llm = RecruitingTestModel()
    manager = SessionMemoryManager()
    service = ChatService(llm=llm, memory_manager=manager)

    service.chat("sessao-a", "A vaga exige Python.")
    service.chat("sessao-b", "A vaga exige Java.")

    history_a = manager.get_or_create("sessao-a", llm).load_memory_variables({})[
        "history"
    ]
    history_b = manager.get_or_create("sessao-b", llm).load_memory_variables({})[
        "history"
    ]
    assert "Python" in history_a and "Java" not in history_a
    assert "Java" in history_b and "Python" not in history_b

    assert service.clear_session("sessao-a") is True
    assert manager.has_session("sessao-a") is False
    assert manager.has_session("sessao-b") is True
    assert service.clear_session("sessao-inexistente") is False


def test_entradas_vazias_sao_rejeitadas() -> None:
    service = ChatService(llm=RecruitingTestModel())

    for session_id, message in (("", "mensagem"), ("sessao", "   ")):
        try:
            service.chat(session_id, message)
        except ValueError:
            pass
        else:
            raise AssertionError("Sessao e mensagem devem ser obrigatorias.")
