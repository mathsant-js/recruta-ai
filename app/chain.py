"""Construcao da chain de conversa com memoria do Recruta AI."""

from threading import RLock

from langchain.chains import ConversationChain
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from app.config import Settings, get_settings
from app.memory_manager import SessionMemoryManager
from app.prompts import (
    CHAT_PROMPT,
    STRUCTURED_ANALYSIS_PROMPT,
    STRUCTURED_RETRY_PROMPT,
)
from app.schemas import AnaliseRecrutamento


def _approximate_token_ids(text: str) -> list[int]:
    """Conta tokens localmente para controlar a janela sem outra chamada de API."""

    estimated_count = max(1, (len(text.encode("utf-8")) + 3) // 4)
    return list(range(estimated_count))


def create_chat_llm(settings: Settings | None = None) -> ChatOllama:
    """Configura o modelo permitido para conversas via Ollama Cloud."""

    current_settings = settings or get_settings()
    if not current_settings.ollama_api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY ausente. Copie .env.example para .env e informe a chave."
        )

    return ChatOllama(
        model=current_settings.ollama_model,
        base_url=current_settings.ollama_host,
        temperature=0.2,
        custom_get_token_ids=_approximate_token_ids,
        client_kwargs={
            "headers": {
                "Authorization": f"Bearer {current_settings.ollama_api_key}",
            }
        },
    )


def create_conversation_chain(
    llm: BaseLanguageModel,
    memory_manager: SessionMemoryManager,
    session_id: str,
) -> ConversationChain:
    """Monta a ConversationChain e injeta a memoria da sessao solicitada."""

    memory = memory_manager.get_or_create(session_id, llm)
    return ConversationChain(
        llm=llm,
        prompt=CHAT_PROMPT,
        memory=memory,
        input_key="input",
        output_key="response",
        verbose=False,
    )


class ChatService:
    """Coordena chains persistentes e independentes para cada sessao."""

    def __init__(
        self,
        llm: BaseLanguageModel | None = None,
        memory_manager: SessionMemoryManager | None = None,
    ) -> None:
        self.llm = llm or create_chat_llm()
        self.memory_manager = memory_manager or SessionMemoryManager()
        self._chains: dict[str, ConversationChain] = {}
        self._lock = RLock()

    def chat(self, session_id: str, message: str) -> str:
        """Processa uma mensagem usando somente o historico da sessao indicada."""

        normalized_id = session_id.strip()
        if not normalized_id:
            raise ValueError("O identificador da sessao nao pode ser vazio.")
        if not message.strip():
            raise ValueError("A mensagem nao pode ser vazia.")

        with self._lock:
            chain = self._chains.get(normalized_id)
            if chain is None:
                chain = create_conversation_chain(
                    self.llm,
                    self.memory_manager,
                    normalized_id,
                )
                self._chains[normalized_id] = chain
            result = chain.invoke({"input": message.strip()})
        return str(result["response"])

    def clear_session(self, session_id: str) -> bool:
        """Limpa a chain e a memoria associadas a uma sessao."""

        normalized_id = session_id.strip()
        with self._lock:
            self._chains.pop(normalized_id, None)
            return self.memory_manager.clear(normalized_id)


class StructuredAnalysisError(RuntimeError):
    """Erro compreensível após esgotar as tentativas de saída estruturada."""


def create_structured_chain(
    llm: BaseLanguageModel,
    prompt: ChatPromptTemplate = STRUCTURED_ANALYSIS_PROMPT,
) -> tuple[Runnable, PydanticOutputParser[AnaliseRecrutamento]]:
    """Compõe explicitamente prompt, modelo e parser pelo operador LCEL."""

    pydantic_parser = PydanticOutputParser(pydantic_object=AnaliseRecrutamento)
    structured_chain = prompt | llm | pydantic_parser
    return structured_chain, pydantic_parser


class StructuredAnalysisService:
    """Executa análise validada e permite no máximo uma tentativa de correção."""

    def __init__(self, llm: BaseLanguageModel | None = None) -> None:
        self.llm = llm or create_chat_llm()
        self.chain, self.parser = create_structured_chain(self.llm)

    def analyze(
        self,
        conversation: str,
        analysis_request: str = "Gere uma análise estruturada desta conversa.",
    ) -> AnaliseRecrutamento:
        """Retorna sempre uma instância Pydantic ou lança erro de domínio."""

        if not conversation.strip():
            raise ValueError("A conversa não pode estar vazia.")

        inputs = {
            "conversation": conversation.strip(),
            "analysis_request": analysis_request.strip(),
            "format_instructions": self.parser.get_format_instructions(),
        }
        try:
            result = self.chain.invoke(inputs)
        except OutputParserException as first_error:
            retry_chain, _ = create_structured_chain(self.llm, STRUCTURED_RETRY_PROMPT)
            try:
                result = retry_chain.invoke(
                    {
                        "conversation": inputs["conversation"],
                        "invalid_output": first_error.llm_output or "Saída indisponível",
                        "validation_error": str(first_error),
                        "format_instructions": inputs["format_instructions"],
                    }
                )
            except OutputParserException as retry_error:
                raise StructuredAnalysisError(
                    "Não foi possível validar a análise após uma tentativa de correção. "
                    "Revise os dados da conversa e tente novamente."
                ) from retry_error

        if not isinstance(result, AnaliseRecrutamento):
            raise StructuredAnalysisError("A análise não retornou o schema esperado.")
        return result
