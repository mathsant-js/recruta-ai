"""Construcao da chain de conversa com memoria do Recruta AI."""

from threading import RLock

from langchain.chains import ConversationChain
from langchain_core.language_models import BaseLanguageModel
from langchain_ollama import ChatOllama

from app.config import Settings, get_settings
from app.memory_manager import SessionMemoryManager
from app.prompts import CHAT_PROMPT


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
