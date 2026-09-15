"""Gerenciamento da memoria conversacional isolada por sessao."""

from threading import RLock
from typing import Final

from langchain.memory import ConversationTokenBufferMemory
from langchain_core.language_models import BaseLanguageModel


DEFAULT_MEMORY_TOKEN_LIMIT: Final = 1_200


class SessionMemoryManager:
    """Cria, recupera e limpa memorias sem compartilhar estado entre sessoes."""

    def __init__(self, token_limit: int = DEFAULT_MEMORY_TOKEN_LIMIT) -> None:
        if token_limit <= 0:
            raise ValueError("O limite de tokens da memoria deve ser positivo.")

        self.token_limit = token_limit
        self._memories: dict[str, ConversationTokenBufferMemory] = {}
        self._lock = RLock()

    def get_or_create(
        self,
        session_id: str,
        llm: BaseLanguageModel,
    ) -> ConversationTokenBufferMemory:
        """Retorna a memoria exclusiva da sessao, criando-a quando necessario."""

        normalized_id = session_id.strip()
        if not normalized_id:
            raise ValueError("O identificador da sessao nao pode ser vazio.")

        with self._lock:
            if normalized_id not in self._memories:
                self._memories[normalized_id] = ConversationTokenBufferMemory(
                    llm=llm,
                    max_token_limit=self.token_limit,
                    memory_key="history",
                    input_key="input",
                    output_key="response",
                    return_messages=False,
                )
            return self._memories[normalized_id]

    def clear(self, session_id: str) -> bool:
        """Remove a memoria da sessao e informa se ela existia."""

        with self._lock:
            memory = self._memories.pop(session_id.strip(), None)
            if memory is None:
                return False
            memory.clear()
            return True

    def has_session(self, session_id: str) -> bool:
        """Informa se ja existe memoria para a sessao."""

        with self._lock:
            return session_id.strip() in self._memories
