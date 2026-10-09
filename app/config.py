"""Configuração central da aplicação."""

from dataclasses import dataclass
from os import getenv

from dotenv import load_dotenv

load_dotenv()

OLLAMA_CHAT_HOST = "https://ollama.com"
OLLAMA_EMBEDDING_HOST = "http://localhost:11434"
OLLAMA_HOST = OLLAMA_CHAT_HOST
OLLAMA_MODEL = "gemma4:cloud"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


@dataclass(frozen=True)
class Settings:
    """Endpoints separados para geração cloud e embeddings locais."""

    ollama_api_key: str
    ollama_host: str = OLLAMA_CHAT_HOST
    ollama_model: str = OLLAMA_MODEL
    ollama_embedding_host: str = OLLAMA_EMBEDDING_HOST
    ollama_embedding_model: str = OLLAMA_EMBEDDING_MODEL


def get_settings() -> Settings:
    """Retorna a configuração carregada do ambiente local."""

    return Settings(
        ollama_api_key=getenv("OLLAMA_API_KEY", ""),
        ollama_host=getenv("OLLAMA_CHAT_HOST", OLLAMA_CHAT_HOST),
        ollama_model=getenv("OLLAMA_CHAT_MODEL", OLLAMA_MODEL),
        ollama_embedding_host=getenv(
            "OLLAMA_EMBEDDING_HOST", OLLAMA_EMBEDDING_HOST
        ),
        ollama_embedding_model=getenv(
            "OLLAMA_EMBEDDING_MODEL", OLLAMA_EMBEDDING_MODEL
        ),
    )
