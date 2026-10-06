"""Configuração central da aplicação."""

from dataclasses import dataclass
from os import getenv

from dotenv import load_dotenv


load_dotenv()

OLLAMA_HOST = "https://ollama.com"
OLLAMA_MODEL = "gemma4:cloud"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


@dataclass(frozen=True)
class Settings:
    """Valores necessários para acessar os modelos aprovados no Ollama Cloud."""

    ollama_api_key: str
    ollama_host: str = OLLAMA_HOST
    ollama_model: str = OLLAMA_MODEL
    ollama_embedding_model: str = OLLAMA_EMBEDDING_MODEL


def get_settings() -> Settings:
    """Retorna a configuração carregada do ambiente local."""

    return Settings(ollama_api_key=getenv("OLLAMA_API_KEY", ""))
