"""Cliente Ollama configurado para o único modelo permitido no projeto."""

from ollama import Client

from app.config import Settings, get_settings


def create_ollama_client(settings: Settings | None = None) -> Client:
    """Cria o cliente da API cloud sem realizar chamadas de rede."""

    current_settings = settings or get_settings()
    headers = None

    if current_settings.ollama_api_key:
        headers = {
            "Authorization": f"Bearer {current_settings.ollama_api_key}",
        }

    return Client(host=current_settings.ollama_host, headers=headers)
