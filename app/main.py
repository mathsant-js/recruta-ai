"""Ponto de entrada oficial da aplicação."""

from app.config import get_settings


def main() -> None:
    """Inicializa a aplicação e confirma sua configuração base."""

    settings = get_settings()
    print(f"Recruta AI iniciado com o modelo {settings.ollama_model}.")


if __name__ == "__main__":
    main()
