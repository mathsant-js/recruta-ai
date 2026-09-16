"""Testes determinísticos dos fluxos expostos pela interface Gradio."""

from typing import Any

import pytest

import app.main as main_module
from app.config import Settings
from app.main import (
    _clear_chat,
    _conversation_text,
    _generate_analysis,
    _send_message,
)
from app.schemas import AnaliseRecrutamento


class FakeChatService:
    """Substitui a chain cloud e registra as interações da interface."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []
        self.cleared_sessions: list[str] = []

    def chat(self, session_id: str, message: str) -> str:
        self.messages.append((session_id, message))
        return "Vamos organizar os requisitos da vaga."

    def clear_session(self, session_id: str) -> bool:
        self.cleared_sessions.append(session_id)
        return True


class FakeAnalysisService:
    """Devolve uma instância Pydantic sem realizar chamada externa."""

    def __init__(self) -> None:
        self.conversation = ""

    def analyze(self, conversation: str) -> AnaliseRecrutamento:
        self.conversation = conversation
        return AnaliseRecrutamento(
            intencao="organizar_requisitos",
            resumo="Vaga backend pleno.",
            competencias_tecnicas=["Python"],
            competencias_comportamentais=["Colaboração"],
            perguntas_sugeridas=["Conte uma entrega relevante."],
            pontos_de_atencao=["Modalidade não informada"],
            proximo_passo="Confirmar a modalidade de trabalho.",
            confianca=0.9,
        )


class FakeDemo:
    """Registra os argumentos de inicialização sem abrir um servidor real."""

    def __init__(self) -> None:
        self.launch_kwargs: dict[str, Any] = {}

    def launch(self, **kwargs: Any) -> None:
        self.launch_kwargs = kwargs


def test_ponto_de_entrada_oficial_inicia_na_porta_7860(monkeypatch) -> None:
    demo = FakeDemo()
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: Settings(ollama_api_key="chave-ficticia-de-teste"),
    )
    monkeypatch.setattr(main_module, "build_interface", lambda: demo)

    main_module.main()

    assert demo.launch_kwargs == {"server_name": "0.0.0.0", "server_port": 7860}


def test_ponto_de_entrada_falha_com_mensagem_clara_sem_chave(monkeypatch) -> None:
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: Settings(ollama_api_key=""),
    )
    monkeypatch.setattr(
        main_module,
        "build_interface",
        lambda: pytest.fail("A interface não deve ser construída sem credencial."),
    )

    with pytest.raises(RuntimeError, match="OLLAMA_API_KEY ausente"):
        main_module.main()


def test_envio_atualiza_chat_e_preserva_id_da_sessao() -> None:
    service = FakeChatService()

    message, history, status = _send_message(
        "  Preciso estruturar uma vaga backend.  ", [], "sessao-ui", service
    )

    assert message == ""
    assert status == ""
    assert service.messages == [("sessao-ui", "Preciso estruturar uma vaga backend.")]
    assert history == [
        {"role": "user", "content": "Preciso estruturar uma vaga backend."},
        {
            "role": "assistant",
            "content": "Vamos organizar os requisitos da vaga.",
        },
    ]


def test_envio_vazio_nao_chama_chain() -> None:
    service = FakeChatService()
    existing_history: list[dict[str, Any]] = [
        {"role": "assistant", "content": "Como posso ajudar?"}
    ]

    message, history, status = _send_message("  ", existing_history, "sessao", service)

    assert message == ""
    assert history == existing_history
    assert "Informe uma mensagem" in status
    assert service.messages == []


def test_limpeza_remove_memoria_e_todos_os_resultados_visuais() -> None:
    service = FakeChatService()

    result = _clear_chat("sessao-ui", service)

    assert service.cleared_sessions == ["sessao-ui"]
    assert result == ("", [], None, "Conversa e memória da sessão foram limpas.")


def test_analise_exibe_model_dump_de_resultado_validado() -> None:
    service = FakeAnalysisService()
    history = [
        {"role": "user", "content": "Vaga backend pleno."},
        {"role": "assistant", "content": "Quais tecnologias são obrigatórias?"},
    ]

    output, status = _generate_analysis(history, service)

    assert output is not None
    assert output["intencao"] == "organizar_requisitos"
    assert output["confianca"] == 0.9
    assert "validada com sucesso" in status
    assert service.conversation == (
        "Usuário: Vaga backend pleno.\n"
        "Recruta AI: Quais tecnologias são obrigatórias?"
    )


def test_analise_vazia_orienta_usuario_sem_chamar_servico() -> None:
    service = FakeAnalysisService()

    output, status = _generate_analysis([], service)

    assert output is None
    assert "Inicie uma conversa" in status
    assert service.conversation == ""


def test_conversa_descarta_papeis_desconhecidos() -> None:
    history = [
        {"role": "system", "content": "texto interno"},
        {"role": "user", "content": "Dado confiável"},
    ]

    assert _conversation_text(history) == "Usuário: Dado confiável"
