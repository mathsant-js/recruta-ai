"""Interface Gradio e ponto de entrada oficial da aplicação."""

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from app.chain import ChatService, StructuredAnalysisService, StructuredAnalysisError
from app.config import get_settings


ChatHistory = list[dict[str, Any]]


def _conversation_text(history: Sequence[dict[str, Any]]) -> str:
    """Converte o histórico visual em texto delimitado para a análise."""

    labels = {"user": "Usuário", "assistant": "Recruta AI"}
    return "\n".join(
        f"{labels.get(str(item.get('role')), str(item.get('role')))}: "
        f"{str(item.get('content')).strip()}"
        for item in history
        if item.get("role") in labels and str(item.get("content", "")).strip()
    )


def _send_message(
    message: str,
    history: ChatHistory | None,
    session_id: str,
    chat_service: ChatService,
) -> tuple[str, ChatHistory, str]:
    """Envia uma mensagem e devolve o histórico pronto para o Chatbot."""

    current_history = list(history or [])
    normalized_message = (message or "").strip()
    if not normalized_message:
        return "", current_history, "Informe uma mensagem antes de enviar."

    try:
        response = chat_service.chat(session_id, normalized_message)
        status = ""
    except Exception as exc:
        response = (
            "Não foi possível consultar o Recruta AI agora. "
            "Confira a conexão e tente novamente."
        )
        status = f"⚠️ Falha no envio: {exc}"

    updated_history = current_history + [
        {"role": "user", "content": normalized_message},
        {"role": "assistant", "content": str(response)},
    ]
    return "", updated_history, status


def _clear_chat(
    session_id: str,
    chat_service: ChatService,
) -> tuple[str, ChatHistory, None, str]:
    """Limpa os componentes visuais e a memória da sessão atual."""

    chat_service.clear_session(session_id)
    return "", [], None, "Conversa e memória da sessão foram limpas."


def _generate_analysis(
    history: ChatHistory | None,
    analysis_service: StructuredAnalysisService,
) -> tuple[dict[str, Any] | None, str]:
    """Gera uma saída legível somente depois da validação Pydantic."""

    conversation = _conversation_text(history or [])
    if not conversation:
        return None, "Inicie uma conversa antes de gerar a análise estruturada."

    try:
        result = analysis_service.analyze(conversation)
        return result.model_dump(mode="json"), "✅ Análise validada com sucesso."
    except (ValueError, StructuredAnalysisError) as exc:
        return None, f"⚠️ {exc}"
    except Exception as exc:
        return None, f"⚠️ Falha ao gerar a análise: {exc}"


def build_interface(
    chat_service: ChatService | None = None,
    analysis_service: StructuredAnalysisService | None = None,
):
    """Monta a interface sem iniciá-la durante a importação do módulo."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio não está instalado. Execute: pip install -r requirements.txt"
        ) from exc

    current_chat_service = chat_service or ChatService()
    current_analysis_service = analysis_service or StructuredAnalysisService(
        llm=current_chat_service.llm
    )

    def send_message(message: str, history: ChatHistory | None, session_id: str):
        return _send_message(message, history, session_id, current_chat_service)

    def clear_chat(session_id: str):
        return _clear_chat(session_id, current_chat_service)

    def generate_analysis(history: ChatHistory | None):
        return _generate_analysis(history, current_analysis_service)

    with gr.Blocks(title="Recruta AI") as demo:
        session_id = gr.State(lambda: str(uuid4()))
        gr.Markdown(
            "# Recruta AI\n\n"
            "Assistente profissional para levantar requisitos, revisar descrições de "
            "vagas e preparar entrevistas de recrutamento e RH."
        )
        chatbot = gr.Chatbot(
            type="messages",
            label="Conversa",
            height=420,
            placeholder="Conte ao Recruta AI qual vaga ou necessidade de RH deseja organizar.",
        )
        message = gr.Textbox(
            label="Mensagem",
            placeholder="Ex.: Preciso estruturar uma vaga de desenvolvedor backend pleno.",
            lines=2,
        )
        with gr.Row():
            send = gr.Button("Enviar", variant="primary")
            clear = gr.Button("Limpar conversa e memória")
            analyze = gr.Button("Gerar análise estruturada")
        analysis_output = gr.JSON(label="Análise validada")
        status = gr.Markdown()
        gr.Markdown(
            "### Uso responsável\n\n"
            "> O Recruta AI é uma ferramenta de apoio e não decide contratações. "
            "Evite inserir dados pessoais desnecessários, não use atributos sensíveis como "
            "critério e submeta toda recomendação à revisão humana."
        )

        send.click(
            send_message,
            [message, chatbot, session_id],
            [message, chatbot, status],
        )
        message.submit(
            send_message,
            [message, chatbot, session_id],
            [message, chatbot, status],
        )
        clear.click(
            clear_chat,
            [session_id],
            [message, chatbot, analysis_output, status],
        )
        analyze.click(generate_analysis, [chatbot], [analysis_output, status])
    return demo


def main() -> None:
    """Inicia a aplicação no endereço exigido pelo checkpoint."""

    settings = get_settings()
    if not settings.ollama_api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY ausente. Copie .env.example para .env e informe a chave."
        )
    build_interface().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
