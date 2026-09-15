"""Interface Gradio e ponto de entrada oficial da aplicação."""

from uuid import uuid4

from app.chain import ChatService, StructuredAnalysisService, StructuredAnalysisError
from app.config import get_settings


def _conversation_text(history: list[dict[str, str]]) -> str:
    """Converte o histórico visual em texto delimitado para a análise."""

    labels = {"user": "Usuário", "assistant": "Recruta AI"}
    return "\n".join(
        f"{labels.get(item['role'], item['role'])}: {item['content']}"
        for item in history
        if item.get("content")
    )


def build_interface():
    """Monta a interface sem iniciá-la durante a importação do módulo."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio não está instalado. Execute: pip install -r requirements.txt"
        ) from exc

    chat_service = ChatService()
    analysis_service = StructuredAnalysisService(llm=chat_service.llm)

    def send_message(message: str, history: list[dict[str, str]], session_id: str):
        history = history or []
        if not message.strip():
            return "", history
        try:
            response = chat_service.chat(session_id, message)
        except Exception as exc:
            response = f"Não foi possível responder: {exc}"
        return "", history + [
            {"role": "user", "content": message.strip()},
            {"role": "assistant", "content": response},
        ]

    def clear_chat(session_id: str):
        chat_service.clear_session(session_id)
        return [], None

    def generate_analysis(history: list[dict[str, str]]):
        try:
            result = analysis_service.analyze(_conversation_text(history or []))
            return result.model_dump(mode="json"), "Análise validada com sucesso."
        except (ValueError, StructuredAnalysisError) as exc:
            return None, str(exc)
        except Exception as exc:
            return None, f"Falha ao gerar a análise: {exc}"

    with gr.Blocks(title="Recruta AI") as demo:
        session_id = gr.State(lambda: str(uuid4()))
        gr.Markdown(
            "# Recruta AI\nAssistente para levantamento de vagas, requisitos e entrevistas."
        )
        chatbot = gr.Chatbot(type="messages", label="Conversa", height=420)
        message = gr.Textbox(label="Mensagem", placeholder="Descreva a vaga ou necessidade de RH")
        with gr.Row():
            send = gr.Button("Enviar", variant="primary")
            clear = gr.Button("Limpar conversa e memória")
            analyze = gr.Button("Gerar análise estruturada")
        analysis_output = gr.JSON(label="Análise validada")
        status = gr.Markdown()
        gr.Markdown(
            "> Ferramenta de apoio: evite dados pessoais desnecessários e submeta resultados "
            "a revisão humana. O Recruta AI não decide contratações."
        )

        send.click(send_message, [message, chatbot, session_id], [message, chatbot])
        message.submit(send_message, [message, chatbot, session_id], [message, chatbot])
        clear.click(clear_chat, [session_id], [chatbot, analysis_output])
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
