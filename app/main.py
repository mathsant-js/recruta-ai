"""Interface Gradio do RAG e ponto de entrada oficial da aplicação."""

from __future__ import annotations

from app.config import get_settings
from app.rag_chain import RAGService, RespostaRAG
from app.reranker import CrossEncoderReranker

CATEGORIES = {
    "Todas as categorias": None,
    "Privacidade e proteção de dados": "privacidade",
    "Recrutamento justo": "recrutamento_justo",
    "Recrutamento e seleção": "recrutamento_e_selecao",
    "Seleção por competências": "selecao_por_competencias",
    "Diversidade e inclusão": "diversidade_e_inclusao",
}


def _format_sources(result: RespostaRAG) -> str:
    """Exibe fontes em área separada, usando apenas metadata validada."""

    if not result.fontes:
        return "_Nenhuma fonte citada._"
    linhas = []
    for source in result.fontes:
        page = f" — p. {source.pagina}" if source.pagina is not None else ""
        linhas.append(
            f"- **{source.chunk_id}**{page} — [{source.titulo}]({source.fonte})"
        )
    return "\n".join(linhas)


def _answer_question(
    question: str,
    category_label: str,
    use_reranking: bool,
    base_service: RAGService,
    reranked_service: RAGService,
) -> tuple[str, str, str, str]:
    """Executa o RAG escolhido e prepara os componentes visuais."""

    normalized = (question or "").strip()
    if not normalized:
        return "", "", "", "⚠️ Informe uma pergunta antes de consultar."

    category = CATEGORIES.get(category_label)
    filters = {"category": category} if category else None
    service = reranked_service if use_reranking else base_service
    try:
        result = service.answer(normalized, filters=filters)
    # A fronteira da UI converte falhas de modelo, índice e rede em estado visível.
    except Exception as exc:  # noqa: BLE001
        return normalized, "", "", f"⚠️ Não foi possível executar o RAG: {exc}"

    method = "com cross-encoder" if use_reranking else "busca vetorial"
    filter_status = category or "sem filtro"
    return (
        normalized,
        result.resposta,
        _format_sources(result),
        f"✅ Consulta concluída ({method}; {filter_status}).",
    )


def _clear_interface() -> tuple[str, str, str, str]:
    """Limpa somente os resultados da consulta; o RAG não mantém memória."""

    return "", "", "", "Consulta limpa."


def build_interface(
    base_service: RAGService | None = None,
    reranked_service: RAGService | None = None,
):
    """Monta a interface sem acessar modelo, índice ou rede durante o import."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio não está instalado. Execute: pip install -r requirements.txt"
        ) from exc

    current_base_service = base_service or RAGService()
    current_reranked_service = reranked_service or RAGService(
        llm=current_base_service.llm,
        reranker=CrossEncoderReranker(),
        candidate_k=10,
        top_k=5,
    )

    def answer(question: str, category: str, reranking: bool):
        return _answer_question(
            question,
            category,
            reranking,
            current_base_service,
            current_reranked_service,
        )

    with gr.Blocks(title="Recruta AI — RAG") as demo:
        gr.Markdown(
            "# Recruta AI — Base documental\n\n"
            "Faça perguntas sobre recrutamento, privacidade, inclusão e gestão de "
            "pessoas. As respostas usam somente os documentos recuperados."
        )
        question = gr.Textbox(
            label="Pergunta",
            placeholder="Ex.: Quais cuidados devo ter com dados de candidatos?",
            lines=2,
        )
        with gr.Row():
            category = gr.Dropdown(
                choices=list(CATEGORIES),
                value="Todas as categorias",
                label="Filtro por categoria",
            )
            reranking = gr.Checkbox(
                value=False,
                label="Reordenar resultados com cross-encoder (experimental)",
            )
        with gr.Row():
            send = gr.Button("Consultar base", variant="primary")
            clear = gr.Button("Limpar")

        answer_output = gr.Markdown(label="Resposta fundamentada")
        sources_output = gr.Markdown(label="Fontes citadas")
        status = gr.Markdown()
        gr.Markdown(
            "### Uso responsável\n\n"
            "> O Recruta AI apoia, mas não decide contratações. Evite dados pessoais "
            "desnecessários e critérios sensíveis. Confirme as fontes e mantenha revisão humana."
        )

        inputs = [question, category, reranking]
        outputs = [question, answer_output, sources_output, status]
        send.click(answer, inputs, outputs)
        question.submit(answer, inputs, outputs)
        clear.click(_clear_interface, outputs=outputs)
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
