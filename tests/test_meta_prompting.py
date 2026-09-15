"""Testes deterministas do fluxo de meta prompting."""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda

from app.meta_prompting import (
    MetaPromptReview,
    build_meta_review_prompt,
    run_meta_review,
    write_review_artifact,
)


VALID_REVIEW = {
    "resumo": "O prompt e robusto, mas tem uma lacuna operacional.",
    "pontos_fortes_a_preservar": ["Limites de decisao claros."],
    "sugestoes": [
        {
            "identificador": "S1",
            "categoria": "lacuna",
            "trecho_ou_secao": "operating_rules",
            "problema": "Nao define como tratar requisitos conflitantes.",
            "risco_associado": "Sintese inconsistente.",
            "mudanca_sugerida": "Pedir confirmacao antes de resolver o conflito.",
            "prioridade": "media",
        }
    ],
    "riscos_residuais": ["Respostas do modelo continuam exigindo revisao humana."],
}


def test_prompt_separa_auditoria_e_conteudo_avaliado() -> None:
    messages = build_meta_review_prompt().format_messages(
        system_prompt="<persona>Prompt em avaliacao</persona>",
        format_instructions="JSON valido",
    )

    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "Prompt em avaliacao" not in messages[0].content
    assert "Prompt em avaliacao" in messages[1].content
    assert "não uma reescrita automática" in messages[0].content


def test_revisao_modelo_e_validada_por_pydantic() -> None:
    parser = PydanticOutputParser(pydantic_object=MetaPromptReview)
    fake_llm = RunnableLambda(lambda _: json.dumps(VALID_REVIEW, ensure_ascii=False))

    review = run_meta_review(fake_llm)

    assert isinstance(review, MetaPromptReview)
    assert review.sugestoes[0].identificador == "S1"


def test_artefato_explicita_que_nao_ha_aceite_automatico(tmp_path: Path) -> None:
    review = MetaPromptReview.model_validate(VALID_REVIEW)
    output = tmp_path / "critica.json"

    write_review_artifact(review, output)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["status"] == "aguardando_revisao_humana"
    assert "não são aceitas automaticamente" in artifact["observacao"]
    assert artifact["critica"]["sugestoes"][0]["identificador"] == "S1"
