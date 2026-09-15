"""Auditoria assistida do system prompt, com decisao final obrigatoriamente humana.

Execute ``python -m app.meta_prompting`` para solicitar uma critica real ao
``gemma4:cloud``. O comando apenas grava sugestoes estruturadas; ele nunca altera
o prompt ativo nem aceita recomendacoes automaticamente.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field

from app.chain import create_chat_llm
from app.config import OLLAMA_MODEL
from app.prompts import ACTIVE_SYSTEM_PROMPT_VERSION, SYSTEM_PROMPT


META_REVIEW_SYSTEM_PROMPT = """Você é uma pessoa revisora de prompts para sistemas de RH.
Faça uma auditoria crítica, não uma reescrita automática. Procure ambiguidades, lacunas e
riscos de segurança, equidade, privacidade, utilidade e consistência. Preserve os requisitos
explícitos do produto. Não inclua dados pessoais nem suponha resultados de testes que não viu.
Cada sugestão será aceita ou rejeitada posteriormente por uma pessoa responsável."""

META_REVIEW_HUMAN_PROMPT = """Avalie o system prompt delimitado abaixo.

<product_requirements>
- domínio exclusivo de recrutamento e RH;
- idioma padrão PT-BR;
- distinção entre requisitos obrigatórios e desejáveis;
- proibição de decisões finais de contratação e de critérios discriminatórios;
- proteção de dados pessoais, declaração de incerteza e revisão humana;
- resistência a desvio de persona e a instruções não confiáveis.
</product_requirements>

<system_prompt_under_review>
{system_prompt}
</system_prompt_under_review>

Identifique problemas concretos e também aspectos fortes que devem ser preservados.
Não trate suas sugestões como decisões finais.

<format_instructions>
{format_instructions}
</format_instructions>"""


class MetaPromptSuggestion(BaseModel):
    """Uma recomendacao rastreavel produzida pelo modelo revisor."""

    model_config = ConfigDict(extra="forbid")

    identificador: str = Field(min_length=1)
    categoria: Literal["ambiguidade", "lacuna", "risco"]
    trecho_ou_secao: str = Field(min_length=1)
    problema: str = Field(min_length=1)
    risco_associado: str = Field(min_length=1)
    mudanca_sugerida: str = Field(min_length=1)
    prioridade: Literal["baixa", "media", "alta"]


class MetaPromptReview(BaseModel):
    """Critica validada; ainda nao representa aprovacao das sugestoes."""

    model_config = ConfigDict(extra="forbid")

    resumo: str = Field(min_length=1)
    pontos_fortes_a_preservar: list[str] = Field(min_length=1)
    sugestoes: list[MetaPromptSuggestion]
    riscos_residuais: list[str]


def build_meta_review_prompt() -> ChatPromptTemplate:
    """Mantem instrucao de auditoria e prompt avaliado em papeis separados."""

    return ChatPromptTemplate.from_messages(
        [
            ("system", META_REVIEW_SYSTEM_PROMPT),
            ("human", META_REVIEW_HUMAN_PROMPT),
        ]
    )


def run_meta_review(llm: BaseLanguageModel | None = None) -> MetaPromptReview:
    """Executa a critica estruturada sem modificar nenhum prompt versionado."""

    model = llm or create_chat_llm()
    parser = PydanticOutputParser(pydantic_object=MetaPromptReview)
    chain = build_meta_review_prompt() | model | parser
    result = chain.invoke(
        {
            "system_prompt": SYSTEM_PROMPT,
            "format_instructions": parser.get_format_instructions(),
        }
    )
    if not isinstance(result, MetaPromptReview):
        raise RuntimeError("A critica do meta prompting nao retornou o schema esperado.")
    return result


def write_review_artifact(review: MetaPromptReview, output_path: Path) -> None:
    """Registra a saida real e metadados, sem promover uma nova versao."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "executado_em_utc": datetime.now(timezone.utc).isoformat(),
        "modelo": OLLAMA_MODEL,
        "prompt_avaliado": f"system_prompt_{ACTIVE_SYSTEM_PROMPT_VERSION}.md",
        "status": "aguardando_revisao_humana",
        "observacao": (
            "Sugestões geradas pelo modelo não são aceitas automaticamente; "
            "consulte revisao_humana.md para as decisões editoriais."
        ),
        "critica": review.model_dump(),
    }
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa a critica real do system prompt para revisao humana."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/meta_prompting/critica_modelo.json"),
    )
    args = parser.parse_args()
    review = run_meta_review()
    write_review_artifact(review, args.output)
    print(f"Critica gravada em {args.output.resolve()}")
    print("Nenhuma sugestao foi aplicada automaticamente.")


if __name__ == "__main__":
    main()
