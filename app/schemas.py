"""Schemas Pydantic para as respostas estruturadas do Recruta AI."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


TextoNaoVazio = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AnaliseRecrutamento(BaseModel):
    """Resultado validado de uma análise de recrutamento e RH."""

    model_config = ConfigDict(extra="forbid")

    intencao: Literal[
        "definir_vaga",
        "revisar_vaga",
        "preparar_entrevista",
        "organizar_requisitos",
        "analisar_solicitacao",
        "fora_do_escopo",
    ] = Field(description="Intenção principal identificada na solicitação.")
    resumo: TextoNaoVazio = Field(description="Síntese fiel aos fatos informados.")
    competencias_tecnicas: list[TextoNaoVazio] = Field(default_factory=list)
    competencias_comportamentais: list[TextoNaoVazio] = Field(default_factory=list)
    perguntas_sugeridas: list[TextoNaoVazio] = Field(default_factory=list)
    pontos_de_atencao: list[TextoNaoVazio] = Field(default_factory=list)
    proximo_passo: TextoNaoVazio
    confianca: float = Field(ge=0, le=1)

    @field_validator(
        "competencias_tecnicas",
        "competencias_comportamentais",
        "perguntas_sugeridas",
        "pontos_de_atencao",
    )
    @classmethod
    def remover_itens_duplicados(cls, valores: list[str]) -> list[str]:
        """Evita repetição sem alterar a ordem produzida pelo modelo."""

        return list(dict.fromkeys(valores))
