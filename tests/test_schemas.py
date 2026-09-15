"""Testes do schema Pydantic da análise de recrutamento."""

import pytest
from pydantic import ValidationError

from app.schemas import AnaliseRecrutamento


def valid_data() -> dict:
    return {
        "intencao": "definir_vaga",
        "resumo": "Vaga backend de nível pleno.",
        "competencias_tecnicas": ["Python", "PostgreSQL"],
        "competencias_comportamentais": ["Colaboração"],
        "perguntas_sugeridas": ["Como você investigaria uma falha?"],
        "pontos_de_atencao": ["Confirmar faixa salarial"],
        "proximo_passo": "Validar requisitos com a gestão.",
        "confianca": 0.85,
    }


def test_schema_valido_e_instancia_pydantic() -> None:
    analysis = AnaliseRecrutamento.model_validate(valid_data())

    assert isinstance(analysis, AnaliseRecrutamento)
    assert analysis.competencias_tecnicas == ["Python", "PostgreSQL"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("confianca", 1.1), ("resumo", "   "), ("intencao", "contratar")],
)
def test_schema_rejeita_dados_invalidos(field: str, value: object) -> None:
    data = valid_data()
    data[field] = value

    with pytest.raises(ValidationError):
        AnaliseRecrutamento.model_validate(data)


def test_schema_rejeita_campos_extras_e_remove_duplicatas() -> None:
    data = valid_data()
    data["competencias_tecnicas"] = ["Python", "Python"]
    analysis = AnaliseRecrutamento.model_validate(data)
    assert analysis.competencias_tecnicas == ["Python"]

    data["decisao_automatica"] = "contratar"
    with pytest.raises(ValidationError):
        AnaliseRecrutamento.model_validate(data)
