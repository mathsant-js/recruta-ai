"""Testes deterministas do desenho experimental de context rot."""

import json
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from app.context_rot import (
    FACT_IDS,
    ExperimentResult,
    build_full_context,
    count_tokens,
    run_experiment,
    truncate_context,
    write_artifacts,
)


def test_contexto_base_e_identico_e_janelas_truncam_a_cauda() -> None:
    context = build_full_context()
    short, original, discarded = truncate_context(context, 256)
    long, original_again, discarded_long = truncate_context(context, 1_500)

    assert original == original_again
    assert count_tokens(short) <= 256
    assert count_tokens(long) <= 1_500
    assert discarded > discarded_long
    assert "cargo_analista_dados" not in short
    assert all(fact_id in long for fact_id in FACT_IDS)


def test_artefatos_sao_gerados_sem_fabricar_resultados(tmp_path: Path) -> None:
    result = ExperimentResult(
        "Teste", 256, 1_430, 256, 1_174, 2, 8, 0.25, "sql_obrigatorio",
        0, 1, 1.25, 0.5, "", '{"requisitos_recuperados":[]}',
    )
    write_artifacts([result], tmp_path)

    assert "taxa_recuperacao" in (tmp_path / "resultados.csv").read_text()
    assert "2/8" in (tmp_path / "tabela_comparativa.md").read_text()
    assert '"modelo": "gemma4:cloud"' in (tmp_path / "metadados.json").read_text()


def test_experimento_executa_as_cinco_janelas_com_saida_validada() -> None:
    resposta = {
        "requisitos_recuperados": [],
        "requisitos_obrigatorios": [],
        "requisitos_desejaveis": [],
        "lacunas": ["Requisitos não encontrados na janela recebida."],
        "informacoes_inventadas": [],
        "resumo": "Não há requisitos recuperáveis nesta resposta de teste.",
        "revisao_humana_recomendada": True,
    }
    llm_local = RunnableLambda(lambda _: json.dumps(resposta, ensure_ascii=False))

    resultados = run_experiment(llm=llm_local)

    assert len(resultados) == 5
    assert [resultado.janela_tokens for resultado in resultados] == [
        256,
        512,
        800,
        1_200,
        1_500,
    ]
    assert all(not resultado.erro for resultado in resultados)
    assert all(resultado.resposta_validada for resultado in resultados)
