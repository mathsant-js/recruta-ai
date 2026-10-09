"""Testes determinísticos das regras de consolidação da fase 4."""

import json
from pathlib import Path

import pytest

from app.evaluation import EvaluationError, _escolher_vencedora, carregar_dataset


def test_dataset_oficial_cobre_minimo_e_casos_criticos() -> None:
    items = carregar_dataset()

    assert len(items) >= 5
    assert any(item["category"] == "privacidade" for item in items)
    assert any("discrimin" in item["category"] for item in items)
    assert any(not item["answerable"] for item in items)


def test_dataset_incompleto_e_rejeitado(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"items": [{"id": "Q1"}]}), encoding="utf-8")

    with pytest.raises(EvaluationError, match="cinco"):
        carregar_dataset(path)


def test_escolha_prioriza_faithfulness_e_ignora_config_sem_metrica() -> None:
    summaries = [
        {
            "chunking_config": "a",
            "faithfulness_mean": 0.82,
            "answer_relevancy_mean": 0.70,
            "expected_source_recall_mean": 1.0,
            "answer_latency_mean_seconds": 2.0,
        },
        {
            "chunking_config": "b",
            "faithfulness_mean": 0.90,
            "answer_relevancy_mean": 0.60,
            "expected_source_recall_mean": 0.8,
            "answer_latency_mean_seconds": 3.0,
        },
        {
            "chunking_config": "c",
            "faithfulness_mean": None,
            "answer_relevancy_mean": None,
            "expected_source_recall_mean": 1.0,
            "answer_latency_mean_seconds": 1.0,
        },
    ]

    assert _escolher_vencedora(summaries)["chunking_config"] == "b"
