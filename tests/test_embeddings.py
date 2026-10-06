"""Testes sem rede da configuracao e validacao de embeddings."""

import pytest

from app.config import Settings
from app.embeddings import (
    EmbeddingConfigurationError,
    create_embeddings,
    validar_embeddings_reais,
)


class EmbeddingsControlados:
    def embed_query(self, texto: str) -> list[float]:
        del texto
        return [0.1, 0.2, 0.3]

    def embed_documents(self, textos: list[str]) -> list[list[float]]:
        return [[float(indice), 0.2, 0.3] for indice, _ in enumerate(textos)]


class EmbeddingsInconsistentes(EmbeddingsControlados):
    def embed_documents(self, textos: list[str]) -> list[list[float]]:
        del textos
        return [[0.1], []]


def test_embeddings_locais_nao_exigem_nem_repassam_chave_cloud() -> None:
    embeddings = create_embeddings(Settings(ollama_api_key="chave-ficticia"))

    assert str(embeddings._client._client.base_url) == "http://localhost:11434"
    assert embeddings._client._client.headers["authorization"] == "Local"
    assert embeddings._client._client.headers["authorization"] != "Bearer chave-ficticia"


def test_valida_query_documentos_e_dimensao() -> None:
    assert validar_embeddings_reais(EmbeddingsControlados()) == 3  # type: ignore[arg-type]


def test_rejeita_vetores_inconsistentes() -> None:
    with pytest.raises(EmbeddingConfigurationError, match="inconsistentes"):
        validar_embeddings_reais(EmbeddingsInconsistentes())  # type: ignore[arg-type]
