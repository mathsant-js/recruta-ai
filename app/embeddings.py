"""Configuracao e validacao do unico modelo de embeddings permitido."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_ollama import OllamaEmbeddings

from app.config import Settings, get_settings


class EmbeddingConfigurationError(RuntimeError):
    """Explica falhas de credencial ou respostas invalidas do servico."""


def create_embeddings(settings: Settings | None = None) -> OllamaEmbeddings:
    """Cria OllamaEmbeddings local sem enviar a chave usada pela nuvem."""

    current = settings or get_settings()
    return OllamaEmbeddings(
        model=current.ollama_embedding_model,
        base_url=current.ollama_embedding_host,
        # O cliente Ollama injeta OLLAMA_API_KEY globalmente quando o cabecalho falta.
        # Um valor local inofensivo impede que a credencial cloud seja enviada ao localhost.
        client_kwargs={"headers": {"Authorization": "Local"}},
        async_client_kwargs={"headers": {"Authorization": "Local"}},
    )


def validar_embeddings_reais(embeddings: OllamaEmbeddings) -> int:
    """Exercita embed_query e embed_documents e devolve a dimensao confirmada."""

    try:
        query_vector = embeddings.embed_query("recrutamento justo e inclusivo")
        document_vectors = embeddings.embed_documents(
            ["Protecao de dados de candidatos.", "Entrevista por competencias."]
        )
    except Exception as exc:
        raise EmbeddingConfigurationError(
            "Falha ao acessar nomic-embed-text no Ollama local. "
            "Verifique se o servico esta ativo e se o modelo foi instalado."
        ) from exc

    vetores: Sequence[Sequence[float]] = [query_vector, *document_vectors]
    dimensoes = {len(vetor) for vetor in vetores}
    if 0 in dimensoes or len(dimensoes) != 1:
        raise EmbeddingConfigurationError("O servico retornou vetores vazios ou inconsistentes.")
    return dimensoes.pop()
