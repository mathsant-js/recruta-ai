"""Reordenação dos chunks recuperados com cross-encoder."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from app.retriever import DocumentoRecuperado

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_SEQUENTIAL_CHUNK = re.compile(r"^(?P<document>.+)-C(?P<number>\d+)$")


class ModeloCrossEncoder(Protocol):
    """Contrato mínimo usado para permitir testes sem baixar o modelo real."""

    def predict(self, sentences: list[list[str]]) -> Sequence[float]: ...


class CrossEncoderReranker:
    """Aplica o cross-encoder indicado no enunciado antes da geração.

    O modelo é carregado apenas na primeira consulta, evitando download e custo de
    inicialização durante imports, testes e abertura da interface.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        *,
        model: ModeloCrossEncoder | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model

    def _get_model(self) -> ModeloCrossEncoder:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "O reranking requer sentence-transformers. "
                    "Execute: pip install -r requirements.txt"
                ) from exc
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: Sequence[DocumentoRecuperado],
        *,
        top_n: int = 5,
    ) -> list[DocumentoRecuperado]:
        """Reordena documentos pela relevância conjunta pergunta-trecho."""

        consulta = query.strip()
        if not consulta:
            raise ValueError("A consulta do reranker não pode estar vazia.")
        if top_n < 1:
            raise ValueError("top_n deve ser positivo.")
        if not documents:
            return []

        pares = [[consulta, documento.conteudo] for documento in documents]
        scores = list(self._get_model().predict(pares))
        if len(scores) != len(documents):
            raise RuntimeError("O cross-encoder retornou quantidade inválida de scores.")

        ordenados = sorted(
            zip(documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [documento for documento, _ in ordenados[:top_n]]


def expand_selected_neighbors(
    selected: Sequence[DocumentoRecuperado],
    candidates: Sequence[DocumentoRecuperado],
    *,
    neighbors: int = 1,
) -> list[DocumentoRecuperado]:
    """Reanexa contexto contíguo sem alterar quais hits o modelo selecionou.

    O cross-encoder pontua chunks isolados e pode escolher a continuação de uma frase.
    Esta etapa recupera os vizinhos já presentes nos candidatos, sem nova busca.
    """

    by_id = {item.chunk_id: item for item in candidates}
    result: list[DocumentoRecuperado] = []
    seen: set[str] = set()
    for item in selected:
        match = _SEQUENTIAL_CHUNK.match(item.chunk_id)
        ids = [item.chunk_id]
        if match is not None:
            number = int(match.group("number"))
            width = len(match.group("number"))
            ids = [
                f"{match.group('document')}-C{current:0{width}d}"
                for current in range(max(1, number - neighbors), number + neighbors + 1)
            ]
        for chunk_id in ids:
            if chunk_id in by_id and chunk_id not in seen:
                result.append(by_id[chunk_id])
                seen.add(chunk_id)
    return result
