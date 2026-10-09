"""Pipeline retrieve → generate fundamentado da fase 3 do CKP02."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser

from app.chain import create_chat_llm
from app.prompts import RAG_PROMPT
from app.reranker import CrossEncoderReranker, expand_selected_neighbors
from app.retriever import DocumentoRecuperado, FiltroMetadata, buscar

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Não encontrei sustentação adequada nos documentos recuperados. Reformule a "
    "pergunta ou adicione uma fonte apropriada à base."
)
_CHUNK_CITATION = re.compile(r"[A-Za-z0-9_-]+-C\d+")


class CitacaoInvalidaError(RuntimeError):
    """Impede que identificadores não recuperados sejam apresentados como fonte."""


@dataclass(frozen=True, slots=True)
class RespostaRAG:
    """Resposta final acompanhada exatamente pelas evidências recuperadas."""

    pergunta: str
    resposta: str
    fontes: tuple[DocumentoRecuperado, ...]
    contextos_recuperados: tuple[DocumentoRecuperado, ...] = ()

    @property
    def texto_formatado(self) -> str:
        """Renderiza resposta e fontes verificáveis no formato pedido pelo plano."""

        if not self.fontes:
            return self.resposta
        linhas = [self.resposta, "", "Fontes:"]
        for fonte in self.fontes:
            pagina = f", p. {fonte.pagina}" if fonte.pagina is not None else ""
            linhas.append(
                f"- [{fonte.chunk_id}{pagina}] {fonte.titulo} — {fonte.fonte}"
            )
        return "\n".join(linhas)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.pergunta,
            "answer": self.resposta,
            "formatted_answer": self.texto_formatado,
            "sources": [fonte.as_dict() for fonte in self.fontes],
            "retrieved_contexts": [
                contexto.as_dict() for contexto in self.contextos_recuperados
            ],
        }


Busca = Callable[[str], Sequence[DocumentoRecuperado]]


def _formatar_contexto(documentos: Sequence[DocumentoRecuperado]) -> str:
    """Delimita cada chunk e sua metadata sem interpolar instruções no system prompt."""

    blocos = []
    for documento in documentos:
        pagina = str(documento.pagina) if documento.pagina is not None else "não informada"
        blocos.append(
            "<document>\n"
            f"<chunk_id>{documento.chunk_id}</chunk_id>\n"
            f"<title>{documento.titulo}</title>\n"
            f"<page>{pagina}</page>\n"
            f"<content>{documento.conteudo}</content>\n"
            "</document>"
        )
    return "\n\n".join(blocos)


def _fontes_citadas(
    resposta: str,
    recuperados: Sequence[DocumentoRecuperado],
) -> tuple[DocumentoRecuperado, ...]:
    """Valida citações do modelo e devolve somente fontes efetivamente citadas."""

    por_id = {documento.chunk_id: documento for documento in recuperados}
    # Aceita tanto ``[DOC-01-C0001]`` quanto citações agrupadas no formato
    # ``[DOC-01-C0001, DOC-01-C0002]`` sem deixar o segundo ID sem validação.
    ids = list(dict.fromkeys(_CHUNK_CITATION.findall(resposta)))
    invalidos = [chunk_id for chunk_id in ids if chunk_id not in por_id]
    if invalidos:
        raise CitacaoInvalidaError(
            "A resposta tentou citar chunks não recuperados: " + ", ".join(invalidos)
        )
    return tuple(por_id[chunk_id] for chunk_id in ids)


class RAGService:
    """Orquestra busca isolada, geração determinística e validação das citações."""

    def __init__(
        self,
        llm: BaseLanguageModel | None = None,
        search: Callable[..., Sequence[DocumentoRecuperado]] = buscar,
        *,
        top_k: int = 5,
        minimum_score: float = 0.2,
        reranker: CrossEncoderReranker | None = None,
        candidate_k: int = 10,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k deve ser positivo.")
        self.llm = llm or create_chat_llm(temperature=0)
        self.search = search
        self.top_k = top_k
        if candidate_k < top_k:
            raise ValueError("candidate_k deve ser maior ou igual a top_k.")
        self.candidate_k = candidate_k
        self.minimum_score = minimum_score
        self.reranker = reranker
        self.chain = RAG_PROMPT | self.llm | StrOutputParser()

    def answer(
        self,
        question: str,
        *,
        filters: FiltroMetadata | None = None,
    ) -> RespostaRAG:
        """Executa retrieve → generate ou recusa quando não há evidência mínima."""

        pergunta = question.strip()
        if not pergunta:
            raise ValueError("A pergunta não pode estar vazia.")
        quantidade = self.candidate_k if self.reranker is not None else self.top_k
        recuperados = tuple(self.search(pergunta, top_k=quantidade, filtros=filters))
        elegiveis = tuple(
            documento
            for documento in recuperados
            if documento.score >= self.minimum_score
        )
        if self.reranker is not None:
            selecionados = self.reranker.rerank(
                pergunta, elegiveis, top_n=self.top_k
            )
            elegiveis = tuple(
                expand_selected_neighbors(selecionados, elegiveis)
            )
        if not elegiveis:
            return RespostaRAG(pergunta, INSUFFICIENT_EVIDENCE_MESSAGE, ())

        resposta = self.chain.invoke(
            {"question": pergunta, "context": _formatar_contexto(elegiveis)}
        ).strip()
        if not resposta:
            raise RuntimeError("O modelo retornou uma resposta vazia.")
        if INSUFFICIENT_EVIDENCE_MESSAGE in resposta:
            return RespostaRAG(
                pergunta,
                INSUFFICIENT_EVIDENCE_MESSAGE,
                (),
                elegiveis,
            )

        fontes = _fontes_citadas(resposta, elegiveis)
        if not fontes:
            raise CitacaoInvalidaError(
                "A resposta fundamentada não citou nenhum chunk recuperado."
            )
        return RespostaRAG(pergunta, resposta, fontes, elegiveis)
