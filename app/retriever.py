"""Contrato público da camada de recuperação documental do CKP02.

Nesta fase o módulo não abre conexões, não carrega documentos e não cria coleções no
ChromaDB durante o import. A implementação concreta será conectada após a ingestão da
base, mantendo este contrato estável para a interface e para o futuro agente do CKP03.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias


FiltroMetadata: TypeAlias = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DocumentoRecuperado:
    """Trecho recuperado com os dados necessários para evidência e citação."""

    conteudo: str
    score: float
    titulo: str
    fonte: str
    pagina: int | None
    categoria: str
    chunk_id: str

    def __post_init__(self) -> None:
        """Impede resultados incompletos de atravessarem a fronteira do recuperador."""

        campos_textuais = {
            "conteudo": self.conteudo,
            "titulo": self.titulo,
            "fonte": self.fonte,
            "categoria": self.categoria,
            "chunk_id": self.chunk_id,
        }
        vazios = [nome for nome, valor in campos_textuais.items() if not valor.strip()]
        if vazios:
            raise ValueError(
                "Campos obrigatórios do documento não podem ser vazios: "
                + ", ".join(vazios)
            )
        if self.pagina is not None and self.pagina < 1:
            raise ValueError("A página deve ser positiva quando informada.")

    def as_dict(self) -> dict[str, Any]:
        """Serializa uma evidencia sem perder os campos do contrato publico."""

        return {
            "conteudo": self.conteudo,
            "score": self.score,
            "titulo": self.titulo,
            "fonte": self.fonte,
            "pagina": self.pagina,
            "categoria": self.categoria,
            "chunk_id": self.chunk_id,
        }


class RecuperadorNaoConfiguradoError(RuntimeError):
    """Indica que a fase de ingestão ainda não conectou um índice ao contrato."""


BackendBusca: TypeAlias = Callable[
    [str, int, FiltroMetadata | None], Sequence[DocumentoRecuperado]
]


def _backend_nao_configurado(
    consulta: str,
    top_k: int,
    filtros: FiltroMetadata | None,
) -> Sequence[DocumentoRecuperado]:
    """Falha de forma explícita enquanto o índice da fase 2 não existe."""

    del consulta, top_k, filtros
    raise RecuperadorNaoConfiguradoError(
        "O índice documental do CKP02 ainda não foi construído. "
        "Execute a fase de ingestão antes de realizar buscas."
    )


_backend_busca: BackendBusca = _backend_nao_configurado


def configurar_backend_busca(backend: BackendBusca) -> None:
    """Conecta uma implementação ao contrato sem expor ChromaDB aos consumidores."""

    if not callable(backend):
        raise TypeError("O backend de busca deve ser chamável.")
    global _backend_busca
    _backend_busca = backend


def buscar(
    consulta: str,
    *,
    top_k: int = 5,
    filtros: FiltroMetadata | None = None,
) -> list[DocumentoRecuperado]:
    """Recupera trechos relevantes por uma API estável e reutilizável no CKP03."""

    consulta_normalizada = consulta.strip()
    if not consulta_normalizada:
        raise ValueError("A consulta não pode estar vazia.")
    if top_k < 1:
        raise ValueError("top_k deve ser maior ou igual a 1.")
    if filtros is not None and not isinstance(filtros, Mapping):
        raise TypeError("filtros deve ser um mapeamento ou None.")

    resultados = list(_backend_busca(consulta_normalizada, top_k, filtros))
    if any(not isinstance(item, DocumentoRecuperado) for item in resultados):
        raise TypeError("O backend retornou um item fora do contrato DocumentoRecuperado.")
    return resultados
