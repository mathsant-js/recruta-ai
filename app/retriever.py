"""Contrato público e inicialização tardia da recuperação documental do CKP02.

O índice e o modelo de embeddings só são abertos na primeira busca. Assim, importar o
módulo continua sem efeitos colaterais e a mesma função poderá virar uma tool no CKP03.
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
    """Indica que o índice persistente não pôde ser conectado ao contrato."""


BackendBusca: TypeAlias = Callable[
    [str, int, FiltroMetadata | None], Sequence[DocumentoRecuperado]
]


def _backend_padrao(
    consulta: str,
    top_k: int,
    filtros: FiltroMetadata | None,
) -> Sequence[DocumentoRecuperado]:
    """Abre o índice somente quando a primeira consulta realmente acontece."""

    global _backend_busca
    try:
        # Imports locais evitam ciclo e acesso ao disco durante importações e testes.
        from app.embeddings import create_embeddings
        from app.vector_store import abrir_indice, buscar_contexto_no_indice

        store = abrir_indice(create_embeddings())
    except (FileNotFoundError, OSError) as exc:
        raise RecuperadorNaoConfiguradoError(
            "O índice documental do CKP02 não está disponível. "
            "Execute python -m app.ingestion antes de realizar buscas."
        ) from exc

    def backend_indice(
        pergunta: str,
        quantidade: int,
        filtro: FiltroMetadata | None,
    ) -> Sequence[DocumentoRecuperado]:
        return buscar_contexto_no_indice(store, pergunta, quantidade, filtro)

    _backend_busca = backend_indice
    return backend_indice(consulta, top_k, filtros)


_backend_busca: BackendBusca = _backend_padrao


def restaurar_backend_padrao() -> None:
    """Descarta a conexão em cache, útil após reconstruir o índice ou em testes."""

    global _backend_busca
    _backend_busca = _backend_padrao


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
