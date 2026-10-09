"""Carregamento validado da base documental do CKP02.

O manifesto e a fonte de metadata e de integridade. Os loaders nao tentam corrigir
silenciosamente arquivos ausentes, duplicados, corrompidos ou sem texto extraivel.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from pypdf import PdfReader

ROOT = Path(__file__).parents[1]
DEFAULT_MANIFEST_PATH = ROOT / "data" / "manifest.json"
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


class DocumentLoadError(RuntimeError):
    """Indica que a base nao pode ser ingerida com seguranca."""


def _normalizar_texto(texto: str) -> str:
    """Remove ruido tecnico sem destruir paragrafos e titulos uteis ao splitter."""

    texto = texto.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    linhas = [re.sub(r"[ \t]+", " ", linha).strip() for linha in texto.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(linhas)).strip()


def _metadata_base(registro: dict[str, Any]) -> dict[str, Any]:
    """Seleciona apenas valores escalares aceitos pelo ChromaDB."""

    return {
        "document_id": registro["document_id"],
        "source": registro["source_page_url"],
        "title": registro["title"],
        "organization": registro["organization"],
        "document_type": registro["document_type"],
        "category": registro["category"],
        "publication_date": registro["publication_date"],
        "language": registro.get("language", "pt-BR"),
    }


def _carregar_pdf(caminho: Path, metadata: dict[str, Any]) -> list[Document]:
    try:
        leitor = PdfReader(caminho)
    except Exception as exc:  # pypdf varia o tipo conforme a corrupcao encontrada
        raise DocumentLoadError(f"PDF invalido ou corrompido: {caminho.name}") from exc

    paginas: list[Document] = []
    for numero, pagina in enumerate(leitor.pages, start=1):
        texto = _normalizar_texto(pagina.extract_text() or "")
        if texto:
            paginas.append(Document(page_content=texto, metadata={**metadata, "page": numero}))
    return paginas


def _carregar_texto(caminho: Path, metadata: dict[str, Any]) -> list[Document]:
    try:
        texto = _normalizar_texto(caminho.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise DocumentLoadError(f"Arquivo de texto ilegivel: {caminho.name}") from exc
    return [Document(page_content=texto, metadata={**metadata, "page": 1})] if texto else []


def carregar_documentos(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    *,
    root: Path = ROOT,
) -> list[Document]:
    """Carrega PDF, TXT e Markdown apos validar manifesto, hash e duplicidade."""

    try:
        manifesto = json.loads(manifest_path.read_text(encoding="utf-8"))
        registros = manifesto["documents"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DocumentLoadError(f"Manifesto invalido: {manifest_path}") from exc

    documentos: list[Document] = []
    ids: set[str] = set()
    hashes: set[str] = set()
    for registro in registros:
        document_id = str(registro.get("document_id", "")).strip()
        if not document_id or document_id in ids:
            raise DocumentLoadError(f"Identificador ausente ou duplicado: {document_id!r}")
        ids.add(document_id)

        caminho = root / registro["local_file"]
        if not caminho.is_file():
            raise DocumentLoadError(f"Documento ausente: {caminho}")
        if caminho.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise DocumentLoadError(f"Formato nao suportado: {caminho.suffix}")

        hash_real = hashlib.sha256(caminho.read_bytes()).hexdigest()
        hash_esperado = str(registro.get("sha256", ""))
        if hash_real != hash_esperado:
            raise DocumentLoadError(f"Hash divergente para {document_id}: {caminho.name}")
        if hash_real in hashes:
            raise DocumentLoadError(f"Conteudo duplicado detectado em {document_id}")
        hashes.add(hash_real)

        metadata = _metadata_base(registro)
        carregados = (
            _carregar_pdf(caminho, metadata)
            if caminho.suffix.lower() == ".pdf"
            else _carregar_texto(caminho, metadata)
        )
        if not carregados:
            raise DocumentLoadError(f"Nenhum texto extraivel em {document_id}")
        documentos.extend(carregados)

    if not documentos:
        raise DocumentLoadError("O manifesto nao contem documentos carregaveis.")
    return documentos
