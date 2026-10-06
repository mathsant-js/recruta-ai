"""Testes determinísticos dos loaders da fase 2."""

import hashlib
import json
from pathlib import Path

import pytest

from app.document_loader import DocumentLoadError, carregar_documentos


def _gravar_manifesto(tmp_path: Path, arquivos: list[Path]) -> Path:
    documentos = []
    for indice, arquivo in enumerate(arquivos, start=1):
        documentos.append(
            {
                "document_id": f"TESTE-{indice}",
                "title": f"Documento {indice}",
                "organization": "Organizacao ficticia para teste",
                "publication_date": "2026",
                "document_type": "manual",
                "category": "teste",
                "language": "pt-BR",
                "source_page_url": "https://example.invalid/documento",
                "local_file": arquivo.name,
                "sha256": hashlib.sha256(arquivo.read_bytes()).hexdigest(),
            }
        )
    caminho = tmp_path / "manifest.json"
    caminho.write_text(json.dumps({"documents": documentos}), encoding="utf-8")
    return caminho


def test_carrega_markdown_e_preserva_metadata(tmp_path: Path) -> None:
    arquivo = tmp_path / "guia.md"
    arquivo.write_text("# Titulo\n\nTexto   com   espacos.\n", encoding="utf-8")
    manifesto = _gravar_manifesto(tmp_path, [arquivo])

    documentos = carregar_documentos(manifesto, root=tmp_path)

    assert len(documentos) == 1
    assert documentos[0].page_content == "# Titulo\n\nTexto com espacos."
    assert documentos[0].metadata["document_id"] == "TESTE-1"
    assert documentos[0].metadata["page"] == 1


def test_rejeita_conteudo_duplicado(tmp_path: Path) -> None:
    primeiro = tmp_path / "a.txt"
    segundo = tmp_path / "b.txt"
    primeiro.write_text("mesmo conteudo", encoding="utf-8")
    segundo.write_text("mesmo conteudo", encoding="utf-8")
    manifesto = _gravar_manifesto(tmp_path, [primeiro, segundo])

    with pytest.raises(DocumentLoadError, match="duplicado"):
        carregar_documentos(manifesto, root=tmp_path)


def test_rejeita_hash_divergente(tmp_path: Path) -> None:
    arquivo = tmp_path / "guia.txt"
    arquivo.write_text("versao inicial", encoding="utf-8")
    manifesto = _gravar_manifesto(tmp_path, [arquivo])
    arquivo.write_text("arquivo alterado", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="Hash divergente"):
        carregar_documentos(manifesto, root=tmp_path)
