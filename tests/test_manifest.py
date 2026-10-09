"""Testes determinísticos do catálogo de fontes reais do CKP02."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "data" / "manifest.json"


def _carregar_documentos() -> list[dict]:
    """Carrega os documentos do manifesto sem depender de rede ou modelos."""

    manifesto = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return manifesto["documents"]


def test_manifesto_possui_cinco_fontes_rastreaveis() -> None:
    """Garante o mínimo do enunciado e identificadores sem duplicidade."""

    documentos = _carregar_documentos()
    ids = [documento["document_id"] for documento in documentos]

    assert len(documentos) >= 5
    assert len(ids) == len(set(ids))
    assert all(documento["source_page_url"].startswith("https://") for documento in documentos)
    assert all(documento["download_url"].startswith("https://") for documento in documentos)


def test_arquivos_correspondem_aos_hashes_do_manifesto() -> None:
    """Detecta arquivos ausentes, trocados, corrompidos ou alterados sem revisão."""

    for documento in _carregar_documentos():
        caminho = ROOT / documento["local_file"]
        conteudo = caminho.read_bytes()

        assert conteudo.startswith(b"%PDF-")
        assert len(conteudo) == documento["file_size_bytes"]
        assert hashlib.sha256(conteudo).hexdigest() == documento["sha256"]


def test_metadata_minima_esta_preenchida() -> None:
    """Mantém os campos necessários à ingestão, aos filtros e às citações."""

    campos_obrigatorios = {
        "document_id",
        "title",
        "organization",
        "publication_date",
        "document_type",
        "category",
        "local_file",
        "rights_note",
        "inclusion_reason",
        "supported_topics",
    }

    for documento in _carregar_documentos():
        assert campos_obrigatorios <= documento.keys()
        assert all(documento[campo] for campo in campos_obrigatorios)
