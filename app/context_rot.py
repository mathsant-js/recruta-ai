"""Experimento reproduzivel de degradacao por truncamento de contexto.

Execute com ``python -m app.context_rot``. A chamada e real e usa exclusivamente o
modelo configurado em :mod:`app.config`; nenhum resultado e gerado sem resposta do LLM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Final, Literal

import tiktoken
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict

from app.chain import create_chat_llm
from app.config import OLLAMA_MODEL
from app.prompts import ACTIVE_SYSTEM_PROMPT_VERSION, SYSTEM_PROMPT


WINDOWS: Final[tuple[tuple[str, int], ...]] = (
    ("Muito curta", 256),
    ("Curta", 512),
    ("Referencia minima", 800),
    ("Escolha do chatbot", 1_200),
    ("Limite superior", 1_500),
)
FACT_IDS: Final[tuple[str, ...]] = (
    "cargo_analista_dados",
    "senioridade_pleno",
    "python_obrigatorio",
    "sql_obrigatorio",
    "airflow_obrigatorio",
    "dbt_desejavel",
    "hibrido_sp_dois_dias",
    "ingles_intermediario",
)
FACT_BLOCK: Final = """REGISTRO OFICIAL DA VAGA (fonte unica de requisitos):
- cargo_analista_dados: cargo de Analista de Dados.
- senioridade_pleno: senioridade plena.
- python_obrigatorio: Python e requisito obrigatorio.
- sql_obrigatorio: SQL e requisito obrigatorio.
- airflow_obrigatorio: Apache Airflow e requisito obrigatorio.
- dbt_desejavel: dbt e requisito desejavel.
- hibrido_sp_dois_dias: trabalho hibrido em Sao Paulo, dois dias presenciais por semana.
- ingles_intermediario: ingles intermediario e requisito obrigatorio.
"""
FINAL_QUESTION: Final = """Com base SOMENTE no historico acima, devolva o JSON solicitado.
Liste em `requisitos_recuperados` apenas os identificadores literais do registro oficial que
voce realmente encontrou. Nao deduza requisitos. Em `informacoes_inventadas`, declare qualquer
afirmacao da sua propria resposta que nao esteja sustentada pelo historico. Separe obrigatorios
de desejaveis, aponte lacunas e preserve as restricoes da persona. O material deve recomendar
revisao humana. {format_instructions}"""

DISTRACTOR = """Nota administrativa {index}: a equipe discutiu a ordem das reunioes, o nome
provisorio das pastas, a cor dos marcadores e a agenda da sala. Esses detalhes operacionais nao
sao requisitos da vaga e devem ser ignorados na analise. Nenhuma tecnologia, competencia,
senioridade, localidade ou modalidade de trabalho adicional foi definida nesta nota.
"""


class ContextRotResponse(BaseModel):
    """Saida pequena e validada usada para pontuar recuperacao sem juiz externo."""

    model_config = ConfigDict(extra="forbid")

    requisitos_recuperados: list[
        Literal[
            "cargo_analista_dados",
            "senioridade_pleno",
            "python_obrigatorio",
            "sql_obrigatorio",
            "airflow_obrigatorio",
            "dbt_desejavel",
            "hibrido_sp_dois_dias",
            "ingles_intermediario",
        ]
    ]
    requisitos_obrigatorios: list[str]
    requisitos_desejaveis: list[str]
    lacunas: list[str]
    informacoes_inventadas: list[str]
    resumo: str
    revisao_humana_recomendada: bool


@dataclass(frozen=True)
class ExperimentResult:
    cenario: str
    janela_tokens: int
    tokens_contexto_original: int
    tokens_contexto_enviado: int
    tokens_descartados: int
    requisitos_corretos: int
    total_requisitos: int
    taxa_recuperacao: float
    requisitos_omitidos: str
    informacoes_inventadas: int
    aderencia_persona: int
    utilidade: float
    tempo_resposta_s: float
    erro: str
    resposta_validada: str


def _encoding() -> tiktoken.Encoding:
    """Usa um tokenizador documentado como aproximacao, pois Gemma nao e exposto."""

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def build_full_context(minimum_tokens: int = 1_430) -> str:
    """Monta um unico historico: fatos no inicio e distracao depois deles."""

    parts = [FACT_BLOCK]
    index = 1
    while count_tokens("\n".join(parts)) < minimum_tokens:
        parts.append(DISTRACTOR.format(index=index))
        index += 1
    return "\n".join(parts)


def truncate_context(text: str, token_limit: int) -> tuple[str, int, int]:
    """Simula TokenBufferMemory, preservando a cauda mais recente do historico."""

    token_ids = _encoding().encode(text)
    kept = token_ids[-token_limit:]
    return _encoding().decode(kept), len(token_ids), len(token_ids) - len(kept)


def build_prompt() -> ChatPromptTemplate:
    """Mantem system prompt e pergunta final identicos em todos os cenarios."""

    return ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "<historico>\n{context}\n</historico>\n\n" + FINAL_QUESTION)]
    )


def score_response(
    scenario: str,
    window: int,
    original_tokens: int,
    sent_tokens: int,
    elapsed: float,
    response: ContextRotResponse,
) -> ExperimentResult:
    recovered = set(response.requisitos_recuperados)
    correct = len(recovered.intersection(FACT_IDS))
    omitted = [fact_id for fact_id in FACT_IDS if fact_id not in recovered]
    invented = len(response.informacoes_inventadas)
    adherence = int(response.revisao_humana_recomendada and invented == 0)
    recovery_rate = correct / len(FACT_IDS)
    usefulness = round(5 * recovery_rate * (1 if adherence else 0.75), 2)
    return ExperimentResult(
        cenario=scenario,
        janela_tokens=window,
        tokens_contexto_original=original_tokens,
        tokens_contexto_enviado=sent_tokens,
        tokens_descartados=original_tokens - sent_tokens,
        requisitos_corretos=correct,
        total_requisitos=len(FACT_IDS),
        taxa_recuperacao=round(recovery_rate, 3),
        requisitos_omitidos="|".join(omitted),
        informacoes_inventadas=invented,
        aderencia_persona=adherence,
        utilidade=usefulness,
        tempo_resposta_s=round(elapsed, 3),
        erro="",
        resposta_validada=response.model_dump_json(),
    )


def run_experiment(llm: BaseLanguageModel | None = None) -> list[ExperimentResult]:
    """Executa uma chamada real e independente para cada janela."""

    model = llm or create_chat_llm()
    parser = PydanticOutputParser(pydantic_object=ContextRotResponse)
    chain = build_prompt() | model | parser
    full_context = build_full_context()
    results: list[ExperimentResult] = []

    for scenario, window in WINDOWS:
        context, original_tokens, discarded = truncate_context(full_context, window)
        started = perf_counter()
        try:
            response = chain.invoke(
                {
                    "context": context,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            elapsed = perf_counter() - started
            results.append(
                score_response(
                    scenario,
                    window,
                    original_tokens,
                    original_tokens - discarded,
                    elapsed,
                    response,
                )
            )
        except Exception as error:  # preserva a falha real no artefato, sem inventar dado
            elapsed = perf_counter() - started
            results.append(
                ExperimentResult(
                    scenario, window, original_tokens, original_tokens - discarded,
                    discarded, 0, len(FACT_IDS), 0.0, "|".join(FACT_IDS), 0, 0,
                    0.0, round(elapsed, 3), f"{type(error).__name__}: {error}", "",
                )
            )
    return results


def write_artifacts(results: list[ExperimentResult], output_dir: Path) -> None:
    """Grava dados brutos em CSV, tabela Markdown e metadados da execucao."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with (output_dir / "resultados.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = [
        "| Cenario | Janela | Enviados | Descartados | Recuperados | Inventados | Persona | Utilidade | Tempo (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        table.append(
            f"| {result.cenario} | {result.janela_tokens} | {result.tokens_contexto_enviado} "
            f"| {result.tokens_descartados} | {result.requisitos_corretos}/{result.total_requisitos} "
            f"| {result.informacoes_inventadas} | {result.aderencia_persona} "
            f"| {result.utilidade:.2f} | {result.tempo_resposta_s:.3f} |"
        )
    (output_dir / "tabela_comparativa.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    metadata = {
        "executado_em_utc": datetime.now(timezone.utc).isoformat(),
        "modelo": OLLAMA_MODEL,
        "system_prompt": f"system_prompt_{ACTIVE_SYSTEM_PROMPT_VERSION}.md",
        "tokenizador_aproximado": "tiktoken cl100k_base (nao e o tokenizador nativo do Gemma)",
        "pergunta_final_sha256": hashlib.sha256(FINAL_QUESTION.encode()).hexdigest(),
        "fatos_base": list(FACT_IDS),
        "observacao": "Resultados provenientes de chamadas reais; linhas com erro permanecem explicitas.",
    }
    (output_dir / "metadados.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o experimento real de context rot.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/context_rot"),
        help="Diretorio dos artefatos (padrao: artifacts/context_rot).",
    )
    args = parser.parse_args()
    results = run_experiment()
    write_artifacts(results, args.output_dir)
    failures = [result for result in results if result.erro]
    print(f"Artefatos gravados em {args.output_dir.resolve()}")
    if failures:
        raise SystemExit(f"{len(failures)} cenario(s) falharam; consulte resultados.csv")


if __name__ == "__main__":
    main()
