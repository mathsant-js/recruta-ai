"""Prompts versionados e templates do domínio de recrutamento e RH."""

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
ACTIVE_SYSTEM_PROMPT_VERSION = "v3"
ACTIVE_CHAT_HUMAN_PROMPT_VERSION = "v1"
ACTIVE_STRUCTURED_HUMAN_PROMPT_VERSION = "v1"
ACTIVE_STRUCTURED_RETRY_PROMPT_VERSION = "v1"

SUPPORTED_CASES = (
    "levantamento e revisão de vagas",
    "briefing com gestores",
    "triagem por critérios profissionais informados",
    "preparação de entrevistas",
    "comunicação profissional com candidatos",
    "organização e análise de informações de recrutamento",
)

DOMAIN_LIMITS = (
    "não decide contratação, rejeição, promoção ou demissão",
    "não usa nem infere atributos sensíveis",
    "não produz diagnósticos psicológicos ou médicos",
    "não inventa informações ausentes",
    "não substitui revisão humana",
)


def _load_prompt(filename: str) -> str:
    """Carrega um artefato de prompt sem permitir caminhos fora do diretório."""

    path = PROMPTS_DIR / filename
    if path.parent != PROMPTS_DIR or not path.is_file():
        raise FileNotFoundError(f"Prompt versionado não encontrado: {filename}")
    return path.read_text(encoding="utf-8").strip()


SYSTEM_PROMPT = _load_prompt(f"system_prompt_{ACTIVE_SYSTEM_PROMPT_VERSION}.md")
CHAT_HUMAN_PROMPT = _load_prompt(
    f"chat_human_{ACTIVE_CHAT_HUMAN_PROMPT_VERSION}.md"
)
STRUCTURED_ANALYSIS_HUMAN_PROMPT = _load_prompt(
    f"structured_analysis_human_{ACTIVE_STRUCTURED_HUMAN_PROMPT_VERSION}.md"
)
STRUCTURED_RETRY_HUMAN_PROMPT = _load_prompt(
    f"structured_retry_human_{ACTIVE_STRUCTURED_RETRY_PROMPT_VERSION}.md"
)


def build_chat_prompt() -> ChatPromptTemplate:
    """Cria o template do chat, com mensagens `system` e `human` separadas."""

    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", CHAT_HUMAN_PROMPT),
        ]
    )


def build_structured_analysis_prompt() -> ChatPromptTemplate:
    """Cria o template que será ligado ao parser Pydantic na etapa da LCEL."""

    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", STRUCTURED_ANALYSIS_HUMAN_PROMPT),
        ]
    )


def build_structured_retry_prompt() -> ChatPromptTemplate:
    """Solicita uma única correção quando a primeira saída não passa no schema."""

    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", STRUCTURED_RETRY_HUMAN_PROMPT),
        ]
    )


CHAT_PROMPT = build_chat_prompt()
STRUCTURED_ANALYSIS_PROMPT = build_structured_analysis_prompt()
STRUCTURED_RETRY_PROMPT = build_structured_retry_prompt()
