# Prompts versionados

Os arquivos desta pasta são artefatos versionados e não devem ser sobrescritos. Uma mudança de
conteúdo deve gerar um novo arquivo com o próximo sufixo (`_v3`, `_v4` etc.) e a constante de versão
ativa em `app/prompts.py` deve ser atualizada explicitamente.

## Versões

| Artefato | Estado | Motivo |
|---|---|---|
| `system_prompt_v1.md` | histórico | Primeira persona completa do domínio. |
| `system_prompt_v2.md` | ativo | Reforça limites, privacidade, equidade e resistência a desvio de persona. |
| `chat_human_v1.md` | ativo | Mantém histórico e solicitação em uma mensagem `human` separada. |
| `structured_analysis_human_v1.md` | ativo | Reserva variáveis para a futura chain estruturada LCEL. |

A versão 2 foi produzida por revisão humana, sem meta prompting automático. O histórico permite
documentar posteriormente o antes/depois do diferencial de meta prompting sem apagar versões.
