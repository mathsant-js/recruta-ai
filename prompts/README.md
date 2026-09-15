# Prompts versionados

Os arquivos desta pasta são artefatos versionados e não devem ser sobrescritos. Uma mudança de
conteúdo deve gerar um novo arquivo com o próximo sufixo (`_v3`, `_v4` etc.) e a constante de versão
ativa em `app/prompts.py` deve ser atualizada explicitamente.

## Versões

| Artefato | Estado | Motivo |
|---|---|---|
| `system_prompt_v1.md` | histórico | Primeira persona completa do domínio. |
| `system_prompt_v2.md` | histórico/antes | Versão submetida à crítica real do meta prompting. |
| `system_prompt_v3.md` | ativo/final | Revisão humana da crítica: evidência em triagem, documentos não confiáveis e resposta a dados sensíveis. |
| `chat_human_v1.md` | ativo | Mantém histórico e solicitação em uma mensagem `human` separada. |
| `structured_analysis_human_v1.md` | ativo | Fornece conversa, pedido e formato à chain estruturada LCEL. |
| `structured_retry_human_v1.md` | ativo | Limita a correção de uma saída inválida a uma nova tentativa. |

A versão 2 foi produzida por revisão humana e preservada como o “antes”. O `gemma4:cloud` gerou
uma crítica estruturada dessa versão; uma pessoa revisou cada sugestão antes de criar a versão 3.
A saída original está em `artifacts/meta_prompting/critica_modelo.json` e as decisões, inclusive
rejeições e ajustes, estão em `artifacts/meta_prompting/revisao_humana.md`.
