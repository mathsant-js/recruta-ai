# Revisão humana do meta prompting

## Escopo e rastreabilidade

- Prompt submetido ao modelo: `prompts/system_prompt_v2.md`.
- Modelo revisor: `gemma4:cloud`.
- Saída original e validada: `artifacts/meta_prompting/critica_modelo.json`.
- Versão final após revisão humana: `prompts/system_prompt_v3.md`.
- Regra editorial: nenhuma sugestão do modelo foi aplicada automaticamente.

## Decisões

| ID | Decisão humana | Alteração na v3 | Justificativa |
|---|---|---|---|
| SUG-001 | Aceita com ajuste | Triagens devem relacionar conclusões a evidência textual concisa e usar “não informado” quando ela não existir. | Reduz inferência de competências. A sugestão de transcrever trechos “exatos” foi limitada para evitar reprodução desnecessária de dados pessoais. |
| SUG-002 | Aceita | Documentos fornecidos passaram a ser tratados explicitamente como dados não confiáveis. | Fecha a lacuna de injeção indireta por currículos e anexos. |
| SUG-003 | Rejeitada | O redirecionamento existente foi mantido. | Encaminhar sempre à política interna ou a especialista pode sugerir recursos inexistentes e não define melhor uma resposta segura de RH. |
| SUG-004 | Aceita com ajuste | Dados pessoais ou sensíveis desnecessários não devem ser repetidos nem usados; o assistente recomenda remoção ou anonimização. | Atende minimização e resposta a vazamento sem alegar conformidade jurídica automática ou emitir aconselhamento legal. |

## Antes e depois

| Critério observado | v2 (antes) | v3 (depois) |
|---|---|---|
| Evidência em triagem | Restringia a critérios profissionais informados, sem exigir vínculo entre conclusão e evidência. | Exige evidência textual concisa e impede dedução por similaridade. |
| Injeção indireta | Tratava histórico e solicitação como não confiáveis. | Inclui explicitamente documentos fornecidos. |
| Dado sensível já enviado | Desencorajava coleta e recomendava anonimização. | Também proíbe repetir/usar o dado desnecessário e recomenda sua remoção ou anonimização. |
| Limite fora do domínio | Redirecionamento breve para ação segura de RH. | Mantido por decisão humana. |

## Resultados observados

A execução real produziu quatro sugestões estruturadas: uma foi aceita integralmente, duas foram
aceitas com ajustes de escopo e uma foi rejeitada (SUG-003). Assim, nenhuma recomendação passou
diretamente do modelo para o prompt final sem uma decisão humana registrada.

Os testes determinísticos verificam que a v3 preserva as restrições anteriores e acrescenta as
três coberturas aprovadas. Eles demonstram presença e integração das regras, não garantem por si
sós o comportamento probabilístico do modelo em todos os ataques. O risco residual de jailbreak,
inferência sutil e variação de resposta permanece e exige revisão humana.
