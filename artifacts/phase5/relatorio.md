# Fase 5 — comparação dos diferenciais

Execução real em `2026-10-08T20:17:42.721859-03:00` sobre 7 perguntas respondíveis. O índice e o cross-encoder foram aquecidos antes da medição.

| Estratégia | Perguntas | Recall médio de fontes | Latência média |
|---|---:|---:|---:|
| Busca vetorial | 7 | 0.929 | 0.237 s |
| Cross-encoder | 7 | 1.000 | 4.252 s |
| Filtro por metadata | 6 | 1.000 | ver CSV |

O filtro foi avaliado apenas quando todos os documentos esperados tinham a mesma categoria. O cross-encoder usa 10 candidatos, seleciona 5 hits e reanexa seus vizinhos disponíveis. O CSV preserva IDs, recalls e latências por pergunta. Recall não mede faithfulness da resposta; ele compara somente a etapa de recuperação. Em um smoke test de geração sobre coleta e guarda de currículos, a busca vetorial respondeu com três chunks válidos, enquanto o contexto reordenado levou o gerador a declarar evidência insuficiente. Por isso, o reranking fica experimental e desativado por padrão na interface.
