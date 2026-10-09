# Relatório da fase 4 - avaliação quantitativa

Execução real em 2026-10-08T19:53:42-03:00, com 8 perguntas.

| Configuração | Chunks | Faithfulness | Answer relevancy | Recall de fontes | Latência média (s) |
|---|---:|---:|---:|---:|---:|
| granular_256 | 2866 | 0.643 ± 0.440 | 0.623 ± 0.399 | 1.000 | 1.051 |
| equilibrada_512 | 1500 | 0.714 ± 0.452 | 0.630 ± 0.407 | 0.929 | 1.012 |

## Escolha

A configuração **equilibrada_512** venceu pela regra ordenada (faithfulness, answer relevancy, recuperação de fontes e latência). Seu faithfulness médio atingiu a meta de 0,7.

A escolha considera somente métricas válidas. Respostas, contextos, citações e eventuais valores N/D permanecem nos artefatos para auditoria.

## Critério das médias

As médias RAGAS usam somente as sete perguntas respondíveis. A pergunta deliberadamente fora da base permanece na tabela por pergunta e é medida separadamente pela taxa de recusa, pois uma recusa correta recebe zero no RAGAS e distorceria a qualidade das respostas fundamentadas.
