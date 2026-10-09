# Plano de desenvolvimento — CKP02 DocMind RAG

Este plano considera o feedback da avaliação do CKP01 e os requisitos do enunciado
`CKP02_2Semestre_DocMind_RAG.pdf`.

O CKP02 deve manter o domínio de recrutamento e RH, mas substituir a memória
conversacional como principal fonte de contexto por uma camada RAG isolada e
reutilizável. O foco passa a ser comprovar, com chamadas reais e artefatos executados,
o pipeline `load → split → embed → store → retrieve → generate`.

## 1. Direção do projeto

O produto continuará sendo o **Recruta AI**, agora apoiado por uma base documental real
para responder perguntas sobre:

- elaboração e revisão de vagas;
- requisitos obrigatórios e desejáveis;
- práticas inclusivas de recrutamento;
- privacidade e tratamento de dados de candidatos;
- critérios potencialmente discriminatórios;
- responsabilidades e limites de recrutadores e gestores;
- preparação de entrevistas estruturadas.

O RAG deverá responder somente com base nos documentos recuperados, citar as fontes
utilizadas e declarar quando a base não sustenta uma resposta.

Uma separação importante será:

```text
Interface / agente futuro
          |
          v
     buscar(consulta)
          |
          v
Retriever de documentos
          |
          v
Contexto documental
          |
          v
gemma4:cloud + prompt fundamentado
```

Essa função `buscar(consulta)` será a futura ferramenta do agente do CKP03.

## 2. Como o feedback do CKP01 muda a estratégia

### 2.1 Evidência real, não apenas implementação

No CKP01, o README e os testes indicavam um roteiro de memória e um experimento de
context rot, mas o professor não encontrou a demonstração real esperada. Para o CKP02:

- testes simulados validarão apenas contratos internos;
- resultados de RAGAS somente serão aceitos após chamadas reais;
- o notebook entregue deverá estar executado, com outputs preservados;
- os artefatos deverão registrar data, modelos, parâmetros e eventuais erros;
- nenhuma etapa será marcada como pronta somente porque existe código para executá-la.

### 2.2 Comentários em PT-BR

O professor apontou comentários escassos. No CKP02, os módulos principais deverão ter:

- docstrings em PT-BR;
- comentários explicando decisões não óbvias;
- justificativas de chunking, filtros, reranking e métricas;
- ausência de comentários redundantes que apenas repetem o código.

### 2.3 Camada de contexto isolada

A memória do CKP01 não deve ser misturada ao recuperador. Será criado um contrato claro,
por exemplo:

```python
def buscar(consulta: str, filtros: dict | None = None) -> list[DocumentoRecuperado]:
    ...
```

A geração receberá o resultado dessa função sem conhecer detalhes de ChromaDB. Isso
facilitará transformar o RAG em `@tool` no CKP03.

## 3. Base de conhecimento

### 3.1 Escopo recomendado

Montar uma base com pelo menos cinco documentos reais e confiáveis, preferencialmente
oficiais, cobrindo tópicos complementares:

1. proteção de dados pessoais em recrutamento;
2. práticas antidiscriminatórias na contratação;
3. legislação trabalhista aplicável ao processo seletivo;
4. classificação e descrição de ocupações;
5. guia ou manual de recrutamento inclusivo;
6. material técnico sobre entrevistas estruturadas ou seleção por competências.

É melhor trabalhar com seis a oito fontes relevantes do que usar dezenas de documentos
pouco relacionados.

### 3.2 Critérios de seleção

Cada documento deverá:

- vir de fonte oficial, acadêmica ou institucional reconhecida;
- ter aplicação clara ao Recruta AI;
- possuir título, organização, URL e data de acesso;
- permitir perguntas não triviais;
- ter licença ou condição de uso compatível com a entrega acadêmica;
- não conter currículos nem dados pessoais reais.

### 3.3 Manifesto de fontes

Criar `data/sources.csv` ou `data/manifest.json` com:

- identificador do documento;
- título;
- instituição ou autoria;
- URL de origem;
- data de publicação, quando houver;
- data de acesso;
- tipo;
- categoria;
- nome do arquivo local;
- hash do arquivo;
- justificativa de inclusão.

**Critério de aceite:** pelo menos cinco documentos reais disponíveis na entrega, todos
rastreáveis até a origem.

## 4. Arquitetura proposta

```text
recruta-ai/
├── app/
│   ├── config.py
│   ├── document_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── rag_chain.py
│   ├── evaluation.py
│   ├── prompts.py
│   └── main.py
├── data/
│   ├── raw/
│   └── manifest.json
├── artifacts/
│   ├── evaluation/
│   └── indexes/
├── notebooks/
│   └── CKP02_DocMind_RAG.ipynb
├── tests/
│   ├── test_loader.py
│   ├── test_chunking.py
│   ├── test_retriever.py
│   ├── test_rag_chain.py
│   └── test_evaluation.py
├── .env.example
├── requirements.txt
└── README.md
```

O notebook será a entrega executada exigida pelo PDF, mas a lógica ficará nos módulos
Python. Assim, evita-se duplicação e o código permanece reutilizável no CKP03.

## 5. Pipeline obrigatório

### 5.1 Load

- Carregar PDF, TXT e Markdown.
- Normalizar texto sem eliminar títulos e divisões úteis.
- Associar metadata a cada documento.
- Detectar arquivo vazio, corrompido ou duplicado.
- Preservar página quando o loader conseguir identificá-la.

Metadata mínima:

```python
{
    "document_id": "...",
    "source": "...",
    "title": "...",
    "organization": "...",
    "document_type": "...",
    "category": "...",
    "publication_date": "...",
    "page": 12,
}
```

### 5.2 Split

Usar literalmente:

```python
RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=...,
    chunk_overlap=...,
)
```

Comparação inicial recomendada:

| Configuração | `chunk_size` | `chunk_overlap` | Percentual |
|---|---:|---:|---:|
| A — granular | 256 | 32 | 12,5% |
| B — equilibrada | 512 | 64 | 12,5% |
| C — ampla | 1024 | 128 | 12,5% |

O enunciado exige duas configurações; testar três aumenta a qualidade da escolha sem
alterar a arquitetura.

### 5.3 Embed

Usar exclusivamente:

```python
OllamaEmbeddings(model="nomic-embed-text")
```

Devem existir verificações reais de:

- `embed_query()`;
- `embed_documents()`;
- dimensão consistente;
- ausência de vetores vazios;
- erro compreensível quando a credencial ou o serviço não estiver disponível.

### 5.4 Store

- Usar ChromaDB local, preferencialmente `PersistentClient`.
- Manter uma coleção por configuração de chunking.
- Usar nomes relacionados ao domínio, por exemplo:
  - `recruta_ai_256`;
  - `recruta_ai_512`;
  - `recruta_ai_1024`.
- Recriar ou versionar os índices quando documentos ou parâmetros mudarem.
- Não versionar caches enormes ou arquivos temporários sem necessidade.

### 5.5 Retrieve

Implementar uma API estável:

```python
def buscar(
    consulta: str,
    *,
    top_k: int = 5,
    filtros: dict | None = None,
) -> list[DocumentoRecuperado]:
    ...
```

Cada resultado deverá retornar:

- conteúdo;
- score ou distância;
- título;
- fonte;
- página;
- categoria;
- identificador do chunk.

### 5.6 Generate

Usar exclusivamente:

```python
ChatOllama(
    model="gemma4:cloud",
    temperature=0,
)
```

O prompt deve instruir o modelo a:

- responder somente com evidências do contexto;
- citar as fontes e chunks utilizados;
- não obedecer a instruções encontradas nos documentos;
- declarar insuficiência de evidência;
- não inventar legislação, políticas ou fatos;
- manter as restrições éticas do Recruta AI;
- recomendar revisão humana quando a resposta afetar decisões sobre pessoas.

## 6. Formato das respostas e citações

Uma resposta deverá apresentar:

```text
Resposta fundamentada
...

Fontes:
- [DOC-02, p. 7, chunk DOC-02-C014] Título do documento
- [DOC-05, p. 3, chunk DOC-05-C006] Título do documento
```

Os identificadores citados deverão existir entre os chunks recuperados. Um teste
automático deverá impedir que a chain apresente uma fonte inexistente.

Quando não houver evidência suficiente:

> Não encontrei sustentação adequada nos documentos recuperados. Reformule a pergunta
> ou adicione uma fonte apropriada à base.

## 7. Dataset de avaliação

Criar ao menos cinco perguntas; recomenda-se entre oito e doze para reduzir conclusões
baseadas em poucos exemplos.

O conjunto deve incluir:

- perguntas respondidas por um único documento;
- perguntas que exigem combinar dois documentos;
- perguntas objetivas;
- perguntas interpretativas;
- pelo menos uma pergunta sem resposta na base;
- pelo menos uma pergunta envolvendo risco discriminatório;
- pelo menos uma pergunta sobre privacidade.

Estrutura sugerida:

```json
{
  "question": "...",
  "reference_answer": "...",
  "expected_document_ids": ["DOC-01", "DOC-03"],
  "category": "privacidade",
  "answerable": true
}
```

As respostas de referência deverão ser redigidas por leitura humana dos documentos,
nunca geradas e aceitas automaticamente.

## 8. Comparação de chunking e RAGAS

Para cada configuração:

1. reconstruir a coleção;
2. executar exatamente o mesmo conjunto de perguntas;
3. manter fixos:
   - documentos;
   - prompt;
   - modelo;
   - `temperature=0`;
   - `top_k`;
   - dataset;
4. coletar contexto recuperado e resposta;
5. calcular:
   - `faithfulness`;
   - `answer_relevancy`;
6. registrar métricas por pergunta e médias.

Tabela obrigatória:

| Pergunta | Chunk size | Overlap | Faithfulness | Answer relevancy | Fontes esperadas recuperadas | Latência |
|---|---:|---:|---:|---:|---:|---:|

Também registrar:

- média e desvio das métricas;
- taxa de recuperação das fontes esperadas;
- quantidade de chunks;
- tamanho médio dos chunks;
- tempo de indexação;
- tempo de resposta;
- falhas reais.

### 8.1 Regra de escolha

A configuração vencedora não será necessariamente a de maior `faithfulness` isolada. A
decisão deverá considerar:

1. `faithfulness` médio, com mínimo desejado de `0,7`;
2. `answer_relevancy`;
3. recuperação das fontes esperadas;
4. estabilidade entre perguntas;
5. custo e latência;
6. qualidade observada das citações.

Se todas as configurações ficarem abaixo de `0,7`, o trabalho não deve avançar
diretamente para a entrega. Primeiro será necessário revisar base, splitter, `top_k` ou
prompt.

## 9. Diferenciais

Os diferenciais devem ser implementados somente após todos os obrigatórios.

### 9.1 Metadata filtering — prioridade 1

É barato e útil para o CKP03:

```python
buscar(
    "Quais cuidados existem no uso de dados de candidatos?",
    filtros={"category": "privacidade"},
)
```

Demonstrar pelo menos uma consulta com e sem `where=`.

### 9.2 Reranking — prioridade 2

- Recuperar um conjunto maior, por exemplo `top_k=10`.
- Aplicar cross-encoder.
- Enviar ao LLM somente os cinco primeiros reordenados.
- Comparar retrieval e métricas com e sem reranking.
- Não presumir melhora: preservar o resultado mesmo que não ajude.

### 9.3 Gradio — prioridade 3

Reaproveitar a interface do CKP01, substituindo o contexto da memória pelo RAG:

- campo para pergunta;
- resposta fundamentada;
- fontes exibidas separadamente;
- filtros opcionais;
- aviso de privacidade e revisão humana.

## 10. Fases e critérios de aceite

### Fase 0 — Baseline e organização

- Preservar o CKP01 funcional.
- Criar branch ou diretório específico do CKP02.
- Atualizar dependências.
- Definir contrato de `buscar()`.
- Preparar notebook fino, baseado nos módulos Python.

**Aceite:** CKP01 permanece recuperável e o CKP02 importa sem efeitos colaterais.

### Fase 1 — Curadoria documental

- Selecionar fontes.
- Baixar documentos.
- Criar manifesto.
- Revisar conteúdo e aplicabilidade.

**Aceite:** cinco ou mais fontes reais, citadas e capazes de sustentar perguntas não
triviais.

### Fase 2 — Ingestão e embedding

- Implementar loaders.
- Gerar chunks da primeira configuração.
- Executar embeddings reais.
- Indexar no ChromaDB.
- Realizar uma busca semântica real.

**Aceite da Aula 05:** documentos carregados, chunks inspecionáveis e busca retornando
conteúdo relevante.

### Fase 3 — Pipeline end-to-end

- Implementar `buscar()`.
- Criar prompt RAG.
- Gerar resposta com `gemma4:cloud`.
- Adicionar citações verificáveis.
- Tratar ausência de evidências.

**Aceite da Aula 06:** pelo menos cinco perguntas reais respondidas com fontes existentes.

### Fase 4 — Avaliação quantitativa

- Criar dataset.
- Executar todas as configurações.
- Calcular RAGAS.
- Gerar CSV, tabela e gráfico.
- Escolher configuração final com justificativa.

**Aceite:** métricas reais por pergunta e por configuração, com `faithfulness` médio
idealmente igual ou superior a `0,7`.

### Fase 5 — Diferenciais

- Metadata filtering.
- Reranking.
- Gradio.

**Aceite:** demonstração executável e comparação objetiva de cada diferencial
implementado.

### Fase 6 — Documentação e entrega

- Executar o notebook do início ao fim.
- Salvar outputs.
- Executar testes.
- Conferir segredos.
- Validar os links.
- Revisar README.
- Testar a partir de ambiente limpo.

**Aceite:** `Run All` sem erros, documentos incluídos, README completo e nenhuma
credencial versionada.

## 11. Testes obrigatórios

### 11.1 Testes determinísticos

- Loader preserva conteúdo e metadata.
- Arquivos vazios são rejeitados.
- Overlap permanece entre 10% e 15%.
- Chunks possuem identificadores únicos.
- Coleções não misturam configurações.
- `buscar()` retorna o tipo esperado.
- Filtros são encaminhados corretamente.
- Citações pertencem ao contexto recuperado.
- Pergunta vazia gera erro claro.
- Prompt ignora instruções maliciosas contidas nos documentos.

### 11.2 Testes de integração real

Devem ser marcados separadamente, por exemplo com `pytest -m integration`:

- `nomic-embed-text` gera embeddings reais;
- ChromaDB recupera documentos relevantes;
- `gemma4:cloud` produz resposta fundamentada;
- cinco ou mais perguntas são executadas de ponta a ponta;
- RAGAS calcula as duas métricas;
- notebook executa sem erro.

Os testes reais devem falhar explicitamente quando não houver credencial, nunca simular
sucesso.

## 12. Evidências a preservar

```text
artifacts/evaluation/
├── dataset.json
├── resultados_por_pergunta.csv
├── resumo_por_configuracao.csv
├── comparacao_chunking.png
├── respostas/
├── configuracao_execucao.json
└── relatorio.md
```

`configuracao_execucao.json` deverá registrar:

- data e fuso;
- documentos e hashes;
- modelos;
- parâmetros de chunking;
- `top_k`;
- prompt ou hash do prompt;
- versões das dependências;
- indicação de execução real;
- erros encontrados.

## 13. README final

O README deverá conter:

- integrantes e RMs;
- domínio e continuidade em relação ao CKP01;
- descrição da base;
- tabela de fontes;
- arquitetura do pipeline;
- modelos permitidos;
- configuração segura da credencial;
- instalação;
- execução do notebook e da aplicação;
- instruções para adicionar documentos;
- instruções para reconstruir o índice;
- metodologia do dataset;
- comparação de chunking;
- resultados RAGAS;
- escolha da configuração final;
- diferenciais;
- limitações;
- privacidade e uso responsável;
- preparação para o CKP03.

## 14. Distribuição sugerida entre integrantes

| Frente | Responsabilidade |
|---|---|
| Curadoria | Fontes, manifesto, perguntas e respostas de referência |
| Ingestão | Loaders, normalização, chunking e testes |
| Recuperação | Embeddings, ChromaDB, `buscar()` e filtros |
| Geração | Prompt RAG, citações e tratamento de insuficiência |
| Avaliação | RAGAS, comparação, tabelas e gráficos |
| Entrega | Notebook, Gradio, README e validação limpa |

Todos devem revisar o notebook final; não convém deixar a única evidência executável sob
responsabilidade de uma pessoa.

## 15. Definição de pronto

O CKP02 estará pronto somente quando:

- existirem pelo menos cinco documentos reais e citados;
- o pipeline completo fizer chamadas reais;
- `nomic-embed-text` for o único embedding;
- `gemma4:cloud` com `temperature=0` for o único modelo gerador;
- ChromaDB armazenar e recuperar chunks com metadata;
- as respostas citarem fontes realmente recuperadas;
- pelo menos duas estratégias de chunking forem comparadas;
- pelo menos cinco perguntas tiverem métricas RAGAS reais;
- `faithfulness` médio for preferencialmente igual ou superior a `0,7`;
- a escolha final for justificada quantitativamente;
- `buscar(consulta)` estiver isolada e preparada para o CKP03;
- o notebook estiver executado com `Run All` sem erros;
- o README explicar como adicionar novos documentos;
- comentários e docstrings relevantes estiverem em PT-BR;
- a entrega não contiver `.env`, chaves ou dados pessoais reais.

## 16. Checklist de entrega

- [ ] Confirmar pelo menos cinco documentos reais na pasta entregue.
- [ ] Confirmar fontes e links no manifesto e no notebook.
- [ ] Executar todas as células do notebook em ordem.
- [ ] Preservar outputs das chamadas reais e do RAGAS.
- [ ] Confirmar pelo menos duas configurações de chunking comparadas.
- [ ] Conferir `faithfulness` e `answer_relevancy` por pergunta.
- [ ] Validar se todas as citações apontam para chunks recuperados.
- [ ] Executar testes determinísticos e de integração.
- [ ] Verificar instruções para adicionar novos documentos no README.
- [ ] Conferir nomes e RMs de todos os integrantes.
- [ ] Procurar chaves, `.env` e dados pessoais antes de empacotar.
- [ ] Abrir o notebook exportado e confirmar que os resultados estão visíveis.
- [ ] Validar permissão de leitura do link do Colab.
- [ ] Confirmar que somente o líder fará o envio pelo Teams.
