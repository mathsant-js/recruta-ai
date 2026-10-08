# Recruta AI

Chatbot profissional de apoio a recrutamento e RH desenvolvido para os checkpoints do
segundo semestre da FIAP. A branch `cp2` preserva a aplicação funcional do CKP01 e inicia
a evolução para o pipeline RAG do CKP02.

## Estado do CKP02

As fases 0, 1, 2, 3, 4 e 5 estão concluídas. A fase 2 foi executada com embeddings locais reais,
depois que o endpoint cloud recusou o modelo obrigatório com `401`. O projeto não apresenta
chamadas simuladas como execução real. O baseline preserva a aplicação do CKP01 e a curadoria
documental fornece cinco PDFs reais e complementares para o RAG:

- o contrato público `app.retriever.buscar()` isola os futuros detalhes de ChromaDB;
- `DocumentoRecuperado` define conteúdo, score e metadata necessária para citações;
- os modelos e endpoints ficam centralizados em `app.config`: `gemma4:cloud` na Ollama Cloud
  para geração e `nomic-embed-text` no Ollama local para embeddings;
- `notebooks/CKP02_DocMind_RAG.ipynb` é um notebook fino, baseado nos módulos Python;
- `data/raw/` contém as cinco fontes oficiais ou institucionais preservadas integralmente;
- `data/manifest.json` registra origem, autoria, categoria, direitos, hash e justificativa;
- `docs/curadoria_fontes_cp2.md` documenta a revisão de aplicabilidade e limitações;
- `app/document_loader.py` valida manifesto, hashes, duplicidade e texto extraível;
- `app/chunking.py` implementa literalmente o `RecursiveCharacterTextSplitter` exigido;
- `app/embeddings.py` configura e valida `embed_query()` e `embed_documents()` reais;
- `app/vector_store.py` persiste e consulta a coleção Chroma `recruta_ai_256`, com
  expansão por chunks vizinhos para recompor frases cortadas;
- `app/ingestion.py` executa a fase inteira e grava sua evidência auditável;
- `app/rag_chain.py` executa retrieve → generate, trata evidência insuficiente e valida
  os IDs citados pelo modelo;
- `app/phase3.py` reproduz as cinco perguntas reais e grava respostas e fontes;
- `app/evaluation.py` compara 256/32 e 512/64 com oito perguntas e RAGAS real;
- `app/reranker.py` reordena dez candidatos com o cross-encoder
  `cross-encoder/ms-marco-MiniLM-L-6-v2` e entrega os cinco melhores ao gerador;
- `app/phase5.py` compara busca base, filtro por metadata e reranking com execução real;
- `app/main.py` integra o RAG, os filtros, o reranking e as fontes em Gradio;
- `artifacts/indexes/` mantém índices locais fora do Git.

`buscar()` abre o índice de forma tardia na primeira consulta, sem efeitos colaterais no
import. Se o índice não existir, falha explicitamente com
`RecuperadorNaoConfiguradoError`, evitando confundir ausência do índice com busca vazia.

## Fase 2 — ingestão e embeddings

A primeira configuração usa `chunk_size=256`, `chunk_overlap=32` (12,5%) e os separadores
obrigatórios `['\n\n', '\n', '. ', ' ', '']`. Cada chunk recebe um identificador estável,
página, fonte, categoria e parâmetros de divisão. O índice usa distância cosseno e a coleção
`recruta_ai_256`; os embeddings são enviados em lotes de 64 para evitar um único payload muito
grande.

Para carregar a base, validar os dois métodos de embedding, indexar e realizar a busca real:

```bash
python -m app.ingestion
```

O comando usa somente `nomic-embed-text` no Ollama local e não repassa a chave cloud ao cliente
de embeddings. O texto dos documentos permanece no computador durante a vetorização. A saída registra dimensão,
quantidade de páginas e chunks, tempo, pergunta de smoke test e os cinco resultados em
`artifacts/ingestion/fase2.json`. O banco vetorial fica em
`artifacts/indexes/granular_256/` e não é versionado; ele deve ser reconstruído no ambiente de
avaliação. Falhas de credencial, serviço, vetor vazio ou dimensão inconsistente produzem erro
explícito.

Na tentativa cloud de **06/10/2026**, a chave autenticou a API e listou os modelos permitidos,
mas `nomic-embed-text` não estava no catálogo acessível pela conta e o endpoint `/api/embed`
respondeu `401 unauthorized`. Nenhum PDF foi enviado nessa tentativa. Por decisão explícita do
grupo, a execução passou a usar o mesmo modelo localmente; essa adaptação deve ser informada ao
professor, pois diverge do requisito literal de embeddings via Cloud.

A execução local real de **06/10/2026** carregou 305 páginas, gerou e indexou 2.866 chunks e
confirmou vetores de 768 dimensões em 3.509,076 segundos. A coleção persistente contém os 2.866
registros. A busca aberta retornou cinco trechos, incluindo evidências da ANPD sobre guarda,
destino e transparência de currículos. Uma busca adicional com filtro `category=privacidade`
recuperou três chunks do documento `DOC-02`, incluindo a fase pré-contratual e consentimento em
plataformas de recrutamento. A evidência completa está em `artifacts/ingestion/fase2.json`.

Para adicionar um documento, coloque um PDF, TXT ou Markdown real em `data/raw/`, acrescente
ao `data/manifest.json` toda a metadata e o SHA-256 corretos, execute os testes e reconstrua o
índice. Arquivos vazios, corrompidos, duplicados ou com hash divergente são recusados.

## Fase 3 — pipeline end-to-end

O fluxo completo usa `buscar()` para recuperar cinco hits semânticos e acrescenta um
chunk anterior e um posterior de cada hit. Essa expansão preserva os IDs e metadados
originais e recompõe afirmações cortadas pela configuração granular de 256 caracteres.
O contexto segue para o `gemma4:cloud` com `temperature=0`; instruções encontradas nos
documentos são tratadas como dados não confiáveis.

O modelo deve citar IDs recuperados. `RAGService` valida citações individuais e agrupadas,
rejeita IDs inexistentes e monta a seção de fontes com título, página e URL diretamente
da metadata do Chroma. Quando não há evidência com score mínimo de `0,2`, a resposta
declara insuficiência sem inventar conteúdo.

Para reproduzir as cinco perguntas reais:

```bash
python -m app.phase3
```

A execução final de **08/10/2026** respondeu as cinco perguntas com fontes existentes,
usando `gemma4:cloud`, `nomic-embed-text`, `top_k=5` e temperatura zero. Levou 12,005
segundos. Perguntas, respostas, latências, chunks, scores, páginas e URLs estão em
`artifacts/rag/fase3.json`.

## Fase 4 — comparação de chunking e RAGAS

A avaliação usa oito perguntas revisadas por leitura humana: sete respondíveis e uma
deliberadamente fora da base. Ela cobre perguntas objetivas, interpretativas, privacidade,
risco discriminatório e combinação de documentos. As mesmas perguntas, documentos, prompt,
`gemma4:cloud`, temperatura zero, `top_k=5` e expansão por um vizinho foram mantidos nas duas
configurações; variaram apenas `chunk_size` e overlap.

Execução real de **08/10/2026**:

| Configuração | Chunks | Faithfulness | Answer relevancy | Recall de fontes | Recusa fora da base |
|---|---:|---:|---:|---:|---:|
| 256 / 32 | 2.866 | 0,643 | 0,623 | 1,000 | 100% |
| 512 / 64 | 1.500 | **0,714** | **0,630** | 0,929 | 100% |

As médias RAGAS consideram as sete perguntas respondíveis. A pergunta sem resposta permanece
na tabela por pergunta e é avaliada separadamente pela taxa de recusa, porque uma recusa
correta recebe zero nessas métricas e distorceria a qualidade das respostas fundamentadas.
A configuração **512/64 venceu**, superou a meta de faithfulness 0,7 e passou a ser a
configuração final do recuperador. A construção inicial desse índice levou 1.923,006 segundos;
reexecuções o reabrem após validar a contagem de 1.500 chunks.

Para reproduzir a avaliação completa:

```bash
python -m app.evaluation
```

O comando exige o Ollama local com `nomic-embed-text`, a chave cloud e conectividade. Os
artefatos auditáveis ficam em `artifacts/evaluation/`: dataset, respostas/contextos, CSV por
pergunta, resumo, configuração com hashes, relatório e gráfico. Os índices ficam ignorados
pelo Git e devem ser reconstruídos quando ausentes. Como RAGAS usa o modelo como juiz, os
valores podem variar entre execuções mesmo com temperatura zero.

## Fase 5 — diferenciais

A interface do CKP02 substitui o contexto conversacional do CKP01 pelo RAG. Ela permite
escolher uma das cinco categorias reais do manifesto, ativar ou desativar o reranking,
consultar a base e ver resposta e fontes citadas em áreas separadas. O RAG recupera dez
candidatos quando o reranking está ativo, aplica o cross-encoder recomendado no enunciado,
seleciona os cinco primeiros e reanexa seus vizinhos para não cortar afirmações antes de
enviar o contexto ao `gemma4:cloud`.

A comparação real de **08/10/2026**, após aquecimento do índice e do cross-encoder, usou as
sete perguntas respondíveis do dataset:

| Estratégia | Perguntas | Recall médio de fontes | Latência média |
|---|---:|---:|---:|
| Busca vetorial | 7 | 0,929 | 0,237 s |
| Cross-encoder | 7 | **1,000** | 4,252 s |
| Filtro por metadata | 6 aplicáveis | **1,000** | detalhada no CSV |

O reranking recuperou o segundo documento esperado da pergunta combinada, mas acrescentou
latência relevante em CPU. O filtro manteve recall 1,0 nos seis casos em que todas as fontes
esperadas pertenciam à mesma categoria e removeu chunks de categorias alheias. Esses números
medem recuperação, não substituem o faithfulness RAGAS da fase 4. Em um smoke test de geração
sobre coleta e guarda de currículos, a busca vetorial respondeu com três chunks válidos,
enquanto o cross-encoder em inglês selecionou contexto menos específico e o gerador recusou a
resposta. Por isso, o reranking permanece experimental e desativado por padrão na interface;
não se presume melhoria automática. IDs e tempos por pergunta estão em
`artifacts/phase5/comparacao_recuperacao.csv`.

Para reproduzir:

```bash
python -m app.phase5
```

Na primeira execução, `sentence-transformers` baixa o cross-encoder. O comando também exige
o índice final e o Ollama local com `nomic-embed-text`, mas não chama o modelo gerador cloud.

## Integrantes

| Nome completo | RM |
|---|----|
| Bernardo Zauza Amorim | 568808 |
| Bruno Almeida de Oliveira | 572648 |
| Gabriel Góes Nunes Pereira | 571735 |
| Guilherme Vinciguerra Carvalho | 571951 |
| Marcos Peterson Martins Pereira | 573857 |
| Matheus Jorge Santana | 574166 |

## Domínio, justificativa e usuários-alvo

O Recruta AI atua no domínio de recrutamento e Recursos Humanos. Ele ajuda a levantar e
esclarecer requisitos, elaborar ou revisar descrições de vagas, separar requisitos
obrigatórios de desejáveis, identificar competências observáveis, sugerir perguntas de
entrevista e consolidar a solicitação em uma análise estruturada.

Esse domínio foi escolhido porque briefings de vagas frequentemente chegam incompletos,
subjetivos ou dispersos. Uma conversa guiada reduz omissões e torna os critérios mais claros
e verificáveis, sem transferir ao modelo a decisão sobre pessoas. Os usuários-alvo são
recrutadores, analistas de RH, gestores solicitantes e consultorias de recrutamento.

## Configuração

Requisitos locais: Python 3.10 ou superior, acesso ao Ollama Cloud e uma chave válida em
`OLLAMA_API_KEY`. Não versione o arquivo `.env`.

### Linux/Mac

1. Crie e ative um ambiente virtual:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Instale as dependências fixadas:

   ```bash
   pip install -r requirements.txt
   ```

3. Copie `.env.example` para `.env` e informe sua chave da API do Ollama:

   ```bash
   cp .env.example .env
   ```

   O arquivo deve conter `OLLAMA_API_KEY=sua_chave`. A chave é carregada com
   `python-dotenv`; `.env` está no `.gitignore` e `.env.example` não possui segredo real.

### Windows

1. Crie e ative um ambiente virtual:

   ```cmd
   python -m venv venv
   venv/bin/activate.bat
   ```

2. Instale as dependências fixadas:

   ```cmd
   pip install -r requirements.txt
   ```

3. Copie `.env.example` para `.env` e informe sua chave da API do Ollama:

   ```bash
   copy .env.example .env
   ```

   O arquivo deve conter `OLLAMA_API_KEY=sua_chave`. A chave é carregada com
   `python-dotenv`; `.env` está no `.gitignore` e `.env.example` não possui segredo real.

## Execução

O comando oficial para iniciar a aplicação é:

```bash
python -m app.main
```

O projeto usa `gemma4:cloud` pela API cloud do Ollama para geração e
`nomic-embed-text` no Ollama local para recuperação semântica.
As versões de Gradio e Pydantic estão fixadas em uma combinação compatível para que
a instalação reproduzível mantenha a validação em Pydantic v2.

Ao iniciar, acesse [http://localhost:7860](http://localhost:7860). A interface oferece
pergunta ao RAG, filtro opcional por categoria, controle do cross-encoder, resposta
fundamentada e fontes citadas separadamente. O envio funciona pelo botão ou pela tecla Enter,
e o aviso permanente no rodapé reforça privacidade, uso responsável e revisão humana.

Se a chave estiver ausente, o processo termina antes de criar a interface e mostra uma
mensagem orientando a configurar `OLLAMA_API_KEY`. Em distribuições nas quais apenas
`python3` existe fora do ambiente virtual, ative o ambiente virtual para usar o comando
oficial `python -m app.main`.

## Arquitetura das duas chains

O projeto mantém dois fluxos separados:

1. **Chat com memória:** `ConversationChain` recebe o prompt de sistema e o histórico de uma
   `ConversationTokenBufferMemory` exclusiva por sessão. `ChatService` cria, reutiliza e
   remove essas chains sob demanda.
2. **Análise estruturada:** a composição LCEL
   `ChatPromptTemplate | ChatOllama | PydanticOutputParser` devolve uma instância validada de
   `AnaliseRecrutamento`. Uma falha de parsing permite uma única correção; a segunda falha
   vira um erro de domínio compreensível.

As chains generativas usam `gemma4:cloud` na Cloud. Templates, schema, memória, regras de negócio,
experimento e interface ficam em módulos separados dentro de `app/`.

## Chat com memória

O chat usa `ConversationChain` com uma `ConversationTokenBufferMemory` limitada a 1.200
tokens. Esse limite mantém fatos recentes úteis para o briefing sem permitir crescimento
ilimitado do contexto e do custo. Cada `session_id` recebe sua própria chain e memória; a
operação `ChatService.clear_session(session_id)` remove ambas e impede reaproveitamento do
histórico após a limpeza.

A contagem local aproxima um token a cada quatro bytes UTF-8, pois o tokenizador exato do
modelo cloud não é exposto por essa integração. O corte de 1.200 deve, portanto, ser entendido
como orçamento operacional aproximado, não como medição exata de cobrança do Ollama.

As duas classes estão marcadas como legadas pelo LangChain atual. O projeto fixa a linha
compatível `langchain==0.3.27` porque o enunciado exige essas APIs literalmente; uma migração
para `RunnableWithMessageHistory` ou LangGraph alteraria a arquitetura avaliada e deve ocorrer
somente depois do checkpoint.

Roteiro de demonstração da recuperação de contexto:

1. “Estou abrindo uma vaga de desenvolvedor backend.”
2. “A senioridade é pleno.”
3. “Precisamos de Python e PostgreSQL.”
4. “O trabalho será híbrido em São Paulo.”
5. “Quais requisitos obrigatórios eu já informei?”
6. “Crie cinco perguntas de entrevista com base neles.”

O teste determinístico `tests/test_memory.py` executa o roteiro completo, comprova que o
último prompt contém senioridade, tecnologias, modalidade e localidade e verifica isolamento
e limpeza de sessões. Ele não é apresentado como uma chamada real ao Ollama Cloud; a validação
real do modelo requer `OLLAMA_API_KEY` e conectividade.

## Prompts e domínio

A persona do Recruta AI atende recrutadores, analistas de RH, gestores solicitantes e
consultorias. Ela apoia definição e revisão de vagas, separação de requisitos, preparação de
entrevistas e organização de informações. O chatbot não toma decisões finais sobre pessoas,
não usa ou infere atributos sensíveis, não produz diagnósticos e exige revisão humana.

Os prompts ficam como arquivos Markdown imutáveis em `prompts/`. A versão ativa é selecionada
explicitamente em `app/prompts.py`; qualquer alteração deve criar um novo arquivo numerado, sem
apagar o histórico. Atualmente, `system_prompt_v3.md` é a versão final ativa. As mensagens
`system` e `human` são separadas por `ChatPromptTemplate`, e histórico, solicitação e instruções
de formato são inseridos por variáveis do próprio template, sem montagem manual com f-strings.

Os testes adversariais determinísticos verificam que tentativas de desvio permanecem na mensagem
humana não confiável e que a mensagem de sistema preserva as regras de domínio, equidade,
privacidade, recusa segura e revisão humana. Chamadas reais dependem da credencial local e são
mantidas separadas dos testes unitários determinísticos.

### Meta prompting

A versão `system_prompt_v2.md` foi preservada como o “antes” e submetida a uma crítica real do
`gemma4:cloud`. O fluxo reproduzível usa:

```bash
python -m app.meta_prompting
```

O modelo identifica ambiguidades, lacunas e riscos em uma saída validada por Pydantic, gravada em
`artifacts/meta_prompting/critica_modelo.json`. O comando não edita nem promove prompts: a saída é
marcada como pendente de revisão humana. Das quatro sugestões recebidas, uma foi aceita, duas foram
aceitas com ajustes e uma foi rejeitada. A versão final `system_prompt_v3.md` acrescenta vínculo de
conclusões de triagem a evidências, proteção contra instruções em documentos e tratamento de dados
sensíveis já enviados. O redirecionamento sugerido para política interna ou especialista foi rejeitado
por poder indicar recursos inexistentes.

As decisões completas e a comparação antes/depois estão em
`artifacts/meta_prompting/revisao_humana.md`. Os testes verificam a integração das novas regras, mas
não eliminam riscos probabilísticos de jailbreak ou viés; as respostas continuam exigindo revisão
humana.

## Análise estruturada LCEL

A segunda chain usa a composição explícita
`ChatPromptTemplate | ChatOllama | PydanticOutputParser`. As instruções JSON são geradas pelo
próprio parser a partir de `AnaliseRecrutamento`, e o retorno interno é uma instância validada
desse modelo Pydantic v2. O schema tipa intenção, resumo, competências, perguntas, pontos de
atenção, próximo passo e confiança; campos desconhecidos e confiança fora de 0 a 1 são rejeitados.

Se a primeira resposta do modelo não puder ser validada, o serviço faz somente uma nova tentativa
com o erro e as instruções de formato. Uma segunda falha resulta em mensagem compreensível, sem
ignorar o erro nem converter silenciosamente a saída em `dict`. Na interface, o `dict` é produzido
apenas depois da validação, por `model_dump`, para que o componente JSON possa exibi-lo.

A interface inclui a ação **Gerar análise estruturada**, que analisa todo o histórico visível da
sessão. Os testes determinísticos em `tests/test_chains.py` cobrem sucesso, tipo final, correção
limitada e falha de parsing sem realizar chamadas externas. A validação com saída real do
`gemma4:cloud` exige `OLLAMA_API_KEY` e conectividade com o Ollama Cloud.

## Experimento de context rot

O experimento reproduzível está em `app/context_rot.py` e é executado com:

```bash
python -m app.context_rot
```

Ele usou exclusivamente `gemma4:cloud` e o `system_prompt_v3.md`, que também é a versão ativa
na aplicação entregue. A mesma pergunta final e os mesmos oito fatos de uma vaga fictícia foram
mantidos em todos os cenários. Um único histórico
de 1.462 tokens aproximados mantém os fatos no início e insere notas administrativas
irrelevantes entre eles e a pergunta final. Para reproduzir o comportamento da
`ConversationTokenBufferMemory`, cada cenário preserva a cauda mais recente e descarta o
excedente conforme a janela. A contagem usa `tiktoken` com `cl100k_base`: ela é uma
aproximação comparável entre cenários, não o tokenizador nativo do Gemma.

Execução real realizada em **15/09/2026**, com saída validada por Pydantic:

| Cenário | Janela | Tokens descartados | Requisitos recuperados | Inventados | Persona | Utilidade | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Muito curta | 256 | 1.206 | 0/8 | 0 | 1 | 0,00 | 1,278 |
| Curta | 512 | 950 | 0/8 | 0 | 1 | 0,00 | 1,418 |
| Referência mínima | 800 | 662 | 0/8 | 0 | 1 | 0,00 | 1,266 |
| Escolha do chatbot | 1.200 | 262 | 0/8 | 0 | 1 | 0,00 | 1,255 |
| Limite superior | 1.500 | 0 | 8/8 | 0 | 1 | 5,00 | 1,998 |

A degradação ocorreu por **truncamento**, não por troca de prompt ou modelo. O bloco de fatos
estava na parte mais antiga do histórico; quando 262 ou mais tokens foram removidos, todos os
oito identificadores ficaram fora do contexto enviado. Por isso, inclusive a janela operacional
de 1.200 tokens perdeu 100% dos requisitos. Com 1.500 tokens, o histórico completo coube na
janela e a recuperação subiu para 100%. Isso demonstra o risco de posicionar fatos essenciais
apenas no começo de conversas longas e justifica recapitulações periódicas ou resumos explícitos.

`artifacts/context_rot/resultados.csv` contém métricas e respostas validadas; a tabela comparativa
está em `artifacts/context_rot/tabela_comparativa.md`, e `metadados.json` registra modelo, versão
do prompt, data, fatos-base e hash da pergunta final. A métrica “informações inventadas” é a
autodeclaração estruturada do modelo e, portanto, não substitui auditoria humana. A utilidade é
determinística (recuperação factual em escala de 0 a 5, penalizada quando há quebra de persona),
e os tempos representam uma única execução por cenário, sem valor de benchmark estatístico.
A execução final com o `system_prompt_v3.md` reproduziu o mesmo padrão de recuperação observado
anteriormente: as janelas de até 1.200 tokens descartaram o bloco de fatos, enquanto a janela de
1.500 tokens preservou o contexto completo. Os tempos e os textos das respostas variaram, como
esperado em chamadas reais a um modelo probabilístico.

## Testes

Execute a suíte determinística com:

```bash
python -m pytest -q
```

Os testes não consomem a API: usam modelos locais controlados para validar contratos, erros e
estado. A evidência do experimento real permanece separada em `artifacts/context_rot/`.

| Item solicitado | Evidência automatizada |
|---|---|
| Inicialização pelo ponto de entrada oficial | `tests/test_interface.py::test_ponto_de_entrada_oficial_inicia_na_porta_7860` verifica que `main()` inicia em `0.0.0.0:7860`; o comando documentado é `python -m app.main`. |
| Ausência da chave de API | `tests/test_interface.py::test_ponto_de_entrada_falha_com_mensagem_clara_sem_chave` garante falha antes da criação da interface. |
| Schema válido e inválido | `tests/test_schemas.py` cobre instância Pydantic, valores fora dos limites, texto vazio, intenção inválida e campos extras. |
| Histórico com mais de cinco turnos | `tests/test_memory.py::test_roteiro_recupera_requisitos_e_embasa_perguntas` executa seis mensagens e inspeciona o contexto recuperado. |
| Limpeza e isolamento da memória | `tests/test_memory.py::test_sessoes_sao_isoladas_e_podem_ser_limpas` e o teste de limpeza da interface. |
| Erro de parsing | `tests/test_chains.py` cobre correção única e erro compreensível após a segunda falha. |
| Comportamento fora do escopo | `tests/test_prompts.py::test_pedido_fora_do_escopo_recebe_regra_de_limite_e_redirecionamento` verifica limite, redirecionamento seguro e resistência à instrução conflitante. |
| Execução do context rot | `tests/test_context_rot.py` percorre as cinco janelas, valida saídas e testa geração dos artefatos sem fabricar chamadas cloud. |

Testes determinísticos demonstram a implementação do contrato, mas não provam que toda
resposta probabilística do modelo obedecerá ao prompt. Para uma validação integrada, configure
a credencial, inicie a interface e execute manualmente o roteiro de memória. O context rot real
pode ser repetido com `python -m app.context_rot`; essa operação faz cinco chamadas cloud e
substitui os artefatos no diretório de saída escolhido.

## Requisitos atendidos

| Requisito | Implementação |
|---|---|
| Projeto Python local e comando oficial | Pacote `app`, executado por `python -m app.main`. |
| Interface Gradio | Chat, envio, limpeza, análise JSON e aviso em `http://localhost:7860`. |
| Modelo e configuração segura | `gemma4:cloud` na Cloud; chave carregada do `.env`, nunca hardcoded. Embeddings locais sem chave. |
| Duas chains | `ConversationChain` com memória e pipeline LCEL estruturado. |
| Memória gerenciada | Limite aproximado de 1.200 tokens, isolamento e limpeza por sessão. |
| Pydantic v2 | `AnaliseRecrutamento` tipado, restrito e integrado ao parser. |
| Prompts | Mensagens `system`/`human` separadas, variáveis de template, persona XML e versões preservadas. |
| Context rot | Mesma tarefa em 256, 512, 800, 1.200 e 1.500 tokens, com CSV, tabela e metadados reais. |
| Meta prompting | Crítica real preservada, revisão humana documentada e v3 promovida manualmente. |
| Testes e documentação | Suíte determinística e rastreabilidade nesta matriz e na tabela de testes. |

## Limitações e uso responsável

- O Recruta AI apoia o trabalho de RH; não decide contratação, rejeição, promoção ou
  demissão. Toda saída que afete uma pessoa exige revisão humana.
- O modelo pode errar, omitir contexto ou variar entre execuções. A validação Pydantic garante
  formato e restrições de tipo, não veracidade factual.
- O limite de 1.200 tokens usa uma aproximação local; ele pode descartar fatos antigos, como o
  experimento demonstra. Recapitule requisitos essenciais em conversas longas.
- Não envie CPF, documentos, dados médicos, endereço completo, fotos ou outros dados pessoais
  desnecessários. Prefira dados fictícios ou anonimizados.
- A ferramenta não deve inferir nem usar idade, raça, gênero, religião, deficiência ou outros
  atributos sensíveis, nem produzir diagnósticos psicológicos.
- O uso depende de disponibilidade, latência e credencial válida do Ollama Cloud. Os tempos do
  context rot representam uma única execução e não constituem benchmark.
