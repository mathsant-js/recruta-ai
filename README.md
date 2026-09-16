# Recruta AI

Chatbot profissional de apoio a recrutamento e RH desenvolvido para o CKP01 do segundo
semestre da FIAP.

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

## Execução

O comando oficial para iniciar a aplicação é:

```bash
python -m app.main
```

O projeto usa exclusivamente o modelo `gemma4:cloud` pela API cloud do Ollama.
As versões de Gradio e Pydantic estão fixadas em uma combinação compatível para que
a instalação reproduzível mantenha a validação em Pydantic v2.

Ao iniciar, acesse [http://localhost:7860](http://localhost:7860). A interface oferece
chat com memória isolada por sessão, envio pelo botão ou pela tecla Enter, limpeza da
conversa e da memória, e geração de uma análise estruturada exibida como JSON somente
depois da validação Pydantic. O aviso permanente no rodapé reforça privacidade, uso
responsável e revisão humana.

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

Ambas usam exclusivamente `gemma4:cloud`. Templates, schema, memória, regras de negócio,
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
| Modelo e configuração segura | Somente `gemma4:cloud`; chave carregada do `.env`, nunca hardcoded. |
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
