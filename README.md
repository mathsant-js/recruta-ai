# Recruta AI

Chatbot para recrutamento e RH.

## Configuração

1. Crie e ative um ambiente virtual.
2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Copie `.env.example` para `.env` e informe sua chave da API do Ollama.

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

Ele usou exclusivamente `gemma4:cloud`, o `system_prompt_v2.md` que estava ativo na data da execução,
a mesma pergunta final e os mesmos oito fatos de uma vaga fictícia em todos os cenários. Um único histórico
de 1.462 tokens aproximados mantém os fatos no início e insere notas administrativas
irrelevantes entre eles e a pergunta final. Para reproduzir o comportamento da
`ConversationTokenBufferMemory`, cada cenário preserva a cauda mais recente e descarta o
excedente conforme a janela. A contagem usa `tiktoken` com `cl100k_base`: ela é uma
aproximação comparável entre cenários, não o tokenizador nativo do Gemma.

Execução real realizada em **15/09/2026**, com saída validada por Pydantic:

| Cenário | Janela | Tokens descartados | Requisitos recuperados | Inventados | Persona | Utilidade | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Muito curta | 256 | 1.206 | 0/8 | 0 | 1 | 0,00 | 0,985 |
| Curta | 512 | 950 | 0/8 | 0 | 1 | 0,00 | 1,010 |
| Referência mínima | 800 | 662 | 0/8 | 0 | 1 | 0,00 | 0,940 |
| Escolha do chatbot | 1.200 | 262 | 0/8 | 0 | 1 | 0,00 | 1,166 |
| Limite superior | 1.500 | 0 | 8/8 | 0 | 1 | 5,00 | 1,402 |

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
Uma nova execução usa a versão ativa atual (`system_prompt_v3.md`) e atualiza os metadados; os
artefatos existentes foram preservados como evidência histórica da execução com a v2.
