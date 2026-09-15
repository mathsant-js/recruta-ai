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
apagar o histórico. Atualmente, `system_prompt_v2.md` é a versão final ativa. As mensagens
`system` e `human` são separadas por `ChatPromptTemplate`, e histórico, solicitação e instruções
de formato são inseridos por variáveis do próprio template, sem montagem manual com f-strings.

Os testes adversariais determinísticos verificam que tentativas de desvio permanecem na mensagem
humana não confiável e que a mensagem de sistema preserva as regras de domínio, equidade,
privacidade, recusa segura e revisão humana. Chamadas reais dependem da credencial local e são
mantidas separadas dos testes unitários determinísticos.

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
