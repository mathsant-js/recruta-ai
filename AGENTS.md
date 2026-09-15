# AGENTS.md - Recruta AI

Este arquivo orienta qualquer agente que trabalhe neste repositorio. O objetivo e desenvolver o CKP01 da FIAP como um chatbot profissional de recrutamento e RH chamado **Recruta AI**.

## 1. Objetivo do produto

O Recruta AI deve auxiliar recrutadores, analistas de RH, gestores solicitantes e consultorias a:

- levantar e esclarecer requisitos de vagas;
- elaborar e revisar descricoes de vagas;
- separar requisitos obrigatorios de desejaveis;
- identificar competencias tecnicas e comportamentais;
- sugerir perguntas de entrevista;
- organizar informacoes da conversa;
- gerar uma analise estruturada e validada da solicitacao.

O chatbot e uma ferramenta de apoio. Ele nao deve tomar a decisao final de contratacao, inventar informacoes sobre candidatos, inferir atributos sensiveis nem usar criterios discriminatorios.

## 2. Fonte de verdade e prioridades

O enunciado do checkpoint e a fonte de verdade funcional:

`/home/mathsant/Downloads/CKP01_2Semestre_Chatbot_Profissional.pdf`

Em caso de conflito, siga nesta ordem:

1. pedido atual do usuario;
2. requisitos obrigatorios do PDF;
3. este `AGENTS.md`;
4. documentacao e decisoes internas do projeto.

Priorize o trabalho conforme a rubrica:

1. pipeline LCEL funcional - 3,0 pontos;
2. memoria gerenciada e context rot - 2,5 pontos;
3. Pydantic v2 integrado - 2,0 pontos;
4. system prompt e dominio - 1,5 ponto;
5. codigo e documentacao - 1,0 ponto;
6. diferenciais somente depois dos obrigatorios - ate +1,0.

## 3. Requisitos obrigatorios

Toda implementacao deve preservar os seguintes requisitos:

- projeto Python local, sem Colab;
- execucao oficial com `python -m app.main`;
- interface Gradio em `http://localhost:7860`;
- uso exclusivo de `gemma4:cloud` via Ollama Cloud;
- `OLLAMA_API_KEY` carregada de `.env` com `python-dotenv`;
- nenhuma chave hardcoded ou versionada;
- `.env` ignorado pelo Git e ausente do ZIP final;
- `.env.example` presente, sem segredo real;
- arquitetura com duas chains;
- chat com `ConversationChain` e um tipo de memoria gerenciada;
- pipeline estruturado com `ChatPromptTemplate | ChatOllama | PydanticOutputParser`;
- templates com variaveis, sem montar prompts manualmente com f-strings;
- mensagens `system` e `human` separadas;
- schema Pydantic v2 com pelo menos quatro campos tipados;
- demonstracao da memoria com pelo menos cinco turnos;
- demonstracao de context rot usando o mesmo prompt e janelas diferentes;
- system prompt com persona, regras, restricoes e secoes em XML;
- README com dominio, usuarios-alvo, integrantes, RMs, execucao e justificativa da memoria;
- codigo modular e comentarios uteis em PT-BR.

Mesmo que uma API do LangChain esteja marcada como legada ou deprecated, nao substitua silenciosamente a arquitetura exigida no enunciado. Se houver incompatibilidade, fixe uma versao compativel, documente a decisao e mantenha os conceitos obrigatorios do checkpoint.

## 4. Arquitetura esperada

O sistema deve conter dois fluxos claramente separados:

### 4.1 Chat com memoria

- usar `ConversationChain`;
- manter historico por sessao;
- preferir `ConversationTokenBufferMemory` com limite inicial de 1.200 tokens;
- oferecer uma acao para limpar a sessao;
- impedir que o estado de um usuario ou sessao vaze para outro;
- comprovar a recuperacao de informacoes depois de cinco ou mais turnos.

Se a versao escolhida do LangChain exigir outra classe equivalente, verifique primeiro se a mudanca continua atendendo literalmente ao PDF e documente a compatibilidade no README.

### 4.2 Analise estruturada LCEL

Implementar explicitamente uma composicao com o operador `|`:

```python
structured_chain = prompt | llm | pydantic_parser
```

O resultado final deve ser uma instancia Pydantic validada, nao apenas um `dict`. Erros de parsing devem gerar uma mensagem compreensivel ou uma nova tentativa limitada; nunca devem ser ignorados silenciosamente.

## 5. Estrutura de arquivos

Manter, no minimo:

```text
recruta-ai/
|-- app/
|   |-- __init__.py
|   |-- main.py
|   |-- chain.py
|   |-- memory_manager.py
|   |-- schemas.py
|   |-- context_rot.py
|   `-- prompts.py
|-- tests/
|   |-- test_schemas.py
|   |-- test_memory.py
|   `-- test_chains.py
|-- artifacts/
|   `-- context_rot/
|-- .env.example
|-- .gitignore
|-- requirements.txt
`-- README.md
```

Responsabilidades:

- `app/main.py`: interface Gradio e ponto de entrada, sem concentrar regras de negocio;
- `app/chain.py`: construcao e composicao das duas chains;
- `app/memory_manager.py`: configuracao, isolamento e limpeza da memoria;
- `app/schemas.py`: modelos e validadores Pydantic v2;
- `app/prompts.py`: prompts centralizados e versionaveis;
- `app/context_rot.py`: experimento reproduzivel de degradacao de contexto;
- `tests/`: testes rapidos e deterministas sempre que possivel;
- `artifacts/context_rot/`: CSV, tabela e/ou grafico gerados pelo experimento.

Evite dependencias circulares e efeitos colaterais durante imports. A chamada da interface deve ficar protegida por `if __name__ == "__main__":` ou por funcao equivalente usada pelo entry point.

## 6. Comportamento e seguranca do dominio

O system prompt deve usar tags XML, por exemplo:

```xml
<persona>...</persona>
<objectives>...</objectives>
<rules>...</rules>
<safety>...</safety>
<response_style>...</response_style>
```

As regras devem obrigar o assistente a:

- pedir esclarecimentos quando faltarem informacoes essenciais;
- diferenciar requisitos obrigatorios de desejaveis;
- declarar incerteza em vez de inventar dados;
- evitar decisao automatica de contratacao;
- nao classificar pessoas por idade, genero, raca, religiao, deficiencia ou outros atributos sensiveis;
- nao produzir diagnosticos psicologicos;
- sinalizar criterios vagos, subjetivos ou potencialmente discriminatorios;
- recomendar revisao humana;
- preservar dados pessoais e desencorajar o envio de dados desnecessarios;
- manter linguagem profissional, objetiva, inclusiva e em PT-BR por padrao.

Nao use dados pessoais reais em exemplos, testes ou artefatos versionados. Use nomes e curriculos ficticios.

## 7. Schema estruturado

O schema principal recomendado e `AnaliseRecrutamento`, com campos equivalentes a:

- `intencao`;
- `resumo`;
- `competencias_tecnicas`;
- `competencias_comportamentais`;
- `perguntas_sugeridas`;
- `pontos_de_atencao`;
- `proximo_passo`;
- `confianca`.

Use tipos restritos quando fizer sentido, como `Literal`, listas tipadas e `Field(ge=0, le=1)`. Adicione validators para regras condicionais relevantes e para impedir listas ou textos invalidos. Preserve pelo menos quatro campos tipados mesmo se o schema for reformulado.

## 8. Experimento de context rot

O experimento deve ser reproduzivel e comparar a mesma tarefa sob janelas diferentes. Como ponto de partida, use 256, 512, 800, 1.200 e 1.500 tokens.

Mantenha constantes:

- modelo;
- system prompt;
- pergunta final;
- conjunto-base de fatos importantes;
- criterio de avaliacao.

Varie apenas a janela e a quantidade/posicao do contexto de distracao. Registre, quando possivel:

- contagem aproximada de tokens com `tiktoken`;
- fatos corretamente recuperados;
- fatos omitidos;
- informacoes inventadas;
- aderencia a persona e restricoes;
- utilidade da resposta;
- tempo de resposta.

O resultado deve gerar CSV e uma tabela ou grafico comparativo. Nao fabrique resultados. Artefatos que dependem de chamadas reais ao modelo devem indicar data, configuracao e limitacoes da execucao.

## 9. Meta prompting

Meta prompting e diferencial e nao deve atrasar requisitos obrigatorios. Quando implementado:

1. preserve a versao inicial do system prompt;
2. solicite ao modelo uma critica estruturada;
3. revise as sugestoes manualmente;
4. salve a versao final;
5. documente antes, depois, justificativas e efeito observado.

Nunca substitua o prompt automaticamente sem revisao humana.

## 10. Interface Gradio

A interface minima deve ter:

- nome e descricao do Recruta AI;
- area de chat;
- campo de mensagem e envio;
- botao para limpar conversa/memoria;
- acao para gerar a analise estruturada;
- exibicao legivel da saida validada;
- aviso de apoio a decisao, privacidade e revisao humana.

Mantenha a interface simples. Nao invista em refinamento visual antes de as duas chains, memoria e validacao estarem funcionando.

## 11. Ordem de desenvolvimento

Siga esta sequencia, salvo pedido explicito em contrario:

1. criar a estrutura do pacote, ambiente e configuracao segura;
2. implementar prompts e persona;
3. implementar chat e memoria;
4. implementar schema, parser e pipeline LCEL;
5. integrar a interface Gradio;
6. implementar e executar o context rot;
7. adicionar testes e documentacao;
8. realizar meta prompting;
9. revisar e preparar o ZIP de entrega.

Nao marque uma etapa como concluida sem atender ao criterio de aceite correspondente.

## 12. Testes e verificacao

Antes de encerrar uma mudanca, execute os testes relevantes e reporte o resultado. A verificacao final deve cobrir:

- importacao dos modulos;
- inicio por `python -m app.main`;
- comportamento quando `OLLAMA_API_KEY` estiver ausente;
- schema valido e invalido;
- integracao real do `PydanticOutputParser`;
- historico com pelo menos cinco turnos;
- limpeza e isolamento da memoria;
- tratamento de falha do parser ou da API;
- prompts fora do escopo e tentativas de quebrar a persona;
- reproducao do experimento de context rot;
- ausencia de segredos no Git.

Testes que exigem Ollama Cloud devem ser distinguiveis dos testes unitarios e devem falhar com mensagem clara quando nao houver credencial. Nao simule uma execucao real nem declare sucesso se a chamada ao modelo nao ocorreu.

## 13. Documentacao e entrega

O README final deve conter:

- nome do projeto;
- nomes e RMs de todos os integrantes;
- dominio, justificativa e usuarios-alvo;
- arquitetura das duas chains;
- tabela de requisitos atendidos;
- instalacao e execucao local;
- justificativa da memoria e seu custo em tokens;
- roteiro de cinco ou mais turnos;
- metodologia e resultados do context rot;
- antes/depois do meta prompting, se implementado;
- limitacoes, privacidade e uso responsavel.

Antes de gerar a entrega:

- execute os testes;
- confira `git status` e procure segredos;
- valide o projeto a partir de um ambiente limpo;
- confirme que `.env` nao esta no pacote;
- inclua `app/`, `.env.example`, `requirements.txt` e `README.md`;
- nomeie o arquivo como `CKP01_recrutamento_RH_grupo.zip`, salvo orientacao diferente do professor.

## 14. Regras para agentes

- Leia este arquivo e o README antes de alterar codigo.
- Inspecione o estado atual do repositorio; nao suponha que etapas anteriores estejam prontas.
- Preserve mudancas do usuario e nao reverta arquivos fora do escopo.
- Prefira alteracoes pequenas, modulares e verificaveis.
- Nao introduza outro provedor ou modelo sem autorizacao explicita.
- Nao exponha chaves em codigo, logs, exemplos, testes ou respostas.
- Nao altere resultados experimentais manualmente para parecerem melhores.
- Se uma dependencia ou API divergir do enunciado, explique a incompatibilidade antes de mudar a arquitetura.
- Mantenha comentarios e documentacao em PT-BR; nomes tecnicos podem permanecer em ingles.
- Ao concluir, informe arquivos alterados, testes executados, limitacoes e trabalho restante.

## 15. Definicao de pronto do CKP01

O checkpoint so esta pronto quando:

- as duas chains estao implementadas e integradas;
- o operador `|` aparece no pipeline LCEL real;
- o chat recupera fatos depois de pelo menos cinco turnos;
- a memoria esta justificada e limitada entre 800 e 1.500 tokens;
- uma saida real e validada por Pydantic v2;
- o context rot possui evidencia comparativa real;
- o system prompt usa XML e resiste a testes basicos de desvio de persona;
- `python -m app.main` inicia a interface;
- testes e documentacao estao atualizados;
- o pacote final nao contem `.env`, credenciais nem dados pessoais reais.
