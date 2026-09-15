## Plano de desenvolvimento — Recruta AI

O repositório contém apenas um README inicial. O plano abaixo parte de um projeto praticamente novo e cobre os requisitos do checkpoint descritos no PDF. :codex-file-citation{path="/home/mathsant/Downloads/CKP01_2Semestre_Chatbot_Profissional.pdf" purpose="source"}

### 1. Escopo do produto

O **Recruta AI** será um assistente para profissionais de recrutamento e RH, voltado a:

- levantar e esclarecer requisitos de uma vaga;
- ajudar a elaborar descrições de vagas;
- identificar competências técnicas e comportamentais;
- sugerir perguntas de entrevista;
- organizar informações discutidas durante a sessão;
- gerar uma análise estruturada da solicitação.

Usuários-alvo:

- recrutadores;
- analistas de RH;
- gestores solicitantes de vagas;
- consultorias de recrutamento.

O chatbot não deverá tomar a decisão final de contratação, inferir atributos sensíveis nem emitir afirmações sem dados suficientes.

---

## 2. Arquitetura proposta

O requisito central é implementar **duas chains**:

```text
Interface Gradio
      |
      +-- Chat livre
      |     └── ConversationChain
      |           └── Memória gerenciada
      |
      +-- Análise estruturada
            └── ChatPromptTemplate
                  | ChatOllama(gemma4:cloud)
                  | PydanticOutputParser
```

### Chain 1 — Conversa com memória

Responsável pelo diálogo contínuo com o recrutador.

- `ConversationChain`;
- histórico gerenciado;
- memória `ConversationTokenBufferMemory`;
- limite sugerido: **1.200 tokens**;
- demonstração obrigatória com pelo menos cinco turnos.

A memória por tokens é uma boa escolha porque conversas de recrutamento podem conter descrições longas de vagas, competências e perguntas. O limite evita crescimento indefinido do histórico e permite demonstrar claramente o efeito do contexto.

### Chain 2 — Saída estruturada

Responsável por transformar uma solicitação em uma análise validada:

```python
prompt | llm | pydantic_parser
```

Ela poderá ser acionada por um botão como **“Gerar análise estruturada”** na interface.

---

## 3. Estrutura planejada

```text
recruta-ai/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── chain.py
│   ├── memory_manager.py
│   ├── schemas.py
│   ├── context_rot.py
│   └── prompts.py
├── tests/
│   ├── test_schemas.py
│   ├── test_memory.py
│   └── test_chains.py
├── artifacts/
│   └── context_rot/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Responsabilidades:

| Arquivo | Responsabilidade |
|---|---|
| `main.py` | Interface Gradio e ponto de entrada |
| `chain.py` | Criação das duas chains |
| `memory_manager.py` | Construção e configuração da memória |
| `schemas.py` | Modelos Pydantic v2 |
| `prompts.py` | System prompt e templates |
| `context_rot.py` | Experimento de degradação do contexto |
| `tests/` | Testes básicos de schema, memória e inicialização |
| `artifacts/context_rot/` | CSV, tabela ou gráfico do experimento |

---

## 4. Schema Pydantic

Sugestão de modelo com mais do que os quatro campos exigidos:

```python
class AnaliseRecrutamento(BaseModel):
    intencao: Literal[
        "criar_vaga",
        "revisar_vaga",
        "planejar_entrevista",
        "analisar_requisitos",
        "outro",
    ]
    resumo: str
    competencias_tecnicas: list[str]
    competencias_comportamentais: list[str]
    perguntas_sugeridas: list[str]
    pontos_de_atencao: list[str]
    proximo_passo: str
    confianca: float
```

Validações recomendadas:

- `confianca` entre `0` e `1`;
- `resumo` com tamanho mínimo;
- remoção de itens vazios das listas;
- pelo menos uma pergunta quando a intenção for `planejar_entrevista`;
- rejeição ou nova tentativa quando a saída do modelo não respeitar o schema.

---

## 5. System prompt

O prompt deverá usar seções XML, como solicitado:

```xml
<persona>
Você é o Recruta AI, assistente de recrutamento e RH...
</persona>

<objectives>
...
</objectives>

<rules>
...
</rules>

<safety>
...
</safety>

<response_style>
...
</response_style>
```

Regras de domínio importantes:

- fazer perguntas quando faltarem dados da vaga;
- separar requisitos obrigatórios de desejáveis;
- não inventar experiência ou qualificações de candidatos;
- não usar idade, gênero, raça, religião, deficiência ou outros atributos sensíveis como critérios;
- não apresentar a resposta como decisão final de contratação;
- evitar diagnósticos psicológicos;
- alertar quando uma exigência parecer subjetiva ou discriminatória;
- preservar a privacidade dos dados recebidos;
- manter linguagem profissional, objetiva e inclusiva.

---

## 6. Plano por etapas

### Etapa 1 — Fundação do projeto

- Criar a estrutura do pacote `app/`.
- Adicionar `.gitignore`.
- Criar `.env.example` contendo apenas:

```env
OLLAMA_API_KEY=
```

- Configurar `python-dotenv`.
- Declarar dependências em `requirements.txt`.
- Configurar exclusivamente `gemma4:cloud`.
- Garantir que `python -m app.main` seja o comando oficial.

**Critério de conclusão:** aplicação importa e inicia sem erro de estrutura ou configuração.

### Etapa 2 — Prompts e domínio

- Escrever a persona completa do Recruta AI.
- Separar mensagens `system` e `human`.
- Usar `ChatPromptTemplate` com variáveis.
- Definir casos suportados e limites do chatbot.
- Não usar f-strings para substituir o mecanismo de templates.

**Critério de conclusão:** respostas permanecem no papel de assistente de RH e respeitam as restrições em testes adversariais simples.

### Etapa 3 — Chat com memória

- Implementar `ConversationChain`.
- Criar a memória com limite de aproximadamente 1.200 tokens.
- Injetar a memória na chain de conversa.
- Implementar limpeza de sessão.
- Executar roteiro de pelo menos cinco turnos.

Roteiro sugerido:

1. “Estou abrindo uma vaga de desenvolvedor backend.”
2. “A senioridade é pleno.”
3. “Precisamos de Python e PostgreSQL.”
4. “O trabalho será híbrido em São Paulo.”
5. “Quais requisitos obrigatórios eu já informei?”
6. “Crie cinco perguntas de entrevista com base neles.”

**Critério de conclusão:** no último turno, o chatbot recupera corretamente senioridade, tecnologias e modalidade.

### Etapa 4 — Pipeline LCEL e Pydantic

- Implementar o schema `AnaliseRecrutamento`.
- Gerar instruções de formato com `PydanticOutputParser`.
- Criar o pipeline com operador `|`.
- Integrar a análise estruturada à interface.
- Tratar erros de validação com mensagem compreensível ou nova tentativa limitada.

**Critério de conclusão:** a saída real é uma instância Pydantic válida e não apenas um `dict`.

### Etapa 5 — Interface Gradio

A interface mínima deve possuir:

- título e descrição do Recruta AI;
- área de chat;
- campo de mensagem;
- botão de envio;
- botão para limpar a conversa;
- botão para gerar análise estruturada;
- apresentação legível do resultado validado;
- aviso sobre uso responsável e decisão humana.

**Critério de conclusão:** todo o fluxo pode ser demonstrado pelo navegador em `http://localhost:7860`.

### Etapa 6 — Experimento de context rot

Usar sempre:

- o mesmo modelo;
- o mesmo system prompt;
- a mesma pergunta final;
- o mesmo conjunto-base de informações;
- janelas diferentes de contexto.

Sugestão de janelas:

| Cenário | Janela |
|---|---:|
| Muito curta | 256 tokens |
| Curta | 512 tokens |
| Referência mínima | 800 tokens |
| Escolha do chatbot | 1.200 tokens |
| Limite superior | 1.500 tokens |

Inserir progressivamente informações irrelevantes entre os dados importantes e a pergunta final. Medir:

- quantidade aproximada de tokens com `tiktoken`;
- requisitos corretamente recuperados;
- informações inventadas;
- aderência à persona;
- utilidade da resposta;
- tempo de resposta, se viável.

Produzir um CSV e uma tabela ou gráfico comparativo. O README deve explicar onde ocorreu a degradação e por quê.

**Critério de conclusão:** há evidência real de perda de informações ou qualidade conforme o contexto cresce ou é truncado.

### Etapa 7 — Meta prompting

Para buscar o diferencial de `+0,5`:

1. preservar a versão inicial do system prompt;
2. pedir ao modelo que identifique ambiguidades, lacunas e riscos;
3. revisar manualmente as sugestões;
4. criar a versão final;
5. documentar o antes, as mudanças e os resultados observados.

Não convém aceitar automaticamente todas as sugestões do modelo.

### Etapa 8 — Testes e documentação

Testar:

- inicialização pelo comando oficial;
- ausência da chave de API;
- schema válido e inválido;
- histórico com mais de cinco turnos;
- limpeza de memória;
- erro de parsing;
- comportamento fora do escopo;
- execução do experimento de context rot.

Completar no README:

- nomes e RMs dos integrantes;
- descrição e justificativa do domínio;
- usuários-alvo;
- arquitetura das duas chains;
- justificativa da memória;
- requisitos atendidos;
- instruções de instalação e execução;
- roteiro de demonstração de memória;
- resultados do context rot;
- comparação do meta prompting;
- limitações e uso responsável.

---

## 7. Cronograma sugerido

| Período | Entrega |
|---|---|
| Dia 1 | Estrutura, ambiente, modelo e prompts |
| Dia 2 | ConversationChain e memória |
| Dia 3 | Pipeline LCEL, Pydantic e interface |
| Dia 4 | Context rot, métricas e meta prompting |
| Dia 5 | Testes, README, revisão e empacotamento |

Se houver três integrantes:

- **Pessoa 1:** prompts, chat e memória;
- **Pessoa 2:** schema, LCEL e testes;
- **Pessoa 3:** Gradio, context rot e documentação.

Todos devem participar da integração e da revisão final.

---

## 8. Priorização pela rubrica

A ordem de esforço deve acompanhar a pontuação:

1. **Pipeline LCEL funcional — 3,0**
2. **Memória e context rot — 2,5**
3. **Pydantic integrado — 2,0**
4. **System prompt e domínio — 1,5**
5. **Código e documentação — 1,0**
6. **Diferenciais — até +1,0**

Os diferenciais só devem ser trabalhados depois que todos os itens obrigatórios estiverem funcionando.

---

## 9. Checklist de entrega

- [ ] `python -m app.main` inicia sem erros.
- [ ] `gemma4:cloud` é o único modelo utilizado.
- [ ] Existe uma `ConversationChain` com memória.
- [ ] Existe uma chain LCEL com operador `|`.
- [ ] O prompt usa `ChatPromptTemplate`.
- [ ] Mensagens system e human estão separadas.
- [ ] O schema possui pelo menos quatro campos tipados.
- [ ] `PydanticOutputParser` está integrado à chain.
- [ ] A memória funciona por cinco ou mais turnos.
- [ ] O context rot compara janelas diferentes.
- [ ] Há tabela ou gráfico de degradação.
- [ ] O system prompt utiliza XML tagging.
- [ ] O README contém integrantes e RMs.
- [ ] O README justifica a estratégia de memória.
- [ ] `.env` está ignorado e fora da entrega.
- [ ] `.env.example` está incluído.

- Implementar a estrutura inicial
- Criar prompts e schemas
- Desenvolver o checkpoint completo