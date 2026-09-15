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
privacidade, recusa segura e revisão humana. Testes reais de comportamento do modelo serão
adicionados junto à integração das chains.
