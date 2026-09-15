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
