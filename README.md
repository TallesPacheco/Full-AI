# Ingestão e Busca Semântica com LangChain e Postgres

Software de RAG (Retrieval-Augmented Generation) que lê um PDF, armazena seus vetores no PostgreSQL com pgVector e permite consultas via linha de comando com respostas baseadas exclusivamente no conteúdo do documento.

## Tecnologias

- Python 3.9+
- LangChain
- PostgreSQL + pgVector
- Docker & Docker Compose
- OpenAI (embeddings + LLM)

## Estrutura do projeto

```
├── docker-compose.yaml   # Banco de dados PostgreSQL com pgVector
├── requirements.txt      # Dependências do projeto
├── .env.example          # Template de variáveis de ambiente
├── document.pdf          # PDF para ingestão
└── src/
    ├── ingest.py         # Script de ingestão do PDF
    ├── search.py         # Módulo de busca vetorial
    └── chat.py           # CLI para interação com o usuário
```

## Pré-requisitos

- Python 3.9+
- Docker e Docker Compose instalados
- Chave de API da OpenAI com créditos disponíveis

## Configuração

1. Clone o repositório:

```bash
git clone <url-do-repositorio>
cd <nome-do-repositorio>
```

2. Crie e ative o ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:

```bash
cp .env.example .env
```

Edite o `.env` com suas chaves:

```env
OPENAI_API_KEY=sua-chave-aqui
PGVECTOR_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PGVECTOR_COLLECTION=minha_collection
OPENAI_MODEL=text-embedding-3-small
```

5. Adicione o arquivo `document.pdf` na raiz do projeto.

## Execução

### 1. Subir o banco de dados

```bash
docker compose up -d
```

### 2. Executar a ingestão do PDF

```bash
python src/ingest.py
```

O script lê o `document.pdf`, divide em chunks de 1000 caracteres (overlap de 150), gera os embeddings e armazena no PostgreSQL.

### 3. Rodar o chat

```bash
python src/chat.py
```

Exemplo de uso:

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Digite `sair` para encerrar o chat.

## Como funciona

1. **Ingestão** (`ingest.py`): carrega o PDF, divide em chunks, gera embeddings via OpenAI e salva no pgVector.
2. **Busca** (`search.py`): vetoriza a pergunta e busca os 10 chunks mais relevantes (`k=10`) usando similaridade de cosseno.
3. **Chat** (`chat.py`): monta um prompt com o contexto recuperado e envia para a LLM (`gpt-5-nano`), que responde apenas com base no conteúdo do documento.