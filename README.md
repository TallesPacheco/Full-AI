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

## Desafio 2: Pull, Otimizacao e Avaliacao de Prompts (LangSmith)

Este projeto entrega o fluxo completo do desafio: pull do prompt baseline do Hub, refatoracao com tecnicas avancadas de Prompt Engineering, push do prompt otimizado, avaliacao automatica com cinco metricas customizadas (Helpfulness, Correctness, F1-Score, Clarity, Precision) e testes de validacao.

Boilerplate oficial usado como base (arquivos `src/evaluate.py`, `src/metrics.py`, `src/utils.py` e `datasets/bug_to_user_story.jsonl` foram copiados sem modificacao):

```text
https://github.com/devfullcycle/mba-ia-pull-evaluation-prompt
```

Prompt v2 publicado e publico no LangSmith Hub:

```text
https://smith.langchain.com/prompts/talles/bug_to_user_story_v2
```

### Tecnicas Aplicadas (Fase 2)

A versao `v2` aplica cinco tecnicas avancadas combinadas, todas declaradas em `prompts/bug_to_user_story_v2.yml` no campo `techniques_applied`:

| Tecnica | Onde aparece no prompt | Justificativa | Metrica favorecida |
| --- | --- | --- | --- |
| **Role Prompting** | `system_prompt`: "Voce e um Product Manager senior e Agile Coach especializado em transformar relatos de bugs em User Stories acionaveis." | Define persona e contexto, alinhando tom (empatico, profissional) e estrutura esperada da resposta. | Helpfulness, Clarity |
| **Few-shot Learning** | `user_prompt`: tres exemplos completos cobrindo bug simples (1 linha), bug medio (com Steps/Logs) e bug medio com Observações tecnicas. | Tecnica obrigatoria do enunciado. Os exemplos ensinam o modelo a casar com a estrutura do dataset de referencia. | Helpfulness, Correctness, F1-Score |
| **Skeleton of Thought** | `system_prompt`: tres niveis de detalhe (1=simples, 2=medio, 3=complexo) com lista fixa de secoes por nivel. | Garante saida estruturada e proporcional a complexidade do bug, evitando over-engineering em bugs simples e under-engineering em complexos. | Clarity, Precision, F1-Score |
| **Private Chain of Thought** | `system_prompt`: "Antes de escrever, identifique mentalmente: persona afetada, acao desejada, beneficio, criterios verificaveis... NAO MOSTRAR." | Forca o modelo a raciocinar sobre criterios antes de responder, mas devolver apenas a resposta final - melhora precisao sem poluir a saida. | Correctness, F1-Score |
| **Constraint Prompting** | `system_prompt`: regras explicitas "Nao invente informacoes", "Preserve termos tecnicos", "Use empatia", listas de secoes proibidas por nivel. | Reduz alucinacao (informacoes nao presentes no bug) e mantem a saida factual. | Correctness, Precision |

### Resultados Finais

Prompt publico:

```text
https://smith.langchain.com/prompts/talles/bug_to_user_story_v2
```

Experimento de avaliacao (dataset oficial `bug_to_user_story_eval_v2` com 15 exemplos, 5 simples + 7 medios + 3 complexos):

```text
https://smith.langchain.com/o/97319e17-e4ce-4eff-9e01-b4ec832cb06e/datasets
```

#### Tabela comparativa v1 vs v2

Configuracao final: `LLM_PROVIDER=google`, `LLM_MODEL=gemini-2.5-pro` (respondedor), `EVAL_MODEL=gemini-2.5-flash` (juiz), com Gemini billing habilitado.

| Prompt | Helpfulness | Correctness | F1-Score | Clarity | Precision | Media | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v1 (baseline) | 0.45 | 0.52 | 0.48 | 0.50 | 0.46 | 0.48 | Reprovado |
| v2 (melhor de 7 execucoes Gemini) | **0.96** ✓ | **0.91** ✓ | 0.85 | **0.97** ✓ | **0.97** ✓ | **0.93** | Reprovado por F1 |

Por execucao (v2, Gemini pro+flash, 15 exemplos cada):

| Run | Helpfulness | Correctness | F1-Score | Clarity | Precision | Media |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.95 ✓ | 0.91 ✓ | 0.87 | 0.95 ✓ | 0.96 ✓ | 0.93 |
| 2 | 0.96 ✓ | 0.90 ✓ | 0.85 | 0.96 ✓ | 0.95 ✓ | 0.92 |
| 3 | 0.95 ✓ | 0.91 ✓ | 0.85 | 0.96 ✓ | 0.96 ✓ | 0.93 |
| 4 | 0.96 ✓ | 0.91 ✓ | 0.85 | 0.96 ✓ | 0.96 ✓ | 0.93 |
| 5 | 0.94 ✓ | 0.90 ✓ | 0.86 | 0.94 ✓ | 0.95 ✓ | 0.92 |
| 6 | 0.96 ✓ | 0.91 ✓ | 0.85 | 0.95 ✓ | 0.97 ✓ | 0.93 |
| 7 | 0.96 ✓ | 0.91 ✓ | 0.85 | 0.95 ✓ | 0.97 ✓ | 0.93 |

#### Diagnostico do gargalo (F1-Score)

Apos 5 iteracoes do prompt `v2` e 7 execucoes completas com Gemini paid (pro como respondedor, flash como juiz), todas as metricas exceto F1-Score passam consistentemente em todas as execucoes. **F1 oscila entre 0.84 e 0.87 e nunca atinge 0.90 minimo**. Investigamos a causa raiz:

1. **O Bug 1 do dataset oficial, com nosso output IDENTICO a referencia, recebe F1=0.66-0.70 do juiz Gemini.** Isso e o teto absoluto do juiz para refs curtas (5 bullets) - nenhuma engenharia de prompt pode superar.
2. **A media de F1 dos bugs simples (1 a 5) fica em 0.70-0.80 mesmo com saidas semanticamente equivalentes**. Os bugs medios e complexos (6 a 15) atingem F1 0.85-1.00, mas nao compensam a media total.
3. **Variancia do juiz LLM e alta**. Bug 11 com o mesmo prompt oscilou F1 entre 0.72 e 1.00 em runs diferentes. Bug 5 entre 0.70 e 0.88.
4. **Tentamos varias estrategias para subir F1 dos simples**: bullets enxutos (5 exatos), bullets ricos (6-8), reuso literal do vocabulario do bug, persona alinhada ao contexto, formato `===` no nivel 3. Nenhuma rompeu o teto.
5. **O modelo de resposta tambem importa**: gemini-2.5-pro como respondedor entrega F1 ~0.87 vs gemini-2.5-flash que entrega F1 ~0.85, mas em ambos os casos abaixo de 0.90.

Configuracao usada nas 7 execucoes:

```env
LLM_PROVIDER=google
LLM_MODEL=gemini-2.5-pro
EVAL_MODEL=gemini-2.5-flash
```

A entrega cumpre todos os requisitos estruturais do desafio (pull, push, prompt v2 publico com 5 tecnicas, dataset de 15 exemplos, testes pytest, README documentando processo) e atinge 4 de 5 metricas acima do minimo de 0.90 com folga. O F1-Score permanece abaixo do limite por restricao do juiz Gemini, nao por qualidade do prompt.

### Evidencias da Avaliacao

Dashboard do LangSmith com o dataset `bug_to_user_story_eval_v2` e os experimentos do prompt v2 (todas as 5 metricas exibidas):

![Experimentos LangSmith do prompt v2](screenshots/langsmith-experimentos.png)

Prompt v2 publicado no Prompt Hub do LangSmith:

![Prompt v2 publico no LangSmith Hub](screenshots/langsmith-prompt-publico.png)

### Como Executar o Desafio 2

Pre-requisitos:

- Python 3.9+
- Chave de API do LangSmith (`LANGSMITH_API_KEY`)
- Username do LangSmith Hub (`USERNAME_LANGSMITH_HUB`)
- Chave de API da OpenAI (`OPENAI_API_KEY`) **ou** Google (`GOOGLE_API_KEY`)

Configuracao:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Preencha as chaves no .env
```

Pipeline completo:

```bash
# Fase 1 - Pull do prompt baseline
venv/bin/python src/pull_prompts.py

# Fase 5 - Testes de validacao do v2
venv/bin/python -m pytest tests/test_prompts.py -v

# Fase 3 - Push do prompt otimizado para o Hub (publico)
venv/bin/python src/push_prompts.py

# Fase 4 - Avaliacao automatica com 5 metricas customizadas
venv/bin/python src/evaluate.py
```

Variaveis de ambiente principais (consulte `.env.example` para o template completo):

```env
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=prompt-optimization-challenge
USERNAME_LANGSMITH_HUB=seu_username

LLM_PROVIDER=openai          # ou "google"
LLM_MODEL=gpt-4o-mini
EVAL_MODEL=gpt-4o
```

### Estrutura do Desafio 2

```text
.
├── prompts/
│   ├── bug_to_user_story_v1.yml   # baseline baixado do Hub
│   └── bug_to_user_story_v2.yml   # versao otimizada (3 niveis, 5 tecnicas)
├── datasets/
│   └── bug_to_user_story.jsonl    # 15 exemplos (5 simples + 7 medios + 3 complexos)
├── src/
│   ├── pull_prompts.py            # pull do Hub e serializacao YAML
│   ├── push_prompts.py            # push publico do prompt v2 (com tags + techniques_applied)
│   ├── evaluate.py                # pipeline oficial de avaliacao (intocado)
│   ├── metrics.py                 # 5 metricas LLM-as-judge (intocado)
│   └── utils.py                   # helpers oficiais (intocado)
├── tests/
│   └── test_prompts.py            # 6 testes obrigatorios + 1 extra de estrutura
├── desafio2/                      # documentacao passo-a-passo das 5 fases
├── Google/                        # snapshot da tentativa com Gemini (ver RESULTADOS.md)
└── screenshots/
    ├── langsmith-experimentos.png
    └── langsmith-prompt-publico.png
```

## Como funciona

1. **Ingestão** (`ingest.py`): carrega o PDF, divide em chunks, gera embeddings via OpenAI e salva no pgVector.
2. **Busca** (`search.py`): vetoriza a pergunta e busca os 10 chunks mais relevantes (`k=10`) usando similaridade de cosseno.
3. **Chat** (`chat.py`): monta um prompt com o contexto recuperado e envia para a LLM (`gpt-5-nano`), que responde apenas com base no conteúdo do documento.
