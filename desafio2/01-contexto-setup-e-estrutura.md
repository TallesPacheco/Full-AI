# Passo 1: Contexto, Setup e Estrutura do Desafio 2

## Objetivo deste passo

Preparar o projeto para o desafio de Pull, Otimizacao e Avaliacao de Prompts com LangChain e LangSmith.

A partir de agora, todo o trabalho deste desafio deve ficar organizado em pastas ou arquivos relacionados a `desafio2`, sem misturar com o material do `desafio1`.

## O que aprendemos no repositorio das aulas

Repositorio analisado:

```text
https://github.com/devfullcycle/mba-ia-prompt-engineering/tree/main
```

O repositorio das aulas esta dividido por capitulos:

```text
1-tipos-de-prompts/
4-prompts-e-workflow-de-agentes/
5-gerenciamento-e-versionamento-de-prompts/
6-prompt-enriquecido/
7-evaluation/
```

Para este desafio, os capitulos mais importantes sao:

| Capitulo | Utilidade para o desafio |
| --- | --- |
| `1-tipos-de-prompts` | Ensina as tecnicas que vamos aplicar no prompt v2: Role Prompting, Few-shot, CoT, ToT, SoT, ReAct. |
| `5-gerenciamento-e-versionamento-de-prompts` | Mostra como organizar prompts em YAML, validar com testes e fazer push/pull no LangSmith. |
| `7-evaluation` | Mostra como avaliar prompts com LangSmith, criterios customizados, correctness, precision, recall e F1. |

## Estado atual do nosso projeto

Hoje o projeto `Full-AI` esta organizado principalmente para o desafio anterior de RAG:

```text
Full-AI/
├── README.md
├── docker-compose.yml
├── documento.pdf
├── requirements.txt
├── chains/
│   └── desafio1/
└── src/
    └── desafio1/
```

Esse novo desafio precisa de uma estrutura propria. A recomendacao e criar uma camada `desafio2` para documentacao e manter os scripts obrigatorios no local esperado pelo enunciado.

## Estrutura alvo

Estrutura recomendada para chegarmos no formato do desafio:

```text
Full-AI/
├── desafio2/
│   ├── 01-contexto-setup-e-estrutura.md
│   ├── 02-pull-prompt-langsmith.md
│   ├── 03-otimizacao-prompt-yaml.md
│   ├── 04-push-versionamento-langsmith.md
│   └── 05-avaliacao-testes-iteracao-entrega.md
│
├── prompts/
│   ├── bug_to_user_story_v1.yml
│   └── bug_to_user_story_v2.yml
│
├── datasets/
│   └── bug_to_user_story.jsonl
│
├── src/
│   ├── desafio1/
│   ├── pull_prompts.py
│   ├── push_prompts.py
│   ├── evaluate.py
│   ├── metrics.py
│   └── utils.py
│
└── tests/
    └── test_prompts.py
```

Observacao: o enunciado exige `src/pull_prompts.py`, `src/push_prompts.py`, `tests/test_prompts.py` e `prompts/bug_to_user_story_v2.yml`. Por isso, mesmo trabalhando conceitualmente em `desafio2`, alguns arquivos precisam ficar nos caminhos esperados.

## Setup de ambiente

O repositorio das aulas reforca que cada capitulo deve ter seu ambiente configurado com:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

No nosso projeto, ja existe um `requirements.txt` com boa parte das dependencias:

- `langchain`
- `langchain-openai`
- `langchain-google-genai`
- `langsmith`
- `python-dotenv`
- `PyYAML`

Precisamos confirmar ou adicionar:

- `pytest`

## Variaveis de ambiente

O desafio usa OpenAI ou Gemini e LangSmith.

O `.env` local deve conter:

```env
OPENAI_API_KEY=
GOOGLE_API_KEY=

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=prompt-management-system

LANGSMITH_USERNAME=
```

Modelos sugeridos pelo enunciado:

| Provider | Uso | Modelo |
| --- | --- | --- |
| OpenAI | Responder | `gpt-4o-mini` |
| OpenAI | Avaliar | `gpt-4o` |
| Gemini | Responder e avaliar | `gemini-2.5-flash` |

## Cuidado com chaves

Antes de qualquer commit:

- `.env` nao deve ser versionado.
- `.env.example` deve ter apenas placeholders.
- Nenhuma chave real deve aparecer em Markdown, README ou codigo.

## Checklist deste passo

- [x] Criar pasta `desafio2/`.
- [x] Criar os 5 documentos de acompanhamento.
- [x] Conferir se `requirements.txt` tem as dependencias necessarias.
- [x] Adicionar `pytest`, se estiver ausente.
- [x] Preparar `.env` local com LangSmith e provider de LLM.
- [x] Limpar `.env.example` para remover qualquer chave real.
- [x] Criar estrutura `prompts/`, `datasets/` e `tests/`.

## Validacao do Checklist

Validacao realizada no projeto local:

| Item | Status | Observacao |
| --- | --- | --- |
| Pasta `desafio2/` | OK | Pasta criada com os documentos do desafio. |
| 5 documentos de acompanhamento | OK | Arquivos `01` a `05` existem em `desafio2/`. |
| Dependencias principais | OK | `langchain`, `langchain-openai`, `langchain-google-genai`, `langsmith`, `python-dotenv` e `PyYAML` estao no `requirements.txt`. |
| `pytest` | OK | Dependencia adicionada ao `requirements.txt`. |
| `.env` local | OK | Arquivo existe localmente. Os valores nao foram expostos na documentacao. |
| `.env.example` seguro | OK | Chave real removida e `LANGSMITH_USERNAME` adicionado como placeholder. |
| Estrutura base | OK | Pastas `prompts/`, `datasets/` e `tests/` criadas. |

## Proxima etapa

Depois do setup, vamos implementar o pull do prompt ruim:

```text
leonanluppi/bug_to_user_story_v1
```

Esse sera o foco do arquivo:

```text
desafio2/02-pull-prompt-langsmith.md
```
