# Desafio: Pull, Otimizacao e Avaliacao de Prompts com LangChain e LangSmith

## 1. Objetivo do Desafio

Este desafio pede a construcao de um fluxo completo de gestao e avaliacao de prompts usando LangChain e LangSmith.

O software final deve ser capaz de:

- Fazer pull de um prompt ruim publicado no LangSmith Prompt Hub.
- Salvar esse prompt localmente em formato YAML.
- Refatorar o prompt usando tecnicas avancadas de Prompt Engineering.
- Fazer push da versao otimizada para o LangSmith Prompt Hub.
- Avaliar a qualidade do prompt usando metricas customizadas.
- Iterar ate que todas as metricas fiquem acima de `0.9`.

As metricas obrigatorias sao:

- `Helpfulness >= 0.9`
- `Correctness >= 0.9`
- `F1-Score >= 0.9`
- `Clarity >= 0.9`
- `Precision >= 0.9`

Importante: nao basta a media ser maior que `0.9`. Todas as cinco metricas precisam atingir o minimo.

## 2. Contexto do Projeto Atual

O projeto atual em `Full-AI` ja possui uma base de estudos com LangChain, OpenAI, Gemini, LangSmith e RAG.

A estrutura existente esta mais voltada para o desafio anterior de ingestao e busca semantica:

```text
Full-AI/
├── README.md
├── docker-compose.yml
├── documento.pdf
├── requirements.txt
├── chains/
│   └── desafio1/
│       ├── agentsCli.py
│       ├── carregamentoPdf.py
│       ├── ingestion-pgvector.py
│       └── webBaseLoader.py
└── src/
    └── desafio1/
        ├── chat.py
        ├── ingest.py
        └── search.py
```

O `README.md` atual descreve um projeto de RAG com:

- LangChain
- PostgreSQL com pgVector
- OpenAI embeddings
- CLI de perguntas e respostas com base em um PDF

Para este novo desafio, precisamos adicionar uma segunda area de trabalho, focada em Prompt Hub e avaliacao no LangSmith.

## 3. Dependencias Ja Presentes

O arquivo `requirements.txt` ja contem boa parte das dependencias exigidas:

- `langchain`
- `langchain-openai`
- `langchain-google-genai`
- `langsmith`
- `python-dotenv`
- `PyYAML`
- `pytest` ainda precisa ser confirmado/adicionado caso nao esteja instalado

Tambem ja existem variaveis relacionadas ao LangSmith em `.env.example`:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=prompt-management-system
```

Atencao: o arquivo `.env.example` deve conter apenas placeholders. Chaves reais de API nao devem ficar versionadas.

## 4. Estrutura Alvo do Desafio

A estrutura esperada para este desafio e:

```text
Full-AI/
├── .env.example
├── requirements.txt
├── README.md
├── DESAFIO_PROMPTS_LANGSMITH.md
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
│   │   ├── chat.py
│   │   ├── ingest.py
│   │   └── search.py
│   │
│   ├── pull_prompts.py
│   ├── push_prompts.py
│   ├── evaluate.py
│   ├── metrics.py
│   └── utils.py
│
└── tests/
    └── test_prompts.py
```

Como o projeto atual ainda nao possui `prompts/`, `datasets/` e `tests/`, esses diretorios precisam ser criados ou importados do boilerplate do desafio.

## 5. Entendimento do Fluxo

O fluxo completo do desafio tem cinco fases.

### Fase 1: Pull do Prompt Ruim

Objetivo: buscar no LangSmith Prompt Hub o prompt inicial de baixa qualidade:

```text
leonanluppi/bug_to_user_story_v1
```

O script `src/pull_prompts.py` deve:

- Carregar variaveis de ambiente com `dotenv`.
- Conectar ao LangSmith.
- Fazer pull do prompt usando `langchain.hub`.
- Converter ou serializar o conteudo para YAML.
- Salvar o resultado em:

```text
prompts/bug_to_user_story_v1.yml
```

Comando esperado:

```bash
python src/pull_prompts.py
```

### Fase 2: Otimizacao do Prompt

Objetivo: criar uma versao otimizada do prompt em:

```text
prompts/bug_to_user_story_v2.yml
```

O prompt otimizado precisa:

- Ter `system_prompt` claro e preenchido.
- Definir persona, por exemplo: Product Manager, Agile Coach ou especialista em escrita de user stories.
- Usar corretamente separacao entre System Prompt e User Prompt.
- Exigir formato de saida padronizado, preferencialmente Markdown.
- Incluir Few-shot Learning com exemplos de entrada e saida.
- Tratar edge cases.
- Listar metadados com pelo menos duas tecnicas de Prompt Engineering.

Tecnicas recomendadas para esta tarefa:

- Few-shot Learning: obrigatoria, com 2 ou 3 exemplos de bugs convertidos em user stories.
- Role Prompting: define a persona do modelo como especialista em produto/agilidade.
- Skeleton of Thought: estrutura a resposta em secoes fixas.
- Chain of Thought privado: orientar o modelo a analisar criterios antes de responder, mas devolver apenas a resposta final.

Sugestao de formato YAML:

```yaml
name: bug_to_user_story_v2
description: Prompt otimizado para converter bugs em user stories claras, testaveis e acionaveis.
metadata:
  version: v2
  techniques:
    - Few-shot Learning
    - Role Prompting
    - Skeleton of Thought
  target_metrics:
    helpfulness: 0.9
    correctness: 0.9
    f1_score: 0.9
    clarity: 0.9
    precision: 0.9
system_prompt: |
  Voce e um Product Manager senior especializado em transformar relatos de bugs em user stories acionaveis.
user_prompt: |
  Converta o bug abaixo em uma user story no formato especificado.

  Bug:
  {bug}
```

### Fase 3: Push do Prompt Otimizado

Objetivo: publicar o prompt `v2` no LangSmith Prompt Hub.

O script `src/push_prompts.py` deve:

- Ler `prompts/bug_to_user_story_v2.yml`.
- Montar o prompt no formato esperado pelo LangChain.
- Fazer push para o LangSmith com nome versionado:

```text
{seu_username}/bug_to_user_story_v2
```

- Adicionar metadados:
  - tags
  - descricao
  - tecnicas utilizadas
  - versao

Comando esperado:

```bash
python src/push_prompts.py
```

Depois do push, verificar no dashboard do LangSmith se:

- O prompt foi criado.
- O prompt esta publico.
- A versao correta aparece no Prompt Hub.

### Fase 4: Avaliacao e Iteracao

Objetivo: executar a avaliacao automatica e iterar ate atingir nota minima em todas as metricas.

Comando esperado:

```bash
python src/evaluate.py
```

Resultado desejado:

```text
Metricas Derivadas:
  - Helpfulness: >= 0.9
  - Correctness: >= 0.9

Metricas Base:
  - F1-Score: >= 0.9
  - Clarity: >= 0.9
  - Precision: >= 0.9

STATUS: APROVADO
```

Processo de iteracao:

1. Rodar avaliacao.
2. Identificar metricas abaixo de `0.9`.
3. Abrir traces no LangSmith.
4. Analisar entradas em que o modelo falhou.
5. Ajustar `prompts/bug_to_user_story_v2.yml`.
6. Fazer push novamente.
7. Rodar avaliacao novamente.

Esperado: de 3 a 5 iteracoes.

### Fase 5: Testes de Validacao

Objetivo: implementar testes em:

```text
tests/test_prompts.py
```

Testes minimos exigidos:

- `test_prompt_has_system_prompt`
- `test_prompt_has_role_definition`
- `test_prompt_mentions_format`
- `test_prompt_has_few_shot_examples`
- `test_prompt_no_todos`
- `test_minimum_techniques`

Comando de validacao:

```bash
pytest tests/test_prompts.py
```

Esses testes devem validar principalmente o arquivo:

```text
prompts/bug_to_user_story_v2.yml
```

## 6. Checklist de Implementacao

### Preparacao

- [ ] Criar branch para o desafio.
- [ ] Conferir se `.env.example` nao contem chaves reais.
- [ ] Criar `.env` local com as chaves corretas.
- [ ] Confirmar `OPENAI_API_KEY` ou `GOOGLE_API_KEY`.
- [ ] Confirmar `LANGCHAIN_API_KEY`.
- [ ] Confirmar `LANGCHAIN_PROJECT`.
- [ ] Instalar dependencias com `pip install -r requirements.txt`.
- [ ] Adicionar `pytest` ao `requirements.txt`, se necessario.

### Estrutura

- [ ] Criar diretorio `prompts/`.
- [ ] Criar diretorio `datasets/`, caso o boilerplate ainda nao tenha sido copiado.
- [ ] Criar diretorio `tests/`.
- [ ] Criar `src/pull_prompts.py`.
- [ ] Criar `src/push_prompts.py`.
- [ ] Copiar ou criar `src/evaluate.py`.
- [ ] Copiar ou criar `src/metrics.py`.
- [ ] Copiar ou criar `src/utils.py`.

### Pull

- [ ] Implementar `src/pull_prompts.py`.
- [ ] Fazer pull de `leonanluppi/bug_to_user_story_v1`.
- [ ] Salvar em `prompts/bug_to_user_story_v1.yml`.
- [ ] Conferir se o YAML ficou legivel.

### Prompt Otimizado

- [ ] Analisar `prompts/bug_to_user_story_v1.yml`.
- [ ] Criar `prompts/bug_to_user_story_v2.yml`.
- [ ] Definir persona no `system_prompt`.
- [ ] Separar claramente System Prompt e User Prompt.
- [ ] Adicionar regras explicitas de comportamento.
- [ ] Adicionar formato de saida em Markdown.
- [ ] Adicionar Few-shot Learning com exemplos de entrada e saida.
- [ ] Adicionar tratamento de edge cases.
- [ ] Preencher metadados com tecnicas utilizadas.

### Push

- [ ] Implementar `src/push_prompts.py`.
- [ ] Ler YAML otimizado.
- [ ] Montar prompt LangChain.
- [ ] Fazer push para `{seu_username}/bug_to_user_story_v2`.
- [ ] Adicionar tags e descricao.
- [ ] Tornar o prompt publico no LangSmith.

### Avaliacao

- [ ] Executar `python src/evaluate.py`.
- [ ] Registrar notas da v1.
- [ ] Registrar notas da v2.
- [ ] Abrir traces com falhas.
- [ ] Ajustar prompt.
- [ ] Repetir ate todas as metricas ficarem `>= 0.9`.

### Testes

- [ ] Implementar `tests/test_prompts.py`.
- [ ] Rodar `pytest tests/test_prompts.py`.
- [ ] Corrigir falhas dos testes.

### Documentacao Final

- [ ] Atualizar `README.md`.
- [ ] Adicionar secao "Tecnicas Aplicadas (Fase 2)".
- [ ] Adicionar secao "Resultados Finais".
- [ ] Adicionar tabela comparativa v1 vs v2.
- [ ] Adicionar link publico do dashboard LangSmith.
- [ ] Adicionar screenshots das avaliacoes.
- [ ] Adicionar secao "Como Executar".

## 7. Plano de Execucao Sugerido

### Passo 1: Organizar o projeto

Criar a estrutura que ainda nao existe:

```bash
mkdir -p prompts datasets tests
```

Criar os scripts esperados pelo desafio:

```bash
touch src/pull_prompts.py src/push_prompts.py tests/test_prompts.py
```

Se o boilerplate oficial tiver `evaluate.py`, `metrics.py`, `utils.py` e o dataset, copiar esses arquivos para este projeto antes de implementar o restante.

### Passo 2: Ajustar ambiente

Criar `.env` a partir do exemplo:

```bash
cp .env.example .env
```

Preencher:

```env
OPENAI_API_KEY=
GOOGLE_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=prompt-management-system
LANGSMITH_USERNAME=
```

Podemos usar OpenAI ou Gemini:

- OpenAI resposta: `gpt-4o-mini`
- OpenAI avaliacao: `gpt-4o`
- Gemini resposta e avaliacao: `gemini-2.5-flash`

### Passo 3: Implementar pull

Implementar um script simples que faca:

1. `load_dotenv()`
2. valide `LANGCHAIN_API_KEY`
3. use `hub.pull("leonanluppi/bug_to_user_story_v1")`
4. salve o conteudo em YAML

### Passo 4: Criar prompt v2

Construir `prompts/bug_to_user_story_v2.yml` com foco nas metricas:

- Helpfulness: incluir informacoes uteis para Product/Dev/QA.
- Correctness: nao inventar dados ausentes.
- F1-Score: capturar bem entidades importantes do bug.
- Clarity: resposta estruturada e sem ambiguidade.
- Precision: evitar conteudo generico ou excessivo.

Formato recomendado da resposta do modelo:

```markdown
## User Story
Como [persona],
quero [acao/necessidade],
para [beneficio/resultado].

## Contexto do Bug
- ...

## Criterios de Aceite
- Dado ...
  Quando ...
  Entao ...

## Regras e Restricoes
- ...

## Casos de Borda
- ...
```

### Passo 5: Implementar push

O script deve publicar o prompt otimizado no LangSmith com:

```text
{LANGSMITH_USERNAME}/bug_to_user_story_v2
```

Caso `LANGSMITH_USERNAME` nao exista no `.env`, o script deve falhar com uma mensagem clara.

### Passo 6: Implementar testes

Usar `pytest` e `yaml.safe_load()` para validar o YAML.

Os testes devem verificar:

- campo `system_prompt`
- persona no prompt
- mencao a Markdown ou User Story
- exemplos few-shot
- ausencia de `[TODO]`
- pelo menos duas tecnicas em `metadata.techniques`

### Passo 7: Rodar avaliacao e iterar

Executar:

```bash
python src/push_prompts.py
python src/evaluate.py
```

Registrar resultados em uma tabela:

| Prompt | Helpfulness | Correctness | F1-Score | Clarity | Precision | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v1 | 0.45 | 0.52 | 0.48 | 0.50 | 0.46 | Reprovado |
| v2 | 0.94 | 0.96 | 0.93 | 0.95 | 0.92 | Aprovado |

Os valores da tabela acima sao ilustrativos. No README final, substituir pelos resultados reais do LangSmith.

## 8. Riscos e Pontos de Atencao

- Nao commitar `.env` com chaves reais.
- Nao deixar chaves reais em `.env.example`.
- Nao alterar o dataset de avaliacao.
- Nao alterar `evaluate.py`, `metrics.py` e `utils.py`, caso venham prontos do boilerplate.
- Garantir que o prompt v2 esteja publico no LangSmith.
- Garantir que os screenshots e links publicos estejam no README final.
- Garantir que o nome do prompt publicado use o username correto.

## 9. Proxima Acao Recomendada

A proxima etapa pratica e criar a estrutura base do desafio dentro deste repositorio:

```text
prompts/
datasets/
tests/
src/pull_prompts.py
src/push_prompts.py
```

Depois disso, o trabalho deve seguir nesta ordem:

1. Corrigir `.env.example` para remover qualquer chave real.
2. Implementar `pull_prompts.py`.
3. Baixar `bug_to_user_story_v1`.
4. Criar `bug_to_user_story_v2.yml`.
5. Implementar testes.
6. Implementar push.
7. Rodar avaliacao.
8. Iterar ate todas as metricas passarem de `0.9`.
9. Atualizar README com tecnicas, resultados, links e evidencias.
