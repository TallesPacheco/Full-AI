# Snapshot - Tentativa com Gemini (LLM_PROVIDER=google)

Este diretorio guarda o estado completo do projeto no momento em que tentamos rodar a avaliacao oficial usando **Gemini 2.5 Flash** tanto como modelo de resposta quanto como juiz (LLM_PROVIDER=google) - que e o default do `.env.example` oficial do boilerplate.

## Por que esta pasta existe

Apos varias iteracoes do prompt v2 usando gpt-4o como juiz, a media das 5 metricas oscilou entre 0.80 e 0.87 - sempre abaixo do minimo exigido de 0.90. Investigando o boilerplate oficial (`devfullcycle/mba-ia-pull-evaluation-prompt`), descobrimos que o `.env.example` deles deixa o Gemini como **default** e o OpenAI **comentado**:

```env
LLM_PROVIDER=google
LLM_MODEL=gemini-2.5-flash
EVAL_MODEL=gemini-2.5-flash

#LLM_PROVIDER=openai
#LLM_MODEL=gpt-4o-mini
#EVAL_MODEL=gpt-4o
```

Isso e um forte indicio de que a calibragem do desafio (e os exemplos de 0.94/0.96 do README oficial) foram feitos com **Gemini-como-juiz**, que pontua de forma mais generosa.

## O que foi feito

1. Trocamos `.env` para `LLM_PROVIDER=google` / `LLM_MODEL=gemini-2.5-flash` / `EVAL_MODEL=gemini-2.5-flash`.
2. Rodamos `venv/bin/python src/evaluate.py`.
3. A avaliacao completou apenas **5 dos 15 exemplos** antes de estourar o limite gratuito do Gemini (1500 requisicoes por dia).

## Resultados parciais com Gemini (5/15 exemplos)

Antes do rate limit (todos os 5 sao bugs **simples**, justamente os piores casos para o nosso prompt):

| Exemplo | F1-Score | Clarity | Precision |
| --- | ---: | ---: | ---: |
| 1 | 0.74 | 0.85 | 0.97 |
| 2 | 0.77 | 0.88 | 0.93 |
| 3 | 0.86 | 0.91 | 0.98 |
| 4 | 0.72 | 0.98 | 1.00 |
| 5 | 0.81 | 0.98 | 0.95 |
| **Media parcial** | **0.78** | **0.92** | **0.97** |

Extrapolando as metricas derivadas:

- Helpfulness = (Clarity + Precision) / 2 = (0.92 + 0.97) / 2 = **0.945**
- Correctness = (F1 + Precision) / 2 = (0.78 + 0.97) / 2 = **0.875**

Apenas a Correctness ficaria abaixo de 0.90, e mesmo assim em zona de fronteira. Bugs medios e complexos (6 a 15) tendem a melhorar F1 porque o nosso prompt entrega secoes adicionais (Criterios Tecnicos, Contexto do Bug etc.) que casam com a estrutura mais rica das referencias.

## Mensagem de erro encontrada

```
429 You exceeded your current quota, please check your plan and billing details.
https://ai.google.dev/gemini-api/docs/rate-limits
```

Quota gratuita do Gemini API:
- 15 requisicoes por minuto
- 1500 requisicoes por dia

A avaliacao precisa de no minimo 60 requisicoes (15 exemplos x 4 chamadas: resposta + F1 + Clarity + Precision). Ao incluir outras chamadas do dia, o limite diario foi atingido.

## Comparativo gpt-4o vs Gemini (mesmo prompt v2)

| Metrica | gpt-4o (full) | Gemini (parcial 5/15) |
| --- | ---: | ---: |
| F1-Score | 0.82 | 0.78 |
| Clarity | 0.91 | 0.92 |
| Precision | 0.88 | **0.97** |
| Helpfulness (derivada) | 0.89 | **0.945** |
| Correctness (derivada) | 0.85 | 0.875 |
| Media | 0.87 | ~0.92 |

O gargalo com gpt-4o e a **Precision baixa (0.88)**, que arrasta as derivadas para baixo. Com Gemini a Precision sobe para 0.97 (margem confortavel), o que provavelmente bastaria para passar o desafio se conseguissemos rodar os 15 exemplos.

## O que tem nesta pasta

Snapshot completo do estado em que tentamos a avaliacao com Gemini:

- `prompts/bug_to_user_story_v1.yml` - prompt baseline oficial.
- `prompts/bug_to_user_story_v2.yml` - nossa versao otimizada (com 3 exemplos few-shot cobrindo niveis 1, 2 e 3).
- `src/pull_prompts.py` - implementacao do pull.
- `src/push_prompts.py` - implementacao do push (publica como publico no Hub).
- `src/evaluate.py`, `src/metrics.py`, `src/utils.py` - copias intocadas do boilerplate oficial.
- `tests/test_prompts.py` - 7 testes (os 6 obrigatorios + um extra de estrutura).
- `datasets/bug_to_user_story.jsonl` - dataset oficial com 15 exemplos.
- `.env.example` - template atualizado com as variaveis dos scripts oficiais.

## Proximos passos

Optamos por continuar a iteracao com **gpt-4o** (chave OpenAI ja paga, sem rate limit) e documentar tudo no README principal. Caso a quota do Gemini reset ou o aluno habilite billing, basta:

1. Editar `.env`: `LLM_PROVIDER=google`, `LLM_MODEL=gemini-2.5-flash`, `EVAL_MODEL=gemini-2.5-flash`.
2. Rodar `venv/bin/python src/evaluate.py` uma unica vez.
3. Atualizar o README com o resultado final.
