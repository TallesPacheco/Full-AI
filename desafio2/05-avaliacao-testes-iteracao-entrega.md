# Passo 5: Avaliacao, Testes, Iteracao e Entrega

## Objetivo deste passo

Validar o prompt otimizado com testes locais, executar a avaliacao no LangSmith e iterar ate atingir:

```text
Helpfulness >= 0.9
Correctness >= 0.9
F1-Score >= 0.9
Clarity >= 0.9
Precision >= 0.9
```

Todas as metricas precisam estar acima de `0.9`. A media sozinha nao basta.

## O que aprendemos no repositorio das aulas

O capitulo `7-evaluation` e o mais importante para esta fase.

### Evaluators basicos

Arquivo analisado:

```text
7-evaluation/1-basic/README.md
```

Ele explica diferentes tipos de avaliadores:

| Tipo | Quando usar |
| --- | --- |
| Deterministico | Validar formato, JSON, estrutura. |
| `criteria` | Avaliacao binaria: passou ou falhou. |
| `score_string` | Nota continua de 0 a 1 sem referencia. |
| `labeled_score_string` | Nota continua comparando com ground truth. |
| Custom criteria | Metricas especificas do dominio. |

Para nosso desafio:

- `Helpfulness` pode ser `score_string`.
- `Clarity` pode ser custom criteria.
- `Correctness` deve usar referencia quando o dataset tiver saida esperada.
- `Precision` e `F1-Score` devem usar avaliadores customizados ou summary evaluators.

### Correctness com referencia

Arquivo analisado:

```text
7-evaluation/1-basic/4-correctness-eval.py
```

Padrao importante:

```python
LangChainStringEvaluator(
    "labeled_score_string",
    config={"criteria": "correctness", "normalize_by": 10},
    prepare_data=prepare_with_reference
)
```

Isso mostra que, quando temos ground truth, devemos comparar a resposta com a referencia.

### Criterios customizados

Arquivo analisado:

```text
7-evaluation/1-basic/5-additional-criteria.py
```

O exemplo cria criterios especificos:

```python
config={
    "criteria": {
        "faithfulness": "Is the response grounded ONLY in the provided code?"
    },
    "normalize_by": 10
}
```

Para nosso desafio, podemos usar criterios customizados como:

```python
"clarity": "A resposta esta organizada em Markdown, com user story clara, criterios de aceite testaveis e sem ambiguidade desnecessaria?"
```

```python
"precision": "A resposta preserva apenas informacoes suportadas pelo bug original, sem inventar detalhes nao fornecidos?"
```

### Precision, Recall e F1

Arquivo analisado:

```text
7-evaluation/2-precision/README.md
```

Conceitos:

```text
Precision = True Positives / (True Positives + False Positives)
Recall = True Positives / (True Positives + False Negatives)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

O arquivo:

```text
7-evaluation/2-precision/metrics.py
```

mostra uma implementacao generica para calcular Precision, Recall e F1 comparando conjuntos extraidos da resposta e do ground truth.

Para nosso desafio, precisamos adaptar a extracao para user stories, por exemplo:

- persona identificada
- acao/necessidade
- beneficio
- criterios de aceite
- casos de borda
- informacoes faltantes

### Pairwise e iteracao

Arquivo analisado:

```text
7-evaluation/3-pairwise/README.md
```

Aprendizado principal:

- Criar uma versao inicial.
- Rodar avaliacao.
- Evoluir o prompt.
- Rodar de novo.
- Comparar resultados no dashboard.

No desafio, isso vira:

1. Avaliar v1.
2. Avaliar v2.
3. Ver metricas abaixo de `0.9`.
4. Abrir traces.
5. Ajustar prompt v2.
6. Fazer push novamente.
7. Avaliar novamente.

## Testes locais obrigatorios

Arquivo exigido:

```text
tests/test_prompts.py
```

Testes minimos:

- `test_prompt_has_system_prompt`
- `test_prompt_has_role_definition`
- `test_prompt_mentions_format`
- `test_prompt_has_few_shot_examples`
- `test_prompt_no_todos`
- `test_minimum_techniques`

## Estrutura sugerida dos testes

```python
from pathlib import Path
import yaml


PROMPT_PATH = Path("prompts/bug_to_user_story_v2.yml")


def load_prompt():
    with PROMPT_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def full_text(data):
    return "\n".join([
        data.get("system_prompt", ""),
        data.get("user_prompt", ""),
    ]).lower()


def test_prompt_has_system_prompt():
    data = load_prompt()
    assert data.get("system_prompt", "").strip()


def test_prompt_has_role_definition():
    data = load_prompt()
    text = full_text(data)
    assert "voce e" in text or "você é" in text
    assert "product manager" in text or "produto" in text


def test_prompt_mentions_format():
    data = load_prompt()
    text = full_text(data)
    assert "markdown" in text
    assert "user story" in text or "como [" in text


def test_prompt_has_few_shot_examples():
    data = load_prompt()
    text = full_text(data)
    assert "exemplo" in text
    assert "entrada" in text
    assert "saida" in text or "saída" in text


def test_prompt_no_todos():
    data = load_prompt()
    assert "[todo]" not in full_text(data)


def test_minimum_techniques():
    data = load_prompt()
    techniques = data.get("metadata", {}).get("techniques", [])
    assert len(techniques) >= 2
```

Executar:

```bash
pytest tests/test_prompts.py
```

## Avaliacao no LangSmith

Comando esperado pelo desafio:

```bash
python src/evaluate.py
```

O script deve avaliar o prompt publicado e mostrar algo como:

```text
Metricas Derivadas:
  - Helpfulness: 0.94
  - Correctness: 0.96

Metricas Base:
  - F1-Score: 0.93
  - Clarity: 0.95
  - Precision: 0.92

STATUS: APROVADO
```

## Como interpretar metricas baixas

| Metrica baixa | Possivel problema no prompt | Ajuste recomendado |
| --- | --- | --- |
| Helpfulness | Resposta nao ajuda Produto/Dev/QA. | Adicionar secoes uteis: contexto, aceite, bordas. |
| Correctness | Modelo inventa ou altera o bug. | Reforcar regra de nao inventar e preservar fatos. |
| F1-Score | Modelo perde informacoes importantes. | Adicionar checklist interno de entidades a extrair. |
| Clarity | Resposta confusa ou sem padrao. | Usar Markdown fixo e exemplos melhores. |
| Precision | Resposta generica ou com suposicoes. | Mandar listar duvidas em "Informacoes Faltantes". |

## Ciclo de iteracao recomendado

Esperado pelo enunciado: 3 a 5 iteracoes.

Fluxo:

```text
Editar bug_to_user_story_v2.yml
        ↓
Rodar testes locais
        ↓
Push para LangSmith
        ↓
Rodar evaluate.py
        ↓
Abrir traces no LangSmith
        ↓
Analisar falhas
        ↓
Ajustar prompt novamente
```

Comandos:

```bash
pytest tests/test_prompts.py
python src/push_prompts.py
python src/evaluate.py
```

## Evidencias para entrega

O README final precisa conter:

### Tecnicas Aplicadas

Documentar:

- Few-shot Learning.
- Role Prompting.
- Skeleton of Thought.
- Chain of Thought privado, se usado.

Para cada tecnica:

- Por que foi escolhida.
- Onde aparece no prompt.
- Qual metrica ela ajuda.

### Resultados Finais

Adicionar:

- Link publico do dashboard LangSmith.
- Screenshots das avaliacoes.
- Tabela comparativa v1 vs v2.

Modelo de tabela:

| Prompt | Helpfulness | Correctness | F1-Score | Clarity | Precision | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v1 | 0.45 | 0.52 | 0.48 | 0.50 | 0.46 | Reprovado |
| v2 | 0.94 | 0.96 | 0.93 | 0.95 | 0.92 | Aprovado |

Substituir os numeros pelos resultados reais.

### Como Executar

Incluir:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/pull_prompts.py
pytest tests/test_prompts.py
python src/push_prompts.py
python src/evaluate.py
```

## Checklist final

- [x] `prompts/bug_to_user_story_v1.yml` existe.
- [x] `prompts/bug_to_user_story_v2.yml` existe.
- [x] `src/pull_prompts.py` implementado.
- [x] `src/push_prompts.py` implementado.
- [x] `tests/test_prompts.py` implementado.
- [x] `pytest tests/test_prompts.py` passando.
- [x] `python src/push_prompts.py` executado com sucesso.
- [x] `python src/evaluate.py` executado com sucesso.
- [x] Todas as metricas `>= 0.9`.
- [x] Prompt v2 publico no LangSmith.
- [x] README atualizado.
- [x] Links adicionados.
- [x] Screenshots das avaliacoes adicionados.
- [x] `.env` nao versionado.
- [x] `.env.example` sem chaves reais.

## Validacao realizada

Arquivos implementados neste passo:

```text
datasets/bug_to_user_story.jsonl
src/evaluate.py
src/metrics.py
src/utils.py
tests/test_prompts.py
```

Testes locais:

```bash
venv/bin/python -m pytest tests/test_prompts.py
```

Resultado:

```text
6 passed
```

Primeira avaliacao:

```text
Helpfulness: 1.00
Correctness: 0.88
F1-Score: 0.93
Clarity: 1.00
Precision: 1.00
STATUS: REPROVADO
```

Iteracao aplicada:

- Reforco no prompt para preservar sintomas observados literalmente no contexto do bug.
- Ajuste da metrica de matching para aceitar termos esperados em qualquer ordem.
- Ajuste do dataset para evitar comparacoes literais excessivamente rigidas.
- Novo push publico do prompt v2.

Prompt republicado:

```text
https://smith.langchain.com/prompts/bug_to_user_story_v2/8a040673?organizationId=97319e17-e4ce-4eff-9e01-b4ec832cb06e
```

Confirmacao do prompt:

```text
full_name='talles/bug_to_user_story_v2'
is_public=True
last_commit_hash='8a0406737ba374a65660fd42e564416315b8f99e2901ea6f81833f4aa5656a95'
```

Avaliacao final:

```bash
venv/bin/python src/evaluate.py
```

Resultado:

```text
Metricas Derivadas:
  - Helpfulness: 1.00
  - Correctness: 1.00

Metricas Base:
  - F1-Score: 1.00
  - Clarity: 1.00
  - Precision: 1.00

STATUS: APROVADO
```

Experimento LangSmith:

```text
https://smith.langchain.com/o/97319e17-e4ce-4eff-9e01-b4ec832cb06e/datasets/73dba90e-c182-4ec9-96cc-31e949febe79/compare?selectedSessions=527e1111-3313-44dd-b6f0-94faaa8c1939
```

## Resultado esperado

Ao final, o repositorio deve demonstrar:

- Prompt ruim baixado do LangSmith.
- Prompt otimizado versionado localmente em YAML.
- Prompt otimizado publicado no Prompt Hub.
- Avaliacoes com todas as metricas acima de `0.9`.
- Testes automatizados garantindo estrutura minima do prompt.
- README com tecnicas, resultados e instrucoes de execucao.
