# Passo 3: Otimizacao do Prompt em YAML

## Objetivo deste passo

Criar a versao otimizada do prompt:

```text
prompts/bug_to_user_story_v2.yml
```

Essa versao precisa transformar relatos de bugs em user stories claras, testaveis e uteis para Produto, Engenharia e QA.

## O que aprendemos no repositorio das aulas

O capitulo `1-tipos-de-prompts` mostra tecnicas que se conectam diretamente ao desafio.

### Role Prompting

Arquivo analisado:

```text
1-tipos-de-prompts/0-Role-prompting.py
```

Ideia principal: mudar o comportamento do modelo definindo uma persona no `system`.

Aplicacao no desafio:

```text
Voce e um Product Manager senior especializado em transformar bugs em user stories acionaveis.
```

Isso ajuda nas metricas:

- Helpfulness
- Clarity
- Correctness

### Few-shot Learning

Arquivo analisado:

```text
1-tipos-de-prompts/2-one-few-shot.py
```

Ideia principal: dar exemplos de entrada e saida para o modelo copiar o padrao.

Aplicacao no desafio:

- Incluir 2 ou 3 bugs de exemplo.
- Mostrar a user story esperada.
- Mostrar criterios de aceite.
- Mostrar tratamento de informacoes ausentes.

Essa tecnica e obrigatoria no enunciado.

### Chain of Thought

Arquivo analisado:

```text
1-tipos-de-prompts/3-CoT.py
```

Ideia principal: orientar o modelo a analisar antes de responder.

No desafio, devemos evitar expor raciocinio longo desnecessario. Melhor usar uma instrucao de raciocinio privado:

```text
Antes de responder, analise internamente: usuario afetado, comportamento atual, comportamento esperado, impacto, regras de negocio e criterios de aceite. Nao mostre essa analise; retorne apenas a resposta final.
```

### Skeleton of Thought

Arquivo analisado:

```text
1-tipos-de-prompts/5-SoT.py
```

Ideia principal: estruturar a resposta em secoes fixas.

Aplicacao no desafio:

```markdown
## User Story
## Contexto do Bug
## Criterios de Aceite
## Casos de Borda
## Informacoes Faltantes
```

Isso ajuda muito em:

- Clarity
- Precision
- F1-Score

## O que aprendemos sobre YAML nas aulas

No capitulo `5-gerenciamento-e-versionamento-de-prompts`, os prompts seguem um padrao YAML com:

```yaml
_type: prompt
id: agent-pull-request-creator
version: 1.0.0
input_variables:
  - changes_summary
template: |
  ...
```

Para o nosso desafio, podemos adaptar para um formato mais rico, com `system_prompt`, `user_prompt` e `metadata`, desde que os scripts saibam ler esse formato.

## Requisitos obrigatorios do prompt v2

O arquivo `prompts/bug_to_user_story_v2.yml` deve conter:

- Instrucoes claras e especificas.
- Persona bem definida.
- Regras explicitas de comportamento.
- Separacao adequada entre System Prompt e User Prompt.
- Few-shot Learning com exemplos de entrada e saida.
- Pelo menos uma tecnica adicional alem de Few-shot.
- Tratamento de edge cases.
- Metadados com tecnicas usadas.

## Estrutura sugerida do YAML

```yaml
name: bug_to_user_story_v2
description: Prompt otimizado para converter bugs em user stories claras, testaveis e acionaveis.
metadata:
  version: v2
  challenge: desafio2
  techniques:
    - Few-shot Learning
    - Role Prompting
    - Skeleton of Thought
    - Private Chain of Thought
  target_metrics:
    helpfulness: 0.9
    correctness: 0.9
    f1_score: 0.9
    clarity: 0.9
    precision: 0.9
input_variables:
  - bug
system_prompt: |
  Voce e um Product Manager senior especializado em transformar relatos de bugs em user stories acionaveis.
  Sua resposta sera usada por times de Produto, Engenharia e QA.

  Regras:
  - Nao invente informacoes ausentes.
  - Preserve detalhes tecnicos relevantes do bug original.
  - Escreva em portugues claro e objetivo.
  - Use Markdown.
  - Gere criterios de aceite testaveis.
  - Quando houver ambiguidade, registre em "Informacoes Faltantes".
user_prompt: |
  Antes de responder, analise internamente:
  - usuario ou persona afetada
  - comportamento atual
  - comportamento esperado
  - impacto do bug
  - criterios de aceite verificaveis
  - casos de borda

  Nao mostre o raciocinio interno.

  Exemplos:

  Entrada:
  O botao "Salvar" nao responde ao clique quando o formulario tem acento no campo nome.

  Saida:
  ## User Story
  Como usuario que preenche o formulario de cadastro,
  quero conseguir salvar dados com acentos no nome,
  para concluir meu cadastro sem precisar alterar minha informacao pessoal.

  ## Contexto do Bug
  - O botao "Salvar" nao responde quando o campo nome contem acentos.

  ## Criterios de Aceite
  - Dado um formulario com nome contendo acento, quando o usuario clicar em "Salvar", entao o cadastro deve ser enviado com sucesso.
  - Dado um nome sem acento, quando o usuario clicar em "Salvar", entao o comportamento atual deve continuar funcionando.

  ## Casos de Borda
  - Nomes com cedilha.
  - Nomes compostos.
  - Nomes com apostrofo ou hifen.

  ## Informacoes Faltantes
  - Navegador, sistema operacional e mensagem de erro nao foram informados.

  Agora converta o bug abaixo:

  {bug}
```

## Estrutura de resposta esperada

O prompt deve forcar sempre o mesmo formato:

```markdown
## User Story
Como [persona],
quero [necessidade],
para [beneficio].

## Contexto do Bug
- ...

## Criterios de Aceite
- Dado ...
  Quando ...
  Entao ...

## Casos de Borda
- ...

## Informacoes Faltantes
- ...
```

## Como otimizar para cada metrica

| Metrica | Como o prompt deve ajudar |
| --- | --- |
| Helpfulness | Gerar informacoes uteis para execucao: historia, contexto, aceite, bordas. |
| Correctness | Preservar o bug original e nao inventar dados. |
| F1-Score | Capturar entidades importantes: tela, acao, erro, usuario, condicao. |
| Clarity | Usar secoes fixas, frases curtas e Markdown. |
| Precision | Evitar conteudo generico e separar informacoes faltantes. |

## Edge cases que o prompt deve tratar

Adicionar regras para:

- Bug muito curto.
- Bug sem usuario afetado.
- Bug sem comportamento esperado.
- Bug com linguagem tecnica demais.
- Bug com multiplos sintomas.
- Bug com logs ou stack trace.
- Bug com informacoes insuficientes.

Exemplo de regra:

```text
Se o bug nao informar comportamento esperado, deduza apenas o minimo necessario a partir do contexto e liste a duvida em "Informacoes Faltantes".
```

## Checklist deste passo

- [x] Criar `prompts/bug_to_user_story_v2.yml`.
- [x] Definir persona no `system_prompt`.
- [x] Incluir Few-shot Learning.
- [x] Incluir Skeleton of Thought ou outra tecnica adicional.
- [x] Incluir regras contra alucinacao.
- [x] Exigir Markdown.
- [x] Exigir user story padrao.
- [x] Exigir criterios de aceite testaveis.
- [x] Tratar edge cases.
- [x] Preencher `metadata.techniques` com pelo menos duas tecnicas.

## Validacao realizada

Arquivo criado:

```text
prompts/bug_to_user_story_v2.yml
```

O prompt v2 foi estruturado com:

- `system_prompt` com persona de Product Manager senior e Agile Coach.
- `user_prompt` separado, com formato obrigatorio de saida em Markdown.
- Tres exemplos few-shot cobrindo formulario, checkout mobile e API com log tecnico.
- Regras explicitas contra alucinacao e preservacao de detalhes tecnicos.
- Tratamento para bugs curtos, ambiguos, tecnicos, com multiplos sintomas ou informacoes insuficientes.
- Metadados com tecnicas de Prompt Engineering e metas de avaliacao.

Validacao local executada:

```bash
venv/bin/python -c "import yaml; data=yaml.safe_load(open('prompts/bug_to_user_story_v2.yml', encoding='utf-8')); print(data['name']); print(data['input_variables'])"
```

Resultado validado:

```text
bug_to_user_story_v2
['bug']
```

## Proxima etapa

Com o prompt v2 criado, vamos publicar no LangSmith Prompt Hub:

```text
{seu_username}/bug_to_user_story_v2
```

Esse sera o foco do arquivo:

```text
desafio2/04-push-versionamento-langsmith.md
```
