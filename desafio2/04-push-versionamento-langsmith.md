# Passo 4: Push e Versionamento no LangSmith Prompt Hub

## Objetivo deste passo

Implementar o script que publica o prompt otimizado no LangSmith Prompt Hub.

Entrada:

```text
prompts/bug_to_user_story_v2.yml
```

Saida esperada no LangSmith:

```text
{seu_username}/bug_to_user_story_v2
```

## O que aprendemos no repositorio das aulas

No capitulo `5-gerenciamento-e-versionamento-de-prompts`, o arquivo:

```text
5-gerenciamento-e-versionamento-de-prompts/src/langsmith_push.py
```

mostra o fluxo de push:

```python
from dotenv import load_dotenv
from langchain_core.prompts.loading import load_prompt
from langsmith import Client

load_dotenv()

prompt_template = load_prompt(prompt.path)

client = Client()
url = client.push_prompt(
    "agent-pull-request-creator",
    object=prompt_template,
    tags=[
        f"v{prompt.version}",
        f"model: {prompt.model}",
    ],
    description=prompt.description,
)
print(url)
```

Os aprendizados principais:

1. O prompt local precisa virar um objeto LangChain.
2. O `Client().push_prompt()` publica no LangSmith.
3. Tags e descricao ajudam no versionamento.
4. O push deve ser reproduzivel via script.

No capitulo `7-evaluation/3-pairwise`, o arquivo `create_prompts.py` mostra uma variacao importante: carregar YAML com `messages` e converter para `ChatPromptTemplate`.

```python
messages = [(msg["role"], msg["content"]) for msg in config["messages"]]
return ChatPromptTemplate.from_messages(messages)
```

Para nosso desafio, essa abordagem combina bem com `system_prompt` e `user_prompt`.

## Arquivo a implementar

```text
src/push_prompts.py
```

## Responsabilidades do script

`src/push_prompts.py` deve:

- Carregar `.env`.
- Validar `LANGCHAIN_API_KEY`.
- Validar `LANGSMITH_USERNAME`.
- Ler `prompts/bug_to_user_story_v2.yml`.
- Criar um `ChatPromptTemplate`.
- Fazer push para `{LANGSMITH_USERNAME}/bug_to_user_story_v2`.
- Enviar descricao e tags.
- Imprimir URL ou identificador publicado.

## Formato esperado do prompt v2

O script pode assumir este formato:

```yaml
name: bug_to_user_story_v2
description: Prompt otimizado para converter bugs em user stories.
metadata:
  version: v2
  techniques:
    - Few-shot Learning
    - Role Prompting
    - Skeleton of Thought
input_variables:
  - bug
system_prompt: |
  ...
user_prompt: |
  ...
```

## Conversao para ChatPromptTemplate

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", data["system_prompt"]),
    ("user", data["user_prompt"]),
])
```

Essa estrutura e melhor que prompt unico porque:

- O system prompt fixa persona e regras.
- O user prompt recebe o bug e os exemplos.
- O LangSmith preserva melhor a intencao do prompt.

## Esqueleto sugerido

```python
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client


PROMPT_PATH = Path("prompts/bug_to_user_story_v2.yml")


def load_prompt_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Prompt nao encontrado: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_chat_prompt(data: dict) -> ChatPromptTemplate:
    system_prompt = data.get("system_prompt")
    user_prompt = data.get("user_prompt")

    if not system_prompt or not user_prompt:
        raise ValueError("YAML precisa conter system_prompt e user_prompt")

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])


def main():
    load_dotenv()

    username = os.getenv("LANGSMITH_USERNAME")
    if not username:
        raise RuntimeError("Configure LANGSMITH_USERNAME no .env")

    data = load_prompt_yaml(PROMPT_PATH)
    prompt_obj = build_chat_prompt(data)

    metadata = data.get("metadata", {})
    version = metadata.get("version", "v2")
    techniques = metadata.get("techniques", [])

    prompt_identifier = f"{username}/bug_to_user_story_v2"

    client = Client()
    url = client.push_prompt(
        prompt_identifier=prompt_identifier,
        object=prompt_obj,
        tags=[
            version,
            "desafio2",
            "bug-to-user-story",
            *[f"technique:{technique}" for technique in techniques],
        ],
        description=data.get("description", "Prompt otimizado do desafio2"),
    )

    print(f"Prompt publicado: {prompt_identifier}")
    print(url)


if __name__ == "__main__":
    main()
```

## Como executar

```bash
python src/push_prompts.py
```

## Validacao no LangSmith

Depois do push:

1. Acessar `https://smith.langchain.com`.
2. Abrir Prompt Hub.
3. Procurar por:

```text
{seu_username}/bug_to_user_story_v2
```

4. Verificar se:

- O prompt existe.
- A versao v2 aparece.
- As tags foram aplicadas.
- A descricao foi preenchida.
- O prompt esta publico, se o desafio exigir link publico.

## Versionamento

Se houver novas iteracoes, manter o mesmo identificador e atualizar tags/metadados:

```yaml
metadata:
  version: v2.1
  iteration: 2
```

Ou manter `bug_to_user_story_v2` como nome final e documentar as iteracoes no README.

## Problemas comuns

### `Nothing to commit`

O LangSmith pode responder que nao ha mudanca se o prompt publicado for igual ao anterior.

Acao:

- Confirmar se o YAML foi alterado.
- Atualizar descricao ou tag da iteracao, se fizer sentido.

### Prompt sem variavel `{bug}`

O prompt precisa receber o bug do dataset. Conferir se `user_prompt` contem:

```text
{bug}
```

### Username errado

O nome final precisa seguir:

```text
{seu_username}/bug_to_user_story_v2
```

Por isso usamos:

```env
LANGSMITH_USERNAME=
```

## Checklist deste passo

- [x] Criar `src/push_prompts.py`.
- [x] Ler `prompts/bug_to_user_story_v2.yml`.
- [x] Converter YAML para `ChatPromptTemplate`.
- [x] Publicar no LangSmith com `Client().push_prompt()`.
- [x] Adicionar tags de versao e tecnicas.
- [x] Verificar o prompt no dashboard.
- [x] Tornar o prompt publico.
- [x] Salvar link para usar no README final.

## Validacao parcial realizada

Script criado:

```text
src/push_prompts.py
```

O script:

- Carrega variaveis com `python-dotenv`.
- Valida `LANGCHAIN_API_KEY` ou `LANGSMITH_API_KEY`.
- Usa `LANGSMITH_USERNAME` quando informado, mas tambem permite publicar apenas como `bug_to_user_story_v2`.
- Le `prompts/bug_to_user_story_v2.yml`.
- Converte `system_prompt` e `user_prompt` para `ChatPromptTemplate`.
- Valida que o `user_prompt` contem `{bug}`.
- Gera tags de versao, desafio e tecnicas usadas.
- Publica com `Client().push_prompt()`, descricao, README e visibilidade mantida por padrao.

Validacao local executada:

```bash
venv/bin/python -c "from pathlib import Path; from src.push_prompts import load_prompt_yaml, build_chat_prompt; data=load_prompt_yaml(Path('prompts/bug_to_user_story_v2.yml')); prompt=build_chat_prompt(data); print(prompt.input_variables)"
```

Resultado:

```text
['bug']
```

Observacao sobre `LANGSMITH_USERNAME`:

```text
Essa variavel e opcional no script local. Quando preenchida, o prompt e publicado como
{LANGSMITH_USERNAME}/bug_to_user_story_v2. Quando ausente, o script publica como
bug_to_user_story_v2 e o LangSmith infere o owner/workspace pela API key.
```

Para executar o push:

```bash
venv/bin/python src/push_prompts.py
```

Push executado com sucesso em modo privado:

```text
Prompt publicado: bug_to_user_story_v2
Visibilidade solicitada: privado
https://smith.langchain.com/prompts/bug_to_user_story_v2/21b70538?organizationId=97319e17-e4ce-4eff-9e01-b4ec832cb06e
```

Depois da criacao do LangChain Hub handle, o prompt foi atualizado para publico e o push foi tentado novamente:

```bash
venv/bin/python src/push_prompts.py --public
```

Resultado:

```text
Nothing to commit: prompt has not changed since latest commit
```

Esse retorno indica que o prompt remoto ja esta sincronizado com o YAML local. Confirmacao via LangSmith Client:

```text
owner='talles'
full_name='talles/bug_to_user_story_v2'
is_public=True
last_commit_hash='21b70538b345f0db2cfc9ed4920ceba7dcf87b07137e07943824dfd954247eb0'
```

Depois da iteracao do passo 05, o prompt foi atualizado e republicado:

```text
last_commit_hash='8a0406737ba374a65660fd42e564416315b8f99e2901ea6f81833f4aa5656a95'
```

Link publico:

```text
https://smith.langchain.com/prompts/talles/bug_to_user_story_v2
```

## Proxima etapa

Com o prompt publicado, vamos avaliar, testar e iterar ate passar em todas as metricas:

```text
Helpfulness, Correctness, F1-Score, Clarity, Precision >= 0.9
```

Esse sera o foco do arquivo:

```text
desafio2/05-avaliacao-testes-iteracao-entrega.md
```
