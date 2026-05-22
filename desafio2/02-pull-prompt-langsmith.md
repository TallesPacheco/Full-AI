# Passo 2: Pull do Prompt Ruim no LangSmith Prompt Hub

## Objetivo deste passo

Implementar o script que baixa o prompt inicial de baixa qualidade do LangSmith Prompt Hub e salva uma copia local em YAML.

Prompt exigido pelo desafio:

```text
leonanluppi/bug_to_user_story_v1
```

Arquivo local esperado:

```text
prompts/bug_to_user_story_v1.yml
```

## O que aprendemos no repositorio das aulas

No capitulo `5-gerenciamento-e-versionamento-de-prompts`, o arquivo:

```text
5-gerenciamento-e-versionamento-de-prompts/src/langsmith_client.py
```

mostra o uso direto do LangSmith Client:

```python
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()
prompt = client.pull_prompt("agent-pull-request-creator:dev")
```

Esse exemplo ensina tres coisas importantes:

1. Carregar `.env` antes de instanciar o client.
2. Usar `Client()` para falar com LangSmith.
3. Usar `pull_prompt()` para buscar um prompt remoto.

O enunciado tambem recomenda:

```python
from langchain import hub
```

Entao podemos usar uma das duas abordagens:

- `hub.pull("leonanluppi/bug_to_user_story_v1")`
- `Client().pull_prompt("leonanluppi/bug_to_user_story_v1")`

Para o desafio, a forma com `hub.pull` fica mais alinhada ao texto oficial.

## Estrutura local necessaria

Antes de implementar o script:

```bash
mkdir -p prompts
```

Arquivo a criar:

```text
src/pull_prompts.py
```

## Responsabilidades do script

`src/pull_prompts.py` deve:

- Carregar variaveis de ambiente com `load_dotenv()`.
- Validar se `LANGCHAIN_API_KEY` existe.
- Fazer pull de `leonanluppi/bug_to_user_story_v1`.
- Converter o prompt para um formato serializavel.
- Salvar em `prompts/bug_to_user_story_v1.yml`.
- Imprimir uma mensagem clara de sucesso.

## Formato YAML recomendado

O prompt baixado pode vir como `PromptTemplate` ou `ChatPromptTemplate`. Para facilitar a etapa seguinte, devemos salvar de forma legivel:

```yaml
name: bug_to_user_story_v1
source: leonanluppi/bug_to_user_story_v1
type: langsmith_prompt
metadata:
  pulled_from: LangSmith Prompt Hub
  challenge: desafio2
content:
  # campos extraidos do prompt remoto
```

Se o objeto tiver mensagens, salvar algo assim:

```yaml
messages:
  - role: system
    content: "..."
  - role: user
    content: "..."
```

Se o objeto tiver apenas template de texto, salvar:

```yaml
template: |
  ...
input_variables:
  - bug
```

## Esqueleto sugerido

```python
from pathlib import Path
import yaml
from dotenv import load_dotenv
from langchain import hub


PROMPT_ID = "leonanluppi/bug_to_user_story_v1"
OUTPUT_PATH = Path("prompts/bug_to_user_story_v1.yml")


def serialize_prompt(prompt):
    data = {
        "name": "bug_to_user_story_v1",
        "source": PROMPT_ID,
        "metadata": {
            "challenge": "desafio2",
            "quality": "baseline_low_quality",
        },
    }

    if hasattr(prompt, "messages"):
        data["messages"] = [
            {
                "role": getattr(message, "prompt", message).__class__.__name__,
                "content": str(message),
            }
            for message in prompt.messages
        ]
    elif hasattr(prompt, "template"):
        data["template"] = prompt.template
        data["input_variables"] = list(getattr(prompt, "input_variables", []))
    else:
        data["raw"] = str(prompt)

    return data


def main():
    load_dotenv()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    prompt = hub.pull(PROMPT_ID)
    data = serialize_prompt(prompt)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)

    print(f"Prompt salvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

Esse esqueleto pode precisar de ajuste depois de vermos o formato real retornado pelo LangSmith.

## Como executar

```bash
python src/pull_prompts.py
```

Depois conferir:

```bash
ls prompts
```

Resultado esperado:

```text
bug_to_user_story_v1.yml
```

## Validacao manual

Abrir o arquivo gerado e verificar:

- O prompt foi salvo.
- O YAML esta valido.
- Existe conteudo suficiente para entender a versao ruim.
- As variaveis de entrada ficaram claras.

## Problemas comuns

### Erro de autenticacao

Verificar:

```env
LANGCHAIN_API_KEY=
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### Prompt nao encontrado

Confirmar o identificador:

```text
leonanluppi/bug_to_user_story_v1
```

### YAML ilegivel

Se o objeto remoto for complexo, salvar tambem uma versao `raw` para diagnostico:

```yaml
raw: |
  ...
```

## Checklist deste passo

- [x] Criar `prompts/`.
- [x] Criar `src/pull_prompts.py`.
- [x] Implementar pull com LangChain Hub ou LangSmith Client.
- [x] Salvar `prompts/bug_to_user_story_v1.yml`.
- [x] Conferir YAML manualmente.
- [x] Registrar no README que o prompt v1 foi baixado do Prompt Hub.

## Validacao realizada

Script criado:

```text
src/pull_prompts.py
```

O script usa `langchain.hub.pull`, carrega variaveis com `python-dotenv`, valida a existencia de `LANGCHAIN_API_KEY` ou `LANGSMITH_API_KEY`, serializa `PromptTemplate` e `ChatPromptTemplate` para YAML e salva o resultado em:

```text
prompts/bug_to_user_story_v1.yml
```

Execucao realizada no projeto local:

```bash
venv/bin/python src/pull_prompts.py
```

Resultado:

```text
Prompt 'bug_to_user_story_v1' salvo em prompts/bug_to_user_story_v1.yml
```

Tambem foi validado que o YAML gerado pode ser carregado com `yaml.safe_load`.

## Proxima etapa

Com o prompt ruim salvo localmente, vamos criar o prompt otimizado:

```text
prompts/bug_to_user_story_v2.yml
```

Esse sera o foco do arquivo:

```text
desafio2/03-otimizacao-prompt-yaml.md
```
