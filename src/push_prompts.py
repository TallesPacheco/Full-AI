"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)
"""

import os
import sys

from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

from utils import load_yaml, check_env_vars, print_section_header


load_dotenv()


PROMPT_PATH = "prompts/bug_to_user_story_v2.yml"
PROMPT_NAME = "bug_to_user_story_v2"


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    errors: list[str] = []

    required_fields = ["description", "system_prompt", "user_prompt", "version"]
    for field in required_fields:
        if field not in prompt_data:
            errors.append(f"Campo obrigatorio faltando: {field}")

    system_prompt = (prompt_data.get("system_prompt") or "").strip()
    user_prompt = (prompt_data.get("user_prompt") or "").strip()

    if not system_prompt:
        errors.append("system_prompt esta vazio")
    if "TODO" in system_prompt or "TODO" in user_prompt:
        errors.append("Prompt ainda contem TODOs")

    techniques = prompt_data.get("techniques_applied", [])
    if len(techniques) < 2:
        errors.append(
            f"Minimo de 2 tecnicas requeridas em techniques_applied, encontradas: {len(techniques)}"
        )

    if "{bug_report}" not in user_prompt and "{bug_report}" not in system_prompt:
        errors.append("O prompt precisa conter a variavel {bug_report}")

    return (len(errors) == 0, errors)


def build_chat_prompt(prompt_data: dict) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", prompt_data["system_prompt"]),
            ("user", prompt_data["user_prompt"]),
        ]
    )


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    username = os.getenv("USERNAME_LANGSMITH_HUB", "").strip()
    if not username:
        print("USERNAME_LANGSMITH_HUB nao configurada no .env")
        return False

    identifier = f"{username}/{prompt_name}"
    chat_prompt = build_chat_prompt(prompt_data)

    techniques = prompt_data.get("techniques_applied", [])
    tags = list(prompt_data.get("tags", []) or [])
    tags.extend(f"technique:{technique}" for technique in techniques)
    tags = list(dict.fromkeys(tag for tag in tags if tag))

    description = prompt_data.get(
        "description", "Prompt otimizado para converter bugs em user stories."
    )

    print(f"\nPublicando em: {identifier}")
    print(f"Tags: {tags}")

    try:
        url = hub.push(
            identifier,
            chat_prompt,
            new_repo_is_public=True,
            new_repo_description=description,
            tags=tags,
        )
        print(f"\nPrompt publicado com sucesso: {url}")
        return True
    except Exception as exc:
        print(f"\nFalha ao publicar o prompt: {exc}")
        return False


def main() -> int:
    print_section_header("PUSH DE PROMPTS OTIMIZADOS PARA O LANGSMITH HUB")

    if not check_env_vars(["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB"]):
        return 1

    raw = load_yaml(PROMPT_PATH)
    if not raw:
        return 1

    prompt_data = raw.get(PROMPT_NAME, raw)

    is_valid, errors = validate_prompt(prompt_data)
    if not is_valid:
        print("\nPrompt invalido. Corrija os erros abaixo:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("\nPrompt validado com sucesso.")
    techniques = prompt_data.get("techniques_applied", [])
    print(f"Tecnicas aplicadas: {', '.join(techniques)}")

    return 0 if push_prompt_to_langsmith(PROMPT_NAME, prompt_data) else 1


if __name__ == "__main__":
    sys.exit(main())
