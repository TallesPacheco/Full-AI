"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull dos prompts do Hub
3. Salva localmente em prompts/bug_to_user_story_v1.yml
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from utils import save_yaml, check_env_vars, print_section_header


load_dotenv()


PROMPT_ID = "leonanluppi/bug_to_user_story_v1"
OUTPUT_PATH = Path("prompts/bug_to_user_story_v1.yml")
PROMPT_NAME = "bug_to_user_story_v1"


def serialize_prompt(prompt) -> dict:
    body: dict = {
        "description": "Prompt baixado do LangSmith Prompt Hub (versao baseline de baixa qualidade).",
    }

    if isinstance(prompt, ChatPromptTemplate):
        messages = []
        for message_template in prompt.messages:
            role = getattr(message_template, "role", None) or message_template.__class__.__name__
            template = getattr(message_template, "prompt", None)
            content = getattr(template, "template", None) if template else None
            if content is None:
                content = str(message_template)
            messages.append({"role": role, "content": content})

            normalized_role = role.lower()
            if "system" in normalized_role and "system_prompt" not in body:
                body["system_prompt"] = content
            elif (
                "human" in normalized_role or "user" in normalized_role
            ) and "user_prompt" not in body:
                body["user_prompt"] = content

        body["messages"] = messages
        body["input_variables"] = list(getattr(prompt, "input_variables", []))
    elif isinstance(prompt, PromptTemplate):
        body["system_prompt"] = prompt.template
        body["user_prompt"] = prompt.template
        body["input_variables"] = list(prompt.input_variables)
    else:
        body["raw"] = str(prompt)

    body["version"] = "v1"
    body["source"] = PROMPT_ID
    body["tags"] = ["bug-analysis", "user-story", "baseline"]

    return {PROMPT_NAME: body}


def pull_prompts_from_langsmith() -> bool:
    print_section_header("PULL DE PROMPTS DO LANGSMITH HUB")
    print(f"Prompt: {PROMPT_ID}")

    try:
        prompt = hub.pull(PROMPT_ID)
    except Exception as exc:
        print(f"\nFalha ao baixar o prompt '{PROMPT_ID}': {exc}")
        return False

    data = serialize_prompt(prompt)
    if not save_yaml(data, str(OUTPUT_PATH)):
        return False

    print(f"\nPrompt salvo em: {OUTPUT_PATH}")
    return True


def main() -> int:
    if not check_env_vars(["LANGSMITH_API_KEY"]):
        return 1

    return 0 if pull_prompts_from_langsmith() else 1


if __name__ == "__main__":
    sys.exit(main())
