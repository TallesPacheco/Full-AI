"""
Testes automatizados para validacao de prompts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure  # noqa: E402


PROMPT_PATH = Path("prompts/bug_to_user_story_v2.yml")
PROMPT_NAME = "bug_to_user_story_v2"


def load_prompts(file_path: Union[str, Path]) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_prompt_data() -> dict:
    raw = load_prompts(PROMPT_PATH)
    return raw.get(PROMPT_NAME, raw)


def full_text(prompt_data: dict) -> str:
    return "\n".join(
        [
            prompt_data.get("system_prompt", "") or "",
            prompt_data.get("user_prompt", "") or "",
        ]
    ).lower()


class TestPrompts:
    def test_prompt_has_system_prompt(self):
        data = get_prompt_data()
        system_prompt = (data.get("system_prompt") or "").strip()
        assert system_prompt, "system_prompt nao pode estar vazio"

    def test_prompt_has_role_definition(self):
        data = get_prompt_data()
        text = full_text(data)
        assert "voce e" in text or "você é" in text, "Persona nao definida (esperado 'Voce e ...')"
        assert (
            "product manager" in text or "produto" in text or "agile coach" in text
        ), "Persona deve mencionar Product Manager / Produto / Agile Coach"

    def test_prompt_mentions_format(self):
        data = get_prompt_data()
        text = full_text(data)
        assert "markdown" in text, "Prompt deve exigir formato Markdown"
        assert "user story" in text or "como [" in text, "Prompt deve mencionar User Story"

    def test_prompt_has_few_shot_examples(self):
        data = get_prompt_data()
        text = full_text(data)
        assert "exemplo" in text, "Prompt deve conter exemplos"
        assert "entrada" in text, "Exemplos devem incluir 'Entrada'"
        assert "saida" in text or "saída" in text, "Exemplos devem incluir 'Saida'"

    def test_prompt_no_todos(self):
        data = get_prompt_data()
        text = full_text(data)
        assert "[todo]" not in text, "Prompt nao deve conter [TODO]"
        assert "todo:" not in text, "Prompt nao deve conter 'TODO:'"

    def test_minimum_techniques(self):
        data = get_prompt_data()
        techniques = data.get("techniques_applied", [])
        assert isinstance(techniques, list), "techniques_applied deve ser uma lista"
        assert len(techniques) >= 2, (
            f"Minimo de 2 tecnicas em techniques_applied, encontradas: {len(techniques)}"
        )

    def test_validate_prompt_structure(self):
        data = get_prompt_data()
        is_valid, errors = validate_prompt_structure(
            {
                **data,
                "techniques_applied": data.get("techniques_applied", []),
            }
        )
        assert is_valid, f"Estrutura invalida: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
