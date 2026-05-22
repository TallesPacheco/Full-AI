"""
Script complementar (nao oficial) que roda a avaliacao do prompt v2 usando
`langsmith.evaluation.evaluate()` para gerar um EXPERIMENTO FORMAL no LangSmith
com as 5 metricas (Helpfulness, Correctness, F1-Score, Clarity, Precision)
persistidas como feedback - visiveis no dashboard como barras coloridas.

Este script:
- Importa as funcoes oficiais de metrics.py (sem modificar)
- Reusa o LLM configurado em utils.get_llm()
- Cria um experiment_prefix proprio para nao colidir com o evaluate.py oficial

Uso:
    venv/bin/python src/log_to_langsmith.py
"""

import os
import sys

from dotenv import load_dotenv
from langchain import hub
from langsmith.evaluation import evaluate

from metrics import evaluate_clarity, evaluate_f1_score, evaluate_precision
from utils import check_env_vars, get_llm as get_configured_llm, print_section_header


load_dotenv()


DATASET_NAME = "prompt-optimization-challenge-eval"
EXPERIMENT_PREFIX = "bug-to-user-story-v2-formal"


def get_llm():
    return get_configured_llm(temperature=0)


def build_predict(prompt_identifier: str):
    prompt = hub.pull(prompt_identifier)
    llm = get_llm()
    chain = prompt | llm

    def predict(inputs: dict) -> dict:
        response = chain.invoke(inputs)
        return {"answer": response.content}

    return predict


def all_metrics_evaluator(run, example) -> dict:
    answer = (run.outputs or {}).get("answer", "")
    reference = (example.outputs or {}).get("reference", "")
    bug = (example.inputs or {}).get("bug_report", "")

    f1 = float(evaluate_f1_score(bug, answer, reference).get("score", 0.0))
    clarity = float(evaluate_clarity(bug, answer, reference).get("score", 0.0))
    precision = float(evaluate_precision(bug, answer, reference).get("score", 0.0))

    helpfulness = round((clarity + precision) / 2, 4)
    correctness = round((f1 + precision) / 2, 4)

    return {
        "results": [
            {"key": "f1_score", "score": f1},
            {"key": "clarity", "score": clarity},
            {"key": "precision", "score": precision},
            {"key": "helpfulness", "score": helpfulness},
            {"key": "correctness", "score": correctness},
        ]
    }


def main() -> int:
    print_section_header("AVALIACAO FORMAL COM FEEDBACK NO LANGSMITH")

    required = ["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB", "LLM_PROVIDER"]
    if not check_env_vars(required):
        return 1

    username = os.getenv("USERNAME_LANGSMITH_HUB", "").strip()
    prompt_identifier = f"{username}/bug_to_user_story_v2"
    print(f"Prompt:  {prompt_identifier}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Prefix:  {EXPERIMENT_PREFIX}\n")

    predict = build_predict(prompt_identifier)

    results = evaluate(
        predict,
        data=DATASET_NAME,
        evaluators=[all_metrics_evaluator],
        experiment_prefix=EXPERIMENT_PREFIX,
        description="Avaliacao formal do prompt v2 com 5 metricas (LLM-as-judge) registradas como feedback.",
        metadata={
            "prompt": prompt_identifier,
            "llm_model": os.getenv("LLM_MODEL"),
            "eval_model": os.getenv("EVAL_MODEL"),
        },
    )

    experiment_name = getattr(results, "experiment_name", "(verifique no dashboard)")
    print(f"\nExperimento criado: {experiment_name}")
    print("Abra o LangSmith Hub para ver as barras de feedback de cada metrica.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
