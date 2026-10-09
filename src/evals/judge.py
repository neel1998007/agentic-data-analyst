"""
src/evals/judge.py
------------------
LLM judge implementation for evaluating agent response quality.

Grades an agent's answer against the original question and the raw
retrieved context on two criteria:
  1. "relevant": Does the answer directly address what was asked?
  2. "grounded": Is every factual claim supported by the retrieved context,
                 with no hallucinations or invented details?
"""

from __future__ import annotations

import json
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

JUDGE_SYSTEM_PROMPT = """You are a strict, independent evaluator of another AI system's answer.
Your job is to critically grade the answer against the user's question and the raw retrieved data.
Do NOT be lenient or assume good faith — actively look for problems, unsupported claims, exaggerations,
or missing information.

Evaluate exactly two boolean criteria:
1. "relevant": Does the answer directly address what was asked in the question? (true/false)
2. "grounded": Is every factual claim, number, or detail in the answer actually supported by the
   provided retrieved_context, with NO invented facts, numbers, or details not present in the context? (true/false)

You must respond with ONLY valid JSON, with no markdown fences, no surrounding commentary, and no extra text.
The JSON must follow this exact format:
{"relevant": true, "grounded": true, "reasoning": "one sentence explanation"}"""


def clean_json_response(raw_text: str) -> str:
    """
    Strip leading/trailing markdown code fences and whitespace from a JSON response string.

    1. Strip leading/trailing whitespace from the raw text
    2. If it starts with "```json", remove that prefix
    3. If it starts with "```" (without "json"), remove that prefix too
    4. If it ends with "```", remove that suffix
    5. Strip whitespace again after removing fences
    """
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[len("```"):]

    if text.endswith("```"):
        text = text[:-len("```")]

    return text.strip()


def judge_answer_quality(
    question: str,
    answer: str,
    retrieved_context: list,
) -> dict:
    """
    Evaluate the relevance and groundedness of an agent answer using an LLM judge.

    Parameters
    ----------
    question : str
        The original user question.
    answer : str
        The agent's generated answer to evaluate.
    retrieved_context : list
        The raw records or data retrieved by the agent's tools.

    Returns
    -------
    dict
        {"relevant": bool, "grounded": bool, "reasoning": str, ...}
    """
    # ── 1. Load API key ───────────────────────────────────────────────────────
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )

    # ── 2. Create fresh Anthropic client instance ─────────────────────────────
    client = anthropic.Anthropic(api_key=anthropic_api_key)

    # ── 3. Build user payload ─────────────────────────────────────────────────
    context_str = json.dumps(retrieved_context, indent=2, default=str)
    user_prompt = f"""Evaluate the following AI system response.

[Original Question]
{question}

[Retrieved Context (Raw Data)]
{context_str}

[Answer to Evaluate]
{answer}

Instructions:
Evaluate exactly two things:
a. "relevant": does the answer directly address what was asked? (true/false)
b. "grounded": is every factual claim in the answer actually supported by the retrieved_context, with no invented facts, numbers, or details not present in the context? (true/false)

Respond with ONLY valid JSON, no other text, in exactly this format:
{{"relevant": true, "grounded": true, "reasoning": "one sentence explanation"}}"""

    # ── 4. Call Anthropic messages API ────────────────────────────────────────
    # max_tokens set to 1024 to give sufficient budget for response generation.
    # Note: Anthropic SDK 1.5.0 / claude-sonnet-5-5 deprecates the temperature
    # parameter, using deterministic output by default for JSON instructions.
    # "thinking" is explicitly omitted so extended reasoning mode is not invoked.
    create_kwargs = {
        "model": MODEL,
        "max_tokens": 1024,
        "system": JUDGE_SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
    }
    create_kwargs.pop("thinking", None)

    response = client.messages.create(**create_kwargs)

    # ── 5. Parse response text as JSON with safe fallback ────────────────────
    # Loop through response.content to find the first text block, skipping
    # any ThinkingBlock or non-text blocks that may appear before it.
    raw_text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            raw_text = block.text
            break

    print(f"[judge debug] Raw response: {repr(raw_text)}")

    cleaned = clean_json_response(raw_text)

    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict) or "relevant" not in parsed or "grounded" not in parsed:
            raise ValueError("Missing required keys in judge response")

        return {
            "relevant": bool(parsed.get("relevant", False)),
            "grounded": bool(parsed.get("grounded", False)),
            "reasoning": str(parsed.get("reasoning", "")),
        }
    except Exception:
        return {
            "relevant": False,
            "grounded": False,
            "reasoning": "Judge response could not be parsed",
            "parse_error": True,
        }


if __name__ == "__main__":
    test_question = "What issues are reported with Tesla Model 3 steering?"
    test_answer = (
        "Drivers reported power steering assist failure making the steering very stiff, "
        "as well as loud grinding noises while turning at low speeds."
    )
    test_context = [
        {
            "odi_number": "1140001",
            "make": "Tesla",
            "model": "Model 3",
            "summary": "Power steering assist failure while driving. Steering became very stiff.",
        },
        {
            "odi_number": "1140002",
            "make": "Tesla",
            "model": "Model 3",
            "summary": "Loud grinding noise when turning the steering wheel at low speeds.",
        },
    ]

    print("Running judge_answer_quality standalone test...")
    eval_result = judge_answer_quality(test_question, test_answer, test_context)
    print("Judge output:")
    print(json.dumps(eval_result, indent=2))
