"""
src/evals/run_evals.py
----------------------
Evaluation harness for the NHTSA complaints agent.

Runs every case in EVAL_CASES through ask_agent(), applies the
appropriate check strategy, and writes a timestamped JSON report to
src/evals/results/.

Run from the project root:
    python -m src.evals.run_evals

Check types
-----------
exact_count  — Calls count_complaints(**ground_truth_query) to obtain the
               real row count from the database, then checks whether that
               integer appears as a substring of the agent's answer.
               Deterministic and fully automated.

llm_judge    — Evaluates response relevance and groundedness against the raw
               retrieved tool results using an independent LLM judge.

no_tool      — Passes iff tools_called is empty.  Used for out-of-scope
               questions where the agent should answer from general
               knowledge without touching the database.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

from src.agents.claude_agent import ask_agent
from src.agents.tools import count_complaints
from src.evals.eval_dataset import EVAL_CASES
from src.evals.judge import judge_answer_quality

# ── Column widths for the terminal summary table ──────────────────────────────
_W_ID      = 30   # case id column
_W_TOOL    = 10   # tool_selection_correct column
_W_CONTENT = 10   # content_check_passed column
_W_OVERALL = 10   # overall_pass column


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tick(value: bool | None) -> str:
    """Return a single printable symbol for a pass/fail/pending value."""
    if value is True:
        return "✓"
    if value is False:
        return "✗"
    return "…"   # None → pending LLM judge


# ── Core evaluation function ──────────────────────────────────────────────────

def run_single_case(case: dict) -> dict:
    """
    Run one evaluation case end-to-end and return a result dict.

    Parameters
    ----------
    case : dict
        A single entry from EVAL_CASES.

    Returns
    -------
    dict
        Evaluation result containing: id, question, answer, tools_called,
        expected_tools, tool_selection_correct, check_type,
        content_check_passed, expected_value (for exact_count cases),
        pending_llm_judge (for llm_judge cases), and overall_pass.
    """
    print(f"\n── Running: {case['id']!r} ──")
    print(f"   Question: {case['question']}")

    # ── a. Call the agent ─────────────────────────────────────────────────
    agent_result = ask_agent(case["question"])
    answer       = agent_result["answer"]
    tools_called = agent_result["tools_called"]
    tool_inputs  = agent_result["tool_inputs"]
    tool_results = agent_result.get("tool_results", [])

    print(f"   Tools called   : {tools_called}")
    print(f"   Answer preview : {answer[:120]}{'…' if len(answer) > 120 else ''}")

    # ── b. CHECK A: tool selection ────────────────────────────────────────
    # Check that every tool name in expected_tools appears somewhere in
    # tools_called (subset check), rather than requiring an exact set match.
    # Calling a tool multiple times, or calling additional reasonable tools
    # beyond the minimum expected, does not cause a false failure.
    if case["expected_tools"]:
        tool_selection_correct: bool = set(case["expected_tools"]).issubset(set(tools_called))
    else:
        # For out-of-scope questions where no tool was expected, none should be called.
        tool_selection_correct: bool = len(tools_called) == 0

    print(f"   Tool selection : {'✓ correct' if tool_selection_correct else '✗ WRONG'}"
          f"  (expected {case['expected_tools']}, got {tools_called})")

    # ── c. CHECK B: content check (strategy depends on check_type) ────────
    check_type = case["check_type"]

    expected_value    = None   # populated for exact_count cases
    judge_result      = None   # populated for llm_judge cases
    content_check_passed: bool = False

    if check_type == "exact_count":
        # Query the database directly with the ground-truth filter args to
        # get the authoritative count.  This is independent of the agent —
        # it re-runs the same query the agent's tool would have run.
        expected_value = count_complaints(**case["ground_truth_query"])

        # The agent's answer passes if it contains the expected integer
        # as a substring (e.g. "47" anywhere in the text).
        content_check_passed = str(expected_value) in answer
        print(f"   Expected count : {expected_value} → "
              f"{'found in answer ✓' if content_check_passed else 'NOT found in answer ✗'}")

    elif check_type == "no_tool":
        # For out-of-scope questions the agent must not call any tool.
        # tool_selection_correct already captures this (expected_tools == []),
        # but we mirror the check here so content_check_passed is consistent.
        content_check_passed = len(tools_called) == 0
        print(f"   No-tool check  : {'✓ passed' if content_check_passed else '✗ FAILED (tool was called)'}")

    elif check_type == "llm_judge":
        # Use an independent LLM judge to evaluate relevance and groundedness
        # against the raw retrieved tool results.
        judge_result = judge_answer_quality(
            question=case["question"],
            answer=answer,
            retrieved_context=tool_results,
        )
        content_check_passed = bool(
            judge_result.get("relevant") and judge_result.get("grounded")
        )
        print(f"   Judge relevant : {'✓' if judge_result.get('relevant') else '✗'}")
        print(f"   Judge grounded : {'✓' if judge_result.get('grounded') else '✗'}")
        print(f"   Judge reasoning: {judge_result.get('reasoning', '')}")

    # ── d. Overall pass ───────────────────────────────────────────────────
    # overall_pass is True only when BOTH checks pass.
    overall_pass = bool(tool_selection_correct and content_check_passed)

    return {
        "id":                     case["id"],
        "question":               case["question"],
        "answer":                 answer,
        "tools_called":           tools_called,
        "tool_inputs":            tool_inputs,
        "tool_results":           tool_results,
        "expected_tools":         case["expected_tools"],
        "tool_selection_correct": tool_selection_correct,
        "check_type":             check_type,
        "content_check_passed":   content_check_passed,
        "expected_value":         expected_value,
        "judge_result":           judge_result,
        "overall_pass":           overall_pass,
    }


# ── Main orchestrator ─────────────────────────────────────────────────────────

def main() -> None:
    """Run the full eval suite, print a summary, and save a JSON report."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 60)
    print(f"NHTSA Agent Eval Run  —  {timestamp}")
    print(f"Total cases: {len(EVAL_CASES)}")
    print("=" * 60)

    # ── a. Run all cases ──────────────────────────────────────────────────
    results: list[dict] = []
    for i, case in enumerate(EVAL_CASES):
        result = run_single_case(case)
        results.append(result)

        # Respect API rate limits between cases (except after the last one).
        if i < len(EVAL_CASES) - 1:
            print("   [sleeping 1 s …]")
            time.sleep(1)

    # ── b. Print summary table ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    header = (
        f"{'Case ID':<{_W_ID}}"
        f"{'Tool ✓/✗':^{_W_TOOL}}"
        f"{'Content':^{_W_CONTENT}}"
        f"{'Overall':^{_W_OVERALL}}"
    )
    print(header)
    print("─" * len(header))

    fully_passed = 0
    failed       = 0

    for r in results:
        tool_sym    = _tick(r["tool_selection_correct"])
        content_sym = _tick(r["content_check_passed"])
        overall_sym = _tick(r["overall_pass"])

        print(
            f"{r['id']:<{_W_ID}}"
            f"{tool_sym:^{_W_TOOL}}"
            f"{content_sym:^{_W_CONTENT}}"
            f"{overall_sym:^{_W_OVERALL}}"
        )

        if r["overall_pass"] is True:
            fully_passed += 1
        else:
            failed += 1

    print("─" * len(header))
    print(
        f"\n{fully_passed}/{len(EVAL_CASES)} cases passed "
        f"({failed} failed)."
    )

    # ── c. Save JSON report ───────────────────────────────────────────────
    results_dir = os.path.join(
        os.path.dirname(__file__),   # same directory as this script
        "results",
    )
    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, f"eval_run_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_timestamp": timestamp,
                "total_cases":   len(EVAL_CASES),
                "fully_passed":  fully_passed,
                "failed":        failed,
                "results":       results,
            },
            f,
            indent=2,
            default=str,   # handles any non-serialisable types gracefully
        )

    print(f"\nFull results saved to: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
