"""
src/evals/eval_dataset.py
--------------------------
Ground-truth evaluation dataset for the NHTSA complaints agent.

Each entry in EVAL_CASES describes one test scenario:

  - "id"                 Unique identifier used to reference the case in reports.
  - "question"           The natural-language question sent verbatim to ask_agent().
  - "expected_tools"     Tool name(s) the agent should call.  Empty list means the
                         agent should answer from general knowledge with no DB call.
  - "check_type"         How the result is evaluated:
                           "exact_count"  — run ground_truth_query against the DB
                                           and assert the agent's answer contains
                                           that integer.
                           "llm_judge"    — pass the answer to a second LLM call
                                           to score relevance/correctness.
                           "no_tool"      — assert tools_called is empty (tool
                                           routing check only, answer not graded).
  - "ground_truth_query" Dict of filter kwargs for count_complaints() used to
                         compute the authoritative count for "exact_count" cases.
                         None for all other check types.
"""

from __future__ import annotations

EVAL_CASES: list[dict] = [
    # ── Exact-count cases (structured filtering) ──────────────────────────────
    {
        "id": "count_crash_civic",
        "question": "How many complaints involved a crash for the Honda Civic?",
        "expected_tools": ["count_complaints"],
        "check_type": "exact_count",
        "ground_truth_query": {"make": "Honda", "model": "Civic", "crash": True},
    },
    {
        "id": "count_fire_tesla",
        "question": "How many fire complaints were filed for the Tesla Model 3?",
        "expected_tools": ["count_complaints"],
        "check_type": "exact_count",
        "ground_truth_query": {"make": "Tesla", "model": "Model 3", "fire": True},
    },
    {
        "id": "count_total_camry",
        "question": "How many total complaints exist for the Toyota Camry?",
        "expected_tools": ["count_complaints"],
        "check_type": "exact_count",
        "ground_truth_query": {"make": "Toyota", "model": "Camry"},
    },
    {
        "id": "count_fire_and_crash",
        "question": (
            "How many complaints involve both a fire and a crash "
            "for the Tesla Model 3?"
        ),
        "expected_tools": ["count_complaints"],
        "check_type": "exact_count",
        "ground_truth_query": {
            "make": "Tesla",
            "model": "Model 3",
            "fire": True,
            "crash": True,
        },
    },

    # ── Semantic-search cases (meaning-based retrieval) ───────────────────────
    {
        "id": "semantic_brakes_own",
        "question": "Find complaints about brakes engaging on their own",
        "expected_tools": ["semantic_search_complaints"],
        "check_type": "llm_judge",
        "ground_truth_query": None,
    },
    {
        "id": "semantic_tire_blowout",
        "question": "Any complaints about tires blowing out while driving?",
        "expected_tools": ["semantic_search_complaints"],
        "check_type": "llm_judge",
        "ground_truth_query": None,
    },
    {
        "id": "semantic_fsd_braking",
        "question": "Describe issues people have had with Tesla's self-driving braking",
        "expected_tools": ["semantic_search_complaints"],
        "check_type": "llm_judge",
        "ground_truth_query": None,
    },
    {
        "id": "semantic_wheels_falling_off",
        "question": "Tell me about problems people had with their wheels falling off",
        "expected_tools": ["semantic_search_complaints"],
        "check_type": "llm_judge",
        "ground_truth_query": None,
    },

    # ── No-tool case (out-of-scope question) ──────────────────────────────────
    {
        "id": "out_of_scope_question",
        "question": "What's the capital of France?",
        "expected_tools": [],
        "check_type": "no_tool",
        "ground_truth_query": None,
    },

    # ── Parallel tool-calling case ────────────────────────────────────────────
    # Calling semantic_search_complaints twice (once per topic) is a valid,
    # preferable way to answer compound descriptive questions without exact counts.
    {
        "id": "parallel_brake_and_tire",
        "question": (
            "Are there complaints about both brake issues and tire issues for Tesla?"
        ),
        "expected_tools": ["semantic_search_complaints"],
        "check_type": "llm_judge",
        "ground_truth_query": None,
    },
]
