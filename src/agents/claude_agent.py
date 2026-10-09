"""
src/agents/claude_agent.py
--------------------------
A minimal but complete Anthropic tool-calling agent loop.

How the loop works (big picture)
---------------------------------
  1. We send Claude a user question + a list of "tools" it may call.
  2. Claude decides whether it needs data before it can answer.
     - If YES  → its response contains a ``tool_use`` block naming the
                 tool and the arguments it wants to pass.
     - If NO   → its response is plain text; we return it immediately.
  3. When Claude requests a tool, WE execute it (not Claude – the model
     never touches your database directly).  We get the real result.
  4. We send that result back to Claude as a ``tool_result`` message.
  5. Claude reads the result and writes its final natural-language answer.
  6. We return that final answer to the caller.

Run directly for a quick smoke-test:
    python -m src.agents.claude_agent
"""

import os
import json

import anthropic
from dotenv import load_dotenv
from langfuse import observe, get_client

from src.agents.tools import count_complaints, semantic_search_complaints

# ── 1. Load secrets from .env ─────────────────────────────────────────────────

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY is not set. "
        "Copy .env.example to .env and add your key."
    )

# ── 2. Anthropic client ───────────────────────────────────────────────────────

# The client automatically picks up ANTHROPIC_API_KEY from the environment,
# but we pass it explicitly here so the missing-key error above fires first
# with a friendlier message.
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

MODEL = "claude-sonnet-5-5"

# ── 3. System prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert assistant that answers questions about
vehicle safety complaints filed with the NHTSA (National Highway Traffic
Safety Administration).

You have access to TWO tools for querying a live database of real complaints:

1. count_complaints — Use this for structured, exact filtering: when the user
   asks how many complaints exist for a specific make, model, year, or asks
   about fire or crash involvement. All filters are optional and combinable.

2. semantic_search_complaints — Use this when the user describes a symptom,
   situation, or experience rather than asking for an exact count or structured
   filter (e.g. "find complaints about sudden loss of power", "what complaints
   mention steering problems", "show me cases where the car caught fire while
   parked"). This searches by meaning, not keywords.

Choose whichever tool best fits the question. If the question can be answered
from general knowledge without querying the database, answer directly without
calling either tool.

Always explain results in plain English. For counts, interpret the number
meaningfully. For semantic results, summarise the key themes you see across
the matching complaints."""

# ── 4. Tool schema ────────────────────────────────────────────────────────────
#
# This is the JSON Schema description of count_complaints() that we send to
# Claude.  Claude reads this and decides when and how to call the function.
# The "name" here is what Claude will put in its tool_use block — we use it
# below to dispatch to the real Python function.

TOOLS: list[dict] = [
    {
        "name": "count_complaints",
        "description": (
            "Count the number of vehicle safety complaints in the NHTSA "
            "database that match the given structured filters. Use this when "
            "the user asks 'how many' complaints exist for a specific make, "
            "model, year, or asks about fire or crash involvement. "
            "All parameters are optional — omitting a parameter means "
            "'don't filter on it'. Returns an integer count."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "make": {
                    "type": "string",
                    "description": (
                        "The vehicle manufacturer/make to filter by, "
                        "e.g. 'Tesla', 'Honda', 'Toyota'. "
                        "Case-insensitive partial match."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": (
                        "The vehicle model name to filter by, "
                        "e.g. 'Model 3', 'Civic', 'Camry'. "
                        "Case-insensitive partial match."
                    ),
                },
                "model_year": {
                    "type": "integer",
                    "description": "The four-digit model year, e.g. 2021.",
                },
                "fire": {
                    "type": "boolean",
                    "description": (
                        "If true, only count complaints where a fire was "
                        "involved. If false, only count complaints with no "
                        "fire. Omit to include both."
                    ),
                },
                "crash": {
                    "type": "boolean",
                    "description": (
                        "If true, only count complaints where a crash was "
                        "involved. If false, only count complaints with no "
                        "crash. Omit to include both."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "semantic_search_complaints",
        "description": (
            "Find complaints whose text is semantically similar to a "
            "natural-language description. Use this when the user describes "
            "a symptom, experience, or situation rather than asking for an "
            "exact count or structured filter — e.g. 'find complaints about "
            "sudden loss of power', 'show me cases where the steering failed', "
            "or 'what complaints mention the battery overheating'. "
            "Returns up to `limit` matching complaint records with their "
            "summaries, ordered by relevance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A natural-language description of the kind of "
                        "complaint to find, e.g. 'engine stalled on highway' "
                        "or 'car accelerated without driver input'. "
                        "Searches by meaning, not exact keywords."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of complaints to return. "
                        "Defaults to 5 if omitted."
                    ),
                },
            },
            "required": ["query"],
        },
    },
]

# ── 5. Tool dispatcher ────────────────────────────────────────────────────────
#
# Maps a tool name (as Claude spells it in its tool_use block) to the actual
# Python callable.  Add new tools here as the agent grows.

TOOL_REGISTRY: dict[str, callable] = {
    "count_complaints":           count_complaints,
    "semantic_search_complaints": semantic_search_complaints,
}


def _execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Look up the requested tool by name, call it with the arguments Claude
    supplied, and return the result as a JSON string.

    Returning JSON (rather than a plain integer) future-proofs this function:
    tools that return dicts or lists will also serialise correctly.

    Parameters
    ----------
    tool_name  : The name Claude used in its tool_use block.
    tool_input : The argument dict Claude constructed from its input_schema.

    Returns
    -------
    str
        A JSON-encoded representation of the tool's return value.

    Raises
    ------
    ValueError
        If Claude requested a tool that isn't in TOOL_REGISTRY.
    """
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(
            f"Claude requested unknown tool '{tool_name}'. "
            f"Available tools: {list(TOOL_REGISTRY.keys())}"
        )

    fn = TOOL_REGISTRY[tool_name]
    result = fn(**tool_input)           # call the real Python function
    return json.dumps(result)           # serialise for the API message


# ── 6. Main agent function ────────────────────────────────────────────────────

@observe()
def ask_agent(user_question: str) -> str:
    """
    Send a natural-language question to Claude and return its answer.

    The function implements a single-step tool-calling loop:

        user question
            ↓
        Claude (may request a tool)
            ↓  [if tool_use block present]
        execute real Python function
            ↓
        Claude (reads result, writes final answer)
            ↓
        return final text

    If Claude answers directly without needing a tool, the intermediate
    steps are skipped and the direct response is returned immediately.

    Parameters
    ----------
    user_question : str
        A natural-language question about NHTSA complaint data.

    Returns
    -------
    str
        Claude's final natural-language answer.
    """

    # ── Step A: First API call ────────────────────────────────────────────
    # We give Claude the user's question and the list of tools it may use.
    # Claude will either answer directly or ask us to run a tool.

    print(f"[agent] Sending question to Claude: {user_question!r}")

    messages = [{"role": "user", "content": user_question}]

    first_response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
    )

    # Record input on the root @observe() span so the Langfuse dashboard
    # shows the full question + system prompt at the top level of the trace.
    # v4 API: get_client().update_current_span() — NOT langfuse_context (removed in v3+).
    get_client().update_current_span(
        input={
            "question": user_question,
            "system_prompt": SYSTEM_PROMPT,
        }
    )

    print(f"[agent] Claude stop reason: {first_response.stop_reason!r}")

    # ── Step B: Did Claude request a tool? ────────────────────────────────
    # When Claude wants to call a tool, it sets stop_reason="tool_use"
    # and includes a content block of type "tool_use" in its response.
    # If stop_reason is "end_turn", Claude answered directly — return now.

    if first_response.stop_reason != "tool_use":
        # No tool needed — extract and return the plain text response.
        direct_text = next(
            block.text
            for block in first_response.content
            if block.type == "text"
        )
        print("[agent] Claude answered directly (no tool call).")
        # Record Claude's direct answer as the output of the root span.
        get_client().update_current_span(output={"answer": direct_text})
        # Flush ensures the trace is sent before the process exits.
        get_client().flush()
        return direct_text

    # ── Step C: Collect ALL tool_use blocks ──────────────────────────────
    # Claude may request multiple tools in parallel in a single turn.
    # The old code used next(...) which grabbed only the FIRST block and
    # silently dropped the rest — causing the API to reject the follow-up
    # message because it referenced tool_use_ids that had no matching
    # tool_result.  We now collect every tool_use block.

    tool_use_blocks = [
        block for block in first_response.content if block.type == "tool_use"
    ]
    print(f"[agent] Claude requested {len(tool_use_blocks)} tool call(s).")

    # ── Step D: Execute EVERY tool, collecting all results ────────────────
    # Each tool is run independently.  A try/except around each call means
    # one failing tool doesn't abort the others — the error becomes the
    # tool_result content so Claude can still reason about partial data.

    tool_results = []
    for tb in tool_use_blocks:
        print(f"[agent] Calling: {tb.name}({tb.input})")
        try:
            result_str = _execute_tool(tb.name, tb.input)
        except Exception as exc:
            result_str = json.dumps({"error": str(exc)})
            print(f"[agent] WARNING tool '{tb.name}' raised: {exc}")
        else:
            print(f"[agent] '{tb.name}' returned: {result_str}")

        tool_results.append({
            "type": "tool_result",
            "tool_use_id": tb.id,   # must match the id in the assistant turn
            "content": result_str,
        })

    # ── Step E: Build the message history for the second API call ─────────
    # The Anthropic API requires this exact 3-turn structure:
    #
    #   Turn 1: user       → original question (str)
    #   Turn 2: assistant  → Claude's first response, serialised as plain
    #                        dicts via model_dump() so the API accepts it.
    #                        Must include ALL tool_use blocks exactly as
    #                        Claude emitted them.
    #   Turn 3: user       → ALL tool_result dicts in ONE message.
    #                        Splitting results across multiple messages is
    #                        an API error — every result for a given
    #                        assistant turn must arrive together.

    messages = [
        # Turn 1 — original question
        {"role": "user", "content": user_question},

        # Turn 2 — Claude's full first response as plain dicts.
        # model_dump() converts each Pydantic content block to a dict the
        # API serialiser can handle cleanly (no __dict__ guesswork needed).
        {
            "role": "assistant",
            "content": [block.model_dump() for block in first_response.content],
        },

        # Turn 3 — ALL tool results in ONE user message.
        {"role": "user", "content": tool_results},
    ]

    # ── Debug: inspect exactly what's being sent ──────────────────────────
    print("\n[debug] Turn 1 — user (original question):")
    print(f"  type : {type(messages[0]['content']).__name__}")
    print(f"  value: {messages[0]['content']!r}")

    print("\n[debug] Turn 2 — assistant (model_dump'd content blocks):")
    print(f"  type : {type(messages[1]['content']).__name__}")
    for i, block_dict in enumerate(messages[1]["content"]):
        print(f"  block[{i}] type={block_dict.get('type', '?')} | "
              f"{json.dumps(block_dict, default=str)}")

    print("\n[debug] Turn 3 — user (all tool_results):")
    print(f"  count: {len(tool_results)} result(s)")
    print(f"  value: {json.dumps(messages[2]['content'], indent=2, default=str)}")

    print("\n[debug] Full messages payload sent to second API call:")
    print(json.dumps(messages, indent=2, default=str))
    print("[debug] ── end of payload ──\n")

    second_response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
    )

    # ── Step F: Extract and return Claude's final answer ──────────────────
    # Claude now has the real data and writes its natural-language answer.

    final_text = next(
        block.text
        for block in second_response.content
        if block.type == "text"
    )

    print("[agent] Claude produced final answer.")
    # Record the tool-assisted final answer as the output of the root span.
    get_client().update_current_span(output={"answer": final_text})
    get_client().flush()
    return final_text


# ── 7. CLI entry-point ───────────────────────────────────────────────────────
#
# Usage:
#   python -m src.agents.claude_agent "How many complaints involved a crash for the Honda Civic?"
#
# If no question is supplied, a default test question is used so the
# script still works as a quick smoke-test with no arguments.

if __name__ == "__main__":
    import argparse

    DEFAULT_QUESTION = "car's tire got puncture while driving"

    parser = argparse.ArgumentParser(
        description="Ask the NHTSA complaints agent a natural-language question.",
    )
    parser.add_argument(
        "question",
        nargs="?",           # 0 or 1 positional args — makes it optional
        default=None,
        help="The question to ask the agent. Wrap in quotes if it contains spaces.",
    )
    args = parser.parse_args()

    if args.question:
        question = args.question
    else:
        print("No question provided, using default test question.")
        question = DEFAULT_QUESTION

    print("=" * 60)
    print(f"Question: {question}")
    print("=" * 60)
    answer = ask_agent(question)
    print("\nAnswer:")
    print(answer)
