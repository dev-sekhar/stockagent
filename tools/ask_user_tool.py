"""
ask_user tool
-------------
A synchronous I/O tool that agents give to the LLM so it can pause
the tool-calling loop and ask the user a clarifying question.

The LLM calls this when it detects genuine ambiguity — e.g. multiple
distinct companies match a query, or one company is listed on several
exchanges.  The tool prints the question, collects input, and returns
the user's answer so the LLM can continue reasoning.

Design principle
----------------
Agents must NEVER hardcode Python input() menus or if-len>1 branching
for user clarification.  All interactive prompts must go through this
tool so the LLM owns the decision of *when* to ask and *what* to ask.
"""


def ask_user(question: str, options: list | None = None) -> dict:
    """
    Pause the agent loop and ask the user a clarifying question.

    Args:
        question: Human-readable question string to display.
        options:  Optional list of labelled choices.  When provided the
                  user picks by number; when omitted free-form input is
                  accepted.

    Returns:
        {"answer": <chosen_label_or_free_text>, "index": <0-based int | null>}
    """
    print()
    if options:
        print(f"  ❓ {question}\n")
        for i, opt in enumerate(options, start=1):
            print(f"    {i}.  {opt}")
        print()
        while True:
            try:
                raw = input("  Enter number: ").strip()
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    chosen = options[idx]
                    print(f"  ✅ Selected: {chosen}\n")
                    return {"answer": chosen, "index": idx}
                print(f"  ⚠️  Please enter a number between 1 and {len(options)}.")
            except ValueError:
                print("  ⚠️  Invalid input — please enter a number.")
            except EOFError:
                # Non-interactive environment: default to first option
                print(f"  ⚠️  No input available — defaulting to: {options[0]}")
                return {"answer": options[0], "index": 0}
    else:
        try:
            answer = input(f"  ❓ {question}\n  > ").strip()
            print()
            return {"answer": answer, "index": None}
        except EOFError:
            return {"answer": "", "index": None}


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": (
            "Pause and ask the user a clarifying question. "
            "Use this ONLY when you detect genuine ambiguity that cannot be "
            "resolved automatically — for example: multiple distinct companies "
            "match the query, or a single company is listed on multiple exchanges. "
            "Provide a clear, concise question and an options list when choices "
            "are known.  Do NOT call this if there is only one possible answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to display to the user.",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Labelled choices for the user to pick from by number. "
                        "Each string should be a human-readable label, e.g. "
                        "'Infosys Limited (NSE India — INFY.NS)'. "
                        "Omit this field for free-form text input."
                    ),
                },
            },
            "required": ["question"],
        },
    },
}
