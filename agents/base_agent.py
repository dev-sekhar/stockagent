"""
BaseAgent
---------
Shared Groq tool-calling loop.  Subclasses only need to set:
  • system_prompt  (str)
  • tools          (list[dict])   – Groq-format tool definitions
  • tool_fn_map    (dict)         – {function_name: callable}
"""
import json
from groq import Groq
from config import LLM_MODEL, TEMP_REASON, TOKENS_AGENT
from tools.access_gateway import gateway


class BaseAgent:
    model: str         = LLM_MODEL
    temperature: float = TEMP_REASON
    max_tokens: int    = TOKENS_AGENT
    system_prompt: str = "You are a helpful assistant."
    tools: list        = []
    tool_fn_map: dict  = {}

    def __init__(self, client: Groq):
        self.client = client

    def run(self, user_message: str) -> str:
        """Drive the tool-calling loop and return the final text response."""
        messages = [
            {"role": "system",  "content": self.system_prompt},
            {"role": "user",    "content": user_message},
        ]

        while True:
            gateway.check("LLM")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools or None,
                tool_choice="auto" if self.tools else None,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                return msg.content or ""

            # Append assistant turn with tool calls
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            # Execute each tool and feed results back
            for tc in msg.tool_calls:
                fn = self.tool_fn_map.get(tc.function.name)
                args = json.loads(tc.function.arguments)
                result = fn(**args) if fn else {"error": f"Unknown tool: {tc.function.name}"}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })
