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


class BaseAgent:
    model: str = "llama-3.3-70b-versatile"
    system_prompt: str = "You are a helpful assistant."
    tools: list = []
    tool_fn_map: dict = {}

    def __init__(self, client: Groq):
        self.client = client

    def run(self, user_message: str) -> str:
        """Drive the tool-calling loop and return the final text response."""
        messages = [
            {"role": "system",  "content": self.system_prompt},
            {"role": "user",    "content": user_message},
        ]

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools or None,
                tool_choice="auto" if self.tools else None,
                max_tokens=4096,
                temperature=0.1,
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
