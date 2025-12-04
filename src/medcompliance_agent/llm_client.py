# src/medcompliance_agent/llm_client.py

from typing import List, Dict, Any
import subprocess
import json


def call_llama(messages: List[Dict[str, str]], model: str = "llama3.1"):
    """
    Wrapper for Ollama using the 'ollama run' command.
    Compatible with older Ollama versions that do NOT support 'ollama chat'.
    """

    try:
        # Convert messages to a single prompt
        # (simple, but works well for checklist generation)
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"

        # Run the model
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if result.returncode != 0:
            print("Ollama error:", result.stderr)
            return None

        return result.stdout.strip()

    except Exception as e:
        print(f"LLM error: {e}")
        return None
