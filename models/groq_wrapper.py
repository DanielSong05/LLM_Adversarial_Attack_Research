import os
import time
import json
import requests
from dotenv import load_dotenv, find_dotenv

class GroqCloudModel:
    """
    Wrapper for Groq API-hosted open-source LLMs.
    Handles prompt building and response parsing.
    """

    MAX_RETRIES = 3
    RETRY_BACKOFF = 2  # seconds multiplier

    def __init__(self, model_name="groq/mistral-saba-24b", system_msg=None, max_tokens=1024):
        load_dotenv(find_dotenv())
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not found in environment.")

        self.model_name = model_name
        self.system_msg = system_msg
        self.max_tokens = max_tokens
        self.reset()

    def reset(self):
        self.messages = []
        if self.system_msg:
            self.messages.append({"role": "system", "content": self.system_msg})

    def query(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "temperature": 0.7,
            "max_tokens": self.max_tokens
        }

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=30,
                )

                response.raise_for_status()
                result = response.json()
                reply = result["choices"][0]["message"]["content"]
                self.messages.append({"role": "assistant", "content": reply})
                return reply.strip()

            except requests.RequestException:
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_BACKOFF * attempt)
                else:
                    raise RuntimeError("Groq API retries exhausted.")

    def get_history(self):
        return self.messages

    def get_token_usage(self):
        return None  # Not supported by Groq yet
