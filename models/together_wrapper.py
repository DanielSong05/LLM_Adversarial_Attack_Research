import os
import time
import requests
from dotenv import load_dotenv, find_dotenv

class TogetherCloudModel:
    """
    Wrapper for Together.xyz hosted open-source LLMs.
    Handles prompt building, error checking, and response parsing uniformly.
    """
    MAX_RETRIES = 3
    RETRY_BACKOFF = 2  # seconds multiplier

    def __init__(self, model_name, system_msg=None, max_tokens=1024):
        # Load API key
        load_dotenv(find_dotenv())
        self.api_key = os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise RuntimeError("TOGETHER_API_KEY not found in environment.")

        self.model_name = model_name
        self.system_msg = system_msg
        self.max_tokens = max_tokens
        # Initialize conversation history
        self.reset()

    def reset(self):
        """Reset conversation with fresh system prompt."""
        self.messages = [{"role": "system", "content": self.system_msg}]

    def query(self, user_input: str) -> str:
        """
        Send a user message, handle retries, and return the assistant reply.
        Raises a RuntimeError for API errors.
        """
        self.messages.append({"role": "user", "content": user_input})
        prompt = self.build_prompt()

        for attempt in range(1, self.MAX_RETRIES + 1):
            response = requests.post(
                "https://api.together.xyz/v1/completions",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": 0.7
                },
                headers={"Authorization": f"Bearer {self.api_key}"}
            )

            # Parse JSON
            try:
                resp = response.json()
            except ValueError:
                raise RuntimeError("Invalid JSON response from Together API.")

            # Check for errors
            if "error" in resp:
                err = resp["error"]
                code = err.get("type")
                msg = err.get("message", "Unknown error")
                # Retry on service_unavailable
                if code == "service_unavailable_error" and attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_BACKOFF * attempt)
                    continue
                raise RuntimeError(f"Together API error [{code}]: {msg}")

            # Extract text from known fields
            if "choices" in resp:
                output = resp["choices"][0].get("text", "")
            elif "completions" in resp:
                output = resp["completions"][0].get("data", {}).get("text", "")
            else:
                keys = list(resp.keys())
                raise KeyError(f"Unexpected response format from Together API, keys: {keys}")

            # Append assistant reply to history and return
            self.messages.append({"role": "assistant", "content": output})
            return output.strip()

        # If we exit the loop without returning, it's a retry failure
        raise RuntimeError("Together API retries exhausted.")

    def build_prompt(self) -> str:
        """
        Construct a single-text prompt with role tags for the Together API.
        """
        prompt = []
        for m in self.messages:
            tag = f"<|{m['role']}|>"
            prompt.append(f"{tag}\n{m['content']}")
        return "\n".join(prompt)

    def get_history(self):
        """Return the full conversation message list."""
        return self.messages

    def get_token_usage(self):
        """
        Together API does not expose token counts; return None.
        """
        return None
