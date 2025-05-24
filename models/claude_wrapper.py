import os
from dotenv import load_dotenv, find_dotenv
import anthropic

class ClaudeModel:
    def __init__(self, model_name="claude-3-7-sonnet-20250219", system_msg="You are a helpful assistant."):
        # Load .env file and get the API key
        load_dotenv(find_dotenv())
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not found in .env file.")

        # Create Anthropic client
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        self.system_msg = system_msg
        self.messages = []
        self.total_tokens = 0  # not exposed by API, but placeholder for symmetry

    def reset(self):
        """Reset the conversation to start fresh."""
        self.messages = []
        self.total_tokens = 0

    def query(self, user_input: str) -> str:
        """Send a user message and receive Claude's reply."""
        self.messages.append({"role": "user", "content": user_input})

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            system=self.system_msg,
            messages=self.messages
        )

        reply = response.content[0].text
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def get_history(self):
        """Return the full message history."""
        return self.messages

    def get_token_usage(self):
        """Token usage is not available in Claude API, so we return None."""
        return None
