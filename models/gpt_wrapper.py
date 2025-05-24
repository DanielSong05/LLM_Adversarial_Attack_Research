# models/gpt4_api.py
import openai
import os
from dotenv import load_dotenv, find_dotenv

class GPT4Model:
    def __init__(self, model_name="gpt-4", system_msg="You are a helpful assistant."):
        load_dotenv(find_dotenv())
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found.")

        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.system_msg = system_msg
        self.messages = [{"role": "system", "content": system_msg}]
        self.total_tokens = 0

    def reset(self):
        self.messages = [{"role": "system", "content": self.system_msg}]
        self.total_tokens = 0

    def query(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        if hasattr(response, "usage"):
            self.total_tokens += response.usage.total_tokens
        return reply

    def get_token_usage(self) -> int:
        return self.total_tokens

    def get_history(self):
        return self.messages
