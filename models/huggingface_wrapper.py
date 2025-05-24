import os
import requests
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer

class HuggingFaceModel:
    """
    Wrapper for Hugging Face Inference API–hosted models.
    Mirrors interface of other wrappers: reset(), query(), get_history(), get_token_usage().
    """
    def __init__(
        self,
        model_name: str,
        system_msg: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
    ):
        # Load environment variables (for HF_TOKEN)
        load_dotenv(find_dotenv())
        token = os.getenv("HF_TOKEN")
        if not token:
            raise EnvironmentError(
                "Please set your Hugging Face token in the HF_TOKEN environment variable"
            )

        # Core settings
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {token}"}
        self.system_msg = system_msg
        self.max_new_tokens = max_new_tokens

        # Conversation state
        self.reset()
        self.total_tokens = 0

        # Load tokenizer for token usage tracking
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def reset(self):
        """Reset the conversation state."""
        self.history = [{"role": "system", "content": self.system_msg}]
        self.total_tokens = 0

    def query(self, user_input: str) -> str:
        """Send a user message and return the model's reply."""
        # Append user turn
        self.history.append({"role": "user", "content": user_input})

        # Build prompt string
        prompt_parts = []
        for turn in self.history:
            prompt_parts.append(f"{turn['role']}: {turn['content']}")
        prompt = "\n".join(prompt_parts)

        # Prepare payload for HF Inference API
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": self.max_new_tokens}
        }

        # Call the API
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()
        data = response.json()

        # Extract generated_text
        if isinstance(data, list) and "generated_text" in data[0]:
            generated = data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            generated = data["generated_text"]
        else:
            raise RuntimeError(f"Unexpected API response format: {data}")

        # Remove prompt prefix if echoed
        if generated.startswith(prompt):
            reply = generated[len(prompt):].strip()
        else:
            reply = generated.strip()

        # Track token usage
        inputs_ids = self.tokenizer(prompt).input_ids
        outputs_ids = self.tokenizer(reply).input_ids
        self.total_tokens += len(inputs_ids) + len(outputs_ids)

        # Append assistant turn
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def get_history(self):
        """Return the conversation history."""
        return self.history

    def get_token_usage(self):
        """Return total tokens used (approximate)."""
        return self.total_tokens