import os
import asyncio
import random
from dotenv import load_dotenv, find_dotenv
import anthropic
from typing import List

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
        self.system_message = system_msg
        self.conversation_histories = {}  # conv_id -> list of messages
        
        # Add retry configuration
        self.max_retries = 3
        self.base_delay = 1  # Base delay in seconds
        self.max_delay = 30  # Maximum delay in seconds

    def reset(self, conversation_id: str = None):
        """Reset conversation history for a specific conversation or all conversations"""
        if conversation_id:
            self.conversation_histories[conversation_id] = []
        else:
            self.conversation_histories.clear()

    async def query(self, prompt: str, conversation_id: str = None) -> str:
        """Query the model with a single prompt, maintaining conversation history"""
        # Initialize or get conversation history
        if conversation_id not in self.conversation_histories:
            self.conversation_histories[conversation_id] = []
        
        # Add user message to history
        self.conversation_histories[conversation_id].append({"role": "user", "content": prompt})
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    system=self.system_message,
                    messages=self.conversation_histories[conversation_id]
                )

                reply = response.content[0].text
                self.conversation_histories[conversation_id].append({"role": "assistant", "content": reply})
                return reply
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    # Calculate exponential backoff with jitter
                    delay = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                    print(f"Error in Claude query, retrying in {delay:.2f} seconds: {str(e)}")
                    await asyncio.sleep(delay)
                else:
                    raise RuntimeError(f"Claude query failed after {self.max_retries} attempts: {str(e)}")

    async def batch_query(self, prompts: List[str], conversation_id: str = None) -> List[str]:
        """Process multiple prompts sequentially"""
        all_replies = []
        for prompt in prompts:
            reply = await self.query(prompt, conversation_id)
            all_replies.append(reply)
        return all_replies

    def get_history(self, conversation_id: str = None):
        """Return the conversation history for a specific conversation or all conversations"""
        if conversation_id:
            return self.conversation_histories.get(conversation_id, [])
        return self.conversation_histories

    def get_token_usage(self) -> int:
        """Token usage is not available in Claude API, so we return None."""
        return None
