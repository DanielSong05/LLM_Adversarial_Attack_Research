import os
import asyncio
import random
import aiohttp
from dotenv import load_dotenv, find_dotenv
from typing import List

class GroqCloudModel:
    """
    Wrapper for Groq API-hosted open-source LLMs.
    Handles prompt building and response parsing.
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        # Load environment variables from .env file
        load_dotenv(find_dotenv())
        
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.model_name = model_name
        self.system_message = "You are a helpful AI assistant."
        self.conversation_histories = {}  # conv_id -> list of messages
        
        # Add retry configuration
        self.max_retries = 3
        self.base_delay = 1  # Base delay in seconds
        self.max_delay = 30  # Maximum delay in seconds
    
    def reset(self, conversation_id: str = None):
        """Reset conversation history for a specific conversation or all conversations"""
        if conversation_id:
            self.conversation_histories[conversation_id] = [{"role": "system", "content": self.system_message}]
        else:
            self.conversation_histories.clear()
    
    async def query(self, prompt: str, conversation_id: str = None) -> str:
        """Query the model with a single prompt, maintaining conversation history"""
        # Initialize or get conversation history
        if conversation_id not in self.conversation_histories:
            self.conversation_histories[conversation_id] = [{"role": "system", "content": self.system_message}]
        
        # Add user message to history
        self.conversation_histories[conversation_id].append({"role": "user", "content": prompt})
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": self.model_name,
                            "messages": self.conversation_histories[conversation_id],
                            "temperature": 0.7,
                            "max_tokens": 1000
                        },
                        timeout=30
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                        reply = result["choices"][0]["message"]["content"]
                        self.conversation_histories[conversation_id].append({"role": "assistant", "content": reply})
                        return reply
                
            except aiohttp.ClientError as e:
                if attempt < self.max_retries - 1:
                    # Calculate exponential backoff with jitter
                    delay = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                    print(f"Error in Groq query, retrying in {delay:.2f} seconds: {str(e)}")
                    await asyncio.sleep(delay)
                else:
                    raise RuntimeError(f"Groq query failed after {self.max_retries} attempts: {str(e)}")

    async def batch_query(self, prompts: List[str], conversation_id: str = None) -> List[str]:
        """Process multiple prompts sequentially"""
        all_replies = []
        for prompt in prompts:
            reply = await self.query(prompt, conversation_id)
            all_replies.append(reply)
        return all_replies

    def get_token_usage(self) -> int:
        return None  # Not supported by Groq yet

    def get_history(self, conversation_id: str = None):
        if conversation_id:
            return self.conversation_histories.get(conversation_id, [])
        return self.conversation_histories
