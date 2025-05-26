# models/gpt4_api.py
import openai
import os
from dotenv import load_dotenv, find_dotenv
from typing import List
import asyncio
import random

class GPT4Model:
    def __init__(self, model_name: str = "gpt-4"):
        # Load environment variables from .env file
        load_dotenv(find_dotenv())
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(api_key=self.api_key)
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
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.conversation_histories[conversation_id],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                # Extract and store response
                reply = response.choices[0].message.content
                self.conversation_histories[conversation_id].append({"role": "assistant", "content": reply})
                
                return reply
                
            except openai.RateLimitError as e:
                if attempt < self.max_retries - 1:
                    # Calculate exponential backoff with jitter
                    delay = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                    print(f"Rate limit hit, retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    raise RuntimeError(f"Rate limit retries exhausted: {str(e)}")
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                    print(f"Error in GPT-4 query, retrying in {delay:.2f} seconds: {str(e)}")
                    await asyncio.sleep(delay)
                else:
                    raise RuntimeError(f"GPT-4 query failed after {self.max_retries} attempts: {str(e)}")

    def get_token_usage(self) -> int:
        return self.total_tokens

    def get_history(self):
        return self.messages
