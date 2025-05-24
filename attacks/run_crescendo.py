#!/usr/bin/env python3
import os
import json
import time
import random
from datetime import datetime
from model_factory import load_model
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from typing import Dict, List, Any

DATA_PATH = "data/bad_conversations.json"
OUT_DIR = "results/raw"

# Define rate limits for each model (requests per minute)
RATE_LIMITS = {
    "gpt-4": 500,  # GPT-4: 500 RPM
    "claude-sonnet": 50,  # Claude: 50 RPM
    "llama-70b": 1000,  # Groq: 1000 RPM
    "llama-8b": 1000,  # Groq: 1000 RPM
}

# Calculate max workers based on rate limits
def calculate_max_workers(model_name: str) -> int:
    """Calculate appropriate number of workers based on rate limits"""
    rpm = RATE_LIMITS.get(model_name.lower(), 100)  # Default to 100 RPM if unknown
    # Convert RPM to requests per second and add safety margin
    rps = rpm / 60
    return max(1, int(rps * 0.8))  # Use 80% of capacity for safety

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.rate_limit = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.interval:
                await asyncio.sleep(self.interval - time_since_last)
            self.last_request_time = time.time()

def extract_user_turns(convo):
    return [
        turn["text"]
        for turn in convo.get("dialogue", [])
        if turn.get("speaker", "").lower() == "user"
    ]

async def process_conversation_with_rate_limit(model, convo, rate_limiter):
    """Process a single conversation with rate limiting"""
    conv_id = convo.get("conversation_id")
    style = convo.get("style", "Crescendo")  # Get style from conversation with default fallback
    user_turns = extract_user_turns(convo)

    model.reset()
    replies = []
    
    for prompt in user_turns:
        await rate_limiter.acquire()  # Wait for rate limit
        start = time.time()
        replies.append(model.query(prompt))
        print(f"Prompt processed in {time.time() - start:.2f}s")

    new_dialogue = [
        {"user": u, "model_response": r}
        for u, r in zip(user_turns, replies)
    ]

    return {
        "conversation_id": conv_id,
        "style": style,  # Use the style from conversation
        "dialogue": new_dialogue
    }

async def process_model(model_name: str, conversations: List[Dict]):
    """Process all conversations for a single model"""
    print(f"\nRunning Crescendo attacks on model: {model_name}")
    model = load_model(model_name)
    
    # Create rate limiter for this model
    rate_limiter = RateLimiter(RATE_LIMITS[model_name.lower()])
    
    # Calculate appropriate number of workers
    max_workers = calculate_max_workers(model_name)
    print(f"Using {max_workers} workers for {model_name}")
    
    results = []
    tasks = []
    
    # Create tasks for all conversations
    for convo in conversations:
        task = asyncio.create_task(
            process_conversation_with_rate_limit(model, convo, rate_limiter)
        )
        tasks.append(task)
    
    # Process tasks with concurrency limit
    for i, completed_task in enumerate(asyncio.as_completed(tasks)):
        try:
            result = await completed_task
            results.append(result)
            print(f"Processed conversation {i+1}/{len(conversations)}")
        except Exception as e:
            print(f"Error processing conversation {i+1}: {e}")
    
    # Save results
    out_file = os.path.join(OUT_DIR, f"crescendo_{model_name}_results.json")
    with open(out_file, "w", encoding="utf-8") as wf:
        json.dump(results, wf, indent=2, ensure_ascii=False)
    
    print(f"Saved results for {model_name} to: {out_file}")
    return results

async def main():
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Load conversations
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            conversations = json.load(f)
    except UnicodeDecodeError:
        print(f"Error: Could not read {DATA_PATH} with UTF-8 encoding. Trying with different encoding...")
        with open(DATA_PATH, "r", encoding="latin-1") as f:
            conversations = json.load(f)
    
    # Limit to 2 conversations for testing
    conversations = conversations[:2]
    print(f"\nRunning test with {len(conversations)} conversations")
    
    # Define models to test
    models = ["gpt-4", "claude-sonnet", "llama-70b", "llama-8b"]
    
    # Process each model
    for model_name in models:
        await process_model(model_name, conversations)

if __name__ == "__main__":
    asyncio.run(main())
