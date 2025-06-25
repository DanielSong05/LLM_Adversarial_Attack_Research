#!/usr/bin/env python3
"""
Modular attack runner for different attack types (CoSafe, MHJ, etc.)
For efficiency and standardization, only conversations with 1-6 turns were included.
"""

import os
import json
import time
import asyncio
import tiktoken
from typing import Dict, List
from models.model_factory import load_model

OUT_DIR = "results/raw"

# Rate limits for each model
RATE_LIMITS = {
    "gpt-4": 500,  # GPT-4: 500 RPM
    "claude-sonnet": 50,  # Claude: 50 RPM
    "llama-70b": 1000,  # Groq: 1000 RPM
    "llama-8b": 1000,  # Groq: 1000 RPM
}

# Token limits for each model (tokens per minute)
TOKEN_LIMITS = {
    "gpt-4": 30000,      # GPT-4: 30k TPM
    "claude-sonnet": 20000,  # Claude: 20k TPM
    "llama-70b": 300000,  # Llama: 300k TPM
    "llama-8b": 250000,   # Llama: 250k TPM
}

def get_tokenizer(model_name: str):
    """Get the appropriate tokenizer for the model"""
    if "gpt-4" in model_name.lower():
        return tiktoken.encoding_for_model("gpt-4")
    elif "claude" in model_name.lower():
        # Claude uses a different tokenizer, but we can use GPT-4's as an approximation
        return tiktoken.encoding_for_model("gpt-4")
    else:
        # For Llama models, use GPT-4's tokenizer as an approximation
        return tiktoken.encoding_for_model("gpt-4")

def count_tokens(text: str, model_name: str) -> int:
    """Count tokens in text using the appropriate tokenizer"""
    tokenizer = get_tokenizer(model_name)
    return len(tokenizer.encode(text))

def calculate_max_workers(conversations: List[Dict], model_name: str) -> int:
    """Calculate maximum number of parallel workers based on rate limits"""
    # Get model-specific rate limit
    rpm_limit = RATE_LIMITS.get(model_name.lower(), 100)  # Default to 100 if unknown
    
    # Convert RPM to requests per second and add 20% safety margin
    rps = rpm_limit / 60
    max_workers = max(1, int(rps * 0.8))  # Use 80% of capacity for safety
    
    print(f"Rate limit: {rpm_limit} RPM")
    print(f"Calculated workers: {max_workers}")
    
    return max_workers

class RateLimiter:
    def __init__(self, rpm_limit: int, tpm_limit: int):
        self.rpm_limit = int(rpm_limit * 0.8)  # 20% safety margin
        self.tpm_limit = int(tpm_limit * 0.8)  # 20% safety margin
        self.request_history = []  # List of timestamps
        self.token_history = []    # List of (timestamp, tokens) pairs
        self.lock = asyncio.Lock()
        print(f"Initialized RateLimiter with {self.rpm_limit} RPM and {self.tpm_limit} TPM limits (with 20% safety margin)")
    
    async def wait_if_needed(self, estimated_tokens: int):
        """Wait if needed to stay within rate limits"""
        async with self.lock:
            current_time = time.time()
            
            # Clean up old history
            self.request_history = [t for t in self.request_history if current_time - t < 60]
            self.token_history = [(t, n) for t, n in self.token_history if current_time - t < 60]
            
            # Calculate current usage
            requests_last_minute = len(self.request_history)
            tokens_last_minute = sum(n for _, n in self.token_history)
            
            # If we're at the limit, wait
            if requests_last_minute >= self.rpm_limit or tokens_last_minute + estimated_tokens > self.tpm_limit:
                if self.request_history:
                    oldest_time = min(self.request_history[0], self.token_history[0][0] if self.token_history else float('inf'))
                    wait_time = 60 - (current_time - oldest_time)
                    if wait_time > 0:
                        print(f"Rate limit reached! Waiting {wait_time:.1f}s...")
                        print(f"Current usage - Requests: {requests_last_minute}/{self.rpm_limit}, Tokens: {tokens_last_minute}/{self.tpm_limit}")
                        await asyncio.sleep(wait_time)
                        # Update current time and clean history after waiting
                        current_time = time.time()
                        self.request_history = [t for t in self.request_history if current_time - t < 60]
                        self.token_history = [(t, n) for t, n in self.token_history if current_time - t < 60]
            
            # Record this request
            self.request_history.append(current_time)
            self.token_history.append((current_time, estimated_tokens))

def extract_user_turns(convo, attack_type: str):
    """Extract user turns based on attack type"""
    if attack_type == "cosafe":
        return [
            turn["text"]
            for turn in convo.get("dialogue", [])
            if turn.get("speaker", "").lower() == "user"
        ]
    else:  # mhj and others
        return [
            turn["text"]
            for turn in convo.get("dialogue", [])
            if turn.get("speaker", "").lower() == "user"
        ]

class ConversationManager:
    def __init__(self, model_name: str, attack_type: str):
        print(f"Initializing ConversationManager for model: {model_name}, attack: {attack_type}")
        self.model = load_model(model_name)
        self.model_name = model_name
        self.attack_type = attack_type
        self.conversation_histories = {}  # conv_id -> list of messages
        self.rate_limiter = RateLimiter(
            RATE_LIMITS[model_name.lower()],
            TOKEN_LIMITS[model_name.lower()]
        )
        self.total_tokens = 0
        self.start_time = None
        self.processed = 0
        self.total_conversations = 0
        self.current_conversation = None
        self.current_turn = 0
        self.total_turns = 0
        self.active_workers = 0  # Track number of active workers
        print("ConversationManager initialized successfully")
        
    async def process_conversation(self, conversation: Dict) -> Dict:
        """Process a single conversation with proper context"""
        conv_id = conversation.get("conversation_id")
        style = conversation.get("style", f"{self.attack_type.upper()}-style")
        user_turns = extract_user_turns(conversation, self.attack_type)
        
        print(f"\n{'='*50}", flush=True)
        print(f"Starting new conversation {conv_id} (Style: {style})", flush=True)
        
        # Update current conversation info
        self.current_conversation = conv_id
        self.total_turns = len(user_turns)
        self.current_turn = 0
        
        print(f"Conversation has {self.total_turns} turns", flush=True)
        
        # Initialize or get conversation history
        if conv_id not in self.conversation_histories:
            self.conversation_histories[conv_id] = [{"role": "system", "content": self.model.system_message}]
            print("Initialized new conversation history", flush=True)
        
        # Process each turn in sequence
        dialogue = []
        conversation_tokens = 0
        
        for i, turn in enumerate(user_turns, 1):
            self.current_turn = i
            print(f"\nProcessing turn {i}/{len(user_turns)} in conversation {conv_id}", flush=True)
            
            # Get token count for this turn
            token_count = count_tokens(turn, self.model_name)
            conversation_tokens += token_count
            print(f"Turn {i} token count: {token_count}", flush=True)
            
            # Wait for rate limits
            await self.rate_limiter.wait_if_needed(token_count)
            
            # Get model response
            response = await self.model.query(turn, conversation_id=conv_id)
            
            # Add to dialogue
            dialogue.append({
                "user": turn,
                "model_response": response
            })
            
            print(f"Completed turn {i}/{len(user_turns)} in conversation {conv_id}", flush=True)
        
        # Update total tokens
        self.total_tokens += conversation_tokens
        print(f"Conversation {conv_id} completed - Total tokens used: {conversation_tokens}", flush=True)
        
        # Clear current conversation info
        self.current_conversation = None
        self.current_turn = 0
        self.total_turns = 0
        
        # Update processed count and print completion
        self.processed += 1
        print(f"Completed {self.processed}/{self.total_conversations} conversations", flush=True)
        
        # Return conversation results
        return {
            "conversation_id": conv_id,
            "style": style,
            "dialogue": dialogue,
            "tokens_used": conversation_tokens
        }
    
    async def process_conversations(self, conversations: List[Dict]) -> List[Dict]:
        """Process multiple conversations in parallel"""
        results = []
        self.total_conversations = len(conversations)
        self.processed = 0
        self.start_time = time.time()
        
        # Calculate max workers based on rate limits
        max_workers = calculate_max_workers(conversations, self.model_name)
        print(f"\nProcessing {self.total_conversations} conversations with {max_workers} workers", flush=True)
        print(f"Model: {self.model_name}, Attack: {self.attack_type}", flush=True)
        print("=" * 50, flush=True)
        
        # Process conversations in parallel with rate limiting
        semaphore = asyncio.Semaphore(max_workers)
        
        async def process_with_semaphore(convo):
            async with semaphore:
                self.active_workers += 1
                try:
                    return await self.process_conversation(convo)
                finally:
                    self.active_workers -= 1
        
        # Process all conversations
        tasks = []
        for convo in conversations:
            task = asyncio.create_task(process_with_semaphore(convo))
            tasks.append(task)
        
        # Wait for tasks and collect results as they complete
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
        
        total_time = time.time() - self.start_time
        print(f"\nProcessing completed in {total_time/60:.1f} minutes", flush=True)
        print(f"Total conversations: {self.total_conversations}", flush=True)
        print(f"Total tokens used: {self.total_tokens:,}", flush=True)
        print(f"Average time per conversation: {total_time/self.total_conversations:.1f}s", flush=True)
        print(f"Average tokens per conversation: {self.total_tokens/self.total_conversations:.0f}", flush=True)
        print(f"Overall speed: {(self.total_conversations/total_time)*60:.1f} conversations/minute", flush=True)
        
        return results

async def process_model(model_name: str, conversations: List[Dict], attack_type: str):
    """Process all conversations for a single model and attack type"""
    print(f"\nRunning {attack_type.upper()} attacks on model: {model_name}")
    
    # Create conversation manager
    manager = ConversationManager(model_name, attack_type)
    
    # Process conversations
    results = await manager.process_conversations(conversations)
    
    # Save results
    out_file = os.path.join(OUT_DIR, f"{attack_type}_{model_name}_results.json")
    with open(out_file, "w", encoding="utf-8") as wf:
        json.dump(results, wf, indent=2, ensure_ascii=False)
    
    print(f"Saved results for {model_name} to: {out_file}")
    return results

async def run_attack(attack_type: str, models: List[str] = None, max_conversations: int = 50):
    """Run attacks for specified attack type and models"""
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Set default models if none specified
    if models is None:
        models = ["gpt-4", "claude-sonnet", "llama-70b", "llama-8b"]
    
    # Load conversations based on attack type
    data_path = f"data/{attack_type}_conversations.json"
    with open(data_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    
    # Filter conversations to only include those with max 6 user turns
    conversations = [
        convo for convo in conversations 
        if len([turn for turn in convo.get("dialogue", []) if turn.get("speaker", "").lower() == "user"]) <= 6
    ]
    
    # Limit conversations
    conversations = conversations[:max_conversations]
    print(f"\nRunning {attack_type} attack with {len(conversations)} conversations (max 6 turns each)")
    
    # Process each model
    for model_name in models:
        await process_model(model_name, conversations, attack_type)

async def main():
    """Main function to run attacks"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run different types of attacks on models")
    parser.add_argument("--models", nargs="+", 
                       default=["gpt-4", "claude-sonnet", "llama-70b", "llama-8b"],
                       help="Models to test (default: all)")
    parser.add_argument("--max-conversations", type=int, default=50,
                       help="Maximum number of conversations to process (default: 50)")
    
    args = parser.parse_args()
    
    # Run both CoSafe and MHJ attacks
    print("=" * 60)
    print("RUNNING COSAFE ATTACKS")
    print("=" * 60)
    await run_attack("cosafe", args.models, args.max_conversations)
    
    print("\n" + "=" * 60)
    print("RUNNING MHJ ATTACKS")
    print("=" * 60)
    await run_attack("mhj", args.models, args.max_conversations)
    
    print("\n" + "=" * 60)
    print("ALL ATTACKS COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main()) 