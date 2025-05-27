import json
from collections import Counter
import argparse
from pathlib import Path

def analyze_conversation_turns(json_file_path):
    """
    Analyze the number of conversations for each unique turn amount in a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file containing conversations
    
    Returns:
        Counter: A counter object containing turn counts and their frequencies
    """
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize counter for turn counts
    turn_counts = Counter()
    
    # Count turns in each conversation
    for conversation in data:
        # Get the dialogue list and count its length
        dialogue = conversation.get('dialogue', [])
        num_turns = len(dialogue)
        turn_counts[num_turns] += 1
    
    return turn_counts

def main():
    parser = argparse.ArgumentParser(description='Analyze conversation turn counts in JSON files')
    parser.add_argument('json_file', type=str, help='Path to the JSON file to analyze')
    args = parser.parse_args()
    
    # Verify file exists
    if not Path(args.json_file).exists():
        print(f"Error: File {args.json_file} does not exist")
        return
    
    # Analyze the file
    turn_counts = analyze_conversation_turns(args.json_file)
    
    # Print results
    print(f"\nAnalysis of {args.json_file}:")
    print("-" * 40)
    print("Number of turns | Number of conversations")
    print("-" * 40)
    
    # Sort by number of turns
    for turns, count in sorted(turn_counts.items()):
        print(f"{turns:^14} | {count:^21}")
    
    # Print total
    total_conversations = sum(turn_counts.values())
    print("-" * 40)
    print(f"Total conversations: {total_conversations}")
    
    # Calculate and display conversations with 6-12 turns
    conversations_6_to_12 = sum(count for turns, count in turn_counts.items() if 6 <= turns <= 12)
    print(f"Conversations with 6-12 turns: {conversations_6_to_12}")

if __name__ == "__main__":
    main() 