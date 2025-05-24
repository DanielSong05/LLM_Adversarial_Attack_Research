import pandas as pd
import json
import os
import argparse
import ast

def transform_harmbench_data(filepath):
    """
    Reads the harmbench_behaviors.csv file and converts each row
    into a structured multi-turn conversation format.
    :return: A list of conversations, where each conversation is a dictionary.
    """
    df = pd.read_csv(filepath)
    conversations = []

    # Iterate over each conversation in the CSV (each row is one conversation)
    for idx, row in df.iterrows():
        # Use the 'Source' column for a conversation ID, or generate one if it doesn't exist
        #conversation_id = row.get("Source", f"conv_{idx}")
        conversation = {
            "conversation_id": f"mhj_{idx:04d}",
            "style": "MHJ-style",
            "dialogue": []
        }

        # Loop over a range of possible message columns (adjust the range if needed)
        for turn in range(1, 101):
            message_col = f"message_{turn}"
            if message_col in df.columns and pd.notnull(row[message_col]):
                raw_text = row[message_col]
                try:
                    parsed = json.loads(raw_text)
                    text = parsed.get("body", raw_text)
                except (json.JSONDecodeError, TypeError):
                    text = raw_text

                # Alternate speakers: odd-numbered turns are 'user', even-numbered are 'model'
                speaker = "user"
                turn_data = {
                    "turns": turn,
                    "speaker": speaker,
                    "text": text
                }
                conversation["dialogue"].append(turn_data)

        # Only add conversation if there are valid message turns
        if conversation["dialogue"]:
            conversations.append(conversation)

    return conversations

def main():
    # Setup argument parsing to specify input and output paths
    parser = argparse.ArgumentParser(
        description="Convert harmbench_behaviors.csv into a structured multi-turn conversation JSON file."
    )
    parser.add_argument(
        "--input", "-i",
        default="data/mhj_raw/harmbench_behaviors.csv",
        help="Path to the harmbench_behaviors.csv file."
    )
    parser.add_argument(
        "--output", "-o",
        default="data/mhj_conversations.json",
        help="Path to the output JSON file."
    )
    args = parser.parse_args()

    # Transform the CSV data into conversations
    conversations = transform_harmbench_data(args.input)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Successfully transformed data! Saved {len(conversations)} conversations to {args.output}")

if __name__ == "__main__":
    main()
