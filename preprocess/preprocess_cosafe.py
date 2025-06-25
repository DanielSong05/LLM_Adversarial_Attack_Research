import os
import json
import argparse

def combine_json_files(input_dir, output_filename="combined_cosafe.json"):
    """
    Combines multiple NDJSON-style .json files from input_dir into a single JSON array file.
    Skips the output file itself to prevent duplication.
    """
    combined_data = []
    output_path = os.path.join(input_dir, output_filename)

    for filename in os.listdir(input_dir):
        if filename.endswith(".json") and filename != output_filename:
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        combined_data.append(obj)
                    except json.JSONDecodeError as e:
                        print(f"[WARN] Skipping invalid line in {filename}: {e}")

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(combined_data, f_out, indent=2)

    print(f"[INFO] Combined {len(combined_data)} objects into {output_path}")


def transform_cosafe_data(input_file, output_file="data/cosafe_conversations.json"):
    """
    Transforms a list of conversation message lists into structured turn-based format.
    Each conversation in the input is a list of message dicts with 'role' and 'content'.
    Only includes user turns since model responses are generated fresh during evaluation.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        all_conversations = json.load(f)

    structured = []

    for i, convo in enumerate(all_conversations):
        conversation = {
            "conversation_id": f"cosafe_{i:04d}",
            "style": "CoSafe-style",
            "dialogue": []
        }

        user_turn_num = 0  # Track user turn numbers separately
        for message in convo:
            role = message.get("role", "").strip().lower()
            if role == "user":
                speaker = "user"
                user_turn_num += 1  # Increment user turn counter
            elif role == "assistant":
                # Skip assistant/model responses since they're not used in evaluation
                continue
            else:
                continue  # skip if role is missing or unknown

            content = message.get("content", "").strip()
            if content:
                conversation["dialogue"].append({
                    "turn": user_turn_num,
                    "speaker": speaker,
                    "text": content
                })

        if conversation["dialogue"]:
            structured.append(conversation)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(structured, f_out, indent=2, ensure_ascii=False)

    print(f"[INFO] Converted {len(structured)} conversations into {output_file}")




def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CoSafe NDJSON files into structured conversation format."
    )
    parser.add_argument(
        "--input_dir", "-i",
        default="data/cosafe_raw",
        help="Directory containing raw NDJSON .json files."
    )
    parser.add_argument(
        "--output", "-o",
        default="data/cosafe_conversations.json",
        help="Output path for the formatted conversation JSON file."
    )
    args = parser.parse_args()

    combine_json_files(args.input_dir)

    combined_path = os.path.join(args.input_dir, "combined_cosafe.json")
    transform_cosafe_data(combined_path, args.output)


if __name__ == "__main__":
    main()
