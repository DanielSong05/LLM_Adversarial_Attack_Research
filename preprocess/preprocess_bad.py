import os
import json

def extract_final_dialogue_lines(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    final_lines = []
    for i in range(1, len(lines)):
        current_text = lines[i].split('\t')[0].removeprefix("text:")
        prev_text = lines[i - 1].split('\t')[0].removeprefix("text:")

        if not current_text.startswith(prev_text):
            final_lines.append(lines[i - 1])

    final_lines.append(lines[-1])  # Always include the last line

    with open(output_path, 'w', encoding='utf-8') as out:
        for line in final_lines:
            out.write(line)

    print(f"Extracted {len(final_lines)} lines from {input_path} → {output_path}")


def combine_cleaned_files(output_dir, splits, combined_filename="combined.txt"):
    combined_lines = []

    for split in splits:
        split_path = os.path.join(output_dir, split)
        with open(split_path, 'r', encoding='utf-8') as f:
            combined_lines.extend(f.readlines())

    combined_path = os.path.join(output_dir, combined_filename)
    with open(combined_path, 'w', encoding='utf-8') as f:
        f.writelines(combined_lines)

    print(f"Combined all cleaned splits into {combined_path} with {len(combined_lines)} total lines.")


def convert_combined_to_json(combined_path, json_path):
    with open(combined_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    conversations = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        text_field = line.split('\t')[0]
        dialogue = text_field.removeprefix("text:").split("\\n")

        turns = []
        speaker = ["user", "model"]
        for turn_num, utterance in enumerate(dialogue):
            turns.append({''
                "turn": turn_num,
                "speaker": speaker[turn_num % 2],
                "text": utterance.strip()
            })

        conversations.append({
            "conversation_id": f"bad_{i:04d}",
            "style": "Crescendo-style",
            "dialogue": turns
        })

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(conversations)} dialogues to JSON format → {json_path}")


def main():
    input_dir = output_dir = "data/bad_raw"
    os.makedirs(output_dir, exist_ok=True)

    splits = ["train.txt", "valid.txt", "test.txt"]

    for split in splits:
        input_path = os.path.join(input_dir, split)
        output_path = os.path.join(output_dir, split)
        extract_final_dialogue_lines(input_path, output_path)

    combine_cleaned_files(output_dir, splits)

    combined_path = os.path.join(output_dir, "combined.txt")
    json_path = os.path.join("data", "bad_conversations.json")
    convert_combined_to_json(combined_path, json_path)


if __name__ == "__main__":
    main()
