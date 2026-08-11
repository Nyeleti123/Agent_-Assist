from pathlib import Path
import json
import pandas as pd

from pathlib import Path

folder = Path(r"C:\work_project_1\UNKNOWN_SKILL")

output_file = Path(r"C:\work_project_1\Output\conversation_flow_dataset.csv")
skipped_file = Path(r"C:\work_project_1\Output\skipped_files.csv")

if not folder.exists():
    raise FileNotFoundError(f"Folder does not exist: {folder}")


def find_messages(obj):
    messages = []

    if isinstance(obj, list):
        for item in obj:
            messages.extend(find_messages(item))

    elif isinstance(obj, dict):
        text_keys = [
            "text",
            "message",
            "content",
            "utterance",
            "transcript",
            "displayText"
        ]

        time_keys = [
            "timestamp",
            "time",
            "start_time",
            "startTime",
            "created_at",
            "createdAt",
            "dateTime"
        ]

        has_text = any(obj.get(k) for k in text_keys)

        if has_text:
            text = next((obj.get(k) for k in text_keys if obj.get(k)), "")
            speaker = (
                obj.get("speaker")
                or obj.get("role")
                or obj.get("participant")
                or obj.get("from")
                or obj.get("sender")
                or "unknown"
            )
            timestamp = next((obj.get(k) for k in time_keys if obj.get(k)), None)

            messages.append({
                "speaker": str(speaker).lower().strip(),
                "timestamp": timestamp,
                "text": str(text).strip()
            })

        for value in obj.values():
            if isinstance(value, (dict, list)):
                messages.extend(find_messages(value))

    return messages


def sort_key(message):
    value = message.get("timestamp")

    if value is None:
        return ""

    return str(value)


rows = []
skipped = []

for file_path in folder.rglob("*.json"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        conversation_id = file_path.stem

        messages = find_messages(data)

        if not messages:
            skipped.append({
                "source_file": str(file_path),
                "reason": "No messages found"
            })
            continue

        messages = sorted(messages, key=sort_key)

        turns = []
        caller_parts = []

        for msg in messages:
            speaker = msg["speaker"]
            text = msg["text"]

            if not text:
                continue

            turns.append(f"{speaker}: {text}")

            if speaker in ["caller", "customer", "client", "user"]:
                caller_parts.append(text)

        if not turns:
            skipped.append({
                "source_file": str(file_path),
                "reason": "Messages found but text was empty"
            })
            continue

        rows.append({
            "conversation_id": conversation_id,
            "source_file": str(file_path),
            "full_conversation": "\n".join(turns),
            "caller_text": " ".join(caller_parts),
            "num_turns": len(turns)
        })

    except Exception as e:
        skipped.append({
            "source_file": str(file_path),
            "reason": str(e)
        })


df = pd.DataFrame(rows)
df.to_csv(output_file, index=False, encoding="utf-8-sig")

skipped_df = pd.DataFrame(skipped)
skipped_df.to_csv(skipped_file, index=False, encoding="utf-8-sig")

print(f"Saved conversations: {len(df)}")
print(f"Skipped files: {len(skipped_df)}")
print(f"Output file: {output_file}")

if len(df) > 0:
    print(df[["conversation_id", "num_turns"]].head())