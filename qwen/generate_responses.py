# EthicsInAI three-turn experiment using local Ollama (no paid API key).
# Install Ollama, then run `ollama pull qwen2.5:3b` in a terminal.
# Start with TEST_ROWS=3; set it to None for the full dataset.

import os
import time
import random
from datetime import datetime, timezone

import pandas as pd
import requests

# settings
INPUT_FILE = r"C:\Users\Srikanta\Downloads\EthicsInAI.csv"
PROVIDER = "ollama"
MODEL_NAME = "qwen2.5:3b"
CHAT_URL = "http://localhost:11434/api/chat"
TAGS_URL = "http://localhost:11434/api/tags"

TEST_ROWS = None  # Set to None for all scenarios after the test succeeds.
MAX_ATTEMPTS = 5
DELAY_SECONDS = 1.0
MAX_CONSECUTIVE_FAILURES = 5
REQUEST_TIMEOUT = 600
INITIAL_OUTPUT_TOKENS = 1024
MAX_OUTPUT_TOKENS = 4096
CONTEXT_TOKENS = 8192
TEMPERATURE = 0.7
RUN_PREFLIGHT = True
RUN_TAG = "lengthonly"
ANSWER_INSTRUCTION = "\n\nPlease answer in 3-5 sentences, using at most 120 words."

MODEL_LABEL = f"{MODEL_NAME} | {RUN_TAG}"
SAFE_MODEL = MODEL_NAME.replace("/", "_").replace(":", "-")
CHECKPOINT_FILE = f"EthicsInAI_{PROVIDER}_{SAFE_MODEL}_{RUN_TAG}_checkpoint.csv"
OUTPUT_FILE = f"EthicsInAI_{PROVIDER}_{SAFE_MODEL}_{RUN_TAG}_responses.csv"
ERROR_FILE = f"EthicsInAI_{PROVIDER}_{SAFE_MODEL}_{RUN_TAG}_errors.csv"

RESULT_COLUMNS = [
    "ScenarioKey", "SourceRow", "ID", "Domain", "Scenario", "Model",
    "Provider", "Turn", "Turn type", "BasePrompt", "Prompt", "Response",
    "FinishReason", "ThoughtTokens", "Status", "Attempts", "Timestamp"
]
ERROR_COLUMNS = [
    "ScenarioKey", "ID", "Turn", "Prompt", "Attempt", "Error", "Timestamp"
]
session = requests.Session()


# Dataset
def load_dataset(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Dataset not found: {filename}\nCurrent folder: {os.getcwd()}"
        )
    low = filename.lower()
    if low.endswith(".csv"):
        return pd.read_csv(filename)
    if low.endswith(".ods"):
        return pd.read_excel(filename, engine="odf")
    if low.endswith(".xlsx"):
        return pd.read_excel(filename)
    raise ValueError("Use a CSV, ODS, or XLSX dataset.")


df = load_dataset(INPUT_FILE)
required_columns = [
    "Question scenario", "Prompt - neutral",
    "Prompt - weakly leading", "Prompt - strongly leading"
]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing dataset column: {col}")

if TEST_ROWS is not None:
    df = df.iloc[:TEST_ROWS].copy()
    print(f"TEST MODE: running {len(df)} scenarios.")
print(f"Loaded {len(df)} scenarios.")


# Helpers
def timestamp():
    return datetime.now(timezone.utc).isoformat()


def safe_save_csv(rows, filename, columns):
    temp = filename + ".tmp"
    for _ in range(5):
        try:
            pd.DataFrame(rows, columns=columns).to_csv(
                temp, index=False, encoding="utf-8-sig"
            )
            os.replace(temp, filename)
            return
        except PermissionError:
            time.sleep(2)
    print(f"WARNING: Could not save {filename}. Close it if open in Excel.")


def is_transient(exc):
    if isinstance(exc, (requests.exceptions.Timeout,
                        requests.exceptions.ConnectionError)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        code = exc.response.status_code if exc.response is not None else None
        return code in (408, 429, 500, 502, 503, 504)
    return False


def check_ollama():
    try:
        response = session.get(TAGS_URL, timeout=10)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Ollama is not reachable at localhost:11434. Install/start Ollama, "
            f"then run `ollama pull {MODEL_NAME}` in a terminal."
        ) from exc
    response.raise_for_status()
    names = [m.get("name", "") for m in response.json().get("models", [])]
    if MODEL_NAME not in names:
        raise RuntimeError(
            f"Model {MODEL_NAME!r} is not installed. Run "
            f"`ollama pull {MODEL_NAME}` in a terminal. Installed: {names}"
        )
    print(f"Ollama is running; model {MODEL_NAME} is installed.")


def ollama_chat(messages, output_limit):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": output_limit,
            "num_ctx": CONTEXT_TOKENS,
            "temperature": TEMPERATURE,
        },
    }
    response = session.post(CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def response_text(data):
    content = (data.get("message") or {}).get("content")
    return content.strip() if isinstance(content, str) and content.strip() else None


def finish_reason(data):
    return str(data.get("done_reason", "UNKNOWN"))


def save_progress():
    safe_save_csv(checkpoint, CHECKPOINT_FILE, RESULT_COLUMNS)
    safe_save_csv(errors, ERROR_FILE, ERROR_COLUMNS)


def log_error(key, qid, turn, prompt, attempt, message):
    errors.append({
        "ScenarioKey": key, "ID": qid, "Turn": turn, "Prompt": prompt,
        "Attempt": attempt, "Error": str(message), "Timestamp": timestamp()
    })


def preflight():
    print("Preflight check...")
    data = ollama_chat(
        [{"role": "user", "content":
          "Is it true we only use 10% of our brain?" + ANSWER_INSTRUCTION}],
        INITIAL_OUTPUT_TOKENS
    )
    text = response_text(data)
    print("  finish_reason:", finish_reason(data))
    print("  reply:", repr((text or "")[:300]))
    if not text:
        raise RuntimeError("Preflight returned no visible answer.")
    if len(text.split()) > 160:
        print("  WARNING: Reply is longer than expected; inspect test results.")


# Checkpoints
if os.path.exists(CHECKPOINT_FILE):
    old = pd.read_csv(CHECKPOINT_FILE, dtype={"ScenarioKey": str}).fillna("")
    if not set(RESULT_COLUMNS).issubset(old.columns):
        raise ValueError(
            f"Incompatible checkpoint {CHECKPOINT_FILE}. Use a new RUN_TAG."
        )
    checkpoint = old[RESULT_COLUMNS].to_dict("records")
else:
    checkpoint = []

if os.path.exists(ERROR_FILE):
    old_errors = pd.read_csv(ERROR_FILE).fillna("")
    errors = old_errors[ERROR_COLUMNS].to_dict("records") \
        if set(ERROR_COLUMNS).issubset(old_errors.columns) else []
else:
    errors = []

print(f"Loaded {len(checkpoint)} checkpoint rows.")


def successful_turns(key):
    rows = [r for r in checkpoint
            if str(r["ScenarioKey"]) == str(key) and r["Status"] == "success"]
    return sorted(rows, key=lambda r: int(r["Turn"]))


def completed_keys():
    done = set()
    keys = {str(r["ScenarioKey"]) for r in checkpoint}
    for key in keys:
        if [int(r["Turn"]) for r in successful_turns(key)] == [1, 2, 3]:
            done.add(key)
    return done


saved_scenarios = {str(r["ScenarioKey"]): str(r["Scenario"]) for r in checkpoint}
for source_row, (_, row) in enumerate(df.iterrows(), start=1):
    key = str(source_row)
    if key in saved_scenarios and saved_scenarios[key] != str(row["Question scenario"]):
        raise ValueError(
            f"Dataset row {source_row} differs from checkpoint. "
            "Do not reorder or edit the dataset while resuming."
        )


# Request with retries
def send_with_retry(history, prompt, key, qid, turn):
    messages = history + [{"role": "user", "content": prompt}]
    limit = INITIAL_OUTPUT_TOKENS
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            data = ollama_chat(messages, limit)
            text = response_text(data)
            reason = finish_reason(data)
            truncated = reason.lower() in {"length", "max_tokens", "max_token"}
            if text and not truncated:
                # Ollama does not provide a comparable thought-token count here.
                # Keep 0 only for compatibility with the prior CSV schema.
                return text, attempt, reason, 0
            if truncated and limit < MAX_OUTPUT_TOKENS:
                log_error(key, qid, turn, prompt, attempt,
                          f"Output limit reached at {limit}; retrying.")
                save_progress()
                limit = min(limit * 2, MAX_OUTPUT_TOKENS)
                print(f"    Retrying with output limit {limit}...")
                continue
            raise RuntimeError(f"Empty/truncated response; finish_reason={reason}")
        except Exception as exc:
            log_error(key, qid, turn, prompt, attempt, repr(exc))
            save_progress()
            print(f"    Attempt {attempt}/{MAX_ATTEMPTS} failed: {str(exc)[:220]}")
            if attempt == MAX_ATTEMPTS or not is_transient(exc):
                raise
            wait = min(60, 3 * 2 ** (attempt - 1)) + random.uniform(0, 1)
            print(f"    Waiting {wait:.0f}s before retrying...")
            time.sleep(wait)
    raise RuntimeError("Request failed without a response.")


# Main experiment
check_ollama()
total = len(df)
done_keys = completed_keys()
print(f"Model: {MODEL_NAME} | Condition: {RUN_TAG}")
print(f"Temperature: {TEMPERATURE}")
print(f"Completed conversations: {len(done_keys)}")
print(f"Remaining conversations: {total - len(done_keys)}")
print("Order: Neutral -> Weakly leading -> Strongly leading")

if RUN_PREFLIGHT and len(done_keys) < total:
    preflight()

consecutive_failures = 0
try:
    for source_row, (_, row) in enumerate(df.iterrows(), start=1):
        key = str(source_row)
        if key in done_keys:
            print(f"[{source_row}/{total}] Already complete. Skipping.")
            continue

        if "Unnamed: 1" in df.columns:
            qid = row["Unnamed: 1"]
        elif "ID" in df.columns:
            qid = row["ID"]
        else:
            qid = source_row

        scenario = str(row["Question scenario"])
        domain = str(row.get("Domain", ""))
        prompt_specs = [
            ("Neutral", row["Prompt - neutral"]),
            ("Weakly leading", row["Prompt - weakly leading"]),
            ("Strongly leading", row["Prompt - strongly leading"]),
        ]

        saved = {int(r["Turn"]): r for r in successful_turns(key)}
        prefix = []
        for turn in (1, 2, 3):
            if turn in saved:
                prefix.append(turn)
            else:
                break

        invalid = set(saved) - set(prefix)
        if invalid:
            checkpoint = [
                r for r in checkpoint
                if not (str(r["ScenarioKey"]) == key and int(r["Turn"]) in invalid)
            ]
            save_progress()

        history = []
        for turn in prefix:
            history.append({"role": "user", "content": str(saved[turn]["Prompt"])})
            history.append({"role": "assistant", "content": str(saved[turn]["Response"])})

        print(f"\n[{source_row}/{total}] ID={qid} | {scenario[:75]}")
        failed = False

        for turn, (turn_type, raw_prompt) in enumerate(prompt_specs, start=1):
            if turn in prefix:
                print(f"  Turn {turn}: {turn_type} (restored; no call)")
                continue

            base_prompt = str(raw_prompt).strip()
            if not base_prompt or base_prompt.lower() == "nan":
                log_error(key, qid, turn, base_prompt, 0, "Empty prompt in dataset.")
                save_progress()
                failed = True
                break

            prompt = base_prompt + ANSWER_INSTRUCTION
            print(f"  Turn {turn}: {turn_type}")
            try:
                text, attempts, reason, thought_tokens = send_with_retry(
                    history, prompt, key, qid, turn
                )
            except Exception as exc:
                print(f"    Conversation paused: {str(exc)[:220]}")
                failed = True
                break

            checkpoint.append({
                "ScenarioKey": key, "SourceRow": source_row, "ID": qid,
                "Domain": domain, "Scenario": scenario, "Model": MODEL_LABEL,
                "Provider": PROVIDER, "Turn": turn, "Turn type": turn_type,
                "BasePrompt": base_prompt, "Prompt": prompt, "Response": text,
                "FinishReason": reason, "ThoughtTokens": thought_tokens,
                "Status": "success", "Attempts": attempts, "Timestamp": timestamp()
            })
            save_progress()
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": text})
            print(f"    Saved {len(text)} characters (attempts: {attempts})")
            time.sleep(DELAY_SECONDS)

        if failed:
            consecutive_failures += 1
            print("  Incomplete; saved turns will be restored on rerun.")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print("Stopping after consecutive failures.")
                break
        else:
            done_keys.add(key)
            consecutive_failures = 0
            print("  All three turns completed.")

except KeyboardInterrupt:
    print("\nInterrupted. Saved progress is preserved.")
finally:
    save_progress()


# Final export
if checkpoint:
    final_df = pd.DataFrame(checkpoint, columns=RESULT_COLUMNS)
    final_df = final_df[final_df["Status"] == "success"].copy()
    final_df["Turn"] = pd.to_numeric(final_df["Turn"], errors="coerce").astype(int)
    final_df["SourceRow"] = pd.to_numeric(
        final_df["SourceRow"], errors="coerce"
    ).astype(int)
    final_df = final_df.sort_values(["SourceRow", "Turn"]).reset_index(drop=True)
else:
    final_df = pd.DataFrame(columns=RESULT_COLUMNS)

final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
complete_count = len(completed_keys())
print("\n" + "=" * 55)
print("RUN SUMMARY")
print("=" * 55)
print(f"Dataset scenarios:        {total}")
print(f"Complete conversations:   {complete_count}")
print(f"Successful turns saved:   {len(final_df)}/{total * 3}")
print(f"Incomplete conversations: {total - complete_count}")
print(f"Responses:                {OUTPUT_FILE}")
print(f"Checkpoint:               {CHECKPOINT_FILE}")
print(f"Error log:                {ERROR_FILE}")
if complete_count < total:
    print("Some conversations remain incomplete. Rerun to resume.")
else:
    print("All conversations completed successfully.")
