# EthicsInAI multi-turn response collection (Gemma 4)
# Runs neutral, weakly leading, and strongly leading prompts per scenario.

import os
import time
import random
import getpass
from datetime import datetime, timezone

import pandas as pd
from google import genai
from google.genai import types

# Configuration

INPUT_FILE = r"C:\Users\Srikanta\Downloads\EthicsInAI.csv"

PROVIDER = "gemini"
MODEL_NAME = "gemma-4-26b-a4b-it"

TEST_ROWS = None                 # None = all scenarios; 3 = test
MAX_ATTEMPTS = 6
DELAY_SECONDS = 4.5              # keeps number of requests under approx 15 requests/minute
MAX_CONSECUTIVE_FAILURES = 5

INITIAL_OUTPUT_TOKENS = 1024
MAX_RETRY_OUTPUT_TOKENS = 4096

# Set to None to send no thinking setting (model default).
THINKING_LEVEL = "MINIMAL"

RUN_PREFLIGHT = True

RUN_TAG = "lengthonly"

ANSWER_INSTRUCTION = (
    "\n\nPlease answer in 3-5 sentences, using at most 120 words."
)

MODEL_LABEL = f"{MODEL_NAME} | {RUN_TAG}"

safe_model = MODEL_NAME.replace("/", "_")

CHECKPOINT_FILE = f"EthicsInAI_{PROVIDER}_{safe_model}_{RUN_TAG}_checkpoint.csv"
OUTPUT_FILE = f"EthicsInAI_{PROVIDER}_{safe_model}_{RUN_TAG}_responses.csv"
ERROR_FILE = f"EthicsInAI_{PROVIDER}_{safe_model}_{RUN_TAG}_errors.csv"

RESULT_COLUMNS = [
    "ScenarioKey", "SourceRow", "ID", "Domain", "Scenario",
    "Model", "Provider", "Turn", "Turn type",
    "BasePrompt", "Prompt", "Response",
    "FinishReason", "ThoughtTokens",
    "Status", "Attempts", "Timestamp"
]

ERROR_COLUMNS = [
    "ScenarioKey", "ID", "Turn", "Prompt",
    "Attempt", "Error", "Timestamp"
]

# API client

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    api_key = getpass.getpass("Paste your Gemini API key (hidden): ").strip()

if not api_key:
    raise ValueError("No API key provided.")

client = genai.Client(api_key=api_key)

# Load dataset

def load_dataset(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Dataset not found: {filename}\n"
            f"Current folder: {os.getcwd()}"
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
print(f"Loaded {len(df)} scenarios.")

required_columns = [
    "Question scenario",
    "Prompt - neutral",
    "Prompt - weakly leading",
    "Prompt - strongly leading",
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing dataset column: {col}")

if TEST_ROWS is not None:
    df = df.iloc[:TEST_ROWS].copy()
    print(f"TEST MODE: running {len(df)} scenarios.")

# Helpers

def timestamp():
    return datetime.now(timezone.utc).isoformat()

def safe_save_csv(data, filename, columns):
    """Write to a temporary file before replacing the target."""
    temp_file = filename + ".tmp"

    for _ in range(5):
        try:
            pd.DataFrame(data, columns=columns).to_csv(
                temp_file, index=False, encoding="utf-8-sig"
            )
            os.replace(temp_file, filename)
            return
        except PermissionError:
            time.sleep(2)

    print(
        f"WARNING: Could not save {filename}. "
        "Close the file in Excel if it is open."
    )

class EmptyResponseError(Exception):
    def __init__(self, message, blocked=False):
        super().__init__(message)
        self.blocked = blocked

def is_transient_error(exc):
    """Identify temporary errors worth retrying."""

    if isinstance(exc, EmptyResponseError):
        return not exc.blocked

    status = getattr(exc, "code", None)

    if status is None:
        status = getattr(exc, "status_code", None)

    if status is not None:
        try:
            status = int(status)
            return status in (408, 429) or 500 <= status <= 599
        except (ValueError, TypeError):
            pass

    message = str(exc).lower()

    return any(term in message for term in [
        "503", "429", "500", "502", "504",
        "timeout", "timed out",
        "temporarily unavailable",
        "connection reset", "connection error",
        "resource_exhausted", "unavailable", "deadline"
    ])

def make_content(role, text):
    return types.Content(role=role, parts=[types.Part(text=text)])

def build_config(output_limit):
    """
    The thinking level is sent through extra_body so it works on old
    SDK versions that do not know the 'thinking_level' field. The same code
    path is used for every request, so all prompts are treated identically.
    """
    if not THINKING_LEVEL:
        return types.GenerateContentConfig(max_output_tokens=output_limit)

    return types.GenerateContentConfig(
        http_options=types.HttpOptions(
            extra_body={
                "generationConfig": {
                    "maxOutputTokens": output_limit,
                    "thinkingConfig": {"thinkingLevel": THINKING_LEVEL},
                }
            }
        )
    )

def get_finish_reason(response):
    try:
        reason = response.candidates[0].finish_reason
        return getattr(reason, "name", str(reason))
    except Exception:
        return "UNKNOWN"

def get_thought_tokens(response):
    try:
        return int(response.usage_metadata.thoughts_token_count or 0)
    except Exception:
        return 0

def get_response_text(response):
    """Return ONLY the visible answer. Thought parts are skipped."""
    try:
        parts = response.candidates[0].content.parts or []
        texts = [
            part.text for part in parts
            if getattr(part, "text", None)
            and not getattr(part, "thought", False)
        ]
        combined = "\n".join(texts).strip()
        return combined or None
    except Exception:
        return None

# Preflight check

def run_preflight():
    test_prompt = (
        "Is it true we only use 10% of our brain?" + ANSWER_INSTRUCTION
    )

    print("Preflight check...")

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=test_prompt,
            config=build_config(INITIAL_OUTPUT_TOKENS)
        )
    except Exception as exc:
        text = str(exc).lower()
        config_problem = (
            isinstance(exc, (ValueError, TypeError))
            or "thinking" in text
            or "invalid argument" in text
            or "400" in text
        )
        if config_problem:
            raise RuntimeError(
                "Preflight failed because of a configuration problem:\n"
                f"  {str(exc)[:300]}\n"
                "If this mentions thinking_level or extra fields, run "
                "'pip install -U google-genai' and restart the kernel. "
                "As a last resort set THINKING_LEVEL = None."
            )
        print(f"  Preflight skipped (temporary error): {str(exc)[:150]}")
        return

    text = get_response_text(response)
    finish = get_finish_reason(response)
    thoughts = get_thought_tokens(response)
    words = len(text.split()) if text else 0

    print(f"  finish_reason:  {finish}")
    print(f"  thought tokens: {thoughts}")
    print(f"  words:          {words}")
    print(f"  reply: {(text or '')[:300]!r}")

    if not text:
        raise RuntimeError(
            "Preflight returned no visible answer."
        )

    if thoughts > 300:
        print(
            "  WARNING: the model is still using many thinking tokens. "
        )

    if words > 160:
        print(
            "  WARNING: the reply is much longer than the 120-word "
            "instruction."
        )

# Load saved progress

if os.path.exists(CHECKPOINT_FILE):
    checkpoint_df = pd.read_csv(
        CHECKPOINT_FILE, dtype={"ScenarioKey": str}
    ).fillna("")

    if not set(RESULT_COLUMNS).issubset(checkpoint_df.columns):
        raise ValueError(
            f"Incompatible checkpoint: {CHECKPOINT_FILE}. "
            "Rename or delete it (and the matching responses/errors "
            "files) and start a new run."
        )

    checkpoint = checkpoint_df[RESULT_COLUMNS].to_dict("records")
else:
    checkpoint = []

if os.path.exists(ERROR_FILE):
    error_df = pd.read_csv(ERROR_FILE).fillna("")

    if set(ERROR_COLUMNS).issubset(error_df.columns):
        errors = error_df[ERROR_COLUMNS].to_dict("records")
    else:
        errors = []
else:
    errors = []

print(f"Loaded {len(checkpoint)} checkpoint rows.")

def save_progress():
    safe_save_csv(checkpoint, CHECKPOINT_FILE, RESULT_COLUMNS)
    safe_save_csv(errors, ERROR_FILE, ERROR_COLUMNS)

def get_successful_turns(scenario_key):
    rows = [
        r for r in checkpoint
        if str(r["ScenarioKey"]) == str(scenario_key)
        and str(r["Status"]) == "success"
    ]
    return sorted(rows, key=lambda r: int(r["Turn"]))

def completed_scenario_keys():
    complete = set()

    for key in {str(r["ScenarioKey"]) for r in checkpoint}:
        turns = [int(r["Turn"]) for r in get_successful_turns(key)]
        if turns == [1, 2, 3]:
            complete.add(key)

    return complete

saved_scenarios = {
    str(r["ScenarioKey"]): str(r["Scenario"]) for r in checkpoint
}

for source_row, (_, row) in enumerate(df.iterrows(), start=1):
    key = str(source_row)

    if (
        key in saved_scenarios
        and saved_scenarios[key] != str(row["Question scenario"])
    ):
        raise ValueError(
            f"Dataset row {source_row} differs from the checkpoint. "
            "Do not reorder or edit the dataset while resuming."
        )

# Send one turn

def log_error(scenario_key, question_id, turn, prompt, attempt, message):
    errors.append({
        "ScenarioKey": scenario_key,
        "ID": question_id,
        "Turn": turn,
        "Prompt": prompt,
        "Attempt": attempt,
        "Error": message,
        "Timestamp": timestamp()
    })
    save_progress()

def send_with_retry(history, prompt, scenario_key, question_id, turn):
    """
    Returns (text, attempts, finish_reason, thought_tokens).

    A reply that hit MAX_TOKENS is never returned. The request is
    repeated with a larger output limit instead.
    """

    contents = history + [make_content("user", prompt)]
    output_limit = INITIAL_OUTPUT_TOKENS

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=build_config(output_limit)
            )

            finish_reason = get_finish_reason(response)
            thought_tokens = get_thought_tokens(response)
            response_text = get_response_text(response)

            truncated = "MAX_TOKENS" in finish_reason.upper()

            if response_text and not truncated:
                return response_text, attempt, finish_reason, thought_tokens

            if truncated:
                log_error(
                    scenario_key, question_id, turn, prompt, attempt,
                    f"MAX_TOKENS at limit {output_limit}; "
                    f"thought_tokens={thought_tokens}; "
                    f"had_text={bool(response_text)}"
                )

                if output_limit < MAX_RETRY_OUTPUT_TOKENS:
                    output_limit = min(
                        output_limit * 2, MAX_RETRY_OUTPUT_TOKENS
                    )
                    print(
                        "    Output hit the token limit. Retrying with "
                        f"limit {output_limit}..."
                    )
                    continue

                raise RuntimeError(
                    "Still truncated at the maximum output limit "
                    f"({output_limit}). Not saved."
                )

            feedback = getattr(response, "prompt_feedback", None)
            reason_text = finish_reason.upper()

            blocked = (
                bool(feedback and getattr(feedback, "block_reason", None))
                or any(
                    marker in reason_text
                    for marker in [
                        "SAFETY", "PROHIBITED", "BLOCKLIST",
                        "SPII", "RECITATION"
                    ]
                )
            )

            raise EmptyResponseError(
                f"Empty response; finish_reason={finish_reason}; "
                f"prompt_feedback={feedback}",
                blocked
            )

        except Exception as exc:
            if "Still truncated" not in str(exc):
                log_error(
                    scenario_key, question_id, turn, prompt,
                    attempt, repr(exc)
                )

            print(
                f"    Attempt {attempt}/{MAX_ATTEMPTS} failed: "
                f"{str(exc)[:220]}"
            )

            if attempt == MAX_ATTEMPTS or not is_transient_error(exc):
                raise

            wait = min(120, 5 * 2 ** (attempt - 1)) + random.uniform(0, 1)
            print(f"    Waiting {wait:.0f}s before retrying...")
            time.sleep(wait)

    raise RuntimeError("Request failed without a response.")

# Run experiment

total_questions = len(df)
complete_keys = completed_scenario_keys()

print(f"Model: {MODEL_NAME}")
print(f"Condition: {RUN_TAG}")
print(f"Thinking level: {THINKING_LEVEL}")
print(f"Completed conversations: {len(complete_keys)}")
print(f"Remaining conversations: {total_questions - len(complete_keys)}")
print("Order: Neutral -> Weakly leading -> Strongly leading")

if RUN_PREFLIGHT and len(complete_keys) < total_questions:
    run_preflight()

consecutive_failures = 0

try:
    for source_row, (_, row) in enumerate(df.iterrows(), start=1):
        scenario_key = str(source_row)

        if scenario_key in complete_keys:
            print(
                f"[{source_row}/{total_questions}] "
                "Already complete. Skipping."
            )
            continue

        if "Unnamed: 1" in df.columns:
            question_id = row["Unnamed: 1"]
        elif "ID" in df.columns:
            question_id = row["ID"]
        else:
            question_id = source_row

        scenario = str(row["Question scenario"])
        domain = str(row.get("Domain", ""))

        prompt_specs = [
            ("Neutral", row["Prompt - neutral"]),
            ("Weakly leading", row["Prompt - weakly leading"]),
            ("Strongly leading", row["Prompt - strongly leading"]),
        ]

        # Restore saved turns and keep only a continuous prefix.
        saved = {
            int(r["Turn"]): r for r in get_successful_turns(scenario_key)
        }

        valid_prefix = []

        for turn_number in (1, 2, 3):
            if turn_number in saved:
                valid_prefix.append(turn_number)
            else:
                break

        invalid_turns = set(saved) - set(valid_prefix)

        if invalid_turns:
            checkpoint = [
                r for r in checkpoint
                if not (
                    str(r["ScenarioKey"]) == scenario_key
                    and int(r["Turn"]) in invalid_turns
                )
            ]
            save_progress()

        # Reconstruct the earlier conversation in order.
        history = []

        for turn_number in valid_prefix:
            saved_row = saved[turn_number]
            history.append(make_content("user", str(saved_row["Prompt"])))
            history.append(make_content("model", str(saved_row["Response"])))

        print(
            f"\n[{source_row}/{total_questions}] "
            f"ID={question_id} | {scenario[:75]}"
        )

        scenario_failed = False

        for turn_number, (turn_type, raw_prompt) in enumerate(
            prompt_specs, start=1
        ):
            if turn_number in valid_prefix:
                print(
                    f"  Turn {turn_number}: {turn_type} "
                    "(restored; no API call)"
                )
                continue

            base_prompt = str(raw_prompt).strip()

            if not base_prompt or base_prompt.lower() == "nan":
                log_error(
                    scenario_key, question_id, turn_number,
                    base_prompt, 0, "Empty prompt in dataset."
                )
                scenario_failed = True
                break

            prompt = base_prompt + ANSWER_INSTRUCTION

            print(f"  Turn {turn_number}: {turn_type}")

            try:
                (
                    response_text,
                    attempts_used,
                    finish_reason,
                    thought_tokens
                ) = send_with_retry(
                    history=history,
                    prompt=prompt,
                    scenario_key=scenario_key,
                    question_id=question_id,
                    turn=turn_number
                )
            except Exception as exc:
                print(f"    Scenario paused: {str(exc)[:220]}")
                scenario_failed = True
                break

            checkpoint.append({
                "ScenarioKey": scenario_key,
                "SourceRow": source_row,
                "ID": question_id,
                "Domain": domain,
                "Scenario": scenario,
                "Model": MODEL_LABEL,
                "Provider": PROVIDER,
                "Turn": turn_number,
                "Turn type": turn_type,
                "BasePrompt": base_prompt,
                "Prompt": prompt,
                "Response": response_text,
                "FinishReason": finish_reason,
                "ThoughtTokens": thought_tokens,
                "Status": "success",
                "Attempts": attempts_used,
                "Timestamp": timestamp()
            })
            save_progress()

            history.append(make_content("user", prompt))
            history.append(make_content("model", response_text))

            print(
                f"    Saved {len(response_text)} characters "
                f"(attempts: {attempts_used}, "
                f"thought tokens: {thought_tokens})"
            )

            time.sleep(DELAY_SECONDS)

        if scenario_failed:
            consecutive_failures += 1
            print(
                "  Conversation incomplete. Saved turns will be "
                "restored on the next run."
            )

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(
                    "\nStopping after consecutive failures. "
                    "Run this cell again later to resume."
                )
                break
        else:
            complete_keys.add(scenario_key)
            consecutive_failures = 0
            print("  All three turns completed.")

except KeyboardInterrupt:
    print("\nInterrupted. Saved progress is preserved.")

finally:
    save_progress()

# Export results

if checkpoint:
    final_df = pd.DataFrame(checkpoint, columns=RESULT_COLUMNS)
    final_df = final_df[final_df["Status"] == "success"].copy()

    final_df["Turn"] = pd.to_numeric(
        final_df["Turn"], errors="coerce"
    ).astype(int)
    final_df["SourceRow"] = pd.to_numeric(
        final_df["SourceRow"], errors="coerce"
    ).astype(int)

    final_df = final_df.sort_values(
        ["SourceRow", "Turn"]
    ).reset_index(drop=True)
else:
    final_df = pd.DataFrame(columns=RESULT_COLUMNS)

final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

complete_count = len(completed_scenario_keys())
successful_turns = len(final_df)

thinking_rows = 0
if successful_turns:
    thinking_rows = int(
        (pd.to_numeric(final_df["ThoughtTokens"], errors="coerce")
         .fillna(0) > 0).sum()
    )

print("\n" + "=" * 60)
print("RUN SUMMARY")
print("=" * 60)
print(f"Dataset scenarios:        {total_questions}")
print(f"Complete conversations:   {complete_count}")
print(f"Successful turns saved:   {successful_turns}/{total_questions * 3}")
print(f"Incomplete conversations: {total_questions - complete_count}")
print(f"Turns that used thinking: {thinking_rows}")
print(f"Responses:                {OUTPUT_FILE}")
print(f"Checkpoint:               {CHECKPOINT_FILE}")
print(f"Error log:                {ERROR_FILE}")

if complete_count < total_questions:
    print(
        "\nSome conversations remain incomplete. "
        "(Run this cell again later to resume.)"
    )
else:
    print("\nAll conversations completed successfully.")
