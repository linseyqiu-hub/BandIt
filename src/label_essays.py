import os
import time
import json
import pandas as pd
import anthropic

KMP_DUPLICATE_LIB_OK = "TRUE"

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_CSV   = "data/ielts_labeled.csv"       # your existing labeled CSV
OUTPUT_CSV  = "data/ielts_relabeledNew.csv"     # new output with Sonnet labels + feedback
BATCH_ID_FILE = "data/batch_id.txt"          # saves batch ID so you can poll later
MODEL       = "claude-sonnet-4-6"
MAX_ROWS    = 50
# ─────────────────────────────────────────────────────────────────────────────

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

PROMPT_TEMPLATE = """You are an expert IELTS examiner with 10+ years of experience.

The overall band score for this essay is {overall}.

Score the essay on all four official IELTS Writing Task 2 criteria:
- Task Response: how fully the question is addressed, position developed, ideas extended and supported
- Coherence and Cohesion: logical organisation, paragraphing, cohesive devices, referencing
- Lexical Resource: range and accuracy of vocabulary, word choice, spelling
- Grammatical Range and Accuracy: range of structures, error frequency and impact

Your scores MUST:
- Average exactly to {overall} when rounded to the nearest 0.5. That is: round((TR + CC + LR + RA) / 4 * 2) / 2 == {overall}
- Reflect genuine differences between criteria — do NOT assign the same score to all four. A writer can have strong vocabulary but weak argument structure; strong organisation but frequent grammar errors. Spread the scores.
- Be multiples of 0.5 between 1.0 and 9.0 (inclusive). Do NOT use 0.0 or 9.5, etc.

Then write 3–4 sentences of examiner feedback. The feedback must:
- Address all four criteria briefly
- Be specific to this essay (not generic advice)
- Use examiner register (professional, constructive)
- Not mention the numerical scores

Question: {question}

Essay:
{essay}

Return ONLY a JSON object, no explanation, no markdown, exactly this format:
{{
  "Task_Response": <band_score>,
  "Coherence_Cohesion": <band_score>,
  "Lexical_Resource": <band_score>,
  "Range_Accuracy": <band_score>,
  "Examiner_Comment": "<3-4 sentence feedback string>"
}}"""


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ── STEP 1: Submit batch ──────────────────────────────────────────────────────

def submit_batch(df: pd.DataFrame) -> str:
    """Build batch requests and submit. Returns batch_id."""
    requests = []
    for i, row in df.iterrows():
        prompt = PROMPT_TEMPLATE.format(
            question=str(row.get("Question", "")),
            essay=str(row.get("Essay", "")),
            overall=str(row.get("Overall", "")).strip() or "unknown"
        )
        requests.append({
            "custom_id": str(i),
            "params": {
                "model": MODEL,
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt}]
            }
        })

    print(f"Submitting batch of {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch submitted. ID: {batch.id}")
    print(f"Status: {batch.processing_status}")

    # Save batch ID so you can resume polling without resubmitting
    with open(BATCH_ID_FILE, "w") as f:
        f.write(batch.id)
    print(f"Batch ID saved to {BATCH_ID_FILE}")

    return batch.id


# ── STEP 2: Poll until done ───────────────────────────────────────────────────

def poll_batch(batch_id: str) -> None:
    """Poll every 60s until batch completes."""
    print(f"\nPolling batch {batch_id}...")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  [{time.strftime('%H:%M:%S')}] Status: {batch.processing_status} | "
              f"done={counts.succeeded + counts.errored + counts.canceled} / "
              f"total={counts.processing + counts.succeeded + counts.errored + counts.canceled}")

        if batch.processing_status == "ended":
            print(f"\nBatch ended. succeeded={counts.succeeded}, errored={counts.errored}")
            break

        time.sleep(60)


# ── STEP 3: Download and merge results ────────────────────────────────────────

def download_results(batch_id: str, df: pd.DataFrame) -> pd.DataFrame:
    """Stream batch results and merge back into df."""
    print(f"\nDownloading results for batch {batch_id}...")

    results = {}
    for result in client.messages.batches.results(batch_id):
        idx = int(result.custom_id)
        if result.result.type == "succeeded":
            raw = result.result.message.content[0].text.strip()
            try:
                parsed = extract_json(raw)
                results[idx] = parsed
            except Exception as e:
                print(f"  ⚠ Row {idx}: JSON parse failed — {e}")
                print(f"    Raw: {raw[:200]}")
        else:
            print(f"  ⚠ Row {idx}: {result.result.type}")

    print(f"Parsed {len(results)} / {len(df)} results successfully")

    # Merge back
    label_cols = ["Task_Response", "Coherence_Cohesion", "Lexical_Resource",
                  "Range_Accuracy", "Examiner_Comment"]

    df_out = df.copy()
    for col in label_cols:
        df_out[col] = None

    for idx, labels in results.items():
        for col in label_cols:
            if col in labels:
                df_out.at[idx, col] = labels[col]

    # Report any rows that failed
    failed = df_out[df_out["Task_Response"].isna()]
    if len(failed) > 0:
        print(f"\n⚠ {len(failed)} rows have no labels (API errors). Row indices: {list(failed.index)}")

    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")
    return df_out


# ── STEP 4: Quick sanity check ────────────────────────────────────────────────

def sanity_check(df: pd.DataFrame) -> None:
    labeled = df[df["Task_Response"].notna()]
    print(f"\n── Sanity Check ──────────────────────────────")
    print(f"Total rows:   {len(df)}")
    print(f"Labeled rows: {len(labeled)}")

    for col in ["Task_Response", "Coherence_Cohesion", "Lexical_Resource", "Range_Accuracy"]:
        corr = labeled[col].astype(float).corr(labeled["Overall"].astype(float))
        print(f"  {col} ↔ Overall correlation: {corr:.3f}")

    print(f"\nSample (first 3 rows):")
    cols = ["Overall", "Task_Response", "Coherence_Cohesion",
            "Lexical_Resource", "Range_Accuracy", "Examiner_Comment"]
    print(labeled[cols].head(3).to_string())
    print("──────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(INPUT_CSV)
    if MAX_ROWS:
        df = df.head(MAX_ROWS)
    print(f"Loaded {len(df)} essays from {INPUT_CSV}")

    # Check if a batch was already submitted
    if os.path.exists(BATCH_ID_FILE):
        with open(BATCH_ID_FILE) as f:
            batch_id = f.read().strip()
        print(f"Found existing batch ID: {batch_id}")
        print("Skipping submission. Delete data/batch_id.txt to resubmit.")
    else:
        batch_id = submit_batch(df)

    # Poll until done
    poll_batch(batch_id)

    # Download and merge
    df_out = download_results(batch_id, df)

    # Sanity check — check correlations dropped from 0.96-0.99
    sanity_check(df_out)


if __name__ == "__main__":
    main()