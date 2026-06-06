import os
import sys
import time
import json
import pandas as pd
import anthropic

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from core.bandDescriptors import format_descriptors_for_prompt

KMP_DUPLICATE_LIB_OK = "TRUE"

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV     = "data/ielts_relabeled.csv"
OUTPUT_CSV    = "data/ielts_relabeled_v3.csv"
BATCH_ID_FILE = "data/batch_id_v3.txt"
MODEL         = "claude-sonnet-4-6"
MAX_ROWS      = None  # set to 20 for test run, None for full
# ─────────────────────────────────────────────────────────────────────────────

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def build_prompt(question: str, essay: str, overall: float) -> str:
    overall_int = round(float(overall))  # descriptors are integer-keyed
    descriptors = format_descriptors_for_prompt(overall_int)

    return f"""{descriptors}

---

You are an expert IELTS examiner. The overall band score for this essay is {overall}.

Score the essay on the four official IELTS Writing Task 2 criteria using the band descriptors above as your reference. Your scores MUST:
- Average exactly to {overall} when rounded to the nearest 0.5: round((TR + CC + LR + RA) / 4 * 2) / 2 == {overall}
- Reflect genuine differences between criteria — do NOT assign the same score to all four
- Be multiples of 0.5 between 1.0 and 9.0 inclusive
- Scores may be whole numbers or halves (e.g. 5.0, 5.5, 6.0) — do not restrict yourself to whole numbers only

Then write an Examiner_Comment of 4–6 sentences following these rules exactly:

COMMENT RULES:
1. Address all four criteria — TR, CC, LR, GRA — each at least once
2. Quote or closely reference specific words, phrases, or sentences from the essay to anchor each observation. Do not make generic claims.
3. Name errors precisely: e.g. "word formation error ('alimentation')", "run-on sentence in paragraph 2", "overuse of 'however'", "memorised phrase pattern"
4. State the scoring impact: e.g. "prevents a higher TR score", "limits CC to band 5"
5. Use professional examiner register — constructive, not encouraging or harsh
6. Do NOT mention the numerical sub-scores in the comment

COMMENT STYLE EXAMPLE (band 5.5 essay):
"A clear position is presented from the outset, supported by relevant ideas, but these would require further development to achieve a higher score; the response is also under-length. Information and ideas are generally arranged coherently with clear overall progression, though paragraphing is not always logical. A range of vocabulary is attempted, but errors in spelling, word choice, and word formation — for example 'alimentation' and 'inequivoque' — suggest first-language interference, though these do not make the response difficult to understand. There is a mix of sentence forms, but the frequency of grammatical errors prevents a higher band score."

Question: {question}

Essay:
{essay}

Return ONLY a JSON object, no explanation, no markdown:
{{
  "Task_Response": <band_score>,
  "Coherence_Cohesion": <band_score>,
  "Lexical_Resource": <band_score>,
  "Range_Accuracy": <band_score>,
  "Examiner_Comment": "<4-6 sentence comment>"
}}"""


def extract_json(text: str) -> dict:
    import re
    text = text.strip()
    # strip ```json ... ``` or ``` ... ``` wrappers
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text.strip())


# ── STEP 1: Submit batch ──────────────────────────────────────────────────────

def submit_batch(df: pd.DataFrame) -> str:
    requests = []
    for i, row in df.iterrows():
        overall = str(row.get("Overall", "")).strip()
        if not overall:
            print(f"  ⚠ Row {i}: missing Overall — skipping")
            continue
        prompt = build_prompt(
            question=str(row.get("Question", "")),
            essay=str(row.get("Essay", "")),
            overall=overall,
        )
        requests.append({
            "custom_id": str(i),
            "params": {
                "model": MODEL,
                "max_tokens": 600,
                "messages": [{"role": "user", "content": prompt}],
            },
        })

    print(f"Submitting batch of {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch submitted. ID: {batch.id}  Status: {batch.processing_status}")

    with open(BATCH_ID_FILE, "w") as f:
        f.write(batch.id)
    print(f"Batch ID saved to {BATCH_ID_FILE}")
    return batch.id


# ── STEP 2: Poll ──────────────────────────────────────────────────────────────

def poll_batch(batch_id: str) -> None:
    print(f"\nPolling batch {batch_id}...")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        done = counts.succeeded + counts.errored + counts.canceled
        total = done + counts.processing
        print(f"  [{time.strftime('%H:%M:%S')}] {batch.processing_status} | "
              f"done={done}/{total}  succeeded={counts.succeeded}  errored={counts.errored}")
        if batch.processing_status == "ended":
            break
        time.sleep(60)


# ── STEP 3: Download + merge ──────────────────────────────────────────────────

def download_results(batch_id: str, df: pd.DataFrame) -> pd.DataFrame:
    print(f"\nDownloading results for {batch_id}...")
    results = {}
    for result in client.messages.batches.results(batch_id):
        idx = int(result.custom_id)
        if result.result.type == "succeeded":
            raw = result.result.message.content[0].text.strip()
            try:
                results[idx] = extract_json(raw)
            except Exception as e:
                print(f"  ⚠ Row {idx}: JSON parse failed — {e} | raw: {raw[:200]}")
        else:
            print(f"  ⚠ Row {idx}: {result.result.type}")

    print(f"Parsed {len(results)}/{len(df)} results")

    label_cols = ["Task_Response", "Coherence_Cohesion", "Lexical_Resource",
                  "Range_Accuracy", "Examiner_Comment"]
    df_out = df.copy()
    for col in label_cols:
        df_out[col] = None

    for idx, labels in results.items():
        for col in label_cols:
            if col in labels:
                df_out.at[idx, col] = labels[col]

    failed = df_out[df_out["Task_Response"].isna()]
    if len(failed):
        print(f"⚠ {len(failed)} rows missing labels: {list(failed.index)}")

    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved → {OUTPUT_CSV}")
    return df_out


# ── STEP 4: Sanity check ──────────────────────────────────────────────────────

def sanity_check(df: pd.DataFrame) -> None:
    labeled = df[df["Task_Response"].notna()].copy()
    print(f"\n── Sanity Check ───────────────────────────────")
    print(f"Total: {len(df)}  Labeled: {len(labeled)}")
    for col in ["Task_Response", "Coherence_Cohesion", "Lexical_Resource", "Range_Accuracy"]:
        corr = labeled[col].astype(float).corr(labeled["Overall"].astype(float))
        print(f"  {col:30s} ↔ Overall: {corr:.3f}")
    print(f"\nSample (first 3):")
    cols = ["Overall", "Task_Response", "Coherence_Cohesion",
            "Lexical_Resource", "Range_Accuracy", "Examiner_Comment"]
    print(labeled[cols].head(3).to_string())
    print("───────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(INPUT_CSV)
    if MAX_ROWS:
        df = df.head(MAX_ROWS)
    print(f"Loaded {len(df)} essays from {INPUT_CSV}")

    if os.path.exists(BATCH_ID_FILE):
        with open(BATCH_ID_FILE) as f:
            batch_id = f.read().strip()
        print(f"Found existing batch ID: {batch_id} — skipping submission")
        print("Delete data/batch_id_v3.txt to resubmit")
    else:
        batch_id = submit_batch(df)

    poll_batch(batch_id)
    df_out = download_results(batch_id, df)
    sanity_check(df_out)


if __name__ == "__main__":
    main()