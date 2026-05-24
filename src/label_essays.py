import os
import time
import json
import pandas as pd
import anthropic

KMP_DUPLICATE_LIB_OK = "TRUE"

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_CSV  = "data/ielts_writing_dataset.csv"
OUTPUT_CSV = "data/ielts_labeled.csv"
SLEEP_SEC  = 0.5          # pause between calls to avoid rate limits
MAX_ROWS   = None 
# ────────────────────────────────────────────────────────────────────────────

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

PROMPT_TEMPLATE = """You are an expert IELTS examiner with 10+ years of experience.

The overall band score for this essay is {overall}.

Score the essay on all four official IELTS criteria. Your scores MUST:
- Be consistent with the overall band of {overall}
- Reflect genuine differences between criteria (do NOT give the same score to all)
- Use the full band range appropriately: a band 5 essay should have sub-scores around 4.5-5.5, a band 8 essay around 7.5-8.5
- Be multiples of 0.5 between 1.0 and 9.0

Question: {question}

Essay:
{essay}

Return ONLY a JSON object, no explanation, no markdown, exactly this format:
{{
  "Task_Response": <band_score>,
  "Coherence_Cohesion": <band_score>,
  "Lexical_Resource": <band_score>,
  "Range_Accuracy": <band_score>
}}"""


def extract_json(text: str) -> dict:
    """Extract JSON from text, handling markdown code blocks."""
    text = text.strip()
    # Remove markdown code block if present
    if text.startswith("```"):
        text = text.split("```")[1]  # Get content between first and second ```
        if text.startswith("json"):
            text = text[4:]  # Remove 'json' language marker
    return json.loads(text.strip())


def label_essay(question: str, essay: str, overall: str, retries: int = 5) -> dict | None:
    models = ["claude-haiku-4-5", "claude-sonnet-4"]
    
    for model in models:
        for attempt in range(retries):
            try:
                message = client.messages.create(
                    model=model,
                    max_tokens=300,
                    messages=[{
                        "role": "user",
                        "content": PROMPT_TEMPLATE.format(question=question, essay=essay, overall=overall)
                    }]
                )
                raw = message.content[0].text.strip()
                if not raw:
                    raise ValueError("Empty response")
                return extract_json(raw)
            except Exception as e:
                wait = min(2 ** attempt * 5, 60)
                print(f"  ⚠ [{model}] Attempt {attempt+1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    print(f"  Waiting {wait}s...")
                    time.sleep(wait)
        print(f"  ↪ Switching from {model} to fallback...")
    
    return None
    


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} essays")

    # Resume support — skip already labeled rows
    if os.path.exists(OUTPUT_CSV):
        done = pd.read_csv(OUTPUT_CSV)
        start_idx = done["Task_Response"].notna().sum()
        print(f"Resuming from row {start_idx}")
    else:
        done = pd.DataFrame()
        start_idx = 0

    for i, row in df.iterrows():
        if i < start_idx:
            continue
        if MAX_ROWS and (i - start_idx) >= MAX_ROWS:
            print(f"\n⏹ Reached MAX_ROWS={MAX_ROWS}, stopping.")
            break

        print(f"[{i+1}/{len(df)}] Labeling essay...")

        result = label_essay(
            question=str(row.get("Question", "")),
            essay=str(row.get("Essay", "")),
            overall=str(row.get("Overall", "")).strip() or "unknown"
        )

        if result:
            row_out = row.to_dict()
            row_out.update(result)
            done = pd.concat([done, pd.DataFrame([row_out])], ignore_index=True)

            # Save after every row (safe resuming)
            done.to_csv(OUTPUT_CSV, index=False)
            print(f"   TR:{result['Task_Response']} CC:{result['Coherence_Cohesion']} "
                    f"LR:{result['Lexical_Resource']} GR:{result['Range_Accuracy']}")
        else:
            print(f"   Skipped row {i}")

        time.sleep(SLEEP_SEC)

    print(f"\n Done! Labeled CSV saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()