import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from core.bandDescriptors import format_descriptors_for_prompt

# ── Config (must match label_essays_v3.py) ────────────────────────────────────
INPUT_CSV  = "data/ielts_relabeled.csv"
MAX_ROWS   = None
MAX_OUTPUT_TOKENS = 600  # per request, as set in label_essays_v3.py

# Sonnet 4.6 batch pricing (50% discount off standard)
# Standard: $3.00 input / $15.00 output per 1M tokens
# Batch:    $1.50 input / $7.50  output per 1M tokens
INPUT_PRICE_PER_1M  = 1.50
OUTPUT_PRICE_PER_1M = 7.50
# ─────────────────────────────────────────────────────────────────────────────


def count_tokens_simple(text: str) -> int:
    """
    Rough token estimate: ~4 chars per token (OpenAI/Anthropic rule of thumb).
    No tokenizer dependency — good enough for cost estimation.
    """
    return len(text) // 4


def build_prompt(question: str, essay: str, overall: float) -> str:
    overall_int = round(float(overall))
    descriptors = format_descriptors_for_prompt(overall_int)
    return f"""{descriptors}

---

You are an expert IELTS examiner. The overall band score for this essay is {overall}.

Score the essay on the four official IELTS Writing Task 2 criteria using the band descriptors above as your reference. Your scores MUST:
- Average exactly to {overall} when rounded to the nearest 0.5: round((TR + CC + LR + RA) / 4 * 2) / 2 == {overall}
- Reflect genuine differences between criteria — do NOT assign the same score to all four
- Be multiples of 0.5 between 1.0 and 9.0 inclusive

Then write an Examiner_Comment of 4–6 sentences following these rules exactly:

COMMENT RULES:
1. Address all four criteria — TR, CC, LR, GRA — each at least once
2. Quote or closely reference specific words, phrases, or sentences from the essay to anchor each observation. Do not make generic claims.
3. Name errors precisely: e.g. "word formation error ('alimentation')", "run-on sentence in paragraph 2", "overuse of 'however'", "memorised phrase pattern"
4. State the scoring impact: e.g. "prevents a higher TR score", "limits CC to band 5", "does not impede communication"
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


def main():
    df = pd.read_csv(INPUT_CSV)
    if MAX_ROWS:
        df = df.head(MAX_ROWS)

    total_rows = len(df)
    total_input_tokens = 0
    skipped = 0
    token_counts = []

    for i, row in df.iterrows():
        overall = str(row.get("Overall", "")).strip()
        if not overall:
            skipped += 1
            continue
        prompt = build_prompt(
            question=str(row.get("Question", "")),
            essay=str(row.get("Essay", "")),
            overall=overall,
        )
        t = count_tokens_simple(prompt)
        token_counts.append(t)
        total_input_tokens += t

    total_output_tokens = (total_rows - skipped) * MAX_OUTPUT_TOKENS

    input_cost  = (total_input_tokens  / 1_000_000) * INPUT_PRICE_PER_1M
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
    total_cost  = input_cost + output_cost

    print(f"── Cost Estimate (Batch API, Sonnet 4.6) ──────────────────")
    print(f"Essays:               {total_rows - skipped:>8}  ({skipped} skipped — missing Overall)")
    print(f"Input tokens total:   {total_input_tokens:>8,}")
    print(f"  avg per prompt:     {total_input_tokens // max(len(token_counts),1):>8,}")
    print(f"  min / max:          {min(token_counts):>8,} / {max(token_counts):,}")
    print(f"Output tokens total:  {total_output_tokens:>8,}  ({MAX_OUTPUT_TOKENS} max per essay)")
    print(f"──────────────────────────────────────────────────────────")
    print(f"Input cost:           ${input_cost:>8.4f}  (${INPUT_PRICE_PER_1M}/1M)")
    print(f"Output cost:          ${output_cost:>8.4f}  (${OUTPUT_PRICE_PER_1M}/1M)")
    print(f"Total estimated:      ${total_cost:>8.4f}")
    print(f"──────────────────────────────────────────────────────────")
    print(f"Note: input tokens estimated at 4 chars/token.")
    print(f"Actual may be ~10% higher. Output cost is worst-case (all {MAX_OUTPUT_TOKENS} tokens used).")


if __name__ == "__main__":
    main()