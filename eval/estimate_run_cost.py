"""
eval/estimate_run_cost.py

Run this BEFORE run_ragas.py to sanity-check call counts and rough token
volume. Does not call any API -- pure estimation from prompt string
lengths, so it's free and instant.

Token estimate uses a rough 4-chars-per-token heuristic (not the real
Groq/Gemini tokenizer) -- good enough for order-of-magnitude budgeting,
not for exact billing.

The one real unknown this can't measure without calling the APIs: how
many claims decompose_claims_prompt will actually produce per feedback
paragraph. This script lets you override that assumption
(--claims-per-arm) and shows the result at a few different values, so you
can see how sensitive the total is to that guess.
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CHARS_PER_TOKEN = 4  # rough heuristic, not a real tokenizer

# Representative generated-feedback length: production's MAX_TOKENS=350
# caps generator output, so use that as the assumed feedback length for
# every downstream judge-prompt-size estimate (decomposition input,
# faithfulness claim size).
ASSUMED_FEEDBACK_TOKENS = 350
ASSUMED_CLAIM_TOKENS = 20  # a single atomic claim, e.g. "Your grammar range shows limited variety."


def est_tokens(char_count):
    return max(1, char_count // CHARS_PER_TOKEN)


def load_test_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def estimate(test_cases, claims_per_arm, n_exemplars=3, avg_exemplar_chars=400):
    """
    Returns a dict of totals. avg_exemplar_chars is a rough guess for a
    single retrieved examiner comment's length -- adjust if your real
    exemplars run longer/shorter.
    """
    n_cases = len(test_cases)

    # --- per-case input sizes (rough, from actual essay/question text) ---
    avg_essay_chars = sum(len(c["essay"]) for c in test_cases) / n_cases
    avg_question_chars = sum(len(c["question"]) for c in test_cases) / n_cases

    exemplars_block_chars = n_exemplars * avg_exemplar_chars

    # --- Groq (generator) calls: 2 per case ---
    # prompt_with_exemplars: question + essay + scores + N exemplars + instructions
    gen_with_exemplars_input_chars = avg_essay_chars + avg_question_chars + exemplars_block_chars + 600
    # prompt_zero_shot: same minus exemplars block
    gen_zero_shot_input_chars = avg_essay_chars + avg_question_chars + 600

    groq_calls = n_cases * 2
    groq_input_tokens = n_cases * (
        est_tokens(gen_with_exemplars_input_chars) + est_tokens(gen_zero_shot_input_chars)
    )
    groq_output_tokens = groq_calls * ASSUMED_FEEDBACK_TOKENS  # MAX_TOKENS ceiling per call

    # --- Gemini (judge) calls ---
    # 1 decomposition call per arm: feedback text + instructions
    decomp_input_chars = (ASSUMED_FEEDBACK_TOKENS * CHARS_PER_TOKEN) + 500
    decomp_calls = n_cases * 2  # 2 arms
    decomp_input_tokens = decomp_calls * est_tokens(decomp_input_chars)
    decomp_output_tokens = decomp_calls * (claims_per_arm * ASSUMED_CLAIM_TOKENS)  # JSON array of claims

    # 1 faithfulness call per claim per arm: question+essay+scores+claim+instructions
    faith_calls = n_cases * 2 * claims_per_arm
    faith_input_chars = avg_essay_chars + avg_question_chars + 400 + (ASSUMED_CLAIM_TOKENS * CHARS_PER_TOKEN)
    faith_input_tokens = faith_calls * est_tokens(faith_input_chars)
    faith_output_tokens = faith_calls * 40  # small {"score","reasoning"} JSON

    # 1 context_precision call per case: question+essay+3 exemplars+instructions
    precision_calls = n_cases
    precision_input_chars = avg_essay_chars + avg_question_chars + exemplars_block_chars + 800
    precision_input_tokens = precision_calls * est_tokens(precision_input_chars)
    precision_output_tokens = precision_calls * 120  # 3 keyed {"score","reasoning"} objects

    gemini_calls = decomp_calls + faith_calls + precision_calls
    gemini_input_tokens = decomp_input_tokens + faith_input_tokens + precision_input_tokens
    gemini_output_tokens = decomp_output_tokens + faith_output_tokens + precision_output_tokens

    return {
        "n_cases": n_cases,
        "claims_per_arm_assumed": claims_per_arm,
        "groq_calls": groq_calls,
        "groq_input_tokens": groq_input_tokens,
        "groq_output_tokens": groq_output_tokens,
        "gemini_calls": gemini_calls,
        "gemini_calls_breakdown": {
            "decomposition": decomp_calls,
            "per_claim_faithfulness": faith_calls,
            "context_precision": precision_calls,
        },
        "gemini_input_tokens": gemini_input_tokens,
        "gemini_output_tokens": gemini_output_tokens,
    }


def print_report(result, rpm_assumed=12):
    print(f"=== estimate for n_cases={result['n_cases']}, "
          f"assumed claims/arm={result['claims_per_arm_assumed']} ===\n")

    print("Groq (generator):")
    print(f"  calls:  {result['groq_calls']}")
    print(f"  tokens: ~{result['groq_input_tokens']:,} in / ~{result['groq_output_tokens']:,} out\n")

    print("Gemini (judge):")
    print(f"  calls:  {result['gemini_calls']}  (breakdown: {result['gemini_calls_breakdown']})")
    print(f"  tokens: ~{result['gemini_input_tokens']:,} in / ~{result['gemini_output_tokens']:,} out")

    est_minutes = result["gemini_calls"] / rpm_assumed
    print(f"  at an assumed {rpm_assumed} RPM ceiling: ~{est_minutes:.0f} min minimum, "
          f"just from throttling (verify your real RPM in AI Studio -- this is a guess)\n")

    print("NOTE: claims-per-arm is a guess, not measured. Re-run with --claims-per-arm "
          "at a few values (e.g. 3, 5, 8) to see how sensitive the Gemini call count is.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-questions", default=str(SCRIPT_DIR / "test_questions.json"),
        help="Path to test_questions.json (default: eval/test_questions.json)",
    )
    parser.add_argument(
        "--claims-per-arm", type=int, default=5,
        help="Assumed number of atomic claims per feedback paragraph (default: 5, a guess)",
    )
    parser.add_argument(
        "--rpm", type=int, default=12,
        help="Assumed Gemini requests-per-minute ceiling for the throttling estimate (default: 12, a guess -- check AI Studio for your real limit)",
    )
    args = parser.parse_args()

    path = Path(args.test_questions)
    if not path.exists():
        print(f"ERROR: {path} not found. Run build_test_set.py first, or pass --test-questions.", file=sys.stderr)
        sys.exit(1)

    test_cases = load_test_cases(path)
    result = estimate(test_cases, claims_per_arm=args.claims_per_arm)
    print_report(result, rpm_assumed=args.rpm)


if __name__ == "__main__":
    main()