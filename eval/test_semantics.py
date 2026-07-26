"""
eval/test_semantics.py

Run this in your eval/ folder: python test_semantics.py

Checks BEHAVIOR, not file contents -- imports your actual run_ragas.py and
calls its real functions with controlled fake inputs, asserting the
semantic contracts we've locked in across this whole debugging session.
No real Groq/Gemini API calls are made (those would cost quota/time) --
only the pure-logic functions (parsing, prompt-building, shape checks,
checkpoint round-trip) are exercised. retrieve_top3's actual Chroma wiring
still needs a real 1-2 case smoke run to fully verify, since mocking
Chroma's internals here wouldn't prove anything about your real database.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_ragas as rr  # your actual file -- this IS the thing being tested

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


print("=" * 70)
print("SEMANTIC TEST SUITE -- exercising real logic, not comparing bytes")
print("=" * 70)
print()

# ---------------------------------------------------------------------------
# 1. case_scores -- correct fields, correct casing
# ---------------------------------------------------------------------------
fake_case = {
    "id": "999", "question": "Q?", "essay": "E.", "band_bin": "developing",
    "overall": 5.5, "task_response": 5.0, "coherence_cohesion": 6.0,
    "lexical_resource": 5.5, "grammatical_range_accuracy": 5.5,
}
s = rr.case_scores(fake_case)
check("case_scores: all 5 fields readable via attribute access",
      s.overall == 5.5 and s.task_response == 5.0 and s.coherence_cohesion == 6.0
      and s.lexical_resource == 5.5 and s.grammatical_range_accuracy == 5.5)

# ---------------------------------------------------------------------------
# 2. rrf_fusion -- THE bug from today. Exemplars must have "text"/"overall",
#    not the old broken "examiner_comment" shape.
# ---------------------------------------------------------------------------
fake_overall_map = {"e1": 6.0, "e2": 5.5}
exemplars = rr.rrf_fusion(
    essay_ids=["e1", "e2"], question_ids=["e1"],
    essay_docs=["comment one", "comment two"], question_docs=["comment one"],
    overall_map=fake_overall_map,
)
check("rrf_fusion: each exemplar has 'text' key (not 'examiner_comment')",
      all("text" in ex for ex in exemplars) and all("examiner_comment" not in ex for ex in exemplars))
check("rrf_fusion: each exemplar has 'overall' key with real values from the map",
      all("overall" in ex for ex in exemplars) and exemplars[0]["overall"] in (6.0, 5.5))

# Prove build_prompt-style consumption doesn't KeyError (the actual crash from today)
try:
    _ = "\n".join(f"Overall: {e['overall']} - {e['text']}" for e in exemplars)
    check("Simulated build_prompt few-shot read (comment['overall']/['text']) succeeds", True)
except KeyError as e:
    check(f"Simulated build_prompt few-shot read succeeds (KeyError: {e})", False)

# ---------------------------------------------------------------------------
# 3. prompt_zero_shot -- exemplar section fully stripped, not left empty
# ---------------------------------------------------------------------------
try:
    _, zero_shot_prompt = rr.prompt_zero_shot(fake_case)
    check("prompt_zero_shot: '## Examiner Reference Comments' section removed",
          "## Examiner Reference Comments" not in zero_shot_prompt)
    check("prompt_zero_shot: '## Task' section still present",
          "## Task" in zero_shot_prompt)
except Exception as e:
    check(f"prompt_zero_shot runs without error (got {type(e).__name__}: {e})", False)

# ---------------------------------------------------------------------------
# 4. faithfulness_judge_prompt -- context has scores, JUDGE_OUTPUT_INSTRUCTION
#    demands strict binary (0 or 1), not a continuous float
# ---------------------------------------------------------------------------
fp = rr.faithfulness_judge_prompt(fake_case, "the essay is well organized")
check("faithfulness_judge_prompt: includes the four criterion scores",
      "5.0" in fp and "6.0" in fp)
check("JUDGE_OUTPUT_INSTRUCTION demands strict binary, not a continuous float",
      '"score": 0 or 1' in rr.JUDGE_OUTPUT_INSTRUCTION
      and "float between 0.0 and 1.0" not in rr.JUDGE_OUTPUT_INSTRUCTION)

# ---------------------------------------------------------------------------
# 5. parse_score -- binary enforcement is REAL, not just documented
# ---------------------------------------------------------------------------
v, ok = rr.parse_score('{"score": 1.0, "reasoning": "x"}')
check("parse_score accepts 1.0", ok and v == 1.0)
v, ok = rr.parse_score('{"score": 0.7, "reasoning": "x"}')
check("parse_score REJECTS 0.7 (partial credit not allowed)", not ok)

# ---------------------------------------------------------------------------
# 6. parse_precision_verdicts + precision_composite
# ---------------------------------------------------------------------------
raw = '{"relevant": {"score": 1, "reasoning": "x"}, "independent": {"score": 1, "reasoning": "x"}, "useful": {"score": 0, "reasoning": "x"}}'
verdicts, ok = rr.parse_precision_verdicts(raw)
check("parse_precision_verdicts parses all 3 keys", ok and set(verdicts) == {"relevant", "independent", "useful"})
composite = rr.precision_composite(verdicts)
check("precision_composite = mean of 3 (2/3 here)", abs(composite - (2 / 3)) < 1e-9)

# missing key -> all-or-nothing failure
bad_raw = '{"relevant": {"score": 1, "reasoning": "x"}}'
_, ok = rr.parse_precision_verdicts(bad_raw)
check("parse_precision_verdicts REJECTS a partial (missing-key) response", not ok)

# ---------------------------------------------------------------------------
# 7. parse_claims -- decomposition parser
# ---------------------------------------------------------------------------
claims, ok = rr.parse_claims('["claim one", "claim two"]')
check("parse_claims parses a valid array", ok and claims == ["claim one", "claim two"])
_, ok = rr.parse_claims('not json')
check("parse_claims REJECTS invalid JSON", not ok)

# ---------------------------------------------------------------------------
# 8. Checkpoint round-trip -- real file I/O, temp path (won't touch your
#    real checkpoint)
# ---------------------------------------------------------------------------
tmp_checkpoint = tempfile.mktemp(suffix=".jsonl")
rr.append_checkpoint(tmp_checkpoint, {"id": "1", "band_bin": "developing"})
rr.append_checkpoint(tmp_checkpoint, {"id": "2", "band_bin": "expert"})
results, completed_ids = rr.load_checkpoint(tmp_checkpoint)
check("checkpoint round-trip: 2 rows written, 2 rows read back", len(results) == 2)
check("checkpoint round-trip: completed_ids correctly extracted", completed_ids == {"1", "2"})
os.remove(tmp_checkpoint)

# ---------------------------------------------------------------------------
# 9. Rate limiter config -- real confirmed numbers, not placeholders
# ---------------------------------------------------------------------------
check("gemini_limiter.rpm == 5 (real free-tier number)", rr.gemini_limiter.rpm == 5)
check("groq_limiter.rpm == 30 and .tpm == 12000 (real numbers, TPM tracked)",
      rr.groq_limiter.rpm == 30 and rr.groq_limiter.tpm == 12_000)

# ---------------------------------------------------------------------------
# 10. _safe wrappers actually exist and are callable (structural check only
#     -- NOT invoking them, that would burn real API calls)
# ---------------------------------------------------------------------------
check("judge_gemini_safe is defined and callable", callable(rr.judge_gemini_safe))
check("generate_groq_safe is defined and callable", callable(rr.generate_groq_safe))

# ---------------------------------------------------------------------------
# 11. generate_groq builds a VALID message shape from the (system, user)
#     tuple -- without making a real API call. This is the bug that
#     slipped through earlier: generate_groq_safe passes the full tuple,
#     and if generate_groq doesn't unpack it into two separate string
#     messages, the tuple gets serialized as a JSON array of two strings,
#     which Groq's API rejects with a 400. Intercept the actual client
#     call to inspect the exact `messages` payload that would be sent.
# ---------------------------------------------------------------------------
captured_messages = {}


class _FakeGroqResponse:
    class _Choice:
        class _Message:
            content = "fake feedback"
        message = _Message()
    choices = [_Choice()]


def _capture_create(model, messages, temperature):
    captured_messages["value"] = messages
    return _FakeGroqResponse()


_real_create = rr.groq_client.chat.completions.create
rr.groq_client.chat.completions.create = _capture_create
try:
    rr.generate_groq(("system text", "user text"))
finally:
    rr.groq_client.chat.completions.create = _real_create

sent = captured_messages.get("value")
check("generate_groq: messages is a list of exactly 2 entries",
      isinstance(sent, list) and len(sent) == 2)
check("generate_groq: every message's 'content' is a plain string (not a tuple/list)",
      sent is not None and all(isinstance(m.get("content"), str) for m in sent))
check("generate_groq: system prompt actually reaches a 'system' role message",
      sent is not None and any(m.get("role") == "system" and m.get("content") == "system text" for m in sent))
check("generate_groq: user prompt actually reaches a 'user' role message",
      sent is not None and any(m.get("role") == "user" and m.get("content") == "user text" for m in sent))

print()
print("=" * 70)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(" -", f)
else:
    print("ALL CHECKS PASSED")
print("=" * 70)
print()
print("NOTE: this does not verify retrieve_top3's real Chroma wiring, or that")
print("generate_groq/judge_gemini actually work against live APIs. Run a real")
print("1-2 case slice through run() to confirm those.")