"""
eval/run_ragas.py

Runs the ungated baseline BandIt RAGAS eval.

Order of operations:
    0. load test_questions.json
    1. preflight: validate + autofix held_out consistency across both
       ChromaDB collections (see preflight.py) -> known_bad_ids
    2. per case:
       - retrieve top-3 exemplars (RRF), held_out-filtered
       - leak check against known_bad_ids; on leak, retry once with
         explicit exclude_ids before giving up
       - generate variant A (with exemplars) and variant B (zero-shot,
         for the ablation) via Llama 3.3 70B / Groq
       - judge faithfulness (essay+question only, exemplars excluded)
         and context_precision (custom style-relevance prompt) via
         Gemini 2.5 Flash
       - parse-check every judge response; log and skip metric on failure
    3. aggregate means/std overall and per band_bin, store ablation pairs
    4. write eval/results_baseline.json (ungated -- no threshold check yet)

FIX APPLIED (see conversation): rrf_fusion/retrieve_top3 previously
returned each exemplar as {"id", "examiner_comment"} only. build_prompt
(src/services/feedback.py) reads comment['overall']/comment['text'], and
context_precision_judge_prompt reads ex['text'] -- neither of which the
old shape provided, so EVERY case crashed with KeyError: 'overall' inside
build_prompt the moment retrieval returned anything. retrieve_top3 now
also fetches Chroma metadatas and builds an overall map (mirroring
production's feedback_service.py pattern), and rrf_fusion returns
{"id", "text", "overall"} per exemplar -- confirmed by reproducing the
exact KeyError with the old shape and confirming it's gone with the new
one.
"""

import json
import os
import sys
import statistics

from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
from types import SimpleNamespace

from common import log_failure, log_fix, count_jsonl
from preflight import run_preflight
from rate_limit import RateLimiter, with_backoff
from src.services.feedback import build_prompt, SYSTEM_PROMPT, get_descriptor

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.config import CHROMA_DB_PATH  # noqa: E402

TEST_QUESTIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_questions.json")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_baseline.json")

RRF_K = 60
FINAL_N = 3
OVERFETCH_N = 10

# Must match ingest.py's EMBEDDING_MODEL exactly -- the collections were
# populated with these embeddings, and embedding_function=None means Chroma
# will not compute embeddings for us at query time. We have to reproduce the
# same embedding space ourselves.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------
# Groq:  https://console.groq.com  -> API Keys -> Create Key
#        export GROQ_API_KEY=...   (or set in your shell / .env)
#        pip install openai --break-system-packages   (Groq is OpenAI-SDK-compatible)
#
# JUDGE MODEL SWAP (was Gemini 2.5 Flash via Google AI Studio):
# Google AI Studio was returning persistent 429 RESOURCE_EXHAUSTED regardless
# of quota state -- a known Google-side bug where projects stay pinned to a
# free_tier_requests limit of 0. Not a code issue and not self-resolving, so
# the judge moved off Google entirely.
#
# Judge is now openai/gpt-oss-120b, served by the SAME Groq client as the
# generator. This preserves the three-distinct-labs property from the design
# doc's section 2: exemplar data (Anthropic/Claude), generator (Meta/Llama),
# judge (OpenAI/gpt-oss). No self-preference overlap. One API key, one SDK,
# no card.

def _require_env(name, setup_hint):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing environment variable {name}. {setup_hint}\n"
            f"Set it via: export {name}=your_key_here"
        )
    return value


groq_client = OpenAI(
    api_key=_require_env(
        "GROQ_API_KEY",
        "Get one at https://console.groq.com -> API Keys -> Create Key.",
    ),
    base_url="https://api.groq.com/openai/v1",
)

GENERATOR_MODEL = "llama-3.3-70b-versatile"
JUDGE_MODEL = "openai/gpt-oss-120b"


def get_collections():
    """Matches ingest.py's client setup exactly."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    essays_collection = client.get_collection("essays", embedding_function=None)
    questions_collection = client.get_collection("questions", embedding_function=None)
    return essays_collection, questions_collection


_embedding_model = None


def get_embedding_model():
    """Lazy-loaded singleton -- avoid loading the model if it's never needed
    (e.g. if a run fails before retrieval starts)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def generate_groq(prompt_tuple, temperature=0.7):
    """
    prompt_tuple is (system_prompt, user_prompt), as returned by
    prompt_with_exemplars()/prompt_zero_shot(). FIXED: previously took a
    single string and stuffed it into one user message with no system
    role at all -- when generate_groq_safe started passing the full
    (system, user) tuple through, that tuple got serialized as a JSON
    array of two strings, which the API rejected with a 400 (neither a
    valid string content nor a valid list-of-objects content). Now
    unpacks the tuple into a proper 2-message system+user request, which
    the OpenAI-compatible /chat/completions endpoint genuinely supports.
    """
    system, user = prompt_tuple
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def judge_llm(prompt):
    """
    Judge call -- same Groq client as the generator, different model.

    Judge prompts are single strings (unlike the generator's
    (system, user) tuple), so this sends one user message. temperature=0.0
    is a deliberate change from the old Gemini call, which set no
    temperature at all and therefore sampled at the provider default:
    scoring should be as close to deterministic as the model allows, so
    repeat runs are comparable and JSON output is less likely to drift.
    """
    resp = groq_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Retrieval -- RRF fusion, wrapped with Chroma queries + exclude_ids
#
# FIXED: now fetches metadatas alongside documents, builds an id->overall
# map, and rrf_fusion returns {"id", "text", "overall"} per exemplar --
# this is the shape build_prompt and context_precision_judge_prompt
# actually need. Previously returned {"id", "examiner_comment"} only,
# which crashed every single case with KeyError: 'overall' inside
# build_prompt.
# ---------------------------------------------------------------------------

def rrf_fusion(essay_ids, question_ids, essay_docs, question_docs, overall_map, k=RRF_K, n=FINAL_N):
    """
    Fuse two ranked lists via Reciprocal Rank Fusion.

    Returns each exemplar as {"id", "text", "overall"} -- "id" for the
    leak-check, "text"/"overall" because that's what build_prompt (in
    src/services/feedback.py) and context_precision_judge_prompt both
    actually read.
    """
    scores: dict[str, float] = {}
    docs: dict[str, str] = {}

    for rank, (id_, doc) in enumerate(zip(essay_ids, essay_docs)):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)
        docs[id_] = doc

    for rank, (id_, doc) in enumerate(zip(question_ids, question_docs)):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)
        docs[id_] = doc

    ranked = sorted(scores.keys(), key=lambda id_: scores[id_], reverse=True)
    return [
        {"id": id_, "text": docs[id_], "overall": overall_map.get(id_)}
        for id_ in ranked[:n]
    ]


def retrieve_top3(question, essay, exclude_ids=None):
    """
    Queries both Chroma collections (held_out-filtered), overfetches so an
    exclude_ids retry still has enough candidates left, then fuses via RRF.

    IMPORTANT: collections were created with embedding_function=None (see
    ingest.py), so Chroma cannot embed query_texts itself. We compute query
    embeddings explicitly with the same model ingest.py used, and pass
    query_embeddings= instead.

    Now also requests metadatas (previously documents/ids only) and builds
    an id -> overall map from it, mirroring production's
    feedback_service.py pattern (overall_map[id_] = meta["overall"]) --
    this is what rrf_fusion needs to attach a real "overall" score to each
    exemplar.
    """
    exclude_ids = exclude_ids or set()
    where_filter = {"held_out": {"$ne": True}}
    model = get_embedding_model()

    essay_embedding = model.encode([essay]).tolist()
    question_embedding = model.encode([question]).tolist()

    essay_results = essays_collection.query(
        query_embeddings=essay_embedding, n_results=OVERFETCH_N, where=where_filter,
        include=["documents", "metadatas"],
    )
    question_results = questions_collection.query(
        query_embeddings=question_embedding, n_results=OVERFETCH_N, where=where_filter,
        include=["documents", "metadatas"],
    )

    # documents ARE the examiner_comment (per ingest.py) -- no need to pull
    # it out of metadata separately.
    essay_ids = essay_results["ids"][0]
    essay_docs = essay_results["documents"][0]
    essay_metas = essay_results["metadatas"][0]
    question_ids = question_results["ids"][0]
    question_docs = question_results["documents"][0]
    question_metas = question_results["metadatas"][0]

    # build id -> overall BEFORE exclusion filtering, so the map covers
    # everything that was originally retrieved
    overall_map: dict[str, float] = {}
    for id_, meta in zip(essay_ids, essay_metas):
        overall_map[id_] = meta.get("overall") if meta else None
    for id_, meta in zip(question_ids, question_metas):
        overall_map[id_] = meta.get("overall") if meta else None

    if exclude_ids:
        essay_pairs = [(i, d) for i, d in zip(essay_ids, essay_docs) if i not in exclude_ids]
        question_pairs = [(i, d) for i, d in zip(question_ids, question_docs) if i not in exclude_ids]
        essay_ids, essay_docs = (list(t) for t in zip(*essay_pairs)) if essay_pairs else ([], [])
        question_ids, question_docs = (list(t) for t in zip(*question_pairs)) if question_pairs else ([], [])

    return rrf_fusion(essay_ids, question_ids, essay_docs, question_docs, overall_map)


# ---------------------------------------------------------------------------
# case is a plain dict loaded from test_questions.json, e.g.:
#   {
#     "id": "254", "question": "...", "essay": "...",
#     "reference": "...", "overall": 5.5, "band_bin": "developing",
#     "task_response": ..., "coherence_cohesion": ...,
#     "lexical_resource": ..., "grammatical_range_accuracy": ...,
#   }
#
# build_prompt() accesses `scores.task_response` etc. via DOT notation (it
# expects an object, not a dict) -- so case_scores() below wraps the four
# flat dict fields into a SimpleNamespace right before use, rather than
# changing build_prompt itself.
# ---------------------------------------------------------------------------


def case_scores(case: dict) -> SimpleNamespace:
    """Wrap a test-set case's flat sub-score fields into an attribute-access
    object, since build_prompt() expects scores.task_response, not
    scores["task_response"]."""
    return SimpleNamespace(
        task_response=case["task_response"],
        coherence_cohesion=case["coherence_cohesion"],
        lexical_resource=case["lexical_resource"],
        grammatical_range_accuracy=case["grammatical_range_accuracy"],
        overall=case["overall"],
    )


def prompt_with_exemplars(case, exemplars):
    """
    Generator prompt, WITH retrieved exemplars -- mirrors production's
    build_prompt() call exactly (same function, same tone default, same
    system prompt). Returns (system_prompt, user_prompt) so the generator
    call can replicate production's client.messages.create(system=..., ...)
    shape rather than just sending a single flattened string.
    """
    user_prompt = build_prompt(
        question=case["question"],
        essay=case["essay"],
        scores=case_scores(case),
        examiner_comments=exemplars,
        tone="coaching",
    )
    return SYSTEM_PROMPT, user_prompt


def prompt_zero_shot(case):
    """
    Generator prompt, ablation arm -- same template, exemplars stripped.
    Calls build_prompt() with examiner_comments=[] (so scores/descriptors/
    tone logic stay byte-identical to production), then trims the resulting
    string to remove the now-empty "## Examiner Reference Comments" section
    entirely, rather than leaving a confusing empty block in the prompt.
    """
    full_prompt = build_prompt(
        question=case["question"],
        essay=case["essay"],
        scores=case_scores(case),
        examiner_comments=[],
        tone="coaching",
    )

    section_start = full_prompt.index("## Examiner Reference Comments")
    section_end = full_prompt.index("## Task")
    zero_shot_prompt = full_prompt[:section_start] + full_prompt[section_end:]

    return SYSTEM_PROMPT, zero_shot_prompt


def faithfulness_judge_prompt(case, claim_text):
    """
    Judge prompt -- is `claim_text` (one decomposed claim from generated
    feedback) grounded in question + essay + the four criterion scores?
    Overall score and exemplars are deliberately excluded: production's own
    generation instruction grounds feedback on the four criteria, NOT on
    the overall band score, and exemplars are style-only references, not
    part of the faithfulness ground truth. One call per claim.
    """
    s = case_scores(case)
    return f"""\
You are evaluating whether a piece of feedback text is faithful to its
source context, i.e. whether every factual claim in it can be traced back
to the context provided -- with no fabricated or unsupported claims.

## Context
### Question
{case["question"]}

### Essay
{case["essay"]}

### Predicted Band Scores
- Task Response:              {s.task_response}
- Coherence and Cohesion:     {s.coherence_cohesion}
- Lexical Resource:           {s.lexical_resource}
- Grammatical Range/Accuracy: {s.grammatical_range_accuracy}

## Claim
{claim_text}

## Task
Determine whether the claim above is fully supported by the context.
A claim is supported if it can be directly verified against the question,
essay content, or the four criterion scores. A claim is NOT supported if
it introduces information, opinions, or specifics not present in or
inferable from the context above.

""" + JUDGE_OUTPUT_INSTRUCTION


def context_precision_judge_prompt(case, exemplars):
    """
    Judge prompt -- custom style-relevance framing, NOT stock RAGAS
    factual-QA precision. One call per case, scoring the retrieved
    exemplar SET as a whole against three independent binary criteria:

      relevant    - is retrieval topically/contextually sensible
      independent - are the exemplars distinct from each other/the target
                    (not near-duplicates or leaked cases)
      useful      - would these plausibly help a writer produce
                    well-grounded, well-styled feedback

    Band level is explicitly excluded from context and named as a
    non-criterion, since production deliberately does not filter/rank
    exemplars by score -- penalizing band mismatch here would reintroduce
    exactly the bias production chose not to have.
    """
    numbered_exemplars = "\n\n".join(
        f"[{i + 1}] {ex['text']}" for i, ex in enumerate(exemplars)
    )

    return f"""\
You are evaluating a set of retrieved examiner comments as reference
material for generating feedback on a target essay.

## Target Essay Context
### Question
{case["question"]}

### Essay
{case["essay"]}

## Retrieved Exemplars
{numbered_exemplars}

## Criteria

1. relevant (0/1): Are the exemplars' contexts (topic, argument
   structure, or essay type) similar enough to the target essay/question
   that this looks like a sensible retrieval match overall -- not
   arbitrary or unrelated results?

2. independent (0/1): Are the exemplars clearly distinct essays/cases --
   from each other and from the target -- rather than near-duplicates or
   the same underlying case reappearing?

3. useful (0/1): Would these exemplars plausibly help a writer produce
   well-grounded, well-styled feedback on the target essay -- e.g. do
   they model clear critique or natural phrasing -- as opposed to being
   vague, malformed, or unhelpful as reference material?

Judge each criterion independently. Band level is NOT a criterion for
any of the above -- do not penalize or reward the exemplar set for
having different Overall scores than the target essay.

Respond with a single JSON object in this exact shape:
{{
  "relevant":    {{"score": 0 or 1, "reasoning": "..."}},
  "independent": {{"score": 0 or 1, "reasoning": "..."}},
  "useful":      {{"score": 0 or 1, "reasoning": "..."}}
}}
No preamble, no markdown fences, JSON only.
"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

JUDGE_OUTPUT_INSTRUCTION = (
    "\n\nRespond with ONLY a JSON object in exactly this format, nothing else, "
    'no markdown code fences: {"score": 0 or 1, '
    '"reasoning": "<one short sentence>"}'
)


def _strip_code_fences(raw_output):
    """Shared fence-stripping for both parsers below. Tolerates the model
    wrapping JSON in ```/```json fences even though we asked it not to."""
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        first_line, _, rest = cleaned.partition("\n")
        if first_line.strip().lower() in ("json", ""):
            cleaned = rest
        cleaned = cleaned.strip()
    return cleaned


def parse_score(raw_output):
    """
    Parses the strict single-score JSON format mandated by
    JUDGE_OUTPUT_INSTRUCTION. Used for faithfulness_judge_prompt (one call
    per claim, per arm).

    Returns (value or None, ok: bool). Value must be exactly 0.0 or 1.0 --
    faithfulness is a binary supported/not-supported determination.
    """
    if raw_output is None:
        return None, False

    cleaned = _strip_code_fences(raw_output)

    try:
        parsed = json.loads(cleaned)
        value = float(parsed["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, False

    if value not in (0.0, 1.0):
        return None, False

    return value, True


def parse_precision_verdicts(raw_output):
    """
    Parses the 3-key JSON format produced by context_precision_judge_prompt.
    One call per CASE -- scores the retrieved exemplar set as a whole.
    Returns (verdicts, ok): ok=True only if all three keys parsed with a
    valid 0/1 score.
    """
    if raw_output is None:
        return None, False

    cleaned = _strip_code_fences(raw_output)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, False

    if not isinstance(parsed, dict):
        return None, False

    verdicts = {}
    for key in ("relevant", "independent", "useful"):
        entry = parsed.get(key)
        if not isinstance(entry, dict):
            return None, False
        try:
            value = int(entry["score"])
        except (KeyError, TypeError, ValueError):
            return None, False
        if value not in (0, 1):
            return None, False
        verdicts[key] = {"score": value, "reasoning": entry.get("reasoning", "")}

    return verdicts, True


def precision_composite(verdicts):
    """
    Single headline number for context_precision: mean of the three
    independent 0/1 criteria. Reporting convenience ON TOP OF the separate
    per-criterion means in write_results, not a replacement for them.
    """
    if verdicts is None:
        return None
    scores = [verdicts[k]["score"] for k in ("relevant", "independent", "useful")]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Retrieval with leak check + one retry
# ---------------------------------------------------------------------------

def retrieve_with_leak_check(case, known_bad_ids):
    """
    Returns (exemplars or None, excluded: bool)
    exemplars is None only if the case must be excluded from this run.
    """
    exemplars = retrieve_top3(case["question"], case["essay"])
    leaked = [e for e in exemplars if e["id"] in known_bad_ids]

    if not leaked:
        return exemplars, False

    # Retry once with explicit exclusion -- not just re-trusting the filter
    exemplars_retry = retrieve_top3(
        case["question"], case["essay"], exclude_ids=known_bad_ids
    )
    leaked_retry = [e for e in exemplars_retry if e["id"] in known_bad_ids]

    if leaked_retry:
        log_failure(
            "retrieval_leak",
            case["id"],
            f"leak persists after explicit-exclude retry: {[e['id'] for e in leaked_retry]}",
            {"original": exemplars, "retry": exemplars_retry},
        )
        return None, True

    log_fix("retrieval_leak_recovered", case["id"], "clean after explicit-exclude retry", None)
    return exemplars_retry, False


# ---------------------------------------------------------------------------
# Rate-limited wrappers -- Groq free-tier numbers (verified against Groq's
# published per-model limits, June 2026). Limits are per model, so the
# generator and judge get separate limiter instances rather than sharing one.
#
#   llama-3.3-70b-versatile (generator): 30 RPM, 12,000 TPM, 1K RPD, 100K TPD
#   openai/gpt-oss-120b     (judge):     30 RPM,  8,000 TPM, 1K RPD, 200K TPD
#
# TPM binds well before RPM on both, so both limiters track tokens. See the
# TPD note below -- daily token budget is the real ceiling for a full run.
# ---------------------------------------------------------------------------

groq_limiter = RateLimiter(rpm=30, tpm=12_000)
judge_limiter = RateLimiter(rpm=30, tpm=8_000)


def judge_safe(prompt):
    """
    Was judge_safe. Now token-aware: the old Gemini limiter was
    RPM-only (rpm=5) because Gemini's TPM ceiling was 250K and never came
    close to binding. The judge model's 8K TPM is tight enough that an
    RPM-only limiter would silently under-protect it and trigger 429s.
    """
    estimated_tokens = len(prompt) // 4 + 300  # rough input + output ceiling
    judge_limiter.wait(estimated_tokens=estimated_tokens)
    return with_backoff(judge_llm, prompt)


def generate_groq_safe(prompt_tuple):
    system, user = prompt_tuple
    estimated_tokens = (len(system) + len(user)) // 4 + 350  # rough input + MAX_TOKENS output ceiling
    groq_limiter.wait(estimated_tokens=estimated_tokens)
    return with_backoff(generate_groq, prompt_tuple)


# ---------------------------------------------------------------------------
# Claim decomposition -- faithfulness is judged per-claim, not on the whole
# feedback paragraph as a single unit.
# ---------------------------------------------------------------------------

def decompose_claims_prompt(feedback_text):
    """
    Asks the judge model to break a piece of generated feedback into
    atomic, independently-checkable claims, so faithfulness can be judged
    per-claim rather than holistically.
    """
    return f"""\
Break the following feedback text into a list of atomic, independently
verifiable claims. Each claim should be a single self-contained statement
that could be checked as true or false on its own, without needing the
other claims for context. Split compound sentences into separate claims
where they contain more than one checkable statement. Do not include
claims that are purely stylistic commentary with no checkable content
(e.g. transitions like "overall" on their own) -- only include statements
that assert something specific about the essay, the question, or the
scores.

## Feedback
{feedback_text}

## Task
Respond with ONLY a JSON array of strings, one per claim, nothing else,
no markdown code fences. Example format:
["claim one text", "claim two text", "claim three text"]
"""


def parse_claims(raw_output):
    """
    Parses the JSON array of claim strings from decompose_claims_prompt.
    Returns (claims: list[str] or None, ok: bool).
    """
    if raw_output is None:
        return None, False

    cleaned = _strip_code_fences(raw_output)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, False

    if not isinstance(parsed, list) or not all(isinstance(c, str) for c in parsed):
        return None, False

    claims = [c.strip() for c in parsed if c.strip()]
    return claims, True


def score_faithfulness(case, feedback_text, arm_label):
    """
    Full per-arm faithfulness pipeline: decompose feedback_text into
    claims, judge each claim independently, average into a single
    per-case, per-arm score.

    Returns (score: float or None, n_parse_failures: int).
    """
    case_id = case["id"]
    n_failures = 0

    raw_claims = judge_safe(decompose_claims_prompt(feedback_text))
    claims, ok = parse_claims(raw_claims)
    if not ok:
        log_failure("judge_parse", case_id, f"claim decomposition unparseable ({arm_label})", raw_claims)
        return None, 1

    if not claims:
        log_failure("judge_parse", case_id, f"zero claims decomposed ({arm_label})", feedback_text)
        return None, 1

    claim_scores = []
    for claim in claims:
        raw = judge_safe(faithfulness_judge_prompt(case, claim))
        score, ok = parse_score(raw)
        if not ok:
            log_failure("judge_parse", case_id, f"faithfulness claim score unparseable ({arm_label})", raw)
            n_failures += 1
            continue
        claim_scores.append(score)

    if not claim_scores:
        return None, n_failures

    return sum(claim_scores) / len(claim_scores), n_failures


# ---------------------------------------------------------------------------
# Checkpointing -- resume-on-restart
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(TEST_QUESTIONS_PATH)), "logs", "run_ragas_checkpoint.jsonl"
)


def load_checkpoint(path):
    """
    Loads previously-completed case results from a JSONL checkpoint file.
    Returns (results: list[dict], completed_ids: set[str]).
    """
    results = []
    completed_ids = set()
    if not os.path.exists(path):
        return results, completed_ids

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            results.append(row)
            completed_ids.add(row["id"])

    return results, completed_ids


def append_checkpoint(path, row):
    """Appends one completed case's result to the checkpoint file
    immediately, forcing it to disk (flush + fsync)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run():
    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    all_held_out_ids = {c["id"] for c in test_cases}

    global essays_collection, questions_collection
    essays_collection, questions_collection = get_collections()

    known_bad_ids = run_preflight(all_held_out_ids, essays_collection, questions_collection)

    results, completed_ids = load_checkpoint(CHECKPOINT_PATH)
    if completed_ids:
        print(f"[resume] found checkpoint with {len(completed_ids)} already-completed cases -- skipping those")

    n_excluded_leak = 0
    n_judge_parse_failures = 0
    n_case_crashes = 0

    for case in test_cases:
        if case["id"] in completed_ids:
            continue

        exemplars, excluded = retrieve_with_leak_check(case, known_bad_ids)
        if excluded:
            n_excluded_leak += 1
            continue

        try:
            variant_A = generate_groq_safe(prompt_with_exemplars(case, exemplars))
            variant_B = generate_groq_safe(prompt_zero_shot(case))

            faithfulness_with_exemplars, n_fail_A = score_faithfulness(case, variant_A, "with_exemplars")
            n_judge_parse_failures += n_fail_A

            faithfulness_zero_shot, n_fail_B = score_faithfulness(case, variant_B, "zero_shot")
            n_judge_parse_failures += n_fail_B

            raw_precision = judge_safe(context_precision_judge_prompt(case, exemplars))
            verdicts, ok = parse_precision_verdicts(raw_precision)
            if not ok:
                log_failure("judge_parse", case["id"], "context_precision verdicts unparseable", raw_precision)
                n_judge_parse_failures += 1
                precision_relevant = precision_independent = precision_useful = precision_composite_score = None
            else:
                precision_relevant = verdicts["relevant"]["score"]
                precision_independent = verdicts["independent"]["score"]
                precision_useful = verdicts["useful"]["score"]
                precision_composite_score = precision_composite(verdicts)

            row = {
                "id": case["id"],
                "band_bin": case["band_bin"],
                "faithfulness_with_exemplars": faithfulness_with_exemplars,
                "faithfulness_zero_shot": faithfulness_zero_shot,
                "context_precision_relevant": precision_relevant,
                "context_precision_independent": precision_independent,
                "context_precision_useful": precision_useful,
                "context_precision_composite": precision_composite_score,
                "variant_A": variant_A,
                "variant_B": variant_B,
            }
        except Exception as exc:
            n_case_crashes += 1
            log_failure("case_crash", case["id"], f"{type(exc).__name__}: {exc}", None)
            print(f"[SKIPPED] case {case['id']} failed ({type(exc).__name__}: {exc}) -- will retry next run")
            continue

        results.append(row)
        append_checkpoint(CHECKPOINT_PATH, row)
        print(f"[progress] case {case['id']} done ({len(results)}/{len(test_cases)})")

    write_results(results, n_excluded_leak, n_judge_parse_failures, len(test_cases))


# ---------------------------------------------------------------------------
# Aggregation + Output
# ---------------------------------------------------------------------------

def _mean_std(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def write_results(results, n_excluded_leak, n_judge_parse_failures, n_total_cases):
    faith_exemplars = _mean_std([r["faithfulness_with_exemplars"] for r in results])
    faith_zero_shot = _mean_std([r["faithfulness_zero_shot"] for r in results])

    prec_relevant = _mean_std([r["context_precision_relevant"] for r in results])
    prec_independent = _mean_std([r["context_precision_independent"] for r in results])
    prec_useful = _mean_std([r["context_precision_useful"] for r in results])
    prec_composite = _mean_std([r["context_precision_composite"] for r in results])

    by_bin = {}
    for r in results:
        by_bin.setdefault(r["band_bin"], []).append(r)

    per_bin_stats = {}
    for band_bin, rows in by_bin.items():
        fe_mean, fe_std = _mean_std([r["faithfulness_with_exemplars"] for r in rows])
        fz_mean, fz_std = _mean_std([r["faithfulness_zero_shot"] for r in rows])
        pr_mean, pr_std = _mean_std([r["context_precision_relevant"] for r in rows])
        pi_mean, pi_std = _mean_std([r["context_precision_independent"] for r in rows])
        pu_mean, pu_std = _mean_std([r["context_precision_useful"] for r in rows])
        pc_mean, pc_std = _mean_std([r["context_precision_composite"] for r in rows])
        per_bin_stats[band_bin] = {
            "n": len(rows),
            "faithfulness_with_exemplars_mean": fe_mean,
            "faithfulness_with_exemplars_std": fe_std,
            "faithfulness_zero_shot_mean": fz_mean,
            "faithfulness_zero_shot_std": fz_std,
            "context_precision_relevant_mean": pr_mean,
            "context_precision_relevant_std": pr_std,
            "context_precision_independent_mean": pi_mean,
            "context_precision_independent_std": pi_std,
            "context_precision_useful_mean": pu_mean,
            "context_precision_useful_std": pu_std,
            "context_precision_composite_mean": pc_mean,
            "context_precision_composite_std": pc_std,
        }

    output = {
        "gated": False,
        "n_total_cases": n_total_cases,
        "n_scored": len(results),
        "n_excluded_leak": n_excluded_leak,
        "n_judge_parse_failures": n_judge_parse_failures,
        "overall": {
            "faithfulness_with_exemplars_mean": faith_exemplars[0],
            "faithfulness_with_exemplars_std": faith_exemplars[1],
            "faithfulness_zero_shot_mean": faith_zero_shot[0],
            "faithfulness_zero_shot_std": faith_zero_shot[1],
            "context_precision_relevant_mean": prec_relevant[0],
            "context_precision_relevant_std": prec_relevant[1],
            "context_precision_independent_mean": prec_independent[0],
            "context_precision_independent_std": prec_independent[1],
            "context_precision_useful_mean": prec_useful[0],
            "context_precision_useful_std": prec_useful[1],
            "context_precision_composite_mean": prec_composite[0],
            "context_precision_composite_std": prec_composite[1],
        },
        "per_band_bin": per_bin_stats,
        "rows": results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n=== summary ===")
    print(f"scored: {len(results)} / {n_total_cases}")
    print(f"excluded (retrieval leak, after retry): {n_excluded_leak}")
    print(f"judge parse failures: {n_judge_parse_failures}")
    print(f"faithfulness (with exemplars): mean={faith_exemplars[0]}, std={faith_exemplars[1]}")
    print(f"faithfulness (zero-shot):      mean={faith_zero_shot[0]}, std={faith_zero_shot[1]}")
    print(f"context_precision relevant:    mean={prec_relevant[0]}, std={prec_relevant[1]}")
    print(f"context_precision independent: mean={prec_independent[0]}, std={prec_independent[1]}")
    print(f"context_precision useful:      mean={prec_useful[0]}, std={prec_useful[1]}")
    print(f"context_precision composite:   mean={prec_composite[0]}, std={prec_composite[1]}")
    print(f"held_out_preflight failures logged: {count_jsonl('held_out_preflight_failures.jsonl')}")
    print(f"retrieval_leak failures logged: {count_jsonl('retrieval_leak_failures.jsonl')}")
    print(f"judge_parse failures logged: {count_jsonl('judge_parse_failures.jsonl')}")
    print(f"results -> {RESULTS_PATH}")


if __name__ == "__main__":
    run()