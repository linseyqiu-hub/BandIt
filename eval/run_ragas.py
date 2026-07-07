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

NOTE: generate(), judge(), retrieve_top3/retrieve_top_n, and the prompt-
builder functions are stubs marked TODO. Wire these to your actual Groq /
Gemini / Chroma clients. The control flow and failure-handling logic around
them is the part that's locked and shouldn't need to change.
"""

import json
import os
import sys
import statistics

from openai import OpenAI
from google import genai
import chromadb
from sentence_transformers import SentenceTransformer

from common import log_failure, log_fix, count_jsonl
from preflight import run_preflight

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
# Gemini: https://aistudio.google.com  -> Get API key
#        export GEMINI_API_KEY=...
#        pip install google-genai --break-system-packages

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
gemini_client = genai.Client(
    api_key=_require_env(
        "GEMINI_API_KEY",
        "Get one at https://aistudio.google.com -> Get API key.",
    )
)


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


def generate_groq(prompt, temperature=0.7):
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def judge_gemini(prompt):
    resp = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return resp.text


# ---------------------------------------------------------------------------
# Retrieval -- your RRF fusion, wrapped with Chroma queries + exclude_ids
# ---------------------------------------------------------------------------

def rrf_fusion(essay_ids, question_ids, essay_docs, question_docs, k=RRF_K, n=FINAL_N):
    """
    Fuse two ranked lists via Reciprocal Rank Fusion.

    NOTE: changed from your original to return dicts (id + comment) instead
    of bare comment strings -- the leak check in retrieve_with_leak_check()
    needs the id of each returned exemplar, not just its text. If you have
    other callers relying on the old list[str] return shape, keep this as a
    separate function name rather than overwriting the original.
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
    return [{"id": id_, "examiner_comment": docs[id_]} for id_ in ranked[:n]]


def retrieve_top3(question, essay, exclude_ids=None):
    """
    Queries both Chroma collections (held_out-filtered), overfetches so an
    exclude_ids retry still has enough candidates left, then fuses via RRF.

    IMPORTANT: collections were created with embedding_function=None (see
    ingest.py), so Chroma cannot embed query_texts itself. We compute query
    embeddings explicitly with the same model ingest.py used, and pass
    query_embeddings= instead.
    """
    exclude_ids = exclude_ids or set()
    where_filter = {"held_out": {"$ne": True}}
    model = get_embedding_model()

    essay_embedding = model.encode([essay]).tolist()
    question_embedding = model.encode([question]).tolist()

    essay_results = essays_collection.query(
        query_embeddings=essay_embedding, n_results=OVERFETCH_N, where=where_filter
    )
    question_results = questions_collection.query(
        query_embeddings=question_embedding, n_results=OVERFETCH_N, where=where_filter
    )

    # documents ARE the examiner_comment (per ingest.py) -- no need to pull
    # it out of metadata separately.
    essay_ids = essay_results["ids"][0]
    essay_docs = essay_results["documents"][0]
    question_ids = question_results["ids"][0]
    question_docs = question_results["documents"][0]

    if exclude_ids:
        essay_pairs = [(i, d) for i, d in zip(essay_ids, essay_docs) if i not in exclude_ids]
        question_pairs = [(i, d) for i, d in zip(question_ids, question_docs) if i not in exclude_ids]
        essay_ids, essay_docs = (list(t) for t in zip(*essay_pairs)) if essay_pairs else ([], [])
        question_ids, question_docs = (list(t) for t in zip(*question_pairs)) if question_pairs else ([], [])

    return rrf_fusion(essay_ids, question_ids, essay_docs, question_docs)


def prompt_with_exemplars(case, exemplars):
    """TODO: system -> question -> essay -> scores -> exemplars -> instructions."""
    raise NotImplementedError


def prompt_zero_shot(case):
    """TODO: same template with exemplar section omitted, for the ablation."""
    raise NotImplementedError


def faithfulness_judge_prompt(case, claim_text):
    """
    TODO: context = essay + question ONLY, exemplars excluded per design.
    IMPORTANT: must end with `+ JUDGE_OUTPUT_INSTRUCTION` (defined above) so
    parse_score() can parse the response deterministically.
    """
    raise NotImplementedError


def context_precision_judge_prompt(case, exemplars):
    """
    TODO: custom style-relevance wording, NOT the default RAGAS factual-QA prompt.
    IMPORTANT: must end with `+ JUDGE_OUTPUT_INSTRUCTION` (defined above) so
    parse_score() can parse the response deterministically.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# We control the judge prompts (see faithfulness_judge_prompt /
# context_precision_judge_prompt, TODO below), so the output format is a
# decision we make once, not something to discover after the fact by
# guessing at whatever Gemini happens to return. Every judge prompt MUST
# end with this instruction appended -- that's what makes parse_score below
# deterministic instead of a best-effort regex.
JUDGE_OUTPUT_INSTRUCTION = (
    "\n\nRespond with ONLY a JSON object in exactly this format, nothing else, "
    'no markdown code fences: {"score": <float between 0.0 and 1.0>, '
    '"reasoning": "<one short sentence>"}'
)


def parse_score(raw_output):
    """
    Parses the strict JSON format mandated by JUDGE_OUTPUT_INSTRUCTION.
    Returns (value or None, ok: bool).

    Tolerates the model wrapping the JSON in ```/```json fences even though
    we asked it not to, since that's a common enough deviation to handle
    rather than fail on -- but anything beyond that (missing 'score' key,
    non-numeric, out of range) is a genuine parse failure and gets logged.
    """
    if raw_output is None:
        return None, False

    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        first_line, _, rest = cleaned.partition("\n")
        if first_line.strip().lower() in ("json", ""):
            cleaned = rest
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        value = float(parsed["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, False

    if not (0.0 <= value <= 1.0):
        return None, False

    return value, True


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
# Main loop
# ---------------------------------------------------------------------------

def run():
    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    all_held_out_ids = {c["id"] for c in test_cases}

    global essays_collection, questions_collection
    essays_collection, questions_collection = get_collections()

    known_bad_ids = run_preflight(all_held_out_ids, essays_collection, questions_collection)

    results = []
    n_excluded_leak = 0
    n_judge_parse_failures = 0

    for case in test_cases:
        exemplars, excluded = retrieve_with_leak_check(case, known_bad_ids)
        if excluded:
            n_excluded_leak += 1
            continue

        variant_A = generate_groq(prompt_with_exemplars(case, exemplars))
        variant_B = generate_groq(prompt_zero_shot(case))  # ablation

        # --- faithfulness: essay + question only, exemplars excluded ---
        raw_faithfulness = judge_gemini(faithfulness_judge_prompt(case, variant_A))
        faithfulness_score, ok = parse_score(raw_faithfulness)
        if not ok:
            log_failure("judge_parse", case["id"], "faithfulness score unparseable", raw_faithfulness)
            n_judge_parse_failures += 1
            faithfulness_score = None

        # --- context_precision: custom style-relevance prompt ---
        raw_precision = judge_gemini(context_precision_judge_prompt(case, exemplars))
        precision_score, ok = parse_score(raw_precision)
        if not ok:
            log_failure("judge_parse", case["id"], "context_precision score unparseable", raw_precision)
            n_judge_parse_failures += 1
            precision_score = None

        results.append({
            "id": case["id"],
            "band_bin": case["band_bin"],
            "faithfulness": faithfulness_score,
            "context_precision": precision_score,
            "variant_A": variant_A,
            "variant_B": variant_B,
        })

    write_results(results, n_excluded_leak, n_judge_parse_failures, len(test_cases))


# ---------------------------------------------------------------------------
# Aggregation + output
# ---------------------------------------------------------------------------

def _mean_std(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def write_results(results, n_excluded_leak, n_judge_parse_failures, n_total_cases):
    overall_faithfulness = _mean_std([r["faithfulness"] for r in results])
    overall_precision = _mean_std([r["context_precision"] for r in results])

    by_bin = {}
    for r in results:
        by_bin.setdefault(r["band_bin"], []).append(r)

    per_bin_stats = {}
    for band_bin, rows in by_bin.items():
        f_mean, f_std = _mean_std([r["faithfulness"] for r in rows])
        p_mean, p_std = _mean_std([r["context_precision"] for r in rows])
        per_bin_stats[band_bin] = {
            "n": len(rows),
            "faithfulness_mean": f_mean,
            "faithfulness_std": f_std,
            "context_precision_mean": p_mean,
            "context_precision_std": p_std,
        }

    output = {
        "gated": False,  # first run is ungated, per design
        "n_total_cases": n_total_cases,
        "n_scored": len(results),
        "n_excluded_leak": n_excluded_leak,
        "n_judge_parse_failures": n_judge_parse_failures,
        "overall": {
            "faithfulness_mean": overall_faithfulness[0],
            "faithfulness_std": overall_faithfulness[1],
            "context_precision_mean": overall_precision[0],
            "context_precision_std": overall_precision[1],
        },
        "per_band_bin": per_bin_stats,
        "rows": results,  # includes variant_A / variant_B for ablation review
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n=== summary ===")
    print(f"scored: {len(results)} / {n_total_cases}")
    print(f"excluded (retrieval leak, after retry): {n_excluded_leak}")
    print(f"judge parse failures: {n_judge_parse_failures}")
    print(f"overall faithfulness: mean={overall_faithfulness[0]}, std={overall_faithfulness[1]}")
    print(f"overall context_precision: mean={overall_precision[0]}, std={overall_precision[1]}")
    print(f"held_out_preflight failures logged: {count_jsonl('held_out_preflight_failures.jsonl')}")
    print(f"retrieval_leak failures logged: {count_jsonl('retrieval_leak_failures.jsonl')}")
    print(f"judge_parse failures logged: {count_jsonl('judge_parse_failures.jsonl')}")
    print(f"results -> {RESULTS_PATH}")


if __name__ == "__main__":
    run()