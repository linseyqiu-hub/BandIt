"""
BandIt feedback service
Retrieves similar examiner comments via RRF fusion across two ChromaDB
collections, then calls Claude API to generate criterion-aligned feedback.
"""

import os
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLAUDE_MODEL    = "claude-sonnet-4-6"
MAX_TOKENS      = 350       # ~4 paragraphs x 3-4 sentences
RRF_K           = 60        # standard RRF constant
RETRIEVE_N      = 10        # candidates per collection before fusion
FINAL_N         = 3         # few-shot examples after RRF

DESCRIPTORS = {
    # maps Overall/sub-score float → band descriptor string
}

def get_descriptor(score: float) -> str:
    if score <= 4.5:
        return "Limited"
    elif score <= 6.0:
        return "Developing"
    elif score <= 7.5:
        return "Competent"
    else:
        return "Expert"

# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------
def rrf_fusion(
    essay_ids:     list[str],
    question_ids:  list[str],
    essay_docs:    list[str],
    question_docs: list[str],
    k: int = RRF_K,
    n: int = FINAL_N,
) -> list[str]:
    """
    Fuse two ranked lists via Reciprocal Rank Fusion.
    Returns top-n examiner comments by fused score.

    essay_ids / question_ids   — ordered by similarity (rank 0 = best)
    essay_docs / question_docs — examiner_comment at same index as id
    """
    scores: dict[str, float] = {}
    docs:   dict[str, str]   = {}

    for rank, (id_, doc) in enumerate(zip(essay_ids, essay_docs)):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)
        docs[id_]   = doc

    for rank, (id_, doc) in enumerate(zip(question_ids, question_docs)):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)
        docs[id_]   = doc

    ranked = sorted(scores.keys(), key=lambda id_: scores[id_], reverse=True)
    return [docs[id_] for id_ in ranked[:n]]

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an experienced IELTS Writing Task 2 examiner with over 10 years of \
experience assessing candidate essays. You provide accurate, criterion-aligned \
feedback based on the four official IELTS marking criteria: Task Response, \
Coherence and Cohesion, Lexical Resource, and Grammatical Range and Accuracy.\
"""

TONE_INSTRUCTIONS = {
    "strict":   "Write in a strict examiner tone. Be direct and precise about weaknesses.",
    "coaching": "Write in a supportive coaching tone. Frame weaknesses as areas for improvement.",
}

def build_prompt(
    question:          str,
    essay:             str,
    scores,
    examiner_comments: list[str],
    tone:              str,
) -> str:
    tr      = scores.task_response
    cc      = scores.coherence_cohesion
    lr      = scores.lexical_resource
    ra      = scores.grammatical_range_accuracy
    overall = scores.overall

    few_shots = "\n\n".join(
        f"[{i+1}] Overall: {comment['overall']}\n    {comment['text']}"
        for i, comment in enumerate(examiner_comments)
    )

    tone_instruction = TONE_INSTRUCTIONS.get(tone, TONE_INSTRUCTIONS["coaching"])

    return f"""\
## Question
{question}

## Essay
{essay}

## Predicted Band Scores
- Task Response:              {tr} ({get_descriptor(tr)})
- Coherence and Cohesion:     {cc} ({get_descriptor(cc)})
- Lexical Resource:           {lr} ({get_descriptor(lr)})
- Grammatical Range/Accuracy: {ra} ({get_descriptor(ra)})
- Overall:                    {overall} ({get_descriptor(overall)})

## Examiner Reference Comments
The following are real examiner comments on essays with similar band levels.
Use them as style and tone references only. Base your feedback on the question,
essay content, and the four criterion scores provided above, not on the overall
band score.

{few_shots}

## Task
{tone_instruction}
Write feedback as a single cohesive paragraph of 6-8 sentences.
Address all four criteria (Task Response, Coherence and Cohesion, Lexical
Resource, Grammatical Range and Accuracy) in order.
Ground your feedback to the essay content. Do not use headers or lists.
Do not repeat the band scores in your feedback.\
"""

# ---------------------------------------------------------------------------
# Main service function
# ---------------------------------------------------------------------------
def generate_feedback(
    question:    str,
    essay:       str,
    scores,
    tone:        str,
    app_state,               # carries embedding_model, essays_col, questions_col
) -> str:
    """
    Full feedback pipeline:
      1. embed essay + question
      2. query both collections
      3. RRF fusion → top-3 examiner comments
      4. build prompt
      5. call Claude API
      6. return feedback text
    """
    # --- 1. embed ---
    embedding_model = app_state.embedding_model
    essay_emb    = embedding_model.encode([essay]).tolist()
    question_emb = embedding_model.encode([question]).tolist()

    # --- 2. query ---
    essay_results = app_state.essays_col.query(
        query_embeddings=essay_emb,
        n_results=RETRIEVE_N,
        include=["documents", "metadatas"],
    )
    question_results = app_state.questions_col.query(
        query_embeddings=question_emb,
        n_results=RETRIEVE_N,
        include=["documents", "metadatas"],
    )

    essay_ids    = essay_results["ids"][0]
    essay_docs   = essay_results["documents"][0]

    question_ids  = question_results["ids"][0]
    question_docs = question_results["documents"][0]

    # --- 3. RRF fusion ---
    # also need overall scores for few-shot display — pull from metadatas
    essay_metas    = essay_results["metadatas"][0]
    question_metas = question_results["metadatas"][0]

    # build id → overall map before fusion
    overall_map: dict[str, float] = {}
    for id_, meta in zip(essay_ids, essay_metas):
        overall_map[id_] = meta["overall"]
    for id_, meta in zip(question_ids, question_metas):
        overall_map[id_] = meta["overall"]

    # RRF — returns top-3 (id, doc) pairs
    scores_rrf: dict[str, float] = {}
    docs_map:   dict[str, str]   = {}

    for rank, (id_, doc) in enumerate(zip(essay_ids, essay_docs)):
        scores_rrf[id_] = scores_rrf.get(id_, 0.0) + 1.0 / (RRF_K + rank)
        docs_map[id_]   = doc

    for rank, (id_, doc) in enumerate(zip(question_ids, question_docs)):
        scores_rrf[id_] = scores_rrf.get(id_, 0.0) + 1.0 / (RRF_K + rank)
        docs_map[id_]   = doc

    ranked_ids = sorted(scores_rrf.keys(), key=lambda x: scores_rrf[x], reverse=True)
    top_ids    = ranked_ids[:FINAL_N]

    examiner_comments = [
        {"text": docs_map[id_], "overall": overall_map[id_]}
        for id_ in top_ids
    ]

    # --- 4. build prompt ---
    user_prompt = build_prompt(question, essay, scores, examiner_comments, tone)

    # --- 5. call Claude API ---
    client   = Anthropic()
    response = client.messages.create(
        model      = CLAUDE_MODEL,
        max_tokens = MAX_TOKENS,
        system     = SYSTEM_PROMPT,
        messages   = [{"role": "user", "content": user_prompt}],
    )

    # --- 6. return ---
    return response.content[0].text