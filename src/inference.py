import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import DebertaV2Tokenizer

from model import BandItScorer


# ------------------------------------------------------------------
# Constants — must match dataset.py exactly
# ------------------------------------------------------------------

MODEL_NAME    = "microsoft/deberta-v3-base"
MAX_LENGTH    = 512
LABEL_COLUMNS = [
    "Task_Response",
    "Coherence_Cohesion",
    "Lexical_Resource",
    "Range_Accuracy",
]


# ------------------------------------------------------------------
# Load phase
# ------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> BandItScorer:
    """
    Loads BandItScorer v4 from a checkpoint.

    v4 checkpoint is best_model_v4.pt — distinct from v3's best_model.pt.
    Both can coexist in checkpoints/.
    """
    print(f"[inference] loading model from {checkpoint_path} ...")

    model = BandItScorer(pretrained=False, dropout=0.0)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.to(device)

    epoch   = checkpoint.get("epoch",   "unknown")
    val_mae = checkpoint.get("val_mae", "unknown")
    if isinstance(val_mae, float):
        print(f"[inference] loaded checkpoint — epoch {epoch}, val MAE {val_mae:.4f}")
    else:
        print(f"[inference] loaded checkpoint — epoch {epoch}, val MAE {val_mae}")
    print(f"[inference] model ready on {device}")

    return model


def load_tokenizer():
    print(f"[inference] loading tokenizer: {MODEL_NAME} ...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    print(f"[inference] tokenizer ready.")
    return tokenizer


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------

def _tokenize(text: str, tokenizer, device: torch.device) -> dict:
    """
    Tokenizes a single string to (1, MAX_LENGTH) tensors on device.
    Internal helper used by preprocess().
    """
    encoded = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids":      encoded["input_ids"].to(device),       # (1, 512)
        "attention_mask": encoded["attention_mask"].to(device),  # (1, 512)
    }


def preprocess(question: str, essay: str, tokenizer, device: torch.device) -> dict:
    """
    Produces two separate tokenizations matching the v4 dual-encoder architecture.

    CRITICAL: must exactly mirror dataset.py __getitem__:
        - question tokenized alone
        - essay tokenized alone
        (no concatenation — that was v3)

    Returns:
        dict with q_input_ids, q_attention_mask, e_input_ids, e_attention_mask
        all shape (1, 512) with batch dim of 1
    """
    q_enc = _tokenize(question, tokenizer, device)
    e_enc = _tokenize(essay,    tokenizer, device)

    return {
        "q_input_ids":      q_enc["input_ids"],
        "q_attention_mask": q_enc["attention_mask"],
        "e_input_ids":      e_enc["input_ids"],
        "e_attention_mask": e_enc["attention_mask"],
    }


# ------------------------------------------------------------------
# Postprocessing
# ------------------------------------------------------------------

def round_to_band(score: float) -> float:
    """Clamps to [1, 9] and rounds to nearest 0.5."""
    score = max(1.0, min(9.0, score))
    return round(score * 2) / 2


def postprocess(raw_output: torch.Tensor) -> dict:
    """
    Converts (1, 4) model output to human-readable band scores.
    Overall is computed deterministically, not predicted.
    """
    scores = raw_output.squeeze(0).detach().tolist()   # (4,)

    result = {}
    for i, col in enumerate(LABEL_COLUMNS):
        result[col] = round_to_band(scores[i])

    sub_avg = sum(result.values()) / 4
    result["Overall"] = round_to_band(sub_avg)

    return result


# ------------------------------------------------------------------
# Predict
# ------------------------------------------------------------------

def predict(
    question:  str,
    essay:     str,
    model:     BandItScorer,
    tokenizer,
    device:    torch.device,
) -> dict:
    """
    Scores one IELTS essay.

    Calls the v4 dual-encoder forward pass:
        model(q_input_ids, q_mask, e_input_ids, e_mask)
    """
    inputs = preprocess(question, essay, tokenizer, device)

    with torch.no_grad():
        raw_output = model(
            inputs["q_input_ids"],
            inputs["q_attention_mask"],
            inputs["e_input_ids"],
            inputs["e_attention_mask"],
        )  # (1, 4)

    return postprocess(raw_output)


# ------------------------------------------------------------------
# BandItInferenceEngine
# ------------------------------------------------------------------

class BandItInferenceEngine:
    """
    Load-once wrapper for server deployment.

    Default checkpoint is best_model_v4.pt.
    Pass checkpoint_path="checkpoints/best_model.pt" to use v3 instead.
    """

    def __init__(self, checkpoint_path: str = "checkpoints/best_model_v4.pt"):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = load_model(checkpoint_path, self.device)
        self.tokenizer = load_tokenizer()
        print(f"[BandItInferenceEngine] ready.")

    def score(self, question: str, essay: str) -> dict:
        return predict(question, essay, self.model, self.tokenizer, self.device)

    def score_with_metadata(self, question: str, essay: str) -> dict:
        word_count  = len(essay.split())
        token_count = len(self.tokenizer.encode(essay))
        truncated   = token_count > MAX_LENGTH

        scores = self.score(question, essay)

        return {
            "scores":          scores,
            "word_count":      word_count,
            "token_count":     token_count,
            "truncated":       truncated,
            "truncation_note": "Essay exceeded 512 tokens and was truncated." if truncated else None,
        }


# ------------------------------------------------------------------
# Smoke test
# python src/inference.py
# ------------------------------------------------------------------

if __name__ == "__main__":

    CHECKPOINT = "checkpoints/best_model_v4.pt"

    if not os.path.exists(CHECKPOINT):
        print(f"[inference] checkpoint not found at {CHECKPOINT}")
        print(f"[inference] run train.py first to generate {CHECKPOINT}")
        exit(1)

    TEST_QUESTION = (
        "Some people think that universities should provide graduates with the knowledge "
        "and skills needed in the workplace. Others think that the true function of a "
        "university is to give access to knowledge for its own sake, regardless of whether "
        "the course is useful to an employer. What, in your opinion, should be the main "
        "function of a university?"
    )

    TEST_ESSAY = (
        "Universities play a pivotal role in shaping the intellectual and professional "
        "landscape of society. While some argue that higher education should primarily "
        "serve as a vocational training ground, I firmly believe that universities should "
        "balance both the pursuit of knowledge for its own sake and practical workplace "
        "preparation.\n\n"
        "On one hand, the acquisition of knowledge independent of its immediate utility "
        "has historically been the cornerstone of academic institutions. Critical thinking, "
        "philosophical inquiry, and theoretical research have produced some of humanity's "
        "greatest intellectual achievements. A student studying ancient history or pure "
        "mathematics may not find direct employment correlates, yet these disciplines "
        "foster analytical skills and cultural awareness that are invaluable in any "
        "professional context.\n\n"
        "On the other hand, the economic realities of modern society cannot be ignored. "
        "With tuition costs rising and graduate employment increasingly competitive, "
        "students reasonably expect their degrees to provide tangible career advantages. "
        "Universities that fail to equip graduates with relevant technical skills risk "
        "producing cohorts ill-prepared for contemporary workplaces.\n\n"
        "In conclusion, the most effective universities are those that integrate rigorous "
        "academic inquiry with practical skill development. Neither approach alone is "
        "sufficient — true education cultivates both the mind and the professional."
    )

    print("=" * 50)
    print("inference.py smoke test — v4")
    print("=" * 50)

    print("\n--- Test 1: functional API ---")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = load_model(CHECKPOINT, device)
    tokenizer = load_tokenizer()
    scores    = predict(TEST_QUESTION, TEST_ESSAY, model, tokenizer, device)

    print("\nPredicted band scores:")
    for criterion, score in scores.items():
        print(f"  {criterion:<25} {score}")

    print("\n--- Test 2: BandItInferenceEngine ---")
    engine = BandItInferenceEngine(CHECKPOINT)
    result = engine.score_with_metadata(TEST_QUESTION, TEST_ESSAY)

    print(f"\n  word count:   {result['word_count']}")
    print(f"  token count:  {result['token_count']}")
    print(f"  truncated:    {result['truncated']}")
    print("\n  scores:")
    for criterion, score in result["scores"].items():
        print(f"    {criterion:<25} {score}")

    print("\n--- Test 3: determinism ---")
    scores_a = engine.score(TEST_QUESTION, TEST_ESSAY)
    scores_b = engine.score(TEST_QUESTION, TEST_ESSAY)
    assert scores_a == scores_b, "predictions not deterministic!"
    print("  same input → same output ✓")

    print("\n--- Test 4: question sensitivity ---")
    # TR should differ when question changes, CC/LR/RA should be identical
    # (essay encoding is the same — only TR head sees different input)
    DIFFERENT_QUESTION = "Do you agree that technology has made modern life easier?"
    scores_diff_q = engine.score(DIFFERENT_QUESTION, TEST_ESSAY)

    print("  Original question scores:")
    for k, v in scores.items():
        print(f"    {k:<25} {v}")
    print("  Different question scores:")
    for k, v in scores_diff_q.items():
        print(f"    {k:<25} {v}")
    print("  (TR should differ, CC/LR/RA should be identical)")

    print("\n" + "=" * 50)
    print("All checks passed. inference.py v4 ready.")
    print("=" * 50)