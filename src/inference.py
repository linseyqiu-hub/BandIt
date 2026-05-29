import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import DebertaV2Tokenizer

from model import BandItScorer


# ------------------------------------------------------------------
# Constants — must match dataset.py exactly
# ------------------------------------------------------------------

MODEL_NAME   = "microsoft/deberta-v3-base"
MAX_LENGTH   = 512
LABEL_COLUMNS = [
    "Task_Response",
    "Coherence_Cohesion",
    "Lexical_Resource",
    "Range_Accuracy",
]


# ------------------------------------------------------------------
# Load phase — call once at server startup, reuse forever
# ------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device):
    """
    Loads BandItScorer architecture and fills it with saved weights.

    Steps:
        1. Instantiate BandItScorer with pretrained=False (architecture only)
        2. Load checkpoint dict from disk
        3. Load saved weights into model via load_state_dict
        4. Switch to eval mode — disables dropout for deterministic predictions
        5. Move to device

    Args:
        checkpoint_path: path to best_model.pt
        device:          torch.device("cpu") or torch.device("cuda")

    Returns:
        model: BandItScorer ready for inference
    """
    print(f"[inference] loading model from {checkpoint_path} ...")

    # Instantiate architecture — no pretrained weights yet, just the blueprint
    # pretrained=False because we're about to load our own fine-tuned weights
    # dropout value doesn't matter here — model.eval() disables dropout anyway
    model = BandItScorer(pretrained=False, dropout=0.0)

    # Load the checkpoint dict saved by train.py's save_checkpoint()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Fill the architecture with the fine-tuned weights
    model.load_state_dict(checkpoint["model_state"])

    # CRITICAL: switches off dropout
    # Without this, every forward pass gives different predictions
    # because dropout randomly zeroes neurons each time
    model.eval()

    # Move all tensors to the correct device
    model.to(device)

    epoch   = checkpoint.get("epoch",   "unknown")
    val_mae = checkpoint.get("val_mae", "unknown")
    print(f"[inference] loaded checkpoint — epoch {epoch}, val MAE {val_mae:.4f}")
    print(f"[inference] model ready on {device}")

    return model


def load_tokenizer():
    """
    Loads the DeBERTa tokenizer.

    Must be the same tokenizer used in dataset.py during training.
    Call once at startup and reuse.
    """
    print(f"[inference] loading tokenizer: {MODEL_NAME} ...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    print(f"[inference] tokenizer ready.")
    return tokenizer


# ------------------------------------------------------------------
# Preprocessing — must be identical to dataset.py __getitem__
# ------------------------------------------------------------------

def preprocess(question: str, essay: str, tokenizer, device: torch.device) -> dict:
    """
    Converts raw text into model-ready tensors.

    CRITICAL: this must exactly mirror what dataset.py __getitem__ does.
    Any difference in format, max_length, padding, or truncation will
    cause the model to receive token sequences it was never trained on,
    producing garbage predictions.

    Training format (dataset.py):
        text = f"Question: {question}\\n\\nEssay: {essay}"
        tokenizer(text, max_length=512, padding="max_length", truncation=True)

    Args:
        question:  IELTS question string
        essay:     candidate essay string
        tokenizer: loaded DebertaV2Tokenizer
        device:    torch.device to move tensors to

    Returns:
        dict with input_ids (1, 512) and attention_mask (1, 512)
        batch dim of 1 because model expects batched input
    """
    # Combine question and essay — identical format to training
    text = f"Question: {question}\n\nEssay: {essay}"

    # Tokenize — identical settings to dataset.py
    encoded = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",      # returns PyTorch tensors
    )

    # Move to device and return
    # Note: we do NOT squeeze here — model expects (batch, 512)
    # During training DataLoader added the batch dim; here we keep it as (1, 512)
    return {
        "input_ids":      encoded["input_ids"].to(device),       # (1, 512)
        "attention_mask": encoded["attention_mask"].to(device),  # (1, 512)
    }


# ------------------------------------------------------------------
# Postprocessing
# ------------------------------------------------------------------

def round_to_band(score: float) -> float:
    """
    Rounds a raw model output to the nearest valid IELTS band.

    IELTS scores are always awarded in 0.5 increments: 1.0, 1.5, 2.0, ..., 9.0
    The model outputs raw floats like 6.734 — these need to be rounded.

    Clamping to [1.0, 9.0] handles edge cases where the model
    predicts slightly outside the valid band range.

    Example:
        6.734 → round(6.734 * 2) / 2 = round(13.468) / 2 = 13 / 2 = 6.5
        8.923 → clamp → 9.0
        0.812 → clamp → 1.0
    """
    score = max(1.0, min(9.0, score))   # clamp to valid range
    return round(score * 2) / 2         # round to nearest 0.5


def postprocess(raw_output: torch.Tensor) -> dict:
    """
    Converts raw model output tensor to human-readable band scores.

    Args:
        raw_output: tensor of shape (1, 5) — model's raw predictions

    Returns:
        dict mapping criterion name → rounded band score
        e.g. {"Overall": 6.5, "Task_Response": 6.5, ...}
    """
    # Detach from computation graph and convert to Python floats
    scores = raw_output.squeeze(0).detach().tolist()  # now (4,) not (5,)

    result = {}
    for i, col in enumerate(LABEL_COLUMNS):
        result[col] = round_to_band(scores[i])

    # Overall is deterministic — compute it, don't predict it
    sub_avg = sum(result.values()) / 4
    result["Overall"] = round_to_band(sub_avg)

    return result


# ------------------------------------------------------------------
# Predict — the main function called per request
# ------------------------------------------------------------------

def predict(
    question:  str,
    essay:     str,
    model:     BandItScorer,
    tokenizer,
    device:    torch.device,
) -> dict:
    """
    Scores one IELTS essay across 5 criteria.

    This is the function FastAPI will call for every /score request.
    Model and tokenizer are passed in — they were loaded once at startup.

    Args:
        question:  IELTS task question
        essay:     candidate's essay response
        model:     loaded BandItScorer in eval mode
        tokenizer: loaded DebertaV2Tokenizer
        device:    torch.device

    Returns:
        dict of band scores, e.g.:
        {
            "Overall":            6.5,
            "Task_Response":      6.5,
            "Coherence_Cohesion": 6.0,
            "Lexical_Resource":   6.5,
            "Range_Accuracy":     7.0,
        }
    """
    # Preprocess — text → tensors
    inputs = preprocess(question, essay, tokenizer, device)

    # Forward pass
    # torch.no_grad() — tells PyTorch not to build the computation graph
    # During training we need the graph for backprop
    # During inference we never backprop — no_grad() saves memory and time
    with torch.no_grad():
        raw_output = model(
            inputs["input_ids"],
            inputs["attention_mask"],
        )  # (1, 5)

    # Postprocess — raw floats → rounded band scores
    scores = postprocess(raw_output)

    return scores


# ------------------------------------------------------------------
# BandItInferenceEngine — load-once wrapper for server use
# ------------------------------------------------------------------

class BandItInferenceEngine:
    """
    Wraps model + tokenizer into a single object for server deployment.

    Usage in FastAPI:
        # at server startup (runs once)
        engine = BandItInferenceEngine("checkpoints/best_model.pt")

        # per request (runs thousands of times, model stays in RAM)
        scores = engine.score(question, essay)

    This pattern keeps the load phase separate from the predict phase,
    which is essential for production — loading DeBERTa takes ~3-5s,
    prediction takes ~0.5s. You never want to reload per request.
    """

    def __init__(self, checkpoint_path: str = "checkpoints/best_model.pt"):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = load_model(checkpoint_path, self.device)
        self.tokenizer = load_tokenizer()
        print(f"[BandItInferenceEngine] ready.")

    def score(self, question: str, essay: str) -> dict:
        """
        Public API — scores one essay.

        Args:
            question: IELTS task question string
            essay:    candidate essay string

        Returns:
            dict of band scores per criterion
        """
        return predict(question, essay, self.model, self.tokenizer, self.device)

    def score_with_metadata(self, question: str, essay: str) -> dict:
        """
        Extended version — returns scores + word count + truncation warning.
        Useful for the frontend to display extra context.
        """
        word_count  = len(essay.split())
        token_count = len(self.tokenizer.encode(essay))
        truncated   = token_count > MAX_LENGTH

        scores = self.score(question, essay)

        return {
            "scores":           scores,
            "word_count":       word_count,
            "token_count":      token_count,
            "truncated":        truncated,
            "truncation_note":  "Essay exceeded 512 tokens and was truncated." if truncated else None,
        }


# ------------------------------------------------------------------
# CLI smoke test — python src/inference.py
# ------------------------------------------------------------------

if __name__ == "__main__":

    CHECKPOINT = "checkpoints/best_model.pt"

    # Check checkpoint exists
    if not os.path.exists(CHECKPOINT):
        print(f"[inference] checkpoint not found at {CHECKPOINT}")
        print(f"[inference] run train.py first to generate best_model.pt")
        exit(1)

    # Sample question and essay for testing
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
    print("inference.py smoke test")
    print("=" * 50)

    # Test 1: functional API
    print("\n--- Test 1: functional API ---")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = load_model(CHECKPOINT, device)
    tokenizer = load_tokenizer()
    scores    = predict(TEST_QUESTION, TEST_ESSAY, model, tokenizer, device)

    print("\nPredicted band scores:")
    for criterion, score in scores.items():
        print(f"  {criterion:<25} {score}")

    # Test 2: engine API (what FastAPI will use)
    print("\n--- Test 2: BandItInferenceEngine ---")
    engine = BandItInferenceEngine(CHECKPOINT)
    result = engine.score_with_metadata(TEST_QUESTION, TEST_ESSAY)

    print(f"\n  word count:   {result['word_count']}")
    print(f"  token count:  {result['token_count']}")
    print(f"  truncated:    {result['truncated']}")
    if result["truncation_note"]:
        print(f"  note:         {result['truncation_note']}")
    print("\n  scores:")
    for criterion, score in result["scores"].items():
        print(f"    {criterion:<25} {score}")

    # Test 3: determinism — same input should always give same output
    print("\n--- Test 3: determinism check ---")
    scores_a = engine.score(TEST_QUESTION, TEST_ESSAY)
    scores_b = engine.score(TEST_QUESTION, TEST_ESSAY)
    assert scores_a == scores_b, "predictions are not deterministic — did you forget model.eval()?"
    print("  same input → same output ✓")

    print("\n" + "=" * 50)
    print("All checks passed. inference.py is ready.")
    print("=" * 50)