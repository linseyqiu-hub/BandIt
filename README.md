# BandIt

An AI-powered IELTS preparation platform. Currently under active development.

---

## What It Does

BandIt is being built as an end-to-end IELTS preparation platform — not just a scorer, but a full feedback and practice system.

**Essay Scoring** *(available)*
Scores IELTS writing submissions across the four official criteria and derives an Overall band score per the official IELTS standard.

**Examiner-Style Feedback** *(in progress)*
Generates detailed, criterion-aligned written feedback grounded in the predicted scores.

**Speaking Assessment** *(in progress)*
Extends the platform to IELTS Speaking evaluation.

> BandIt is under active construction. Features marked *in progress* are not yet available.

---

## Tech Stack

- **Scoring model:** fine-tuned transformer (local inference)
- **Feedback pipeline:** RAG + Claude API
- **Backend:** FastAPI *(in progress)*
- **Frontend:** React *(in progress)*
- **Deployment:** HuggingFace Spaces + Vercel *(in progress)*

---

## Honest Limitations

**Dataset size.** The current model is trained on a limited dataset centered around band 6–7. Performance is strongest in this range and weaker at the extremes (bands 1–4 and 8–9).

**Synthetic sub-score labels.** Only Overall band scores in the training data come from real human examiners. The four sub-criterion labels are AI-generated proxies. This is the primary bottleneck for sub-score accuracy and is the target improvement when real examiner data becomes available.

**No question context in v1.** The first model version scored essays without grounding in the question prompt. This has been corrected in the current version — question and essay are both provided as input.

**CPU inference only.** No GPU required. Suitable for demo and small-scale use.

---

## Status

| Component | Status |
|---|---|
| Dataset pipeline | ✅ Complete |
| Scoring model (Phase 1) | ✅ Complete |
| Inference engine | ✅ Complete |
| Feedback generation (Phase 2) | 🔧 In progress |
| Backend API | 🔧 In progress |
| Frontend | 🔧 In progress |
| Deployment | 🔧 In progress |

---

## Quickstart

```bash
pip install torch transformers sentencepiece protobuf
```

```python
from src.inference import BandItInferenceEngine

engine = BandItInferenceEngine("checkpoints/best_model.pt")

result = engine.score_with_metadata(
    question="...",
    essay="..."
)

print(result["scores"])
# {
#   "Task_Response":      6.5,
#   "Coherence_Cohesion": 6.0,
#   "Lexical_Resource":   6.5,
#   "Range_Accuracy":     6.5,
#   "Overall":            6.5
# }
```

---

## Acknowledgements

Base model: [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)
