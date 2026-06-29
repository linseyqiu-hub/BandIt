# BandIt

An AI-powered IELTS preparation platform. Currently under active development.

---

## What It Does

BandIt is being built as an end-to-end IELTS preparation platform — not just a scorer, but a full feedback and practice system.

**Essay Scoring** *(available)*
Scores IELTS writing submissions across the four official criteria and derives an Overall band score per the official IELTS standard.

**Examiner-Style Feedback** *(available)*
Generates detailed, criterion-aligned written feedback grounded in the predicted scores.

**Speaking Assessment** *(planned)*
Extends the platform to IELTS Speaking evaluation.

> BandIt is under active construction. Features marked *in progress* or *planned* are not yet available.

---


## Honest Limitations

**Dataset size.** The current model is trained on a limited dataset centered around band 6–7. Performance is strongest in this range and weaker at the extremes (bands 1–4 and 8–9).

**Synthetic sub-score labels.** Only Overall band scores in the training data come from real human examiners. The four sub-criterion labels are AI-generated proxies anchored to Overall, with correlations of 0.96–0.99 (real IELTS sub-score correlations are 0.75–0.85). This is the primary bottleneck for sub-score accuracy and is the target improvement when real examiner data becomes available.

**Regression to the mean.** Because training data is centered around band 6–7, the model under-penalises essays at the extremes — particularly very weak essays. It correctly identifies relative quality differences across criteria but compresses absolute scores toward the middle of the band range.

**No question context in v1.** The first model version scored essays without grounding in the question prompt. This has been corrected in the current version — question and essay are both provided as input.

**CPU inference only.** No GPU required. Suitable for demo and small-scale use.

---



## Acknowledgements

Base model: [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)
