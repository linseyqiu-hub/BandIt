import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/ielts_labeled.csv")

# only look at labeled rows
labeled = df[df["Task_Response"].notna()].copy()
print(f"=== Labeled rows: {len(labeled)} ===\n")

# === Distributions ===
score_cols = ["Overall", "Task_Response", "Coherence_Cohesion", "Lexical_Resource", "Range_Accuracy"]

print("=== Score Distributions ===")
for col in score_cols:
    print(f"\n{col}:")
    print(f"  mean={labeled[col].mean():.2f}  std={labeled[col].std():.2f}  "
          f"min={labeled[col].min():.1f}  max={labeled[col].max():.1f}")
    print(f"  values: {sorted(labeled[col].unique())}")

# === Correlations ===
print("\n=== Correlations with Overall ===")
for col in score_cols[1:]:
    print(f"  Overall ↔ {col}: {labeled['Overall'].corr(labeled[col]):.3f}")

# === Value counts — spot the 6.5/6.0 pattern ===
print("\n=== Task_Response value counts ===")
print(labeled["Task_Response"].value_counts().sort_index())

# === Sample rows — visual check ===
print("\n=== Sample: Overall vs Sub-scores ===")
print(labeled[["Overall"] + score_cols[1:]].head(20).to_string())

# === Plot ===
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, col in enumerate(score_cols):
    labeled[col].hist(bins=15, ax=axes[idx], edgecolor='black')
    axes[idx].set_title(col)
    axes[idx].set_xlabel("Band Score")
    axes[idx].set_ylabel("Count")
    axes[idx].axvline(labeled[col].mean(), color='red', linestyle='--', label='mean')
    axes[idx].legend()

# === Scatter: Overall vs each sub-score ===
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
axes2 = axes2.flatten()

for idx, col in enumerate(score_cols[1:]):
    axes2[idx].scatter(labeled["Overall"], labeled[col], alpha=0.5)
    axes2[idx].set_xlabel("Overall")
    axes2[idx].set_ylabel(col)
    axes2[idx].set_title(f"Overall vs {col} (r={labeled['Overall'].corr(labeled[col]):.2f})")
    
    # trend line
    z = np.polyfit(labeled["Overall"], labeled[col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(labeled["Overall"].min(), labeled["Overall"].max(), 100)
    axes2[idx].plot(x_line, p(x_line), "r--")

plt.tight_layout()
fig.savefig("notebooks/score_distributions_50.png")
fig2.savefig("notebooks/score_correlations_50.png")
print("\nPlots saved!")