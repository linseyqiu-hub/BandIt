import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/ielts_labeled.csv")

# === Basic Info ===
print("=== Shape ===")
print(df.shape)

print("\n=== Columns ===")
print(df.columns.tolist())

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Score Distributions ===")
for col in ["Overall", "Task_Response", "Coherence_Cohesion", "Lexical_Resource", "Range_Accuracy"]:
    print(f"\n{col}:")
    print(df[col].describe())

print("\n=== Sample Examiner Comment ===")
print(df["Examiner_Commen"].iloc[0])

# === Correlation between scores ===
print("\n=== Score Correlations ===")
score_cols = ["Overall", "Task_Response", "Coherence_Cohesion", "Lexical_Resource", "Range_Accuracy"]
print(df[score_cols].corr())

# === Plot distributions ===
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, col in enumerate(score_cols):
    df[col].hist(bins=20, ax=axes[idx])
    axes[idx].set_title(col)
    axes[idx].set_xlabel("Band Score")
    axes[idx].set_ylabel("Count")

plt.tight_layout()
plt.savefig("notebooks/score_distributions.png")
print("\n=== Plot saved to notebooks/score_distributions.png ===")